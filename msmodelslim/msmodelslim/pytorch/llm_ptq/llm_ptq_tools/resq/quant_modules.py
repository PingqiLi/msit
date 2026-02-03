# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
ResQ Linear Quantizer for mixed-precision 4/8-bit quantization.

This module provides a specialized LinearQuantizer for ResQ that:
- Separates weights into high-precision (8-bit) and low-precision (4-bit) parts
- Saves dual weights and dual scales for each precision level
- Supports real int8 storage (not fake quantization)
- Supports CPU/NPU kernels without GPU dependency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs import (
    StatMinMaxObserver,
    fake_quantize,
    linear_quantization_params,
)


def symmetric_quantize_to_int8(tensor: torch.Tensor, bits: int, quant_dim: int = 1) -> tuple:
    """
    Symmetric quantization to int8 storage.

    Args:
        tensor: Input tensor to quantize
        bits: Number of bits for quantization (4 or 8)
        quant_dim: Dimension along which to compute scale (default: 1 for per-row)

    Returns:
        Tuple of (quantized_int8, scale)
        - quantized_int8: int8 tensor with values in range [-2^(bits-1), 2^(bits-1)-1]
        - scale: float32 scale tensor for dequantization

    Note:
        For 4-bit, values are in range [-8, 7], stored as int8
        For 8-bit, values are in range [-128, 127], stored as int8
    """
    # Compute symmetric range: max absolute value per row
    n = 2 ** (bits - 1) - 1  # e.g., 127 for 8-bit, 7 for 4-bit

    # Per-row max absolute value
    abs_max = tensor.abs().max(dim=quant_dim, keepdim=True)[0]

    # Compute scale: scale = max_abs / n
    # Add small epsilon to avoid division by zero
    scale = abs_max / n
    scale = torch.clamp(scale, min=1e-8)

    # Quantize: q = round(x / scale), clamped to [-n-1, n]
    quant_float = torch.round(tensor / scale)
    quant_float = torch.clamp(quant_float, -(n + 1), n)

    # Convert to int8 for storage
    quant_int8 = quant_float.to(torch.int8)

    return quant_int8, scale


class ResQWeightQuantizer(nn.Module):
    """
    Mixed-precision weight quantizer for ResQ.

    Separates weight tensor into high-precision (8-bit) and low-precision (4-bit) parts,
    quantizing each with symmetric quantization and storing as actual int8 tensors.
    """

    def __init__(
        self,
        high_bits: int = 8,
        low_bits: int = 4,
        high_fraction: float = 0.125,
        split_dim: int = 1,  # 1 for input projections, 0 for output projections
        logger=None,
        high_dim_override: int = None,
    ):
        super().__init__()
        self.high_bits = high_bits
        self.low_bits = low_bits
        self.high_fraction = high_fraction
        self.split_dim = split_dim  # dimension to split: 0 for rows, 1 for columns
        self.logger = logger
        self.high_dim_override = high_dim_override

        # Quantization parameters for each precision level (symmetric - no offset needed)
        self.high_weight_scale = None
        self.low_weight_scale = None

        # Quantized weights (int8 storage)
        self.high_weight = None
        self.low_weight = None

        # Dequantized weights (for inference)
        self.high_weight_dequant = None
        self.low_weight_dequant = None

        # Dimension info
        self.high_dim = None
        self.low_dim = None
        self.total_dim = None

        self.has_init_quant_para = False

    def compute_dimensions(self, out_features: int) -> None:
        """Compute the dimensions for high and low precision regions."""
        self.total_dim = out_features
        self.high_dim = int(self.high_fraction * out_features)
        self.low_dim = out_features - self.high_dim

    def quantize_weight(self, weight: torch.Tensor) -> tuple:
        """
        Quantize weight tensor with mixed precision using symmetric int8 storage.

        Args:
            weight: Weight tensor of shape [out_features, in_features]

        Returns:
            Tuple of (quantized_weight_for_inference, low_weight, high_weight)
            where low_weight and high_weight are int8 tensors
        """
        # Determine split dimension based on layer type
        # split_dim=1: split along in_features (columns) for input projections (q,k,v,up,gate)
        # split_dim=0: split along out_features (rows) for output projections (o,down)

        if self.split_dim == 1:
            # Split along in_features (columns)
            out_features, in_features = weight.shape
            self.total_dim = in_features
            if self.high_dim_override is not None:
                self.high_dim = self.high_dim_override
            else:
                self.high_dim = int(self.high_fraction * in_features)
            self.low_dim = in_features - self.high_dim

            weight_low = weight[:, :self.low_dim]   # [out_features, low_dim]
            weight_high = weight[:, self.low_dim:]  # [out_features, high_dim]
            cat_dim = 1
        else:
            # Split along out_features (rows)
            out_features, in_features = weight.shape
            self.total_dim = out_features
            if self.high_dim_override is not None:
                self.high_dim = self.high_dim_override
            else:
                self.high_dim = int(self.high_fraction * out_features)
            self.low_dim = out_features - self.high_dim

            weight_low = weight[:self.low_dim, :]   # [low_dim, in_features]
            weight_high = weight[self.low_dim:, :]  # [high_dim, in_features]
            cat_dim = 0

        # Per-row quantization (dim=1)
        quant_dim = 1

        # Quantize low precision part (4-bit) using symmetric int8 storage
        weight_low_dequant = None
        if self.low_dim > 0:
            self.low_weight, self.low_weight_scale = symmetric_quantize_to_int8(
                weight_low, self.low_bits, quant_dim
            )
            # Compute dequantized version for inference (store as original dtype to save memory)
            weight_low_dequant = (self.low_weight.float() * self.low_weight_scale).to(weight.dtype)
            self.low_weight_dequant = weight_low_dequant

        # Quantize high precision part (8-bit) using symmetric int8 storage
        weight_high_dequant = None
        if self.high_dim > 0:
            self.high_weight, self.high_weight_scale = symmetric_quantize_to_int8(
                weight_high, self.high_bits, quant_dim
            )
            # Compute dequantized version for inference (store as original dtype to save memory)
            weight_high_dequant = (self.high_weight.float() * self.high_weight_scale).to(weight.dtype)
            self.high_weight_dequant = weight_high_dequant

        # Store actual high_fraction for later retrieval (for safetensor)
        self.actual_high_fraction = self.high_dim / self.total_dim if self.total_dim > 0 else 0.0

        # Combine dequantized weights for inference
        if self.low_dim > 0 and self.high_dim > 0:
            quantized_weight = torch.cat([weight_low_dequant, weight_high_dequant], dim=cat_dim)
        elif self.high_dim > 0:
            quantized_weight = weight_high_dequant
        else:
            quantized_weight = weight_low_dequant

        self.has_init_quant_para = True
        return quantized_weight, self.low_weight, self.high_weight

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """Forward pass with weight quantization."""
        if not self.has_init_quant_para:
            quantized_weight, _, _ = self.quantize_weight(weight)
            return quantized_weight

        # Use cached dequantized weights for inference
        # When use_real_quant=True, low_weight/high_weight are int8,
        # so we must use the dequantized versions for float computation
        cat_dim = 1 if self.split_dim == 1 else 0

        # Get dequantized weights (or original weights if not using real quant)
        low_w = self.low_weight_dequant if self.low_weight_dequant is not None else self.low_weight
        high_w = self.high_weight_dequant if self.high_weight_dequant is not None else self.high_weight

        if low_w is not None and high_w is not None:
            return torch.cat([low_w, high_w], dim=cat_dim)
        elif high_w is not None:
            return high_w
        else:
            return low_w


class ResQActQuantizer(nn.Module):
    """
    Mixed-precision activation quantizer for ResQ.

    Applies different quantization to different activation regions
    based on eigenvalue-derived importance.
    """

    def __init__(
        self,
        high_bits: int = 8,
        low_bits: int = 4,
        high_fraction: float = 0.125,
        is_sym: bool = False,
        is_dynamic: bool = True,
        logger=None,
    ):
        super().__init__()
        self.high_bits = high_bits
        self.low_bits = low_bits
        self.high_fraction = high_fraction
        self.is_sym = is_sym
        self.is_dynamic = is_dynamic
        self.logger = logger

        # Static quantization parameters (for non-dynamic mode)
        self.high_input_scale = None
        self.high_input_offset = None
        self.low_input_scale = None
        self.low_input_offset = None

        self.observer = None
        self.is_calib = True
        self.is_enable = True

    def init_observer(self):
        """Initialize min-max observer for calibration."""
        self.observer = StatMinMaxObserver(self.low_bits, True, self.is_sym)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with mixed-precision activation quantization."""
        if not self.is_enable:
            return x

        hidden_dim = x.shape[-1]
        high_dim = int(self.high_fraction * hidden_dim)
        low_dim = hidden_dim - high_dim

        # Split activation into regions
        x_low = x[..., :low_dim]
        x_high = x[..., low_dim:]

        if self.is_dynamic:
            # Per-token dynamic quantization
            if low_dim > 0:
                low_min = x_low.min(dim=-1, keepdim=True)[0]
                low_max = x_low.max(dim=-1, keepdim=True)[0]
                low_scale, low_offset = linear_quantization_params(
                    self.low_bits, low_min, low_max,
                    integral_zero_point=True, q_signed=True, sym=self.is_sym
                )
                _, x_low = fake_quantize(x_low, low_scale, low_offset, self.low_bits, is_signed=True)

            if high_dim > 0:
                high_min = x_high.min(dim=-1, keepdim=True)[0]
                high_max = x_high.max(dim=-1, keepdim=True)[0]
                high_scale, high_offset = linear_quantization_params(
                    self.high_bits, high_min, high_max,
                    integral_zero_point=True, q_signed=True, sym=self.is_sym
                )
                _, x_high = fake_quantize(x_high, high_scale, high_offset, self.high_bits, is_signed=True)
        else:
            # Static quantization
            if self.is_calib and self.observer is not None:
                self.observer(x)

            if self.low_input_scale is not None and low_dim > 0:
                _, x_low = fake_quantize(x_low, self.low_input_scale, self.low_input_offset,
                                         self.low_bits, is_signed=True)

            if self.high_input_scale is not None and high_dim > 0:
                _, x_high = fake_quantize(x_high, self.high_input_scale, self.high_input_offset,
                                          self.high_bits, is_signed=True)

        # Combine quantized activations
        if low_dim > 0 and high_dim > 0:
            return torch.cat([x_low, x_high], dim=-1)
        elif high_dim > 0:
            return x_high
        else:
            return x_low


class LinearResQQuantizer(nn.Module):
    """
    Linear layer quantizer for ResQ with mixed-precision 4/8-bit support.

    This quantizer:
    - Separates weights into high-precision (8-bit) and low-precision (4-bit) parts
    - Applies mixed-precision activation quantization
    - Saves dual weights and dual scales for each precision level
    - Uses real int8 storage with symmetric quantization
    """

    def __init__(
        self,
        cfg=None,
        logger=None,
        high_bits: int = 8,
        low_bits: int = 4,
        high_fraction: float = 0.125,
        split_dim: int = 1,  # 1 for input projections (q,k,v,up,gate), 0 for output projections (o,down)
        high_dim_override: int = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.logger = logger
        self.high_bits = high_bits
        self.low_bits = low_bits
        self.high_fraction = high_fraction
        self.split_dim = split_dim
        self.high_dim_override = high_dim_override

        # Linear layer parameters
        self.in_features = None
        self.out_features = None
        self.weight = None
        self.bias = None

        # Quantizers
        self.quant_weight = ResQWeightQuantizer(
            high_bits=high_bits,
            low_bits=low_bits,
            high_fraction=high_fraction,
            split_dim=split_dim,
            logger=logger,
            high_dim_override=high_dim_override,
        )

        self.quant_input = ResQActQuantizer(
            high_bits=high_bits,
            low_bits=low_bits,
            high_fraction=high_fraction,
            is_sym=False,  # Asymmetric for activations
            is_dynamic=getattr(cfg, 'is_dynamic', True) if cfg else True,
            logger=logger,
        )

        self.is_calib = True
        self.is_enable = True

    def set_param(self, linear: nn.Linear) -> None:
        """Set parameters from a linear layer."""
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = nn.Parameter(linear.weight.data.clone())
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone())
        else:
            self.bias = None

    def enable_quantization(self, name=None, range_param=None):
        """Enable quantization."""
        self.is_enable = True

    def disable_quantization(self, name=None):
        """Disable quantization."""
        self.is_enable = False

    def disable_calib(self):
        """Disable calibration mode."""
        self.is_calib = False
        self.quant_input.is_calib = False

    def enable_calib(self):
        """Enable calibration mode."""
        self.is_calib = True
        self.quant_input.is_calib = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with mixed-precision quantization."""
        # Quantize activation
        if self.is_enable:
            x = self.quant_input(x)

        # Quantize weight
        if self.is_enable:
            weight = self.quant_weight(self.weight)
            # After first quantization, delete original weight to save memory
            # The quantized weights are stored in self.quant_weight
            if self.quant_weight.has_init_quant_para and self.weight is not None:
                del self.weight
                self.weight = None
        else:
            weight = self.weight

        return F.linear(x, weight, self.bias)

    def get_quant_weights(self) -> dict:
        """
        Get quantized weights and scales for saving.

        Returns:
            Dictionary with:
            - 'weight_low': Low-precision quantized weight (4-bit as int8)
            - 'weight_high': High-precision quantized weight (8-bit as int8)
            - 'scale_low': Scale for low-precision weight
            - 'scale_high': Scale for high-precision weight

        Note: Uses symmetric quantization (no offsets needed).
        """
        if not self.quant_weight.has_init_quant_para:
            self.quant_weight.quantize_weight(self.weight)

        result = {}

        if self.quant_weight.low_weight is not None:
            result['weight_low'] = self.quant_weight.low_weight.cpu()
            result['scale_low'] = self.quant_weight.low_weight_scale.cpu()

        if self.quant_weight.high_weight is not None:
            result['weight_high'] = self.quant_weight.high_weight.cpu()
            result['scale_high'] = self.quant_weight.high_weight_scale.cpu()

        # Add high_fraction to result for safetensor
        if hasattr(self.quant_weight, 'actual_high_fraction'):
            result['high_fraction'] = self.quant_weight.actual_high_fraction
        else:
            result['high_fraction'] = self.quant_weight.high_fraction

        return result


class LinearW8A8DynamicQuantizer(nn.Module):
    """
    W8A8 dynamic quantizer for use in ResQ context (after rotation fusion).

    - Per-channel symmetric weight quantization (int8)
    - Per-token dynamic activation quantization at runtime
    """

    def __init__(self, cfg=None, logger=None):
        super().__init__()
        self.cfg = cfg
        self.logger = logger
        self.weight_scale = None
        self.weight_quant = None  # int8
        self.has_init_quant_para = False

        # Linear layer parameters
        self.in_features = None
        self.out_features = None
        self.weight = None
        self.bias = None

    def set_param(self, linear: nn.Linear) -> None:
        """Set parameters from a linear layer."""
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = nn.Parameter(linear.weight.data.clone())
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone())
        else:
            self.bias = None

    def quantize_weight(self) -> None:
        """Quantize weight to int8 with per-channel symmetric quantization."""
        n = 127  # int8 symmetric range
        abs_max = self.weight.abs().max(dim=1, keepdim=True)[0]
        scale = abs_max / n
        scale = torch.clamp(scale, min=1e-8)

        quant_float = torch.round(self.weight / scale)
        quant_float = torch.clamp(quant_float, -128, 127)

        self.weight_scale = scale.squeeze(-1)
        self.weight_quant = quant_float.to(torch.int8)
        self.has_init_quant_para = True

    def get_quant_weights(self) -> dict:
        """Return quantized weights in w8a8_dynamic format."""
        if not self.has_init_quant_para:
            self.quantize_weight()
        # weight_scale: reshape to [out_features, 1] for 2D per-channel format
        # weight_offset: zeros tensor for symmetric quantization
        weight_scale_2d = self.weight_scale.unsqueeze(-1)  # [out_features] -> [out_features, 1]
        weight_offset = torch.zeros_like(weight_scale_2d, dtype=torch.int8)  # zeros for symmetric
        return {
            'weight': self.weight_quant.cpu(),
            'weight_scale': weight_scale_2d.cpu(),
            'weight_offset': weight_offset.cpu(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with weight dequantization."""
        if not self.has_init_quant_para:
            self.quantize_weight()
        weight_dequant = self.weight_quant.float() * self.weight_scale.unsqueeze(1)
        return F.linear(x, weight_dequant.to(x.dtype), self.bias)


def add_resq_quantizers(model: nn.Module, cfg=None, logger=None,
                        high_bits: int = 8, low_bits: int = 4,
                        high_fraction: float = 0.125,
                        skip_names: list = None,
                        ratio_dict: dict = None,
                        splits_dict: dict = None,
                        mix_cfg: dict = None) -> nn.Module:
    """
    Replace linear layers with ResQ quantizers.

    Args:
        model: The transformer model
        cfg: Quantization configuration
        logger: Logger instance
        high_bits: Bits for high precision (default: 8)
        low_bits: Bits for low precision (default: 4)
        high_fraction: Default fraction of dimensions at high precision
        skip_names: Layer names to skip
        ratio_dict: Optional dictionary of per-layer/per-transform ratios.
                   Keys are in format: 'Ua', 'layer.{i}.Ub', 'layer.{i}.Uc', 'layer.{i}.Ud'
                   If None or key not found, uses high_fraction.
        splits_dict: Optional dictionary of scaled dimension splits for actual weight dimensions.
                    Keys are same format as ratio_dict, values are (low_dim, high_dim) tuples.
                    If provided, overrides ratio-based dimension calculation for alignment.
        mix_cfg: Optional dictionary mapping layer name patterns to quant types.
                Keys are fnmatch patterns, values are 'resq', 'w8a8_dynamic', or 'float'.

    Returns:
        Model with ResQ quantizers
    """
    import fnmatch

    skip_names = skip_names or []
    mix_cfg = mix_cfg or {}

    def get_layer_quant_type(name: str) -> str:
        """Determine quant type from mix_cfg patterns."""
        if not mix_cfg:
            return 'resq'
        # Check exact match first
        if name in mix_cfg:
            return mix_cfg[name].lower()
        # Check pattern matches
        for pattern, quant_type in mix_cfg.items():
            if fnmatch.fnmatchcase(name, pattern):
                return quant_type.lower()
        return 'resq'  # default

    # All projections split along in_features (dim=1, the last dimension)
    # This is because ResQ rotations are applied to the INPUT of each linear layer
    # - q,k,v,up,gate projections: input is hidden_size, split on hidden_size
    # - o_proj: input is num_heads * head_dim, split on that dimension
    # - down_proj: input is intermediate_size, split on intermediate_size

    def _set_module(ori_mod, submodule_key, module):
        tokens = submodule_key.split('.')
        sub_tokens = tokens[:-1]
        cur_mod = ori_mod
        for s in sub_tokens:
            cur_mod = getattr(cur_mod, s)
        setattr(cur_mod, tokens[-1], module)

    for name, mod in list(model.named_modules()):
        if name in skip_names:
            continue
        if isinstance(mod, nn.Linear):
            # Determine quantization type from mix_cfg
            quant_type = get_layer_quant_type(name)

            if quant_type == 'float':
                # No quantization - keep original Linear layer
                continue

            elif quant_type == 'w8a8_dynamic':
                # W8A8 dynamic quantization
                quant_mod = LinearW8A8DynamicQuantizer(cfg=cfg, logger=logger)
                quant_mod.set_param(mod)
                _set_module(model, name, quant_mod)

            else:  # 'resq' - default ResQ mixed-precision quantization
                # All projections split along in_features (dim=1, the last dimension)
                # ResQ rotations are applied to inputs, so we split on input dimension
                split_dim = 1

                # Determine per-layer ratio and high_dim_override based on layer type
                layer_fraction = high_fraction  # Default
                high_dim_override = None
                ratio_key = None

                # Parse layer index from name (e.g., "model.layers.0.self_attn.q_proj")
                import re
                layer_match = re.search(r'layers\.(\d+)\.', name)
                layer_idx = int(layer_match.group(1)) if layer_match else None

                if layer_idx is not None:
                    # Map projection type to transform type
                    if any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj']):
                        # Uses Ua ratio (shared) or fallback to default
                        ratio_key = 'Ua'
                    elif 'o_proj' in name:
                        # Uses Ub ratio (per-layer)
                        ratio_key = f'layer.{layer_idx}.Ub'
                    elif 'down_proj' in name:
                        # Uses Ud ratio (per-layer)
                        ratio_key = f'layer.{layer_idx}.Ud'
                    else:
                        ratio_key = 'Ua'  # Default to Ua

                # Get ratio from ratio_dict if available
                if ratio_dict is not None and ratio_key is not None and ratio_key in ratio_dict:
                    layer_fraction = ratio_dict[ratio_key]

                # Get aligned high_dim from splits_dict if available
                if splits_dict is not None and ratio_key is not None and ratio_key in splits_dict:
                    _, high_dim = splits_dict[ratio_key]  # (low_dim, high_dim)
                    high_dim_override = high_dim

                quant_mod = LinearResQQuantizer(
                    cfg=cfg,
                    logger=logger,
                    high_bits=high_bits,
                    low_bits=low_bits,
                    high_fraction=layer_fraction,
                    split_dim=split_dim,
                    high_dim_override=high_dim_override,
                )
                quant_mod.set_param(mod)
                _set_module(model, name, quant_mod)

    return model
