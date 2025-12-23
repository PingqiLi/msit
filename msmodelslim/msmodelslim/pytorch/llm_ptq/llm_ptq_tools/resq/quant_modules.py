# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
ResQ Linear Quantizer for mixed-precision 4/8-bit quantization.

This module provides a specialized LinearQuantizer for ResQ that:
- Separates weights into high-precision (8-bit) and low-precision (4-bit) parts
- Saves dual weights and dual scales for each precision level
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


class ResQWeightQuantizer(nn.Module):
    """
    Mixed-precision weight quantizer for ResQ.

    Separates weight tensor into high-precision and low-precision parts,
    quantizing each with different bit-widths.
    """

    def __init__(
        self,
        high_bits: int = 8,
        low_bits: int = 4,
        high_fraction: float = 0.125,
        is_sym: bool = True,
        logger=None,
    ):
        super().__init__()
        self.high_bits = high_bits
        self.low_bits = low_bits
        self.high_fraction = high_fraction
        self.is_sym = is_sym
        self.logger = logger

        # Quantization parameters for each precision level
        self.high_weight_scale = None
        self.high_weight_offset = None
        self.low_weight_scale = None
        self.low_weight_offset = None

        # Quantized weights
        self.high_weight = None
        self.low_weight = None

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
        Quantize weight tensor with mixed precision.

        Args:
            weight: Weight tensor of shape [out_features, in_features]

        Returns:
            Tuple of (quantized_weight, high_weight, low_weight)
        """
        out_features = weight.shape[0]
        self.compute_dimensions(out_features)

        # Split weight into low and high precision regions
        # ResQ layout: [low_precision_dims | high_precision_dims]
        weight_low = weight[:self.low_dim, :]
        weight_high = weight[self.low_dim:, :]

        # Quantize low precision part (4-bit)
        if self.low_dim > 0:
            low_min = weight_low.min(dim=1, keepdim=True)[0]
            low_max = weight_low.max(dim=1, keepdim=True)[0]
            self.low_weight_scale, self.low_weight_offset = linear_quantization_params(
                self.low_bits, low_min, low_max,
                integral_zero_point=True, q_signed=True, sym=self.is_sym
            )

            _, weight_low_quant = fake_quantize(
                weight_low, self.low_weight_scale, self.low_weight_offset,
                self.low_bits, is_signed=True
            )
            self.low_weight = weight_low_quant.clone()

        # Quantize high precision part (8-bit)
        if self.high_dim > 0:
            high_min = weight_high.min(dim=1, keepdim=True)[0]
            high_max = weight_high.max(dim=1, keepdim=True)[0]
            self.high_weight_scale, self.high_weight_offset = linear_quantization_params(
                self.high_bits, high_min, high_max,
                integral_zero_point=True, q_signed=True, sym=self.is_sym
            )

            _, weight_high_quant = fake_quantize(
                weight_high, self.high_weight_scale, self.high_weight_offset,
                self.high_bits, is_signed=True
            )
            self.high_weight = weight_high_quant.clone()

        # Combine quantized weights
        if self.low_dim > 0 and self.high_dim > 0:
            quantized_weight = torch.cat([weight_low_quant, weight_high_quant], dim=0)
        elif self.high_dim > 0:
            quantized_weight = weight_high_quant
        else:
            quantized_weight = weight_low_quant

        self.has_init_quant_para = True
        return quantized_weight, self.low_weight, self.high_weight

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """Forward pass with weight quantization."""
        if not self.has_init_quant_para:
            quantized_weight, _, _ = self.quantize_weight(weight)
            return quantized_weight

        # Use cached quantized weights
        if self.low_weight is not None and self.high_weight is not None:
            return torch.cat([self.low_weight, self.high_weight], dim=0)
        elif self.high_weight is not None:
            return self.high_weight
        else:
            return self.low_weight


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
    """

    def __init__(
        self,
        cfg=None,
        logger=None,
        high_bits: int = 8,
        low_bits: int = 4,
        high_fraction: float = 0.125,
    ):
        super().__init__()
        self.cfg = cfg
        self.logger = logger
        self.high_bits = high_bits
        self.low_bits = low_bits
        self.high_fraction = high_fraction

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
            is_sym=getattr(cfg, 'w_sym', True) if cfg else True,
            logger=logger,
        )

        self.quant_input = ResQActQuantizer(
            high_bits=high_bits,
            low_bits=low_bits,
            high_fraction=high_fraction,
            is_sym=getattr(cfg, 'a_sym', False) if cfg else False,
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
        else:
            weight = self.weight

        return F.linear(x, weight, self.bias)

    def get_quant_weights(self) -> dict:
        """
        Get quantized weights and scales for saving.

        Returns:
            Dictionary with:
            - 'weight_low': Low-precision quantized weight (4-bit)
            - 'weight_high': High-precision quantized weight (8-bit)
            - 'scale_low': Scale for low-precision weight
            - 'scale_high': Scale for high-precision weight
            - 'offset_low': Offset for low-precision weight
            - 'offset_high': Offset for high-precision weight
        """
        if not self.quant_weight.has_init_quant_para:
            self.quant_weight.quantize_weight(self.weight)

        result = {}

        if self.quant_weight.low_weight is not None:
            result['weight_low'] = self.quant_weight.low_weight.cpu()
            result['scale_low'] = self.quant_weight.low_weight_scale.cpu()
            result['offset_low'] = self.quant_weight.low_weight_offset.cpu()

        if self.quant_weight.high_weight is not None:
            result['weight_high'] = self.quant_weight.high_weight.cpu()
            result['scale_high'] = self.quant_weight.high_weight_scale.cpu()
            result['offset_high'] = self.quant_weight.high_weight_offset.cpu()

        return result


def add_resq_quantizers(model: nn.Module, cfg=None, logger=None,
                        high_bits: int = 8, low_bits: int = 4,
                        high_fraction: float = 0.125,
                        skip_names: list = None) -> nn.Module:
    """
    Replace linear layers with ResQ quantizers.

    Args:
        model: The transformer model
        cfg: Quantization configuration
        logger: Logger instance
        high_bits: Bits for high precision (default: 8)
        low_bits: Bits for low precision (default: 4)
        high_fraction: Fraction of dimensions at high precision
        skip_names: Layer names to skip

    Returns:
        Model with ResQ quantizers
    """
    skip_names = skip_names or []

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
            quant_mod = LinearResQQuantizer(
                cfg=cfg,
                logger=logger,
                high_bits=high_bits,
                low_bits=low_bits,
                high_fraction=high_fraction,
            )
            quant_mod.set_param(mod)
            _set_module(model, name, quant_mod)

    return model
