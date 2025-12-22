# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted for msmodelslim from ResQ: https://github.com/facebookresearch/resq

"""
Mixed-precision activation quantizer for ResQ.

This module implements activation quantization with support for:
- Mixed-precision quantization (2-bit/4-bit/8-bit)
- Per-token symmetric/asymmetric quantization
- Straight-Through Estimator (STE) for gradients
- Groupwise quantization support
- Online Hadamard transform for outlier suppression
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional


def get_minq_maxq(bits: int, sym: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get quantization range based on bits and symmetry.

    Args:
        bits: Number of quantization bits
        sym: Whether to use symmetric quantization

    Returns:
        Tuple of (minq, maxq) tensors
    """
    if sym:
        maxq = torch.tensor(2 ** (bits - 1) - 1)
        minq = -maxq - 1
    else:
        maxq = torch.tensor(2 ** bits - 1)
        minq = torch.tensor(0)
    return minq, maxq


def asym_quant(x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor,
               maxq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric quantization."""
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero


def asym_dequant(q: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    """Asymmetric dequantization."""
    return scale * (q - zero)


def asym_quant_dequant(x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor,
                       maxq: torch.Tensor) -> torch.Tensor:
    """Asymmetric quantize-dequantize."""
    return asym_dequant(*asym_quant(x, scale, zero, maxq))


def sym_quant(x: torch.Tensor, scale: torch.Tensor,
              maxq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric quantization."""
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
    return q, scale


def sym_dequant(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Symmetric dequantization."""
    return scale * q


def sym_quant_dequant(x: torch.Tensor, scale: torch.Tensor,
                      maxq: torch.Tensor) -> torch.Tensor:
    """Symmetric quantize-dequantize."""
    return sym_dequant(*sym_quant(x, scale, maxq))


def stoch_round(tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies stochastic rounding to the elements of the input tensor.

    Args:
        tensor: Input tensor to round

    Returns:
        Stochastically rounded tensor
    """
    floor_values = tensor.floor()
    ceil_values = tensor.ceil()
    fractional_part = tensor - floor_values
    random_values = torch.rand_like(tensor)
    rounded_tensor = torch.where(
        random_values < fractional_part, ceil_values, floor_values
    )
    return rounded_tensor


class STEQuantize(torch.autograd.Function):
    """Symmetric quantization with Straight-Through Estimator."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor, maxq: torch.Tensor,
                stoch: bool = False):
        scale = scale.to(x.device)
        if stoch:
            q = torch.clamp(stoch_round(x / scale), -(maxq + 1), maxq)
        else:
            q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
        return scale * q

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: just pass the gradient through
        return grad_output, None, None, None


class AsymSTEQuantize(torch.autograd.Function):
    """Asymmetric quantization with Straight-Through Estimator."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor,
                maxq: torch.Tensor, stoch: bool = False):
        scale = scale.to(x.device)
        zero = zero.to(x.device)
        if stoch:
            q = torch.clamp(stoch_round(x / scale) + zero, 0, maxq)
        else:
            q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


class ActQuantizer(nn.Module):
    """
    Mixed-precision activation quantizer for ResQ.

    Supports per-token quantization with three precision tiers:
    - Low precision: low_bits_length dimensions at low_bits
    - Middle precision: remaining dimensions at bits
    - High precision: high_bits_length dimensions at high_bits

    The precision grouping is determined by eigenvalues from basis computation,
    with low variance channels quantized to lower bits.
    """

    def __init__(self) -> None:
        super(ActQuantizer, self).__init__()

        # Middle precision (default)
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(1))
        self.register_buffer("zero", torch.zeros(1))

        # High precision
        self.register_buffer("maxq_h", torch.tensor(0))
        self.register_buffer("scale_h", torch.zeros(1))
        self.register_buffer("zero_h", torch.zeros(1))

        # Low precision
        self.register_buffer("maxq_l", torch.tensor(0))
        self.register_buffer("scale_l", torch.zeros(1))
        self.register_buffer("zero_l", torch.zeros(1))

        # Configuration
        self.bits = 16
        self.high_bits = 16
        self.low_bits = 16
        self.high_bits_length = 0
        self.low_bits_length = 0
        self.groupsize = -1
        self.sym = False
        self.clip_ratio = 1.0
        self.stoch = False

    def free(self) -> None:
        """Free memory by clearing scale/zero buffers."""
        self.zero = None
        self.scale = None
        self.zero_h = None
        self.scale_h = None
        self.zero_l = None
        self.scale_l = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with mixed-precision quantization.

        Args:
            x: Input tensor of shape [..., hidden_dim]

        Returns:
            Quantized tensor with same shape as input
        """
        x_dtype = x.dtype

        if self.bits == 16:
            return x

        # Handle groupwise quantization
        if self.groupsize > 0:
            init_shape = x.shape
            x = x.reshape(
                x.shape[0], x.shape[1], x.shape[2] // self.groupsize, self.groupsize
            )

        # Split into precision tiers: [low | middle | high]
        low_dim, high_dim = self.low_bits_length, x.shape[-1] - self.high_bits_length
        x_l, x_m, x_h = x[..., :low_dim], x[..., low_dim:high_dim], x[..., high_dim:]

        if self.sym:
            # Symmetric quantization
            x_m = STEQuantize.apply(x_m, self.scale, self.maxq, self.stoch)

            if self.high_bits_length != 0:
                x_h = STEQuantize.apply(x_h, self.scale_h, self.maxq_h, self.stoch)
                x_m = torch.cat([x_m, x_h], dim=-1)

            if self.low_bits_length != 0:
                x_l = STEQuantize.apply(x_l, self.scale_l, self.maxq_l, self.stoch)
                x_m = torch.cat([x_l, x_m], dim=-1)

        else:
            # Asymmetric quantization
            x_m = AsymSTEQuantize.apply(x_m, self.scale, self.zero, self.maxq, self.stoch)

            if self.high_bits_length != 0:
                x_h = AsymSTEQuantize.apply(x_h, self.scale_h, self.zero_h, self.maxq_h, self.stoch)
                x_m = torch.cat([x_m, x_h], dim=-1)

            if self.low_bits_length != 0:
                x_l = AsymSTEQuantize.apply(x_l, self.scale_l, self.zero_l, self.maxq_l, self.stoch)
                x_m = torch.cat([x_l, x_m], dim=-1)

        # Reshape back if groupwise
        if self.groupsize > 0:
            x_m = x_m.reshape(init_shape)

        return x_m.to(x_dtype)

    def quantize(self, x: torch.Tensor, return_low: bool = False,
                 return_high: bool = False) -> Tuple:
        """
        Quantize and return integers + scales (without dequantization).

        Args:
            x: Input tensor
            return_low: Return low precision group
            return_high: Return high precision group

        Returns:
            Tuple of (quantized_int, scale[, zero if asymmetric])
        """
        low_dim, high_dim = self.low_bits_length, x.shape[-1] - self.high_bits_length
        x_l, x_m, x_h = x[..., :low_dim], x[..., low_dim:high_dim], x[..., high_dim:]

        if self.sym:
            if return_low:
                return sym_quant(x_l, self.scale_l, self.maxq_l)
            elif return_high:
                return sym_quant(x_h, self.scale_h, self.maxq_h)
            else:
                return sym_quant(x_m, self.scale, self.maxq)
        else:
            if return_low:
                return asym_quant(x_l, self.scale_l, self.zero_l, self.maxq_l)
            elif return_high:
                return asym_quant(x_h, self.scale_h, self.zero_h, self.maxq_h)
            else:
                return asym_quant(x_m, self.scale, self.zero, self.maxq)

    def configure(
        self,
        bits: int,
        groupsize: int = -1,
        sym: bool = False,
        clip_ratio: float = 1.0,
        stoch: bool = False,
        high_bits_length: int = 0,
        high_bits: int = 16,
        low_bits_length: int = 0,
        low_bits: int = 16,
    ) -> None:
        """
        Configure quantization parameters.

        Args:
            bits: Middle precision bits
            groupsize: Group size for groupwise quantization (-1 for per-token)
            sym: Use symmetric quantization
            clip_ratio: Ratio for clipping outliers (0, 1]
            stoch: Use stochastic rounding
            high_bits_length: Number of high precision dimensions
            high_bits: Bit width for high precision
            low_bits_length: Number of low precision dimensions
            low_bits: Bit width for low precision
        """
        _, self.maxq = get_minq_maxq(bits, sym)
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        self.stoch = stoch

        self.high_bits_length = high_bits_length
        self.high_bits = high_bits
        _, self.maxq_h = get_minq_maxq(high_bits, sym)

        self.low_bits_length = low_bits_length
        self.low_bits = low_bits
        _, self.maxq_l = get_minq_maxq(low_bits, sym)

        assert (
            self.clip_ratio <= 1 and self.clip_ratio > 0
        ), "Clip ratio should be in (0, 1]"

    def find_params_per_token_groupwise(
        self, x: torch.Tensor, maxq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find quantization parameters for per-token groupwise quantization.

        Args:
            x: Input tensor reshaped to [..., num_groups, groupsize]
            maxq: Maximum quantization value

        Returns:
            Tuple of (scale, zero) tensors
        """
        xmax = torch.amax(x, dim=3, keepdim=True) * self.clip_ratio
        xmin = torch.amin(x, dim=3, keepdim=True) * self.clip_ratio

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            scale = xmax / maxq
            scale[tmp] = 1
            zero = torch.zeros_like(scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            scale = (xmax - xmin) / maxq
            zero = torch.round(-xmin / scale)

        return scale, zero

    def find_params(self, x: torch.Tensor, residual_dim: int = -1) -> None:
        """
        Find quantization parameters (scales/zeros) for input tensor.

        Args:
            x: Input tensor of shape [..., hidden_dim]
            residual_dim: Reserved for future use
        """
        if self.groupsize > 0:
            # Per-group mixed precision quantization
            init_shape = x.shape
            x_reshaped = x.reshape(
                x.shape[0], x.shape[1], x.shape[2] // self.groupsize, self.groupsize
            )
            low_dim, high_dim = (
                self.low_bits_length,
                x_reshaped.shape[-1] - self.high_bits_length,
            )
            x_l = x_reshaped[..., :low_dim]
            x_m = x_reshaped[..., low_dim:high_dim]
            x_h = x_reshaped[..., high_dim:]

            self.scale, self.zero = self.find_params_per_token_groupwise(x_m, self.maxq)

            if self.high_bits_length != 0:
                self.scale_h, self.zero_h = self.find_params_per_token_groupwise(
                    x_h, self.maxq_h
                )

            if self.low_bits_length != 0:
                self.scale_l, self.zero_l = self.find_params_per_token_groupwise(
                    x_l, self.maxq_l
                )

            return

        # Per-token quantization
        low_dim, high_dim = self.low_bits_length, x.shape[-1] - self.high_bits_length
        x_l, x_m, x_h = x[..., :low_dim], x[..., low_dim:high_dim], x[..., high_dim:]

        self.scale, self.zero = self._find_params(x_m, self.maxq)

        if self.high_bits_length != 0:
            self.scale_h, self.zero_h = self._find_params(x_h, self.maxq_h)

        if self.low_bits_length != 0:
            self.scale_l, self.zero_l = self._find_params(x_l, self.maxq_l)

    def _find_params(
        self, x: torch.Tensor, maxq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Internal method to find per-token quantization parameters.

        Args:
            x: Input tensor
            maxq: Maximum quantization value

        Returns:
            Tuple of (scale, zero) tensors
        """
        if self.bits == 16:
            return None, None

        dev = x.device
        init_shape = x.shape
        reshaped_x = x.reshape((-1, x.shape[-1]))

        tmp = torch.zeros(reshaped_x.shape[0], device=dev)
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            scale = (xmax / maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
            scale[tmp] = 1
            scale = scale.reshape(init_shape)
            zero = torch.zeros_like(scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            scale = (xmax - xmin) / maxq
            zero = torch.round(-xmin / scale)
            scale = scale.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
            zero = zero.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)

        return scale, zero


class ActQuantWrapper(nn.Module):
    """
    Activation quantization wrapper for linear layers.

    This class wraps a linear layer and applies activation quantization
    to both inputs and outputs. Supports:
    - Mixed-precision activation quantization
    - Online Hadamard transform for outlier suppression
    - Rotation matrices during training
    - Column reordering for precision grouping
    """

    def __init__(self, module: nn.Linear) -> None:
        super(ActQuantWrapper, self).__init__()
        self.module = module
        self.weight = module.weight
        self.bias = module.bias

        # Quantizers
        self.quantizer = ActQuantizer()  # Input quantizer
        self.hadK_quantizer = ActQuantizer()  # Reserved for Hadamard quantization
        self.out_quantizer = ActQuantizer()  # Output quantizer

        # Hadamard transform buffers
        self.register_buffer("had_K", torch.tensor(0))
        self._buffers["had_K"] = None
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.had_dim = 0
        self.fp32_had = False
        self.residual = 0
        self.no_had = False

    def extra_repr(self) -> str:
        """Return string representation of quantization configuration."""
        str_ = (
            f"Input Quantizer Bits: {self.quantizer.bits}, "
            f"High Bits/Dim: {self.quantizer.high_bits}/{self.quantizer.high_bits_length}, "
            f"Low Bits/Dim: {self.quantizer.low_bits}/{self.quantizer.low_bits_length}"
        )
        if self.quantizer.bits < 16:
            str_ += (
                " (Asymmetric Per-Token)"
                if not self.quantizer.sym
                else " (Symmetric Per-Token)"
            )

        str_ += (
            f"\nOutput Quantizer Bits: {self.out_quantizer.bits}, "
            f"High Bits/Dim: {self.out_quantizer.high_bits}/{self.out_quantizer.high_bits_length}, "
            f"Low Bits/Dim: {self.out_quantizer.low_bits}/{self.out_quantizer.low_bits_length}"
        )
        if self.out_quantizer.bits < 16:
            str_ += (
                " (Asymmetric Per-Token)"
                if not self.out_quantizer.sym
                else " (Symmetric Per-Token)"
            )

        return str_

    def forward(
        self,
        x: torch.Tensor,
        R1: Optional[torch.Tensor] = None,
        R2: Optional[torch.Tensor] = None,
        transpose: bool = False,
        column_order: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with activation quantization.

        Args:
            x: Input tensor
            R1: Rotation matrix 1 (for training)
            R2: Rotation matrix 2 (for training)
            transpose: Whether to transpose rotations
            column_order: Column reordering indices

        Returns:
            Output tensor after linear layer and quantization
        """
        x_dtype = x.dtype

        # Apply Hadamard transform if enabled
        if self.online_full_had:
            # Import here to avoid circular dependency
            from ..utils import hadamard_utils
            if self.fp32_had:
                x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K, self.K).to(x_dtype)
            else:
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K)

        elif self.online_partial_had:
            if self.fp32_had:
                x = x.float()

            init_shape = x.shape
            if self.K == 1:
                # Use fast Hadamard transform
                from ..utils.hadamard_utils import HadamardTransform
                x = (
                    HadamardTransform.apply(
                        x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim).transpose(1, 2)
                    )
                    / math.sqrt(init_shape[-1] // self.had_dim)
                ).transpose(1, 2)
            else:
                x = (
                    self.had_K.to(x.dtype)
                    @ x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim)
                ) / math.sqrt(init_shape[-1] // self.had_dim)

            if self.fp32_had:
                x = x.to(x_dtype)
            x = x.reshape(init_shape)

        # Quantize input if needed
        if self.quantizer.bits < 16:
            self.quantizer.find_params(x)
            x = self.quantizer(x).to(x_dtype)
            self.quantizer.free()

        # Reorder columns if specified
        if column_order is not None:
            x = x[..., column_order]

        # Forward through linear layer
        if R1 is not None:
            assert column_order is None  # column order only used when not training rotations
            x = self.module(x, R1, R2, transpose).to(x_dtype)
        else:
            x = self.module(x).to(x_dtype)

        # Quantize output if needed
        if self.out_quantizer.bits < 16:
            self.out_quantizer.find_params(x)
            x = self.out_quantizer(x).to(x_dtype)
            self.out_quantizer.free()

        return x

    def get_rotated_weight(
        self,
        R1: Optional[torch.Tensor] = None,
        R2: Optional[torch.Tensor] = None,
        R4: Optional[torch.Tensor] = None,
        transpose: bool = False,
    ) -> torch.Tensor:
        """
        Get weight matrix with rotations applied (for training).

        Args:
            R1, R2, R4: Rotation matrices
            transpose: Whether to transpose rotations

        Returns:
            Rotated weight matrix
        """
        return self.module.get_rotated_weight(R1, R2, R4, transpose)
