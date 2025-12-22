# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted for msmodelslim from ResQ: https://github.com/facebookresearch/resq
# Original WeightQuantizer from GPTQ

"""
Mixed-precision weight quantizer for ResQ.

This module implements weight quantization with support for:
- Mixed-precision quantization (4-bit/8-bit)
- Per-channel or per-tensor quantization
- Symmetric/asymmetric quantization
- MSE-based scale optimization
"""

import torch
import torch.nn as nn
from typing import Optional
from .act_quantizer import (
    get_minq_maxq,
    sym_quant_dequant,
    asym_quant_dequant,
    STEQuantize,
    AsymSTEQuantize,
)


class WeightQuantizer(nn.Module):
    """
    Weight quantizer with optional mixed-precision support.

    Supports per-channel quantization with two precision tiers:
    - 4-bit for low variance channels
    - 8-bit for high variance channels
    """

    def __init__(self, shape: int = 1) -> None:
        """
        Initialize weight quantizer.

        Args:
            shape: Shape of scale/zero buffers (1 for per-tensor, out_features for per-channel)
        """
        super(WeightQuantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

        # Mixed precision configuration
        self.mixed_precision = False
        self.high_bits_indices = None  # Indices for 8-bit quantization

    def configure(
        self,
        bits: int,
        perchannel: bool = False,
        sym: bool = True,
        mse: bool = False,
        norm: float = 2.4,
        grid: int = 100,
        maxshrink: float = 0.8,
        mixed_precision: bool = False,
        high_bits: int = 8,
    ) -> None:
        """
        Configure quantization parameters.

        Args:
            bits: Base quantization bits (4-bit default)
            perchannel: Use per-channel quantization
            sym: Use symmetric quantization
            mse: Use MSE-based scale optimization
            norm: Norm for MSE optimization
            grid: Grid size for MSE search
            maxshrink: Maximum shrinkage for MSE search
            mixed_precision: Enable mixed-precision (4-bit/8-bit)
            high_bits: Bits for high variance channels (8-bit default)
        """
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.mixed_precision = mixed_precision
        self.high_bits = high_bits if mixed_precision else bits

        if sym:
            self.maxq = torch.tensor(2 ** (bits - 1) - 1)
        else:
            self.maxq = torch.tensor(2 ** bits - 1)

    def find_params(self, x: torch.Tensor) -> None:
        """
        Find quantization parameters (scales and zeros) for weights.

        Args:
            x: Weight tensor of shape [out_features, in_features]
        """
        if self.bits == 16:
            return

        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            # MSE-based scale optimization
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:
                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(
                        x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq
                    )

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]

        if not self.perchannel:
            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize and dequantize weight tensor.

        Args:
            x: Input weight tensor

        Returns:
            Quantized-dequantized weight tensor
        """
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            if self.sym:
                return STEQuantize.apply(x, self.scale, self.maxq, False).to(x_dtype)
            return AsymSTEQuantize.apply(x, self.scale, self.zero, self.maxq, False).to(
                x_dtype
            )
        return x

    def set_high_bits_indices(self, indices: torch.Tensor) -> None:
        """
        Set indices for high-precision (8-bit) quantization.

        Args:
            indices: Boolean tensor or index tensor for high precision channels
        """
        self.high_bits_indices = indices

    def enabled(self) -> bool:
        """Check if quantizer is enabled."""
        return self.maxq > 0

    def ready(self) -> bool:
        """Check if quantizer parameters are ready."""
        return torch.all(self.scale != 0)


class MixedPrecisionWeightQuantizer(nn.Module):
    """
    Mixed-precision weight quantizer for ResQ (4-bit + 8-bit).

    This quantizer splits weight channels into two groups based on variance:
    - Low variance channels → 4-bit quantization
    - High variance channels → 8-bit quantization
    """

    def __init__(self, shape: int = 1) -> None:
        """
        Initialize mixed-precision weight quantizer.

        Args:
            shape: Shape of scale/zero buffers
        """
        super(MixedPrecisionWeightQuantizer, self).__init__()

        # 4-bit quantizer for low/middle variance
        self.quantizer_4bit = WeightQuantizer(shape)

        # 8-bit quantizer for high variance
        self.quantizer_8bit = WeightQuantizer(shape)

        # Precision grouping
        self.high_precision_length = 0
        self.channel_order = None  # Reordering indices

    def configure(
        self,
        bits: int = 4,
        high_bits: int = 8,
        high_precision_fraction: float = 0.2,
        perchannel: bool = True,
        sym: bool = True,
        mse: bool = False,
        norm: float = 2.4,
        grid: int = 100,
        maxshrink: float = 0.8,
    ) -> None:
        """
        Configure mixed-precision quantization.

        Args:
            bits: Base bits for low variance (4-bit default)
            high_bits: Bits for high variance (8-bit default)
            high_precision_fraction: Fraction of channels for high precision
            perchannel: Use per-channel quantization
            sym: Use symmetric quantization
            mse: Use MSE-based optimization
            norm: Norm for MSE optimization
            grid: Grid size for MSE search
            maxshrink: Maximum shrinkage for MSE search
        """
        self.bits = bits
        self.high_bits = high_bits
        self.high_precision_fraction = high_precision_fraction

        # Configure 4-bit quantizer
        self.quantizer_4bit.configure(
            bits=bits,
            perchannel=perchannel,
            sym=sym,
            mse=mse,
            norm=norm,
            grid=grid,
            maxshrink=maxshrink,
        )

        # Configure 8-bit quantizer
        self.quantizer_8bit.configure(
            bits=high_bits,
            perchannel=perchannel,
            sym=sym,
            mse=mse,
            norm=norm,
            grid=grid,
            maxshrink=maxshrink,
        )

    def set_channel_order(
        self, eigenvalues: torch.Tensor, total_channels: int
    ) -> None:
        """
        Set channel ordering based on eigenvalues (variance).

        Channels are reordered: [low variance (4-bit) | high variance (8-bit)]

        Args:
            eigenvalues: Eigenvalues from basis computation (sorted low to high)
            total_channels: Total number of channels
        """
        # Compute high precision length
        self.high_precision_length = int(self.high_precision_fraction * total_channels)

        # Sort by eigenvalue (descending) - high variance first
        sorted_indices = torch.argsort(eigenvalues, descending=True)

        # High precision: first high_precision_length channels
        high_indices = sorted_indices[: self.high_precision_length]

        # Low precision: remaining channels
        low_indices = sorted_indices[self.high_precision_length :]

        # Channel order: [low | high]
        self.channel_order = torch.cat([low_indices, high_indices])

    def find_params(self, x: torch.Tensor, eigenvalues: Optional[torch.Tensor] = None) -> None:
        """
        Find quantization parameters for mixed-precision weights.

        Args:
            x: Weight tensor of shape [out_features, in_features]
            eigenvalues: Optional eigenvalues for automatic channel grouping
        """
        if self.bits == 16:
            return

        # Set channel order if eigenvalues provided
        if eigenvalues is not None:
            self.set_channel_order(eigenvalues, x.shape[1])

        # Reorder columns if channel order is set
        if self.channel_order is not None:
            x_reordered = x[:, self.channel_order]
        else:
            x_reordered = x
            # Default: last high_precision_fraction are high precision
            total_channels = x.shape[1]
            self.high_precision_length = int(
                self.high_precision_fraction * total_channels
            )

        # Split into low and high precision groups
        low_end = x_reordered.shape[1] - self.high_precision_length
        x_low = x_reordered[:, :low_end]
        x_high = x_reordered[:, low_end:]

        # Find parameters for each group
        if x_low.numel() > 0:
            self.quantizer_4bit.find_params(x_low)

        if x_high.numel() > 0:
            self.quantizer_8bit.find_params(x_high)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize weights with mixed precision.

        Args:
            x: Input weight tensor

        Returns:
            Quantized-dequantized weight tensor
        """
        if self.bits == 16:
            return x

        x_dtype = x.dtype

        # Reorder columns if channel order is set
        if self.channel_order is not None:
            x_reordered = x[:, self.channel_order]
        else:
            x_reordered = x

        # Split into low and high precision groups
        low_end = x_reordered.shape[1] - self.high_precision_length
        x_low = x_reordered[:, :low_end]
        x_high = x_reordered[:, low_end:]

        # Quantize each group
        if x_low.numel() > 0:
            x_low_q = self.quantizer_4bit.quantize(x_low)
        else:
            x_low_q = x_low

        if x_high.numel() > 0:
            x_high_q = self.quantizer_8bit.quantize(x_high)
        else:
            x_high_q = x_high

        # Concatenate
        x_q = torch.cat([x_low_q, x_high_q], dim=1)

        # Restore original column order
        if self.channel_order is not None:
            # Create inverse permutation
            inverse_order = torch.argsort(self.channel_order)
            x_q = x_q[:, inverse_order]

        return x_q.to(x_dtype)

    def enabled(self) -> bool:
        """Check if quantizer is enabled."""
        return self.quantizer_4bit.enabled() or self.quantizer_8bit.enabled()

    def ready(self) -> bool:
        """Check if quantizer parameters are ready."""
        return self.quantizer_4bit.ready() and self.quantizer_8bit.ready()


def add_actquant(
    module: nn.Module,
    name: str = "",
    layers: list = None,
) -> None:
    """
    Recursively add activation quantization wrappers to linear layers.

    Args:
        module: Module to process
        name: Current module name
        layers: List of layer types to wrap
    """
    if layers is None:
        layers = [nn.Linear]

    # Import ActQuantWrapper here to avoid circular dependency
    from .act_quantizer import ActQuantWrapper

    if isinstance(module, ActQuantWrapper):
        return

    for attr in dir(module):
        try:
            tmp = getattr(module, attr)
        except:
            continue

        if type(tmp) in layers:
            setattr(module, attr, ActQuantWrapper(tmp))
        elif type(tmp) == nn.Sequential:
            replaced = []
            for child in tmp.children():
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, nn.Sequential(*replaced))
        elif type(tmp) == nn.ModuleList:
            replaced = []
            for child in tmp.children():
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, nn.ModuleList(replaced))

    for name1, child in module.named_children():
        add_actquant(child, name + "." + name1 if name != "" else name1, layers)
