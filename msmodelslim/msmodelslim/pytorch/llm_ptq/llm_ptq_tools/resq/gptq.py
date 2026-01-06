# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Adapted from ResQ (https://github.com/facebookresearch/resq)
"""
GPTQ quantization for ResQ.

This module implements the GPTQ (Generative Pre-trained Transformer Quantization)
algorithm with support for mixed-precision quantization.

GPTQ performs column-by-column quantization with error compensation using
the Hessian matrix computed from calibration data.
"""

import gc
import logging
import math
import time
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def cleanup_memory():
    """Clean up NPU/CPU memory."""
    gc.collect()
    try:
        import torch_npu
        if torch.npu.is_available():
            torch.npu.empty_cache()
    except ImportError:
        pass


def symmetric_quantize(tensor: torch.Tensor, bits: int, scale: torch.Tensor = None) -> torch.Tensor:
    """
    Symmetric quantization using pre-computed scale.

    Args:
        tensor: Input tensor to quantize
        bits: Number of bits (4 or 8)
        scale: Pre-computed scale tensor (if None, compute from tensor)

    Returns:
        Dequantized tensor (float) for GPTQ error compensation
    """
    n = 2 ** (bits - 1) - 1  # e.g., 127 for 8-bit, 7 for 4-bit

    if scale is None:
        abs_max = tensor.abs().max(dim=-1, keepdim=True)[0]
        scale = abs_max / n
        scale = torch.clamp(scale, min=1e-8)

    # Quantize and dequantize
    q = torch.clamp(torch.round(tensor / scale), -(n + 1), n)
    return scale * q


class GPTQWeightQuantizer(nn.Module):
    """
    Weight quantizer for GPTQ with symmetric quantization.

    This is a simplified quantizer that computes scale from weights
    and performs symmetric quantization.
    """

    def __init__(self, bits: int = 4, perchannel: bool = True, sym: bool = True):
        super().__init__()
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.scale = None
        self.maxq = 2 ** (bits - 1) - 1

    def ready(self) -> bool:
        """Check if quantizer is ready (scale computed)."""
        return self.scale is not None

    def find_params(self, x: torch.Tensor) -> None:
        """Compute quantization scale from weight tensor."""
        if self.bits >= 16:
            return

        shape = x.shape
        if self.perchannel:
            x_flat = x.flatten(1)
        else:
            x_flat = x.flatten().unsqueeze(0)

        # Symmetric quantization: scale = max_abs / maxq
        xmax = x_flat.abs().max(dim=1)[0].clamp(min=1e-8)
        self.scale = xmax / self.maxq

        # Reshape scale for broadcasting
        if not self.perchannel:
            self.scale = self.scale.repeat(shape[0])

        shape_out = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape_out)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize and dequantize (for GPTQ error compensation)."""
        if not self.ready() or self.bits >= 16:
            return x

        # Symmetric quantization: q = round(x / scale), dequant = scale * q
        q = torch.clamp(torch.round(x / self.scale), -(self.maxq + 1), self.maxq)
        return self.scale * q

    def cpu(self):
        """Move quantizer to CPU."""
        if self.scale is not None:
            self.scale = self.scale.cpu()
        return self


class GPTQ:
    """
    GPTQ quantizer for a single linear layer.

    Performs column-by-column quantization with Hessian-based error compensation.
    Supports mixed-precision quantization with different bit-widths for
    different regions of the weight matrix.

    Args:
        layer: The linear layer to quantize
        mixed_precision: Whether to use mixed-precision quantization
        high_bits_length: Number of columns for high precision (int8)
        low_bits_length: Number of columns for low precision (extra low, usually 0)
    """

    def __init__(
        self,
        layer: nn.Linear,
        mixed_precision: bool = False,
        high_bits_length: int = 0,
        low_bits_length: int = 0,
    ):
        self.layer = layer
        self.dev = layer.weight.device
        W = layer.weight.data.clone()

        self.mixed_precision = mixed_precision
        self.high_bits_length = high_bits_length  # high precision (int8)
        self.low_bits_length = low_bits_length    # extra low precision (usually 0)

        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.float32)
        self.nsamples = 0

        # Quantizers (to be configured before fasterquant)
        self.quantizer = None      # Main quantizer (4-bit for mid region)
        self.high_quantizer = None  # High precision quantizer (8-bit)
        self.low_quantizer = None   # Low precision quantizer (extra low, usually not used)

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor = None) -> None:
        """
        Accumulate Hessian from a batch of inputs.

        The Hessian approximation is H = X^T @ X where X is the input activation.

        Args:
            inp: Input activation tensor [batch, seq, hidden] or [batch*seq, hidden]
            out: Output tensor (not used, kept for API compatibility)
        """
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))

        inp = inp.t().float()  # [hidden, batch*seq]

        # Running average of Hessian
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self,
        blocksize: int = 128,
        percdamp: float = 0.01,
        groupsize: int = -1,
        actorder: bool = False,
    ) -> None:
        """
        Perform GPTQ quantization with error compensation.

        This method quantizes the weight matrix column by column, compensating
        for quantization error using the inverse Hessian.

        Args:
            blocksize: Number of columns to process at once
            percdamp: Dampening percentage for numerical stability
            groupsize: Group size for grouped quantization (-1 for per-channel)
            actorder: Whether to reorder columns by activation magnitude
        """
        W_org = self.layer.weight.data.clone().float()
        W = W_org.clone()

        # Dimension boundaries for mixed precision
        # Column layout: [low_dim | mid_dim | high_dim]
        high_dim = W_org.shape[-1] - self.high_bits_length
        low_dim = self.low_bits_length
        mp = self.mixed_precision

        # Find quantization parameters
        if mp:
            W_l = W[:, :low_dim] if low_dim > 0 else None
            W_m = W[:, low_dim:high_dim]
            W_h = W[:, high_dim:] if self.high_bits_length > 0 else None

            if not self.quantizer.ready():
                self.quantizer.find_params(W_m)
            if self.high_quantizer is not None and not self.high_quantizer.ready() and self.high_bits_length > 0:
                self.high_quantizer.find_params(W_h)
            if self.low_quantizer is not None and not self.low_quantizer.ready() and low_dim > 0:
                self.low_quantizer.find_params(W_l)
        else:
            if not self.quantizer.ready():
                self.quantizer.find_params(W)

        H = self.H
        del self.H

        # Handle dead columns (zero diagonal)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # Optional: reorder by activation magnitude
        invperm = None
        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Q = torch.zeros_like(W)

        # Damping and Cholesky decomposition
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp

        try:
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            Hinv = torch.linalg.cholesky(H, upper=True)
        except RuntimeError as e:
            logger.warning(f"Cholesky failed, adding epsilon: {e}")
            epsilon = 1e-5
            H = H + epsilon * torch.eye(H.size(0), device=H.device)
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            Hinv = torch.linalg.cholesky(H, upper=True)

        # Block-wise quantization with error compensation
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                col_idx = i1 + i
                if actorder and invperm is not None:
                    col_idx = invperm[col_idx].item() if hasattr(invperm[col_idx], 'item') else invperm[col_idx]

                # Select quantizer based on column position (mixed precision)
                if mp and col_idx >= high_dim:
                    # High precision region (int8)
                    q = self.high_quantizer.quantize(w.unsqueeze(1)).flatten()
                elif mp and col_idx < low_dim:
                    # Extra low precision region (usually not used)
                    q = self.low_quantizer.quantize(w.unsqueeze(1)).flatten()
                else:
                    # Mid precision region (int4)
                    q = self.quantizer.quantize(w.unsqueeze(1)).flatten()

                Q1[:, i] = q

                # Error compensation
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        # Revert column order if actorder was used
        if actorder and invperm is not None:
            Q = Q[:, invperm]

        # Update layer weight with quantized values
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        if torch.any(torch.isnan(self.layer.weight.data)):
            logger.error("NaN detected in quantized weights!")
            raise ValueError("NaN in weights after GPTQ quantization")

    def free(self) -> None:
        """Free memory used by GPTQ."""
        self.H = None
        cleanup_memory()


def create_gptq_quantizers(
    layer: nn.Linear,
    high_bits: int = 8,
    low_bits: int = 4,
    high_fraction: float = 0.125,
    low_fraction: float = 0.0,
    sym: bool = True,
    mse: bool = False,
) -> GPTQ:
    """
    Create a GPTQ object with configured quantizers for a layer.

    Args:
        layer: Linear layer to quantize
        high_bits: Bits for high precision region (default: 8)
        low_bits: Bits for low/mid precision region (default: 4)
        high_fraction: Fraction of columns for high precision
        low_fraction: Fraction of columns for extra low precision (usually 0)
        sym: Whether to use symmetric quantization
        mse: Whether to use MSE optimization for scale

    Returns:
        Configured GPTQ object
    """
    in_features = layer.in_features
    high_bits_length = int(high_fraction * in_features)
    low_bits_length = int(low_fraction * in_features)
    mixed_precision = high_bits_length > 0 or low_bits_length > 0

    gptq = GPTQ(
        layer,
        mixed_precision=mixed_precision,
        high_bits_length=high_bits_length,
        low_bits_length=low_bits_length,
    )

    # Main quantizer (for mid region, usually 4-bit)
    gptq.quantizer = GPTQWeightQuantizer(bits=low_bits, perchannel=True, sym=sym)

    if mixed_precision:
        # High precision quantizer (8-bit)
        gptq.high_quantizer = GPTQWeightQuantizer(bits=high_bits, perchannel=True, sym=sym)

        # Low precision quantizer (extra low, usually not used when low_fraction=0)
        if low_bits_length > 0:
            gptq.low_quantizer = GPTQWeightQuantizer(bits=low_bits, perchannel=True, sym=sym)

    return gptq
