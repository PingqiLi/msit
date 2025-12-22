# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Adapted from ResQ (https://github.com/facebookresearch/resq)
"""
ResQ: Residual Quantization for Low-Bit LLM Quantization

This module provides the ResQ quantization method implementation
adapted for Ascend NPU and CPU platforms.
"""

__all__ = ['ResQConfig', 'resq_quantize', 'compute_basis']

from .config import ResQConfig
from .processors.resq_processor import resq_quantize
from .processors.basis_processor import compute_basis
