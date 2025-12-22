# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Adapted from ResQ (https://github.com/facebookresearch/resq)
"""
ResQ: Residual Quantization for Low-Bit LLM Quantization

This module provides the ResQ quantization method implementation
adapted for Ascend NPU and CPU platforms.

ResQ uses eigenvalue decomposition to identify low-rank subspaces
where activation variances are highest, enabling mixed-precision
quantization (4-bit/8-bit) with minimal accuracy loss.

Key Features:
- 4/8-bit hybrid quantization for activations
- Eigenvalue-based channel importance ranking
- CPU/NPU kernel support (no GPU dependency)
- Per-head rotation for value projections
- Hadamard transform for outlier suppression

Usage:
    from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.resq import (
        ResQConfig,
        resq_quantize,
        compute_basis,
        resq_train,
    )

    # Configure
    config = ResQConfig()
    config.a_bits = 4
    config.high_bits = 8
    config.high_fraction = 0.125

    # Quantize model
    model = resq_quantize(model, config)

    # Or use trainer for full pipeline
    model = resq_train(model, dataloader, config)
"""

__all__ = [
    # Configuration
    'ResQConfig',
    # Main functions
    'resq_quantize',
    'compute_basis',
    'resq_train',
    # Processors
    'apply_rotations',
    'rearrange_columns',
    # Components
    'ActQuantizer',
    'ActQuantWrapper',
    'WeightQuantizer',
    'MixedPrecisionWeightQuantizer',
]

from .config import ResQConfig
from .processors.resq_processor import resq_quantize, apply_rotations, rearrange_columns
from .processors.basis_processor import compute_basis
from .trainer import resq_train
from .components.act_quantizer import ActQuantizer, ActQuantWrapper
from .components.weight_quantizer import WeightQuantizer, MixedPrecisionWeightQuantizer
