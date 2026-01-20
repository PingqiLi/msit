# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
ResQ processors module.

This module provides the main processing functions for ResQ quantization:
- Basis computation (eigenvalue decomposition)
- Model quantization with rotations
- Adaptive precision ratio computation
"""

from .basis_processor import compute_basis, perform_eigen_decomp
from .resq_processor import resq_quantize, apply_rotations, rearrange_columns
from .adaptive_ratio import (
    AdaptiveRatioConfig,
    AdaptiveRatioComputer,
    AdaptiveRatioResult,
    HessianTraceAlgorithm,
    KurtosisAlgorithm,
    CEVAlgorithm,
    HybridAlgorithm,
    create_algorithm,
    align_dimension_split,
    compute_aligned_fraction,
    save_adaptive_ratios,
    load_adaptive_ratios,
)

__all__ = [
    # Basis processor
    'compute_basis',
    'perform_eigen_decomp',
    # ResQ processor
    'resq_quantize',
    'apply_rotations',
    'rearrange_columns',
    # Adaptive ratio
    'AdaptiveRatioConfig',
    'AdaptiveRatioComputer',
    'AdaptiveRatioResult',
    'HessianTraceAlgorithm',
    'KurtosisAlgorithm',
    'CEVAlgorithm',
    'HybridAlgorithm',
    'create_algorithm',
    'align_dimension_split',
    'compute_aligned_fraction',
    'save_adaptive_ratios',
    'load_adaptive_ratios',
]
