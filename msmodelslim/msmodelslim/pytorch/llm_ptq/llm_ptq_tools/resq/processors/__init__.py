# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
ResQ processors module.

This module provides the main processing functions for ResQ quantization:
- Basis computation (eigenvalue decomposition)
- Model quantization with rotations
"""

from .basis_processor import compute_basis, perform_eigen_decomp
from .resq_processor import resq_quantize, apply_rotations, rearrange_columns

__all__ = [
    'compute_basis',
    'perform_eigen_decomp',
    'resq_quantize',
    'apply_rotations',
    'rearrange_columns',
]
