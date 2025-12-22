# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""ResQ utility modules."""

from .fuse_norm_utils import fuse_layer_norms, fuse_ln_linear
from .common import get_device, cleanup_memory, get_logger, set_seed, get_local_rank
from .hadamard_utils import (
    HadamardTransform,
    hadamard_transform_cpu,
    get_hadamard_matrix,
    random_hadamard_matrix,
    random_orthogonal_matrix,
    get_hadK,
    matmul_hadU,
    matmul_hadU_cpu,
    apply_exact_had_to_linear,
    is_pow2,
)

__all__ = [
    # Common utilities
    'get_device',
    'cleanup_memory',
    'get_logger',
    'set_seed',
    'get_local_rank',
    # Fuse norm utilities
    'fuse_layer_norms',
    'fuse_ln_linear',
    # Hadamard utilities
    'HadamardTransform',
    'hadamard_transform_cpu',
    'get_hadamard_matrix',
    'random_hadamard_matrix',
    'random_orthogonal_matrix',
    'get_hadK',
    'matmul_hadU',
    'matmul_hadU_cpu',
    'apply_exact_had_to_linear',
    'is_pow2',
]
