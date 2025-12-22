# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
ResQ models module.

This module provides model structure utilities and bridges for ResQ.
"""

from .model_utils import (
    get_model_layers,
    get_module_by_name,
    set_module_by_name,
    ModelStructure,
)

__all__ = [
    'get_model_layers',
    'get_module_by_name',
    'set_module_by_name',
    'ModelStructure',
]
