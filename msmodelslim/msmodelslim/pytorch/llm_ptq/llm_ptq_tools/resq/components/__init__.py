# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
ResQ components module.

This module provides quantization components for ResQ:
- ActQuantizer: Mixed-precision activation quantizer
- ActQuantWrapper: Wrapper for linear layers with activation quantization
- WeightQuantizer: Weight quantizer with MSE optimization
- MixedPrecisionWeightQuantizer: 4/8-bit hybrid weight quantizer
"""

from .act_quantizer import (
    ActQuantizer,
    ActQuantWrapper,
    STEQuantize,
    AsymSTEQuantize,
    get_minq_maxq,
    sym_quant,
    sym_dequant,
    sym_quant_dequant,
    asym_quant,
    asym_dequant,
    asym_quant_dequant,
)
from .weight_quantizer import (
    WeightQuantizer,
    MixedPrecisionWeightQuantizer,
    add_actquant,
)

__all__ = [
    # Activation quantization
    'ActQuantizer',
    'ActQuantWrapper',
    'STEQuantize',
    'AsymSTEQuantize',
    'get_minq_maxq',
    'sym_quant',
    'sym_dequant',
    'sym_quant_dequant',
    'asym_quant',
    'asym_dequant',
    'asym_quant_dequant',
    # Weight quantization
    'WeightQuantizer',
    'MixedPrecisionWeightQuantizer',
    'add_actquant',
]
