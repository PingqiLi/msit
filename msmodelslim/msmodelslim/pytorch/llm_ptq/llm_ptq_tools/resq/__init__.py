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
- 4/8-bit hybrid quantization for activations and weights
- Eigenvalue-based channel importance ranking
- CPU/NPU kernel support (no GPU dependency)
- Per-head rotation for value projections
- Hadamard transform for outlier suppression
- Dual weights and dual scales output format

Usage:
    from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.resq import (
        ResQConfig,
        ResQCalibrator,
        resq_calibrate,
    )

    # Configure
    config = ResQConfig()
    config.high_bits = 8
    config.low_bits = 4
    config.high_fraction = 0.125

    # Use calibrator
    calibrator = ResQCalibrator(model, config, calib_data)
    calibrator.run()
    calibrator.save(output_path)

    # Or use convenience function
    model = resq_calibrate(model, calib_data, config, output_path)
"""

__all__ = [
    # Configuration
    'ResQConfig',
    # Calibrator
    'ResQCalibrator',
    'resq_calibrate',
    # Main functions
    'resq_quantize',
    'compute_basis',
    'resq_train',
    # Processors
    'apply_rotations',
    'rearrange_columns',
    # Adaptive ratio
    'AdaptiveRatioConfig',
    'AdaptiveRatioComputer',
    'AdaptiveRatioResult',
    'create_algorithm',
    'align_dimension_split',
    'save_adaptive_ratios',
    'load_adaptive_ratios',
    # Quantizer modules
    'LinearResQQuantizer',
    'ResQWeightQuantizer',
    'ResQActQuantizer',
    'add_resq_quantizers',
    # Components
    'ActQuantizer',
    'ActQuantWrapper',
    'WeightQuantizer',
    'MixedPrecisionWeightQuantizer',
]

from .config import ResQConfig
from .calibrator import ResQCalibrator, resq_calibrate
from .processors.resq_processor import resq_quantize, apply_rotations, rearrange_columns
from .processors.basis_processor import compute_basis
from .processors.adaptive_ratio import (
    AdaptiveRatioConfig,
    AdaptiveRatioComputer,
    AdaptiveRatioResult,
    create_algorithm,
    align_dimension_split,
    save_adaptive_ratios,
    load_adaptive_ratios,
)
from .trainer import resq_train
from .quant_modules import (
    LinearResQQuantizer,
    ResQWeightQuantizer,
    ResQActQuantizer,
    add_resq_quantizers,
)
from .components.act_quantizer import ActQuantizer, ActQuantWrapper
from .components.weight_quantizer import WeightQuantizer, MixedPrecisionWeightQuantizer
