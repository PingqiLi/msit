# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Adapted from ResQ (https://github.com/facebookresearch/resq)
"""
ResQ configuration class.
"""


class ResQConfig:
    """Configuration for ResQ quantization method."""

    def __init__(self):
        # General Arguments
        self.seed = 0

        # Rotation mode: 'resq', 'quarot', 'spinquant', 'none'
        self.rotate_mode = "resq"
        # Rotation granularity: 'full_shared', 'per_layer', 'one_per_decoder'
        self.rotation_granularity = "full_shared"

        # Paths for pre-computed rotations and basis
        self.optimized_rotation_path = None
        self.optimized_basis_path = None

        # Mixed precision fractions
        self.high_fraction = 0.03125  # 1/32, high precision portion
        self.low_fraction = 0.03125   # 1/32, low precision portion
        self.sparse_fraction = 0.0    # sparse fraction within low precision

        # Activation Quantization Arguments
        self.a_bits = 4
        self.a_groupsize = -1
        self.a_asym = False
        self.a_clip_ratio = 1.0

        # High/Low precision bits for activations
        self.high_bits = 8
        self.low_bits = 8

        # Weight Quantization Arguments
        self.w_bits = 4
        self.w_groupsize = -1
        self.w_asym = False
        self.w_clip = True  # Use MSE for weight quantization
        self.w_rtn = True   # Use RTN (round-to-nearest) instead of GPTQ

        # Value cache quantization
        self.v_bits = 4
        self.v_asym = False
        self.v_clip_ratio = 1.0

        # Key cache quantization
        self.k_bits = 4
        self.k_groupsize = -1
        self.k_asym = False
        self.k_clip_ratio = 1.0
        self.k_pre_rope = False

        # GPTQ settings
        self.percdamp = 0.01
        self.act_order = False
        self.nsamples = 128

        # Hadamard settings
        self.fp32_had = True  # Use FP32 for Hadamard transform
        self.int8_down_proj = True  # Use INT8 for down_proj

        # Down projection block size for rotation
        self.down_proj_blocksize = 256

        # Training rotations (for rotation optimization)
        self.train_rotations = False

        # Calibration dataset
        self.calib_dataset = "wikitext2"

        # Batch size
        self.bsz = 1

        # Data type
        self.amp_dtype = "bfloat16"

    def validate(self):
        """Validate configuration parameters."""
        assert self.rotate_mode in ['resq', 'quarot', 'spinquant', 'none'], \
            f"Invalid rotate_mode: {self.rotate_mode}"
        assert self.rotation_granularity in ['full_shared', 'per_layer', 'one_per_decoder'], \
            f"Invalid rotation_granularity: {self.rotation_granularity}"
        assert 0.0 <= self.high_fraction <= 1.0, "high_fraction must be between 0 and 1"
        assert 0.0 <= self.low_fraction <= 1.0, "low_fraction must be between 0 and 1"
        assert self.high_fraction + self.low_fraction <= 1.0, \
            "high_fraction + low_fraction must be <= 1"

        if self.rotate_mode == 'resq':
            assert self.optimized_basis_path is not None, \
                "optimized_basis_path is required for ResQ mode"
            if not self.train_rotations:
                assert self.optimized_rotation_path is not None, \
                    "optimized_rotation_path is required when not training rotations"
