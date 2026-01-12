# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Adapted from ResQ (https://github.com/facebookresearch/resq)
"""
ResQ configuration class.
"""


class ResQConfig:
    """Configuration for ResQ quantization method."""

    def __init__(
        self,
        high_bits: int = 8,
        low_bits: int = 4,
        high_fraction: float = 0.125,
        seed: int = 0,
        dev_type: str = 'npu',
        dev_id: int = 0,
        rotate_mode: str = 'resq',
        **kwargs
    ):
        # General Arguments
        self.seed = seed

        # Device settings
        self.dev_type = dev_type
        self.dev_id = dev_id

        # Rotation mode: 'resq' or 'none'
        self.rotate_mode = rotate_mode
        # Rotation granularity: 'full_shared' (default), 'per_layer', or 'one_per_decoder'
        # - full_shared: single shared basis for attn+mlp across all layers (recommended)
        # - per_layer: separate basis for each layer's attn and mlp
        # - one_per_decoder: per-layer shared basis for attn+mlp
        self.rotation_granularity = kwargs.get('rotation_granularity', 'full_shared')

        # Paths for pre-computed rotations and basis
        self.optimized_rotation_path = kwargs.get('optimized_rotation_path', None)
        self.optimized_basis_path = kwargs.get('optimized_basis_path', None)

        # Mixed precision fractions
        self.high_fraction = high_fraction  # high precision portion (e.g., 0.125 = 1/8)

        # Activation Quantization Arguments
        self.a_bits = kwargs.get('a_bits', 4)
        self.a_groupsize = kwargs.get('a_groupsize', -1)
        self.a_clip_ratio = kwargs.get('a_clip_ratio', 1.0)

        # High/Low precision bits for activations and weights
        self.high_bits = high_bits
        self.low_bits = low_bits

        # Weight Quantization Arguments
        self.w_bits = kwargs.get('w_bits', 4)
        self.w_groupsize = kwargs.get('w_groupsize', -1)
        self.w_sym = kwargs.get('w_sym', True)
        self.w_clip = kwargs.get('w_clip', True)
        self.w_rtn = kwargs.get('w_rtn', True)

        # Value cache quantization
        self.v_bits = kwargs.get('v_bits', 4)
        self.v_clip_ratio = kwargs.get('v_clip_ratio', 1.0)

        # Key cache quantization
        self.k_bits = kwargs.get('k_bits', 4)
        self.k_groupsize = kwargs.get('k_groupsize', -1)
        self.k_clip_ratio = kwargs.get('k_clip_ratio', 1.0)
        self.k_pre_rope = kwargs.get('k_pre_rope', False)

        # GPTQ settings
        # w_rtn: True = RTN (Round-to-Nearest), False = GPTQ (column-by-column with Hessian)
        self.percdamp = kwargs.get('percdamp', 0.01)
        self.act_order = kwargs.get('act_order', False)
        self.nsamples = kwargs.get('nsamples', 128)
        self.gptq_blocksize = kwargs.get('gptq_blocksize', 128)  # Block size for GPTQ column processing

        # Hadamard settings
        self.fp32_had = kwargs.get('fp32_had', True)
        self.int8_down_proj = kwargs.get('int8_down_proj', True)

        # Down projection block size for rotation
        self.down_proj_blocksize = kwargs.get('down_proj_blocksize', 256)

        # Ud rotation type: 'hadamard' or 'random'
        # - hadamard: Ud = block_diag(Pd) @ H, save Pd per layer + H globally
        # - random: Ud = block_diag(Pd) @ Rd, save full Ud per layer
        self.ud_rotation_type = kwargs.get('ud_rotation_type', 'hadamard')

        # Training rotations (for rotation optimization)
        self.train_rotations = kwargs.get('train_rotations', False)

        # Calibration dataset
        self.calib_dataset = kwargs.get('calib_dataset', 'wikitext2')

        # Batch size
        self.bsz = kwargs.get('bsz', 1)

        # Data type
        self.amp_dtype = kwargs.get('amp_dtype', 'bfloat16')

        # Dynamic quantization mode (for activations)
        self.is_dynamic = kwargs.get('is_dynamic', True)
        self.a_sym = kwargs.get('a_sym', False)

        # Output mode: determines what to save
        # - 'fused': Save fused weights + online transforms (default, production use)
        # - 'transforms_only': Save only decomposed P and R matrices (no fusion)
        # - 'debug': Save both fused weights and decomposed P/R matrices
        output_mode = kwargs.get('output_mode', 'fused')

        # Backward compatibility: convert boolean save_transforms_only to output_mode
        if 'save_transforms_only' in kwargs and kwargs['save_transforms_only'] is not None:
            import warnings
            warnings.warn(
                "save_transforms_only is deprecated, use output_mode='transforms_only' instead",
                DeprecationWarning,
                stacklevel=2
            )
            if kwargs['save_transforms_only']:
                output_mode = 'transforms_only'
            else:
                output_mode = 'fused'

        self.output_mode = output_mode

    @property
    def should_skip_fusion(self) -> bool:
        """Return True if weight fusion should be skipped (transforms_only mode)."""
        return self.output_mode == 'transforms_only'

    @property
    def should_save_transforms(self) -> bool:
        """Return True if decomposed P/R transform matrices should be saved."""
        return self.output_mode in ['transforms_only', 'debug']

    @property
    def should_save_fused_weights(self) -> bool:
        """Return True if fused weights and online transforms should be saved."""
        return self.output_mode in ['fused', 'debug']

    def validate(self, strict: bool = False):
        """
        Validate configuration parameters.

        Args:
            strict: If True, require basis_path for ResQ mode.
                   If False, allow simplified mode without basis.
        """
        assert self.rotate_mode in ['resq', 'none'], \
            f"Invalid rotate_mode: {self.rotate_mode}. Only 'resq' and 'none' are supported."
        assert self.ud_rotation_type in ['hadamard', 'random'], \
            f"Invalid ud_rotation_type: {self.ud_rotation_type}. Only 'hadamard' and 'random' are supported."
        assert self.output_mode in ['fused', 'transforms_only', 'debug'], \
            f"Invalid output_mode: {self.output_mode}. Valid options: 'fused', 'transforms_only', 'debug'"
        assert 0.0 <= self.high_fraction <= 1.0, "high_fraction must be between 0 and 1"

        if strict and self.rotate_mode == 'resq':
            assert self.optimized_basis_path is not None, \
                "optimized_basis_path is required for ResQ mode (use strict=False for simplified mode)"
