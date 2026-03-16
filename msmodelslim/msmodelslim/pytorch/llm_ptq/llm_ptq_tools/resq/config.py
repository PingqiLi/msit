# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Adapted from ResQ (https://github.com/facebookresearch/resq)
"""
ResQ configuration class.
"""

from typing import Optional, Dict


# Allowed quantization types for mix_cfg
RESQ_ALLOWED_MIX_TYPES = {"resq", "w8a8_dynamic", "float", "int4_hadamard"}


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
        mix_cfg: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        # General Arguments
        self.seed = seed

        # Device settings
        self.dev_type = dev_type
        self.dev_id = dev_id

        # Mixed-precision layer configuration
        # Keys are layer name patterns (fnmatch), values are quant types: 'resq', 'w8a8_dynamic', 'float'
        self.mix_cfg = mix_cfg or {}

        # Paths for pre-computed rotations and basis
        self.optimized_rotation_path = kwargs.get('optimized_rotation_path', None)
        self.optimized_basis_path = kwargs.get('optimized_basis_path', None)

        # Mixed precision fractions
        self.high_fraction = high_fraction  # high precision portion (e.g., 0.125 = 1/8)

        # Adaptive ratio configuration
        # When enabled, high_fraction is determined per-layer based on quantization difficulty
        self.adaptive_ratio = kwargs.get('adaptive_ratio', False)
        self.adaptive_algorithm = kwargs.get('adaptive_algorithm', 'hybrid')  # 'hessian', 'kurtosis', 'cev', 'hybrid'
        self.adaptive_min_ratio = kwargs.get('adaptive_min_ratio', 0.0625)  # 1/16
        self.adaptive_max_ratio = kwargs.get('adaptive_max_ratio', 0.25)    # 1/4
        self.cev_target_variance = kwargs.get('cev_target_variance', 0.95)
        self.adaptive_alignment = kwargs.get('adaptive_alignment', 512)
        # Per-transform algorithm override (optional)
        # e.g., {'Ua': 'cev', 'Ub': 'kurtosis'}
        self.transform_algorithms = kwargs.get('transform_algorithms', {})
        # Ub aggregation across heads: 'max' (conservative) or 'mean'
        self.ub_head_aggregation = kwargs.get('ub_head_aggregation', 'max')
        # Path to pre-computed adaptive ratios
        self.adaptive_ratio_path = kwargs.get('adaptive_ratio_path', None)
        # Compute kurtosis during basis computation (requires extra memory)
        self.compute_kurtosis = kwargs.get('compute_kurtosis', False)

        # Hessian log-scale: use log mapping instead of sigmoid for better inter-layer differentiation
        self.hessian_log_scale = kwargs.get('hessian_log_scale', True)
        # Kurtosis adaptive thresholds: use data-driven percentile-based thresholds
        self.kurtosis_adaptive_thresholds = kwargs.get('kurtosis_adaptive_thresholds', True)
        self.kurtosis_percentile_low = kwargs.get('kurtosis_percentile_low', 10.0)
        self.kurtosis_percentile_high = kwargs.get('kurtosis_percentile_high', 90.0)

        # Down projection ratio threshold for adaptive quantization type selection
        # When adaptive_ratio is enabled, down_proj layers with ratio < threshold use int4_hadamard,
        # and those with ratio >= threshold use w8a8_dynamic.
        # Default: midpoint of (adaptive_min_ratio + adaptive_max_ratio) / 2
        self.down_proj_ratio_threshold = kwargs.get('down_proj_ratio_threshold', None)

        # Remove Ub (basis) mode - use rotation only for value projection
        # When enabled:
        # - V_proj output is rotated with Rb only (no Pb basis)
        # - O_proj absorbs Rb^(-1) only (no Pb^(-1))
        # - rearrange_o_proj() is skipped (no column reordering)
        self.remove_ub = kwargs.get('remove_ub', False)

        # Save online rotation matrices (Uc, Pd, Hd, Ud) to the weight file.
        # When False (default), these matrices are NOT saved — producing
        # inference-ready checkpoints that vLLM can load directly.
        # When True, rotation matrices are saved based on the actual
        # quantizer types (e.g., Uc only if o_proj uses ResQ).
        # Use True only for debugging or research purposes.
        self.save_online_rotations = kwargs.get('save_online_rotations', False)

        # NPU optimization for rotation operations
        # When enabled (default), rotation matmul operations are performed on NPU
        # instead of CPU, providing significant speedup for large models
        self.use_npu_rotation = kwargs.get('use_npu_rotation', True)

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
        self.gptq_device = kwargs.get('gptq_device', None)  # Override device for GPTQ layer processing

        # Hadamard settings
        self.fp32_had = kwargs.get('fp32_had', True)
        self.int8_down_proj = kwargs.get('int8_down_proj', True)

        # Down projection block size for rotation
        self.down_proj_blocksize = kwargs.get('down_proj_blocksize', 256)

        # Ud rotation type: 'hadamard' or 'random'
        # - hadamard: Ud = block_diag(Pd) @ H, save Pd per layer + H globally
        # - random: Ud = block_diag(Pd) @ Rd, save full Ud per layer
        self.ud_rotation_type = kwargs.get('ud_rotation_type', 'hadamard')

        # FFN rotation mode: controls how down_proj rotation is performed
        # - 'ud': (default) Current scheme: Ud = Hd @ block_diag(Pd), fully
        #         fused into down_proj weight, activation quantized in raw space.
        # - 'perm_rd': New scheme: Perm (permutation) forward-fused through
        #         SwiGLU into gate/up weights; Rd (block Hadamard) applied
        #         online to down_proj input before mixed-precision quantization.
        #         Perm computed from per-channel variance of down_proj input.
        self.ffn_rotation_mode = kwargs.get('ffn_rotation_mode', 'ud')

        # Block size for online Rd (block Hadamard) in perm_rd mode.
        # Each precision group is partitioned into blocks of this size,
        # and a normalized Hadamard matrix H_{rd_block_size} is applied
        # within each block.  Only used when ffn_rotation_mode='perm_rd'.
        self.rd_block_size = kwargs.get('rd_block_size', 32)

        # Maximum tensor parallelism supported by the checkpoint.
        # Used in perm_rd mode to compute per-group permutation:
        #   group_size = intermediate_size / max_tp
        # Each TP rank receives exactly N/max_tp contiguous channels,
        # and the [low | high] split is applied within each group.
        self.max_tp = kwargs.get('max_tp', 2)

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
        assert self.ud_rotation_type in ['hadamard', 'random'], \
            f"Invalid ud_rotation_type: {self.ud_rotation_type}. Only 'hadamard' and 'random' are supported."
        assert self.ffn_rotation_mode in ['ud', 'perm_rd'], \
            f"Invalid ffn_rotation_mode: {self.ffn_rotation_mode}. Only 'ud' and 'perm_rd' are supported."
        if self.ffn_rotation_mode == 'perm_rd':
            assert self.rd_block_size > 0 and (self.rd_block_size & (self.rd_block_size - 1)) == 0, \
                f"rd_block_size must be a positive power of 2, got {self.rd_block_size}"
            assert self.max_tp >= 1 and (self.max_tp & (self.max_tp - 1)) == 0, \
                f"max_tp must be a positive power of 2, got {self.max_tp}"
        assert self.output_mode in ['fused', 'transforms_only', 'debug'], \
            f"Invalid output_mode: {self.output_mode}. Valid options: 'fused', 'transforms_only', 'debug'"
        assert 0.0 <= self.high_fraction <= 1.0, "high_fraction must be between 0 and 1"

        # Validate adaptive ratio configuration
        if self.adaptive_ratio:
            valid_algorithms = ['hessian', 'kurtosis', 'cev', 'hybrid']
            assert self.adaptive_algorithm in valid_algorithms, \
                f"Invalid adaptive_algorithm: {self.adaptive_algorithm}. Valid options: {valid_algorithms}"
            assert 0.0 <= self.adaptive_min_ratio <= 1.0, \
                "adaptive_min_ratio must be between 0 and 1"
            assert 0.0 <= self.adaptive_max_ratio <= 1.0, \
                "adaptive_max_ratio must be between 0 and 1"
            assert self.adaptive_min_ratio <= self.adaptive_max_ratio, \
                "adaptive_min_ratio must be <= adaptive_max_ratio"
            assert 0.0 < self.cev_target_variance <= 1.0, \
                "cev_target_variance must be in (0, 1]"
            assert self.adaptive_alignment > 0, \
                "adaptive_alignment must be positive"
            assert self.ub_head_aggregation in ['max', 'mean'], \
                f"Invalid ub_head_aggregation: {self.ub_head_aggregation}. Valid options: 'max', 'mean'"
            # Validate per-transform algorithm overrides
            for transform, alg in self.transform_algorithms.items():
                assert transform in ['Ua', 'Ub', 'Uc', 'Ud'], \
                    f"Invalid transform key: {transform}. Valid keys: 'Ua', 'Ub', 'Uc', 'Ud'"
                assert alg in valid_algorithms, \
                    f"Invalid algorithm for {transform}: {alg}. Valid options: {valid_algorithms}"
            # Validate kurtosis adaptive threshold percentiles
            if self.kurtosis_adaptive_thresholds:
                assert 0.0 <= self.kurtosis_percentile_low < self.kurtosis_percentile_high <= 100.0, \
                    f"kurtosis_percentile_low ({self.kurtosis_percentile_low}) must be < " \
                    f"kurtosis_percentile_high ({self.kurtosis_percentile_high}), both in [0, 100]"

        # Validate remove_ub configuration
        assert isinstance(self.remove_ub, bool), \
            f"remove_ub must be a boolean, got {type(self.remove_ub)}"

        assert isinstance(self.save_online_rotations, bool), \
            f"save_online_rotations must be a boolean, got {type(self.save_online_rotations)}"

        # Validate mix_cfg configuration
        if self.mix_cfg:
            for pattern, quant_type in self.mix_cfg.items():
                assert quant_type.lower() in RESQ_ALLOWED_MIX_TYPES, \
                    f"mix_cfg type '{quant_type}' not in allowed types: {RESQ_ALLOWED_MIX_TYPES}"

        if strict:
            assert self.optimized_basis_path is not None, \
                "optimized_basis_path is required for ResQ mode (use strict=False for simplified mode)"
