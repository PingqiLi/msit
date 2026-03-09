# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
Adaptive precision ratio algorithms for ResQ quantization.

This module provides algorithms to adaptively determine the high/low precision
ratio (high_fraction) per layer and per transformation based on quantization
difficulty metrics, replacing the fixed ratio of 0.125.

Algorithms:
- HessianTraceAlgorithm: Uses eigenvalues (Hessian diagonal approximation)
- KurtosisAlgorithm: Uses activation distribution peakedness
- CEVAlgorithm: Cumulative Explained Variance algorithm
- HybridAlgorithm: Weighted combination of all three

Transformations:
- Ua: Shared attention/MLP basis (hidden_dim)
- Ub: Per-head value projection (head_dim per head)
- Uc: Key position basis (head_dim)
- Ud: Down projection basis (down_proj_blocksize)
"""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveRatioConfig:
    """Configuration for adaptive ratio algorithms."""

    # Enable adaptive ratio computation
    enabled: bool = False

    # Default algorithm to use: 'hessian', 'kurtosis', 'cev', 'hybrid'
    algorithm: str = 'hybrid'

    # Ratio bounds
    min_ratio: float = 0.0625  # 1/16
    max_ratio: float = 0.25    # 1/4

    # CEV algorithm parameters
    cev_target_variance: float = 0.95

    # Hessian algorithm parameters
    hessian_scale_factor: float = 1.0
    hessian_log_scale: bool = True        # Use log-scale instead of sigmoid

    # Kurtosis algorithm parameters
    kurtosis_threshold_low: float = 3.0   # Normal distribution kurtosis
    kurtosis_threshold_high: float = 10.0  # High kurtosis threshold
    kurtosis_adaptive_thresholds: bool = True   # Use data-driven thresholds
    kurtosis_percentile_low: float = 10.0       # Low percentile for threshold
    kurtosis_percentile_high: float = 90.0      # High percentile for threshold

    # Hybrid algorithm weights
    hybrid_weight_hessian: float = 0.4
    hybrid_weight_kurtosis: float = 0.3
    hybrid_weight_cev: float = 0.3

    # Hardware alignment
    alignment: int = 512

    # Per-transform algorithm override (optional)
    # Keys: 'Ua', 'Ub', 'Uc', 'Ud'
    transform_algorithms: Dict[str, str] = field(default_factory=dict)

    # Ub aggregation across heads: 'max' or 'mean'
    ub_head_aggregation: str = 'max'


def align_dimension_split(
    dim: int,
    high_fraction: float,
    alignment: int = 512,
    mode: str = 'round'
) -> Tuple[int, int]:
    """
    Align dimension split to hardware boundary.

    Args:
        dim: Total dimension to split
        high_fraction: Target fraction for high precision
        alignment: Alignment boundary (default 512)
        mode: Alignment mode - 'round', 'ceil', or 'floor'

    Returns:
        Tuple of (low_dim, high_dim) aligned to boundary
    """
    if dim < alignment:
        # Fallback for small dimensions - no alignment
        high_dim = int(high_fraction * dim)
        low_dim = dim - high_dim
        return low_dim, high_dim

    target = high_fraction * dim

    if mode == 'round':
        high_dim = round(target / alignment) * alignment
    elif mode == 'ceil':
        high_dim = math.ceil(target / alignment) * alignment
    elif mode == 'floor':
        high_dim = math.floor(target / alignment) * alignment
    else:
        raise ValueError(f"Unknown alignment mode: {mode}")

    # Ensure high_dim is at least one alignment block (never 0)
    high_dim = max(alignment, min(dim, high_dim))
    low_dim = dim - high_dim

    return low_dim, high_dim


def compute_aligned_fraction(
    dim: int,
    high_fraction: float,
    alignment: int = 512,
    mode: str = 'round'
) -> float:
    """
    Compute the actual high_fraction after alignment.

    Args:
        dim: Total dimension
        high_fraction: Target fraction
        alignment: Alignment boundary
        mode: Alignment mode

    Returns:
        Actual high_fraction after alignment
    """
    low_dim, high_dim = align_dimension_split(dim, high_fraction, alignment, mode)
    return high_dim / dim if dim > 0 else high_fraction


class BaseRatioAlgorithm(ABC):
    """Base class for adaptive ratio algorithms."""

    def __init__(self, config: AdaptiveRatioConfig):
        self.config = config

    @abstractmethod
    def compute_ratio(
        self,
        eigenvalues: Optional[torch.Tensor] = None,
        kurtosis: Optional[float] = None,
        dim: int = None,
        **kwargs
    ) -> float:
        """
        Compute the high precision ratio.

        Args:
            eigenvalues: Eigenvalues from covariance decomposition (sorted ascending)
            kurtosis: Kurtosis value of activations
            dim: Total dimension
            **kwargs: Additional algorithm-specific parameters

        Returns:
            Computed high_fraction ratio in [min_ratio, max_ratio]
        """
        pass

    def clamp_ratio(self, ratio: float) -> float:
        """Clamp ratio to configured bounds."""
        return max(self.config.min_ratio, min(self.config.max_ratio, ratio))


class HessianTraceAlgorithm(BaseRatioAlgorithm):
    """
    Hessian Trace Algorithm.

    Uses eigenvalues (Hessian diagonal approximation) to measure quantization
    sensitivity. Higher trace indicates more sensitive layers that need higher
    precision.

    Formula:
        normalized_trace = sum(eigenvalues) / dim
        ratio = min_ratio + (max_ratio - min_ratio) * scale(normalized_trace)
    """

    def __init__(self, config: AdaptiveRatioConfig, reference_trace: float = None):
        super().__init__(config)
        self.reference_trace = reference_trace
        self._trace_history = []

    def compute_ratio(
        self,
        eigenvalues: Optional[torch.Tensor] = None,
        kurtosis: Optional[float] = None,
        dim: int = None,
        **kwargs
    ) -> float:
        if eigenvalues is None:
            logger.warning("HessianTraceAlgorithm: No eigenvalues provided, using default ratio")
            return (self.config.min_ratio + self.config.max_ratio) / 2

        # Compute normalized trace
        total_trace = eigenvalues.sum().item()
        normalized_trace = total_trace / len(eigenvalues)

        # Store for reference computation
        self._trace_history.append(normalized_trace)

        # Compute relative trace using configured method
        if self.config.hessian_log_scale:
            # Log-scale mapping: preserves inter-layer differences
            log_trace = math.log(1.0 + normalized_trace)
            if self.reference_trace is not None and self.reference_trace > 0:
                log_ref = math.log(1.0 + self.reference_trace)
                relative_trace = log_trace / (2.0 * log_ref)
            else:
                # Pass 1 placeholder: return midpoint, will be recomputed in pass 2
                return (self.config.min_ratio + self.config.max_ratio) / 2
        elif self.reference_trace is not None and self.reference_trace > 0:
            relative_trace = normalized_trace / self.reference_trace
        else:
            # Use sigmoid-like scaling for relative importance
            # Scale factor adjusts sensitivity
            scale_factor = self.config.hessian_scale_factor
            relative_trace = 2.0 / (1.0 + math.exp(-scale_factor * normalized_trace)) - 1.0

        # Map to ratio range
        ratio_range = self.config.max_ratio - self.config.min_ratio
        ratio = self.config.min_ratio + ratio_range * min(1.0, max(0.0, relative_trace))

        return self.clamp_ratio(ratio)

    def set_reference_trace(self, reference_trace: float):
        """Set reference trace for normalization."""
        self.reference_trace = reference_trace

    def compute_reference_from_history(self):
        """Compute reference trace from collected history."""
        if self._trace_history:
            self.reference_trace = sum(self._trace_history) / len(self._trace_history)


class KurtosisAlgorithm(BaseRatioAlgorithm):
    """
    Kurtosis Algorithm.

    Measures activation distribution "peakedness" - higher kurtosis means more
    outliers, which require higher precision for accurate quantization.

    Formula:
        kurtosis = E[(x - mean)^4] / std^4 - 3
        ratio = interpolate(kurtosis, [threshold_low, threshold_high], [min_ratio, max_ratio])
    """

    def __init__(self, config: AdaptiveRatioConfig):
        super().__init__(config)
        self._kurtosis_history = []
        self._adaptive_threshold_low = None
        self._adaptive_threshold_high = None

    def compute_ratio(
        self,
        eigenvalues: Optional[torch.Tensor] = None,
        kurtosis: Optional[float] = None,
        dim: int = None,
        **kwargs
    ) -> float:
        if kurtosis is None:
            logger.warning("KurtosisAlgorithm: No kurtosis provided, using default ratio")
            return (self.config.min_ratio + self.config.max_ratio) / 2

        # Collect kurtosis values for adaptive threshold computation
        if self.config.kurtosis_adaptive_thresholds:
            self._kurtosis_history.append(kurtosis)

        # Use adaptive thresholds if available, otherwise use configured thresholds
        if self._adaptive_threshold_low is not None and self._adaptive_threshold_high is not None:
            threshold_low = self._adaptive_threshold_low
            threshold_high = self._adaptive_threshold_high
        else:
            threshold_low = self.config.kurtosis_threshold_low
            threshold_high = self.config.kurtosis_threshold_high

        if kurtosis <= threshold_low:
            ratio = self.config.min_ratio
        elif kurtosis >= threshold_high:
            ratio = self.config.max_ratio
        else:
            # Linear interpolation
            t = (kurtosis - threshold_low) / (threshold_high - threshold_low)
            ratio = self.config.min_ratio + t * (self.config.max_ratio - self.config.min_ratio)

        return self.clamp_ratio(ratio)

    def set_adaptive_thresholds(self, all_kurtosis_values: List[float]):
        """Compute adaptive thresholds from collected kurtosis values using percentiles."""
        if not all_kurtosis_values:
            logger.warning("KurtosisAlgorithm: No kurtosis values for adaptive thresholds")
            return

        sorted_values = sorted(all_kurtosis_values)
        n = len(sorted_values)

        # Compute percentile indices
        low_idx = max(0, int(self.config.kurtosis_percentile_low / 100.0 * n) - 1)
        high_idx = min(n - 1, int(self.config.kurtosis_percentile_high / 100.0 * n))

        self._adaptive_threshold_low = sorted_values[low_idx]
        self._adaptive_threshold_high = sorted_values[high_idx]

        # Guard against degenerate case
        if self._adaptive_threshold_high <= self._adaptive_threshold_low:
            self._adaptive_threshold_high = self._adaptive_threshold_low + 1.0

        logger.info(
            f"KurtosisAlgorithm: Adaptive thresholds set to "
            f"[{self._adaptive_threshold_low:.4f}, {self._adaptive_threshold_high:.4f}] "
            f"(from {n} values, p{self.config.kurtosis_percentile_low:.0f}/p{self.config.kurtosis_percentile_high:.0f})"
        )

    def compute_reference_from_history(self):
        """Compute adaptive thresholds from collected kurtosis history."""
        if self._kurtosis_history:
            self.set_adaptive_thresholds(self._kurtosis_history)


class CEVAlgorithm(BaseRatioAlgorithm):
    """
    Cumulative Explained Variance (CEV) Algorithm.

    Finds the number of dimensions needed to explain a target variance
    (e.g., 95%). Layers where variance is concentrated in few dimensions
    need higher precision for those critical dimensions.

    Formula:
        cum_var = cumsum(sorted_eigenvalues_desc) / total_variance
        high_dim = first index where cum_var >= target_variance
        ratio = clamp(high_dim / total_dim, min_ratio, max_ratio)
    """

    def compute_ratio(
        self,
        eigenvalues: Optional[torch.Tensor] = None,
        kurtosis: Optional[float] = None,
        dim: int = None,
        **kwargs
    ) -> float:
        if eigenvalues is None:
            logger.warning("CEVAlgorithm: No eigenvalues provided, using default ratio")
            return (self.config.min_ratio + self.config.max_ratio) / 2

        # Sort eigenvalues in descending order
        sorted_evals, _ = torch.sort(eigenvalues, descending=True)

        # Compute cumulative explained variance
        total_variance = sorted_evals.sum().item()
        if total_variance <= 0:
            return self.config.min_ratio

        cum_variance = torch.cumsum(sorted_evals, dim=0) / total_variance

        # Find first index where cum_var >= target_variance
        target = self.config.cev_target_variance
        indices = torch.where(cum_variance >= target)[0]

        if len(indices) > 0:
            high_dim = indices[0].item() + 1  # +1 because we want count, not index
        else:
            high_dim = len(eigenvalues)

        # Compute ratio
        total_dim = dim if dim is not None else len(eigenvalues)
        ratio = high_dim / total_dim

        return self.clamp_ratio(ratio)


class HybridAlgorithm(BaseRatioAlgorithm):
    """
    Hybrid Algorithm.

    Weighted combination of Hessian, Kurtosis, and CEV algorithms for
    robust ratio estimation.

    Formula:
        ratio = w_hessian * hessian_ratio + w_kurtosis * kurtosis_ratio + w_cev * cev_ratio
    """

    def __init__(self, config: AdaptiveRatioConfig):
        super().__init__(config)
        self.hessian_alg = HessianTraceAlgorithm(config)
        self.kurtosis_alg = KurtosisAlgorithm(config)
        self.cev_alg = CEVAlgorithm(config)

    def compute_ratio(
        self,
        eigenvalues: Optional[torch.Tensor] = None,
        kurtosis: Optional[float] = None,
        dim: int = None,
        **kwargs
    ) -> float:
        ratios = []
        weights = []

        # Compute individual ratios based on available data
        if eigenvalues is not None:
            hessian_ratio = self.hessian_alg.compute_ratio(eigenvalues=eigenvalues, dim=dim)
            ratios.append(hessian_ratio)
            weights.append(self.config.hybrid_weight_hessian)

            cev_ratio = self.cev_alg.compute_ratio(eigenvalues=eigenvalues, dim=dim)
            ratios.append(cev_ratio)
            weights.append(self.config.hybrid_weight_cev)

        if kurtosis is not None:
            kurtosis_ratio = self.kurtosis_alg.compute_ratio(kurtosis=kurtosis)
            ratios.append(kurtosis_ratio)
            weights.append(self.config.hybrid_weight_kurtosis)

        if not ratios:
            logger.warning("HybridAlgorithm: No data available, using default ratio")
            return (self.config.min_ratio + self.config.max_ratio) / 2

        # Weighted average
        total_weight = sum(weights)
        weighted_ratio = sum(r * w for r, w in zip(ratios, weights)) / total_weight

        return self.clamp_ratio(weighted_ratio)

    def set_reference_trace(self, reference_trace: float):
        """Set reference trace for Hessian algorithm."""
        self.hessian_alg.set_reference_trace(reference_trace)

    def compute_reference_from_history(self):
        """Propagate reference computation to sub-algorithms."""
        self.hessian_alg.compute_reference_from_history()
        self.kurtosis_alg.compute_reference_from_history()


def create_algorithm(name: str, config: AdaptiveRatioConfig) -> BaseRatioAlgorithm:
    """
    Factory function to create ratio algorithm by name.

    Args:
        name: Algorithm name ('hessian', 'kurtosis', 'cev', 'hybrid')
        config: Algorithm configuration

    Returns:
        Algorithm instance
    """
    algorithms = {
        'hessian': HessianTraceAlgorithm,
        'kurtosis': KurtosisAlgorithm,
        'cev': CEVAlgorithm,
        'hybrid': HybridAlgorithm,
    }

    if name not in algorithms:
        raise ValueError(f"Unknown algorithm: {name}. Valid options: {list(algorithms.keys())}")

    return algorithms[name](config)


@dataclass
class AdaptiveRatioResult:
    """Result of adaptive ratio computation."""

    # Per-layer/per-transform ratios
    ratios: Dict[str, float] = field(default_factory=dict)

    # Aligned dimension splits
    splits: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    # Metrics used for computation
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Algorithm used for each transform
    algorithms_used: Dict[str, str] = field(default_factory=dict)


class AdaptiveRatioComputer:
    """
    Main class for computing adaptive ratios across all layers and transforms.

    This class coordinates the ratio computation for all transformations:
    - Ua: Shared attention/MLP basis
    - Ub: Per-head value projection (aggregated across heads)
    - Uc: Key position basis
    - Ud: Down projection basis
    """

    def __init__(self, config: AdaptiveRatioConfig):
        self.config = config

        # Create algorithms for each transform type
        self.algorithms = {}
        for transform in ['Ua', 'Ub', 'Uc', 'Ud']:
            alg_name = config.transform_algorithms.get(transform, config.algorithm)
            self.algorithms[transform] = create_algorithm(alg_name, config)

    def _needs_two_pass(self) -> bool:
        """Check if any algorithm requires a two-pass approach."""
        return self.config.hessian_log_scale or self.config.kurtosis_adaptive_thresholds

    def _compute_references_from_history(self):
        """Compute GLOBAL references from all transforms' histories."""
        hessian_algs = []
        kurtosis_algs = []

        for transform, algorithm in self.algorithms.items():
            if isinstance(algorithm, HybridAlgorithm):
                hessian_algs.append((transform, algorithm.hessian_alg))
                kurtosis_algs.append((transform, algorithm.kurtosis_alg))
            elif isinstance(algorithm, HessianTraceAlgorithm):
                hessian_algs.append((transform, algorithm))
            elif isinstance(algorithm, KurtosisAlgorithm):
                kurtosis_algs.append((transform, algorithm))

        # Global hessian reference: pool traces from all transforms
        all_traces = []
        for transform, alg in hessian_algs:
            all_traces.extend(alg._trace_history)

        if all_traces:
            global_ref = sum(all_traces) / len(all_traces)
            logger.info(f"Global hessian reference_trace: {global_ref:.6f} "
                        f"(from {len(all_traces)} entries across {len(hessian_algs)} transforms)")
            for _, alg in hessian_algs:
                alg.set_reference_trace(global_ref)

        # Global kurtosis thresholds: pool kurtosis from all transforms
        all_kurtosis = []
        for transform, alg in kurtosis_algs:
            all_kurtosis.extend(alg._kurtosis_history)

        if all_kurtosis and self.config.kurtosis_adaptive_thresholds:
            logger.info(f"Global kurtosis thresholds from {len(all_kurtosis)} values "
                        f"across {len(kurtosis_algs)} transforms")
            for _, alg in kurtosis_algs:
                alg.set_adaptive_thresholds(all_kurtosis)

    def _run_pass(
        self,
        eval_dict: Dict[str, torch.Tensor],
        kurtosis_dict: Optional[Dict[str, float]],
        model_config: Any,
        collect_only: bool = False,
    ) -> AdaptiveRatioResult:
        """
        Run a single pass of ratio computation.

        Args:
            eval_dict: Dictionary of eigenvalues
            kurtosis_dict: Dictionary of kurtosis values (optional)
            model_config: Model configuration for dimension info
            collect_only: If True, skip computing splits (pass 1 optimization)

        Returns:
            AdaptiveRatioResult with computed ratios and metrics
        """
        result = AdaptiveRatioResult()

        # Get dimensions from model config
        if model_config is not None:
            hidden_dim = model_config.hidden_size
            num_kv_heads = getattr(model_config, 'num_key_value_heads',
                                   model_config.num_attention_heads)
            head_dim = getattr(model_config, 'head_dim',
                              hidden_dim // model_config.num_attention_heads)
            intermediate_size = getattr(model_config, 'intermediate_size', None)
            # Infer number of layers from eval_dict
            nlayers = sum(1 for k in eval_dict.keys() if k.startswith('layer.') and 'value' in k)
        else:
            hidden_dim = None
            num_kv_heads = None
            head_dim = None
            intermediate_size = None
            nlayers = 0

        # Compute Ua ratio (shared attn_mlp basis)
        if 'attn_mlp' in eval_dict:
            per_layer_traces = eval_dict.get('attn_mlp_per_layer_traces')
            per_layer_kurtosis = kurtosis_dict.get('attn_mlp') if kurtosis_dict else None

            if collect_only and per_layer_traces:
                # Pass 1: feed per-layer traces into Ua's history (64 entries)
                algorithm = self.algorithms['Ua']
                for i, trace in enumerate(per_layer_traces):
                    layer_kurtosis = None
                    if isinstance(per_layer_kurtosis, list) and i < len(per_layer_kurtosis):
                        layer_kurtosis = per_layer_kurtosis[i]
                    # Create a 1-element tensor whose sum/len equals the trace
                    synthetic_evals = torch.tensor([trace])
                    algorithm.compute_ratio(
                        eigenvalues=synthetic_evals,
                        kurtosis=layer_kurtosis,
                        dim=hidden_dim
                    )
                # Placeholder ratio for pass 1
                result.ratios['Ua'] = (self.config.min_ratio + self.config.max_ratio) / 2
                result.metrics['Ua'] = {}
                result.algorithms_used['Ua'] = self.config.transform_algorithms.get(
                    'Ua', self.config.algorithm
                )
            else:
                # Pass 2 or single-pass: use shared eigenvalues for actual ratio
                ua_kurtosis_for_ratio = None
                if isinstance(per_layer_kurtosis, list) and per_layer_kurtosis:
                    ua_kurtosis_for_ratio = sum(per_layer_kurtosis) / len(per_layer_kurtosis)
                elif isinstance(per_layer_kurtosis, (int, float)):
                    ua_kurtosis_for_ratio = per_layer_kurtosis

                ua_ratio, ua_metrics = self._compute_ratio_for_key(
                    'Ua', 'attn_mlp', eval_dict,
                    {'attn_mlp': ua_kurtosis_for_ratio} if ua_kurtosis_for_ratio is not None else kurtosis_dict,
                    hidden_dim
                )
                result.ratios['Ua'] = ua_ratio
                result.metrics['Ua'] = ua_metrics
                result.algorithms_used['Ua'] = self.config.transform_algorithms.get(
                    'Ua', self.config.algorithm
                )
                if hidden_dim:
                    result.splits['Ua'] = align_dimension_split(
                        hidden_dim, ua_ratio, self.config.alignment
                    )

        # Compute per-layer ratios
        for i in range(nlayers):
            # Ub: value projection (per-head, aggregated)
            value_key = f'layer.{i}.self_attn.value'
            if value_key in eval_dict:
                ub_ratio, ub_metrics = self._compute_ub_ratio(
                    i, eval_dict, kurtosis_dict, num_kv_heads, head_dim,
                    collect_only=collect_only
                )
                result.ratios[f'layer.{i}.Ub'] = ub_ratio
                result.metrics[f'layer.{i}.Ub'] = ub_metrics
                result.algorithms_used[f'layer.{i}.Ub'] = self.config.transform_algorithms.get(
                    'Ub', self.config.algorithm
                )
                if not collect_only and head_dim:
                    result.splits[f'layer.{i}.Ub'] = align_dimension_split(
                        head_dim, ub_ratio, self.config.alignment
                    )

            # Uc: key position
            key_pos_key = f'layer.{i}.self_attn.key_pos'
            if key_pos_key in eval_dict:
                uc_ratio, uc_metrics = self._compute_ratio_for_key(
                    'Uc', key_pos_key, eval_dict, kurtosis_dict, head_dim
                )
                result.ratios[f'layer.{i}.Uc'] = uc_ratio
                result.metrics[f'layer.{i}.Uc'] = uc_metrics
                result.algorithms_used[f'layer.{i}.Uc'] = self.config.transform_algorithms.get(
                    'Uc', self.config.algorithm
                )
                if not collect_only and head_dim:
                    result.splits[f'layer.{i}.Uc'] = align_dimension_split(
                        head_dim, uc_ratio, self.config.alignment
                    )

            # Ud: down projection
            down_proj_key = f'layer.{i}.mlp.down_proj'
            if down_proj_key in eval_dict:
                ud_ratio, ud_metrics = self._compute_ratio_for_key(
                    'Ud', down_proj_key, eval_dict, kurtosis_dict
                )
                result.ratios[f'layer.{i}.Ud'] = ud_ratio
                result.metrics[f'layer.{i}.Ud'] = ud_metrics
                result.algorithms_used[f'layer.{i}.Ud'] = self.config.transform_algorithms.get(
                    'Ud', self.config.algorithm
                )
                # Compute aligned split at full intermediate_size level (NOT blocksize)
                if not collect_only and intermediate_size:
                    result.splits[f'layer.{i}.Ud'] = align_dimension_split(
                        intermediate_size, ud_ratio, self.config.alignment
                    )

        return result

    def compute_all_ratios(
        self,
        eval_dict: Dict[str, torch.Tensor],
        kurtosis_dict: Optional[Dict[str, float]] = None,
        model_config: Any = None,
    ) -> AdaptiveRatioResult:
        """
        Compute adaptive ratios for all layers and transforms.

        When hessian_log_scale or kurtosis_adaptive_thresholds is enabled,
        uses a two-pass approach:
        - Pass 1: Collect statistics (trace history, kurtosis values)
        - Compute references from collected statistics
        - Pass 2: Compute final ratios using calibrated references

        Args:
            eval_dict: Dictionary of eigenvalues from compute_basis()
                Keys: 'attn_mlp', 'layer.{i}.self_attn.value',
                      'layer.{i}.self_attn.key_pos', 'layer.{i}.mlp.down_proj'
            kurtosis_dict: Dictionary of kurtosis values (optional)
                Keys: Same as eval_dict
            model_config: Model configuration for dimension info

        Returns:
            AdaptiveRatioResult with computed ratios and metrics
        """
        if self._needs_two_pass():
            logger.info("Two-pass adaptive ratio: Pass 1/2 — collecting statistics...")
            # Pass 1: collect statistics (trace history, kurtosis values)
            self._run_pass(eval_dict, kurtosis_dict, model_config, collect_only=True)

            # Compute references from collected statistics
            logger.info("Two-pass adaptive ratio: Computing references from collected statistics...")
            self._compute_references_from_history()

            # Pass 2: compute final ratios with calibrated references
            logger.info("Two-pass adaptive ratio: Pass 2/2 — computing final ratios...")
            result = self._run_pass(eval_dict, kurtosis_dict, model_config, collect_only=False)
            return result
        else:
            # Single pass (identical to previous behavior)
            return self._run_pass(eval_dict, kurtosis_dict, model_config, collect_only=False)

    def _compute_ratio_for_key(
        self,
        transform: str,
        key: str,
        eval_dict: Dict[str, torch.Tensor],
        kurtosis_dict: Optional[Dict[str, float]],
        dim: Optional[int] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute ratio for a single key."""
        eigenvalues = eval_dict.get(key)
        kurtosis = kurtosis_dict.get(key) if kurtosis_dict else None

        algorithm = self.algorithms[transform]
        ratio = algorithm.compute_ratio(
            eigenvalues=eigenvalues,
            kurtosis=kurtosis,
            dim=dim
        )

        metrics = {}
        if eigenvalues is not None:
            metrics['trace'] = eigenvalues.sum().item()
            metrics['mean_eigenvalue'] = eigenvalues.mean().item()
            metrics['max_eigenvalue'] = eigenvalues.max().item()
        if kurtosis is not None:
            metrics['kurtosis'] = kurtosis

        return ratio, metrics

    def _compute_ub_ratio(
        self,
        layer_idx: int,
        eval_dict: Dict[str, torch.Tensor],
        kurtosis_dict: Optional[Dict[str, float]],
        num_kv_heads: Optional[int],
        head_dim: Optional[int],
        collect_only: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute Ub ratio with per-head aggregation.

        When collect_only=True (pass 1), selects the representative head and
        calls compute_ratio once to add 1 history entry per layer.
        When collect_only=False (pass 2 / single-pass), uses full per-head
        computation with max/mean aggregation.
        """
        value_key = f'layer.{layer_idx}.self_attn.value'
        eigenvalues = eval_dict.get(value_key)

        if eigenvalues is None:
            return (self.config.min_ratio + self.config.max_ratio) / 2, {}

        # Eigenvalues shape: [num_kv_heads, head_dim]
        if eigenvalues.dim() == 2:
            algorithm = self.algorithms['Ub']

            if collect_only:
                # Pass 1: select representative head, call compute_ratio once
                # This gives 1 history entry per layer (64 total) instead of 512
                per_head_traces = eigenvalues.sum(dim=1)  # [num_kv_heads]

                if self.config.ub_head_aggregation == 'max':
                    rep_head = per_head_traces.argmax().item()
                else:  # 'mean'
                    mean_trace = per_head_traces.mean()
                    rep_head = (per_head_traces - mean_trace).abs().argmin().item()

                rep_evals = eigenvalues[rep_head]
                rep_kurtosis = None
                if kurtosis_dict and value_key in kurtosis_dict:
                    kurtosis_data = kurtosis_dict[value_key]
                    if isinstance(kurtosis_data, (list, torch.Tensor)):
                        rep_kurtosis = kurtosis_data[rep_head] if rep_head < len(kurtosis_data) else None
                    else:
                        rep_kurtosis = kurtosis_data

                algorithm.compute_ratio(
                    eigenvalues=rep_evals,
                    kurtosis=rep_kurtosis,
                    dim=head_dim
                )
                # Placeholder ratio for pass 1
                ratio = (self.config.min_ratio + self.config.max_ratio) / 2
            else:
                # Pass 2 or single-pass: full per-head computation
                per_head_ratios = []

                for h in range(eigenvalues.shape[0]):
                    head_evals = eigenvalues[h]
                    kurtosis = None
                    if kurtosis_dict and value_key in kurtosis_dict:
                        kurtosis_data = kurtosis_dict[value_key]
                        if isinstance(kurtosis_data, (list, torch.Tensor)):
                            kurtosis = kurtosis_data[h] if h < len(kurtosis_data) else None
                        else:
                            kurtosis = kurtosis_data

                    head_ratio = algorithm.compute_ratio(
                        eigenvalues=head_evals,
                        kurtosis=kurtosis,
                        dim=head_dim
                    )
                    per_head_ratios.append(head_ratio)

                # Aggregate across heads
                if self.config.ub_head_aggregation == 'max':
                    ratio = max(per_head_ratios)
                else:  # 'mean'
                    ratio = sum(per_head_ratios) / len(per_head_ratios)
        else:
            # Single eigenvalue set (already aggregated)
            kurtosis = kurtosis_dict.get(value_key) if kurtosis_dict else None
            ratio = self.algorithms['Ub'].compute_ratio(
                eigenvalues=eigenvalues,
                kurtosis=kurtosis,
                dim=head_dim
            )

        metrics = {
            'trace': eigenvalues.sum().item() if eigenvalues.dim() == 1 else eigenvalues.sum().item(),
            'aggregation': self.config.ub_head_aggregation,
        }

        return ratio, metrics


def compute_kurtosis(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute kurtosis of a tensor along specified dimension.

    Kurtosis measures the "tailedness" of the distribution.
    Normal distribution has kurtosis = 3, so excess kurtosis = kurtosis - 3.

    Args:
        tensor: Input tensor
        dim: Dimension along which to compute kurtosis

    Returns:
        Excess kurtosis values
    """
    mean = tensor.mean(dim=dim, keepdim=True)
    centered = tensor - mean
    std = tensor.std(dim=dim, keepdim=True)

    # Avoid division by zero
    std = torch.clamp(std, min=1e-8)

    # Fourth moment
    m4 = (centered ** 4).mean(dim=dim)
    # Excess kurtosis = E[(x-mean)^4] / std^4 - 3
    kurtosis = m4 / (std.squeeze(dim) ** 4) - 3

    return kurtosis


class StreamingKurtosisComputer:
    """
    Streaming kurtosis computation for memory-efficient batch processing.

    Uses Welford's online algorithm extended for higher moments.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.n = 0
        self.mean = torch.zeros(dim, dtype=torch.float64)
        self.m2 = torch.zeros(dim, dtype=torch.float64)  # Sum of squared deviations
        self.m4 = torch.zeros(dim, dtype=torch.float64)  # Sum of fourth-power deviations

    def update(self, x: torch.Tensor):
        """
        Update statistics with new batch of data.

        Args:
            x: Input tensor of shape [..., dim]
        """
        x = x.view(-1, self.dim).to(torch.float64)
        batch_size = x.shape[0]

        for i in range(batch_size):
            self.n += 1
            delta = x[i] - self.mean
            delta_n = delta / self.n
            delta_n2 = delta_n * delta_n
            term1 = delta * delta_n * (self.n - 1)

            self.mean = self.mean + delta_n
            self.m4 = self.m4 + term1 * delta_n2 * (self.n * self.n - 3 * self.n + 3) + \
                      6 * delta_n2 * self.m2 - 4 * delta_n * self.m2 / (self.n - 1) if self.n > 1 else self.m4
            self.m2 = self.m2 + term1

    def get_kurtosis(self) -> torch.Tensor:
        """
        Get excess kurtosis from accumulated statistics.

        Returns:
            Excess kurtosis tensor of shape [dim]
        """
        if self.n < 4:
            return torch.zeros(self.dim, dtype=torch.float64)

        variance = self.m2 / (self.n - 1)
        variance = torch.clamp(variance, min=1e-8)

        # Unbiased kurtosis estimator
        kurtosis = (self.n * (self.n + 1) * self.m4 / ((self.n - 1) * (self.n - 2) * (self.n - 3) * variance ** 2)) - \
                   (3 * (self.n - 1) ** 2 / ((self.n - 2) * (self.n - 3)))

        return kurtosis

    def reset(self):
        """Reset accumulated statistics."""
        self.n = 0
        self.mean = torch.zeros(self.dim, dtype=torch.float64)
        self.m2 = torch.zeros(self.dim, dtype=torch.float64)
        self.m4 = torch.zeros(self.dim, dtype=torch.float64)


def save_adaptive_ratios(result: AdaptiveRatioResult, path: str):
    """
    Save adaptive ratio results to JSON file.

    Args:
        result: AdaptiveRatioResult to save
        path: Output file path
    """
    import json

    output = {
        'per_layer_ratios': result.ratios,
        'aligned_splits': {k: {'low_dim': v[0], 'high_dim': v[1]}
                          for k, v in result.splits.items()},
        'metrics': result.metrics,
        'algorithms_used': result.algorithms_used,
    }

    with open(path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved adaptive ratios to {path}")


def load_adaptive_ratios(path: str) -> Dict[str, float]:
    """
    Load adaptive ratio results from JSON file.

    Args:
        path: Input file path

    Returns:
        Dictionary of ratios
    """
    import json

    with open(path, 'r') as f:
        data = json.load(f)

    return data.get('per_layer_ratios', {})
