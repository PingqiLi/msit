# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Adapted from ResQ (https://github.com/facebookresearch/resq)
"""
Basis computation for ResQ quantization.

This module computes the eigenvalue basis matrices from activation
covariances. The basis matrices are used to identify high-variance
channels for mixed-precision quantization.
"""

import logging
from typing import Dict, Optional, Tuple, List, Any

import torch
import torch.nn as nn
from tqdm import tqdm

from ..utils.common import get_device, cleanup_memory

logger = logging.getLogger(__name__)


def perform_eigen_decomp(
    H: torch.Tensor,
    damp_percent: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform eigenvalue decomposition on covariance matrix.

    Args:
        H: Covariance matrix of shape [n, n]
        damp_percent: Dampening percentage for numerical stability

    Returns:
        Tuple of (eigenvalues, eigenvectors) sorted in ascending order
    """
    # Add dampening for numerical stability
    diag = torch.diag(H)
    damp = damp_percent * diag.mean()
    H_damp = H + damp * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)

    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(H_damp)

    # Sort by eigenvalue (ascending - smallest first)
    sorted_indices = torch.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    return eigenvalues, eigenvectors


def compute_covariance(activations: torch.Tensor) -> torch.Tensor:
    """
    Compute covariance matrix from activations.

    Args:
        activations: Activation tensor of shape [num_samples, hidden_dim]

    Returns:
        Covariance matrix of shape [hidden_dim, hidden_dim]
    """
    # Center the data
    mean = activations.mean(dim=0, keepdim=True)
    centered = activations - mean

    # Compute covariance
    n_samples = activations.shape[0]
    cov = (centered.T @ centered) / (n_samples - 1)

    return cov


class BasisCollector:
    """
    Collects activations and computes basis matrices for ResQ.

    This class hooks into model forward passes to collect activations
    at key points (QKV projections, MLP layers) and computes eigenvalue
    decomposition to find the optimal rotation basis.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        device: torch.device = None,
    ):
        """
        Initialize the basis collector.

        Args:
            model: The transformer model
            config: ResQ configuration
            device: Device to use for computation
        """
        self.model = model
        self.config = config
        self.device = device or get_device()

        # Storage for collected activations
        self.activations: Dict[str, List[torch.Tensor]] = {}
        self.hooks = []

        # Model structure
        self._analyze_model_structure()

    def _analyze_model_structure(self):
        """Analyze model structure to find layers."""
        self.hidden_dim = self.model.config.hidden_size
        self.num_heads = self.model.config.num_attention_heads
        self.head_dim = self.hidden_dim // self.num_heads
        self.num_kv_heads = getattr(
            self.model.config, 'num_key_value_heads', self.num_heads
        )

        # Find decoder layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            self.layers = list(self.model.model.layers)
        else:
            raise ValueError("Unsupported model architecture")

        self.num_layers = len(self.layers)

    def _create_hook(self, name: str, collect_input: bool = True):
        """Create a forward hook to collect activations."""
        def hook_fn(module, inp, output):
            if collect_input:
                tensor = inp[0] if isinstance(inp, tuple) else inp
            else:
                tensor = output[0] if isinstance(output, tuple) else output

            if name not in self.activations:
                self.activations[name] = []
            self.activations[name].append(tensor.detach().cpu())

        return hook_fn

    def register_hooks(self, layer_idx: int):
        """
        Register forward hooks for a specific layer.

        Args:
            layer_idx: Index of the layer to hook
        """
        self.clear_hooks()
        layer = self.layers[layer_idx]

        # Hook QKV input (attention input)
        if hasattr(layer, 'self_attn'):
            self.hooks.append(
                layer.self_attn.q_proj.register_forward_hook(
                    self._create_hook(f'layer.{layer_idx}.attn_input', collect_input=True)
                )
            )

            # Hook value projection output
            self.hooks.append(
                layer.self_attn.v_proj.register_forward_hook(
                    self._create_hook(f'layer.{layer_idx}.v_output', collect_input=False)
                )
            )

            # Hook key projection output
            self.hooks.append(
                layer.self_attn.k_proj.register_forward_hook(
                    self._create_hook(f'layer.{layer_idx}.k_output', collect_input=False)
                )
            )

        # Hook MLP input
        if hasattr(layer, 'mlp'):
            self.hooks.append(
                layer.mlp.up_proj.register_forward_hook(
                    self._create_hook(f'layer.{layer_idx}.mlp_input', collect_input=True)
                )
            )

    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear_activations(self):
        """Clear collected activations."""
        self.activations = {}

    def compute_layer_basis(
        self,
        layer_idx: int,
        rotation_granularity: str = 'full_shared',
    ) -> Dict[str, torch.Tensor]:
        """
        Compute basis matrices for a specific layer.

        Args:
            layer_idx: Layer index
            rotation_granularity: Granularity of rotation matrices

        Returns:
            Dictionary of basis matrices
        """
        basis_dict = {}

        # Attention input basis
        attn_key = f'layer.{layer_idx}.attn_input'
        if attn_key in self.activations:
            attn_acts = torch.cat(self.activations[attn_key], dim=0)
            attn_acts = attn_acts.view(-1, self.hidden_dim).to(torch.float64)
            cov = compute_covariance(attn_acts)
            eigenvalues, eigenvectors = perform_eigen_decomp(cov.cpu())
            basis_dict[f'layer.{layer_idx}.self_attn'] = eigenvectors.T
            basis_dict[f'layer.{layer_idx}.self_attn.eigenvalues'] = eigenvalues

        # MLP input basis
        mlp_key = f'layer.{layer_idx}.mlp_input'
        if mlp_key in self.activations:
            mlp_acts = torch.cat(self.activations[mlp_key], dim=0)
            mlp_acts = mlp_acts.view(-1, self.hidden_dim).to(torch.float64)
            cov = compute_covariance(mlp_acts)
            eigenvalues, eigenvectors = perform_eigen_decomp(cov.cpu())
            basis_dict[f'layer.{layer_idx}.mlp'] = eigenvectors.T
            basis_dict[f'layer.{layer_idx}.mlp.eigenvalues'] = eigenvalues

        # Combined attn_mlp basis for full_shared granularity
        if rotation_granularity == 'full_shared':
            if attn_key in self.activations and mlp_key in self.activations:
                combined_acts = torch.cat([
                    torch.cat(self.activations[attn_key], dim=0).view(-1, self.hidden_dim),
                    torch.cat(self.activations[mlp_key], dim=0).view(-1, self.hidden_dim),
                ], dim=0).to(torch.float64)
                cov = compute_covariance(combined_acts)
                eigenvalues, eigenvectors = perform_eigen_decomp(cov.cpu())
                basis_dict[f'layer.{layer_idx}.self_attn_mlp'] = eigenvectors.T
                basis_dict[f'layer.{layer_idx}.self_attn_mlp.eigenvalues'] = eigenvalues

        # Value projection basis (per head)
        v_key = f'layer.{layer_idx}.v_output'
        if v_key in self.activations:
            v_acts = torch.cat(self.activations[v_key], dim=0)
            # Reshape to [batch * seq, num_kv_heads, head_dim]
            v_acts = v_acts.view(-1, self.num_kv_heads, self.head_dim).to(torch.float64)

            v_basis_list = []
            for head_idx in range(self.num_kv_heads):
                head_acts = v_acts[:, head_idx, :]
                cov = compute_covariance(head_acts)
                eigenvalues, eigenvectors = perform_eigen_decomp(cov.cpu())
                v_basis_list.append(eigenvectors.T)

            basis_dict[f'layer.{layer_idx}.self_attn.value'] = torch.stack(v_basis_list)

        return basis_dict


def compute_basis(
    model: nn.Module,
    dataloader,
    config: Any,
    device: torch.device = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute all basis matrices for the model.

    This is the main entry point for basis computation.

    Args:
        model: The transformer model
        dataloader: Calibration data loader
        config: ResQ configuration
        device: Device to use

    Returns:
        Dictionary of all basis matrices
    """
    if device is None:
        device = get_device()

    collector = BasisCollector(model, config, device)
    all_basis = {}

    model.eval()
    model.to(device)

    # Collect activations and compute basis for each layer
    for layer_idx in tqdm(range(collector.num_layers), desc="Computing basis"):
        # Clear previous activations
        collector.clear_activations()

        # Register hooks for this layer
        collector.register_hooks(layer_idx)

        # Move layer to device
        layer = collector.layers[layer_idx].to(device)

        # Run calibration data through the model up to this layer
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                    try:
                        model(**batch)
                    except Exception:
                        pass  # We only need to collect activations
                else:
                    batch = batch.to(device)
                    try:
                        model(batch)
                    except Exception:
                        pass

        # Compute basis for this layer
        layer_basis = collector.compute_layer_basis(
            layer_idx,
            rotation_granularity=config.rotation_granularity,
        )
        all_basis.update(layer_basis)

        # Move layer back to CPU to save memory
        layer.cpu()
        cleanup_memory(verbos=False)

    # Clear hooks
    collector.clear_hooks()

    # Compute shared basis for full_shared granularity
    if config.rotation_granularity == 'full_shared':
        # Combine all layer bases
        attn_mlp_bases = []
        for layer_idx in range(collector.num_layers):
            key = f'layer.{layer_idx}.self_attn_mlp'
            if key in all_basis:
                attn_mlp_bases.append(all_basis[key])

        if attn_mlp_bases:
            # Use the mean basis as the shared basis
            shared_basis = torch.stack(attn_mlp_bases).mean(dim=0)
            # Re-orthogonalize
            q, r = torch.linalg.qr(shared_basis.T)
            shared_basis = q.T
            all_basis['attn_mlp'] = shared_basis

    return all_basis


def generate_random_rotations(
    hidden_dim: int,
    head_dim: int,
    high_fraction: float = 0.125,
    low_fraction: float = 0.0,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Generate random orthogonal rotation matrices for ResQ.

    The rotations are block-diagonal to preserve precision groupings.

    Args:
        hidden_dim: Hidden dimension of the model
        head_dim: Dimension per attention head
        high_fraction: Fraction for high precision (8-bit)
        low_fraction: Fraction for low precision (2-bit)
        seed: Random seed for reproducibility

    Returns:
        Dictionary of rotation matrices
    """
    from ..utils.hadamard_utils import random_orthogonal_matrix

    torch.manual_seed(seed)

    rotation_dict = {}

    # Compute dimension splits
    high_dim = int(high_fraction * hidden_dim)
    low_dim = int(low_fraction * hidden_dim)
    mid_dim = hidden_dim - high_dim - low_dim

    # Generate block-diagonal R1 for hidden dimension
    R1_parts = []
    if low_dim > 0:
        R1_0 = random_orthogonal_matrix(low_dim)
        rotation_dict['R1_0'] = R1_0
        R1_parts.append(R1_0)
    else:
        rotation_dict['R1_0'] = None

    R1_1 = random_orthogonal_matrix(mid_dim)
    R1_2 = random_orthogonal_matrix(high_dim)
    rotation_dict['R1_1'] = R1_1
    rotation_dict['R1_2'] = R1_2

    # Compute dimension splits for head_dim
    high_head_dim = int(high_fraction * head_dim)
    low_head_dim = int(low_fraction * head_dim)
    mid_head_dim = head_dim - high_head_dim - low_head_dim

    # Generate block-diagonal R2 for head dimension
    R2_parts = []
    if low_head_dim > 0:
        R2_0 = random_orthogonal_matrix(low_head_dim)
        rotation_dict['R2_0'] = R2_0
        R2_parts.append(R2_0)
    else:
        rotation_dict['R2_0'] = None

    R2_1 = random_orthogonal_matrix(mid_head_dim)
    R2_2 = random_orthogonal_matrix(high_head_dim)
    rotation_dict['R2_1'] = R2_1
    rotation_dict['R2_2'] = R2_2

    return rotation_dict


def save_basis(basis_dict: Dict[str, torch.Tensor], path: str):
    """Save basis matrices to file."""
    torch.save(basis_dict, path)
    logger.info(f"Saved basis matrices to {path}")


def load_basis(path: str) -> Dict[str, torch.Tensor]:
    """Load basis matrices from file."""
    basis_dict = torch.load(path, map_location='cpu')
    logger.info(f"Loaded basis matrices from {path}")
    return basis_dict
