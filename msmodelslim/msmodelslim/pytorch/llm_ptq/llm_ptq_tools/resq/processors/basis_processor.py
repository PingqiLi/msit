# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Adapted from ResQ (https://github.com/facebookresearch/resq)
"""
Basis computation for ResQ quantization.

This module computes the eigenvalue basis matrices from activation
covariances using layer-by-layer processing to minimize memory usage.
"""

import gc
import logging
from typing import Dict, Optional, Tuple, List, Any

import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


def cleanup_memory():
    """Clean up GPU/NPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        import torch_npu
        if torch.npu.is_available():
            torch.npu.empty_cache()
    except ImportError:
        pass


def get_device():
    """Get the best available device."""
    try:
        import torch_npu
        if torch.npu.is_available():
            return torch.device("npu")
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def perform_eigen_decomp(
    cov_matrix: torch.Tensor,
    damp_percent: float = 0.01,
    per_head: bool = False,
    num_heads: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform eigenvalue decomposition on covariance matrix.

    Args:
        cov_matrix: Covariance matrix of shape [n, n] or [num_heads, head_dim, head_dim]
        damp_percent: Dampening percentage for numerical stability
        per_head: Whether to perform per-head decomposition
        num_heads: Number of attention heads (required if per_head=True)

    Returns:
        Tuple of (eigenvalues, eigenvectors) sorted in ascending order
    """
    device = get_device()

    if per_head:
        eval_list = []
        evec_list = []
        for hd in range(num_heads):
            H = cov_matrix[hd].to(device).to(torch.float64)
            # Add dampening for numerical stability
            damp = damp_percent * torch.mean(torch.diag(H))
            diag_idx = torch.arange(H.shape[-1], device=H.device)
            H[diag_idx, diag_idx] = H[diag_idx, diag_idx] + damp

            # Eigendecomposition
            eigenvalues, eigenvectors = torch.linalg.eigh(H)
            # Sort by eigenvalue (ascending)
            sorted_idx = torch.argsort(eigenvalues)
            eval_list.append(eigenvalues[sorted_idx].cpu())
            evec_list.append(eigenvectors[:, sorted_idx].cpu())

        return torch.stack(eval_list), torch.stack(evec_list)
    else:
        H = cov_matrix.to(device).to(torch.float64)
        # Add dampening
        damp = damp_percent * torch.mean(torch.diag(H))
        diag_idx = torch.arange(H.shape[-1], device=H.device)
        H[diag_idx, diag_idx] = H[diag_idx, diag_idx] + damp

        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(H)
        # Sort by eigenvalue (ascending)
        sorted_idx = torch.argsort(eigenvalues)

        return eigenvalues[sorted_idx].cpu(), eigenvectors[:, sorted_idx].cpu()


class InputCatcher(nn.Module):
    """Catches inputs to the first decoder layer."""

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.inputs = []
        self.attention_masks = []  # Store per batch
        self.position_ids_list = []  # Store per batch
        self.position_embeddings_list = []  # Store per batch
        self.cache_positions = []  # Store per batch

    def __getattr__(self, name):
        # Forward attribute access to wrapped module for model-specific attributes
        # (e.g., Qwen3 accesses decoder_layer.attention_type)
        if name in ('module', 'inputs', 'attention_masks', 'position_ids_list',
                    'position_embeddings_list', 'cache_positions', 'training',
                    '_parameters', '_buffers', '_modules'):
            return super().__getattr__(name)
        try:
            return getattr(self.module, name)
        except AttributeError:
            return super().__getattr__(name)

    def forward(self, inp, **kwargs):
        self.inputs.append(inp.cpu())
        # Store kwargs per batch (move to CPU to save device memory)
        if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
            self.attention_masks.append(kwargs['attention_mask'].cpu())
        else:
            self.attention_masks.append(None)
        if 'position_ids' in kwargs and kwargs['position_ids'] is not None:
            self.position_ids_list.append(kwargs['position_ids'].cpu())
        else:
            self.position_ids_list.append(None)
        if 'position_embeddings' in kwargs and kwargs['position_embeddings'] is not None:
            # position_embeddings can be a tuple of tensors
            if isinstance(kwargs['position_embeddings'], tuple):
                self.position_embeddings_list.append(
                    tuple(pe.cpu() if pe is not None else None for pe in kwargs['position_embeddings'])
                )
            else:
                self.position_embeddings_list.append(kwargs['position_embeddings'].cpu())
        else:
            self.position_embeddings_list.append(None)
        if 'cache_position' in kwargs and kwargs['cache_position'] is not None:
            self.cache_positions.append(kwargs['cache_position'].cpu())
        else:
            self.cache_positions.append(None)
        raise ValueError("Catcher stop")  # Stop forward pass here


@torch.no_grad()
def compute_basis(
    model: nn.Module,
    dataloader,
    config: Any,
    device: torch.device = None,
    cov_device: str = 'cpu',
) -> Dict[str, torch.Tensor]:
    """
    Compute all basis matrices for the model using layer-by-layer processing.

    This follows the original ResQ pattern:
    1. Capture first layer inputs using Catcher
    2. Process layer by layer to save memory
    3. Use hooks to collect activations
    4. Accumulate covariance matrices
    5. Perform eigendecomposition

    Args:
        model: The transformer model
        dataloader: Calibration data loader
        config: ResQ configuration
        device: Device to use for layer processing
        cov_device: Device for covariance matrices ('cpu' recommended for large models)

    Returns:
        Dictionary of all basis matrices
    """
    if device is None:
        device = get_device()

    model.eval()

    # Get model structure
    hidden_dim = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    num_kv_heads = getattr(model.config, 'num_key_value_heads', num_heads)
    head_dim = hidden_dim // num_heads

    # Get layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = list(model.model.layers)
    else:
        raise ValueError("Unsupported model architecture")

    nlayers = len(layers)
    logger.info(f"Processing {nlayers} layers, hidden_dim={hidden_dim}, num_heads={num_heads}")

    # Move model to CPU first to free device memory
    model.cpu()
    cleanup_memory()

    # ========== Step 1: Capture first layer inputs using Catcher ==========
    logger.info("Capturing first layer inputs...")

    # Move only necessary parts to device
    if hasattr(model, 'model'):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        if hasattr(model.model, 'rotary_emb'):
            model.model.rotary_emb = model.model.rotary_emb.to(device)

    # Replace first layer with catcher - must modify actual model, not the list copy
    original_first_layer = model.model.layers[0]
    catcher = InputCatcher(original_first_layer)
    model.model.layers[0] = catcher

    # Run data through to capture inputs
    for batch in tqdm(dataloader, desc="Capturing inputs"):
        try:
            if isinstance(batch, dict):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                model(**batch)
            else:
                model(batch.to(device))
        except ValueError:
            pass  # Expected - catcher raises ValueError

    # Restore first layer
    model.model.layers[0] = catcher.module
    layers[0] = catcher.module  # Also update our local reference

    # Get captured data
    inps = catcher.inputs
    attention_masks = catcher.attention_masks
    position_ids_list = catcher.position_ids_list
    position_embeddings_list = catcher.position_embeddings_list
    cache_positions = catcher.cache_positions

    nbatches = len(inps)
    seqlen = inps[0].shape[1] if len(inps) > 0 else 0

    logger.info(f"Captured {nbatches} batches, seqlen={seqlen}")

    # Move embeddings back to CPU
    if hasattr(model, 'model'):
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        if hasattr(model.model, 'rotary_emb'):
            model.model.rotary_emb = model.model.rotary_emb.cpu()

    cleanup_memory()

    # ========== Step 2: Initialize covariance matrices ==========
    logger.info(f"Initializing covariance matrices on {cov_device}...")

    # Covariance matrices for attention and MLP inputs
    H_attn = torch.zeros((nlayers, hidden_dim, hidden_dim), device=cov_device, dtype=torch.float64)
    H_mlp = torch.zeros((nlayers, hidden_dim, hidden_dim), device=cov_device, dtype=torch.float64)

    # Per-head covariance for value projection outputs
    H_value = torch.zeros((nlayers, num_kv_heads, head_dim, head_dim), device=cov_device, dtype=torch.float64)

    # Prepare output buffer
    outs = [None] * nbatches

    # ========== Step 3: Layer-by-layer processing with hooks ==========
    logger.info("Computing covariance matrices layer by layer...")

    # Global variables for hook-captured values
    captured = {}

    def make_hook_fn(name, capture_input=True):
        def hook_fn(module, inp, output):
            if capture_input:
                captured[name] = inp[0].detach()
            else:
                captured[name] = output.detach()
        return hook_fn

    for layer_idx in tqdm(range(nlayers), desc="Processing layers"):
        layer = layers[layer_idx]

        # Move layer to device
        layer = layer.to(device)

        # Register hooks
        hooks = []

        # Hook for attention input (q_proj input)
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'q_proj'):
            hooks.append(layer.self_attn.q_proj.register_forward_hook(
                make_hook_fn('attn_input', capture_input=True)
            ))

        # Hook for value projection output
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'v_proj'):
            hooks.append(layer.self_attn.v_proj.register_forward_hook(
                make_hook_fn('v_output', capture_input=False)
            ))

        # Hook for MLP input (up_proj input)
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'up_proj'):
            hooks.append(layer.mlp.up_proj.register_forward_hook(
                make_hook_fn('mlp_input', capture_input=True)
            ))

        # Process all batches for this layer
        for batch_idx in range(nbatches):
            inp = inps[batch_idx].to(device)

            # Prepare kwargs for layer forward (per-batch)
            kwargs = {}
            if attention_masks[batch_idx] is not None:
                kwargs['attention_mask'] = attention_masks[batch_idx].to(device)
            if position_ids_list[batch_idx] is not None:
                kwargs['position_ids'] = position_ids_list[batch_idx].to(device)
            if position_embeddings_list[batch_idx] is not None:
                pe = position_embeddings_list[batch_idx]
                if isinstance(pe, tuple):
                    kwargs['position_embeddings'] = tuple(
                        p.to(device) if p is not None else None for p in pe
                    )
                else:
                    kwargs['position_embeddings'] = pe.to(device)
            if cache_positions[batch_idx] is not None:
                kwargs['cache_position'] = cache_positions[batch_idx].to(device)

            # Forward pass through layer
            try:
                out = layer(inp, **kwargs)
                if isinstance(out, tuple):
                    out = out[0]
                outs[batch_idx] = out.cpu()
            except Exception as e:
                logger.warning(f"Layer {layer_idx} batch {batch_idx} forward failed: {e}")
                outs[batch_idx] = inp.cpu()
                continue

            # Accumulate covariance from captured activations
            if 'attn_input' in captured:
                x = captured['attn_input'].view(-1, hidden_dim).to(torch.float64)
                H_attn[layer_idx] += (x.T @ x).to(cov_device)

            if 'mlp_input' in captured:
                x = captured['mlp_input'].view(-1, hidden_dim).to(torch.float64)
                H_mlp[layer_idx] += (x.T @ x).to(cov_device)

            if 'v_output' in captured:
                # Reshape to [batch * seq, num_kv_heads, head_dim]
                v = captured['v_output']
                v = v.view(-1, num_kv_heads, head_dim).to(torch.float64)
                for hd in range(num_kv_heads):
                    head_v = v[:, hd, :]  # [batch * seq, head_dim]
                    H_value[layer_idx, hd] += (head_v.T @ head_v).to(cov_device)

            # Clear captured values
            captured.clear()

            # Free input memory
            del inp
            cleanup_memory()

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Move layer back to CPU
        layers[layer_idx] = layer.cpu()
        cleanup_memory()

        # Swap inputs and outputs for next layer
        inps, outs = outs, [None] * nbatches

    # ========== Step 4: Eigenvalue decomposition ==========
    logger.info("Performing eigenvalue decomposition...")

    basis_dict = {}
    eval_dict = {}

    # Normalize covariances
    normalizer = nbatches * seqlen if seqlen > 0 else 1

    rotation_granularity = getattr(config, 'rotation_granularity', 'full_shared')

    if 'per_layer' in rotation_granularity.lower():
        # Per-layer basis
        for i in tqdm(range(nlayers), desc="Eigendecomp per layer"):
            # Attention basis
            eval_attn, evec_attn = perform_eigen_decomp(H_attn[i] / normalizer)
            basis_dict[f'layer.{i}.self_attn'] = evec_attn
            eval_dict[f'layer.{i}.self_attn'] = eval_attn

            # MLP basis
            eval_mlp, evec_mlp = perform_eigen_decomp(H_mlp[i] / normalizer)
            basis_dict[f'layer.{i}.mlp'] = evec_mlp
            eval_dict[f'layer.{i}.mlp'] = eval_mlp

            # Value basis (per head)
            eval_value, evec_value = perform_eigen_decomp(
                H_value[i] / normalizer, per_head=True, num_heads=num_kv_heads
            )
            basis_dict[f'layer.{i}.self_attn.value'] = evec_value
            eval_dict[f'layer.{i}.self_attn.value'] = eval_value

    elif 'full_shared' in rotation_granularity.lower():
        # Combined basis for all layers
        H_combined = (H_attn.sum(0) + H_mlp.sum(0)) / (2 * nlayers * normalizer)
        eval_combined, evec_combined = perform_eigen_decomp(H_combined)
        basis_dict['attn_mlp'] = evec_combined
        eval_dict['attn_mlp'] = eval_combined

        # Per-layer value basis
        for i in range(nlayers):
            eval_value, evec_value = perform_eigen_decomp(
                H_value[i] / normalizer, per_head=True, num_heads=num_kv_heads
            )
            basis_dict[f'layer.{i}.self_attn.value'] = evec_value
            eval_dict[f'layer.{i}.self_attn.value'] = eval_value

    else:
        # Default: one basis per decoder (average all layers)
        H_combined = (H_attn.sum(0) + H_mlp.sum(0)) / (2 * nlayers * normalizer)
        eval_combined, evec_combined = perform_eigen_decomp(H_combined)
        basis_dict['attn_mlp'] = evec_combined
        eval_dict['attn_mlp'] = eval_combined

    logger.info(f"Basis computation complete. Keys: {list(basis_dict.keys())}")

    return basis_dict


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
        low_fraction: Fraction for low precision (extra low)
        seed: Random seed for reproducibility

    Returns:
        Dictionary of rotation matrices
    """
    torch.manual_seed(seed)

    def random_orthogonal_matrix(size):
        """Generate a random orthogonal matrix using QR decomposition."""
        random_matrix = torch.randn(size, size, dtype=torch.float64)
        q, r = torch.linalg.qr(random_matrix)
        # Ensure proper orthogonal matrix (det = 1)
        d = torch.diag(r)
        ph = d.sign()
        q = q * ph.unsqueeze(0)
        return q

    rotation_dict = {}

    # Compute dimension splits for hidden_dim
    high_dim = int(high_fraction * hidden_dim)
    low_dim = int(low_fraction * hidden_dim)
    mid_dim = hidden_dim - high_dim - low_dim

    # Generate block-diagonal R1 for hidden dimension
    if low_dim > 0:
        R1_0 = random_orthogonal_matrix(low_dim)
        rotation_dict['R1_0'] = R1_0
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
    if low_head_dim > 0:
        R2_0 = random_orthogonal_matrix(low_head_dim)
        rotation_dict['R2_0'] = R2_0
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
