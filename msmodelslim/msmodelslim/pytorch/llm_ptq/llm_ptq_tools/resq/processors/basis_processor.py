# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Adapted from ResQ (https://github.com/facebookresearch/resq)
"""
Basis computation for ResQ quantization.

This module computes the eigenvalue basis matrices from activation
covariances using layer-by-layer processing to minimize memory usage.

Optimization strategies for eigendecomposition:
1. Use NPU/GPU for covariance computation (matrix multiplication is fast)
2. Use CPU or GPU for eigendecomposition (NPU may fallback to CPU anyway)
3. Parallel processing of independent eigendecompositions using ThreadPoolExecutor
4. Detailed timing information for each step
"""

import gc
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Tuple, List, Any

import torch
import torch.nn as nn
from tqdm import tqdm

from ..utils.hadamard_utils import get_hadK, random_orthogonal_matrix as had_random_orthogonal

logger = logging.getLogger(__name__)


def cleanup_memory():
    """Clean up NPU memory."""
    gc.collect()
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
    return torch.device("cpu")


def get_eigen_device():
    """
    Get the best device for eigendecomposition.

    Note: torch.linalg.eigh is typically efficient on CPU with multi-threading.
    On NPU, it may fallback to CPU. We use CPU for eigh to avoid unnecessary
    data transfer.
    """
    # NPU's eigh may fallback to CPU anyway, so use CPU directly
    # to avoid unnecessary data transfer
    return torch.device("cpu")


def perform_eigen_decomp(
    cov_matrix: torch.Tensor,
    damp_percent: float = 0.01,
    per_head: bool = False,
    num_heads: int = 0,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform eigenvalue decomposition on covariance matrix.

    Args:
        cov_matrix: Covariance matrix of shape [n, n] or [num_heads, head_dim, head_dim]
        damp_percent: Dampening percentage for numerical stability
        per_head: Whether to perform per-head decomposition
        num_heads: Number of attention heads (required if per_head=True)
        device: Device to use for computation (default: auto-detect best device)

    Returns:
        Tuple of (eigenvalues, eigenvectors) sorted in ascending order
    """
    if device is None:
        device = get_eigen_device()

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
            # Move to CPU before sorting (NPU doesn't support argsort on float64)
            eigenvalues = eigenvalues.cpu()
            eigenvectors = eigenvectors.cpu()
            # Sort by eigenvalue (ascending)
            sorted_idx = torch.argsort(eigenvalues)
            eval_list.append(eigenvalues[sorted_idx])
            evec_list.append(eigenvectors[:, sorted_idx])

        return torch.stack(eval_list), torch.stack(evec_list)
    else:
        H = cov_matrix.to(device).to(torch.float64)
        # Add dampening
        damp = damp_percent * torch.mean(torch.diag(H))
        diag_idx = torch.arange(H.shape[-1], device=H.device)
        H[diag_idx, diag_idx] = H[diag_idx, diag_idx] + damp

        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(H)
        # Move to CPU before sorting (NPU doesn't support argsort on float64)
        eigenvalues = eigenvalues.cpu()
        eigenvectors = eigenvectors.cpu()
        # Sort by eigenvalue (ascending)
        sorted_idx = torch.argsort(eigenvalues)

        return eigenvalues[sorted_idx], eigenvectors[:, sorted_idx]


def perform_eigen_decomp_timed(
    name: str,
    cov_matrix: torch.Tensor,
    damp_percent: float = 0.01,
    per_head: bool = False,
    num_heads: int = 0,
    device: torch.device = None,
) -> Tuple[str, torch.Tensor, torch.Tensor, float]:
    """
    Perform eigenvalue decomposition with timing.

    Returns:
        Tuple of (name, eigenvalues, eigenvectors, elapsed_time)
    """
    start_time = time.time()
    eigenvalues, eigenvectors = perform_eigen_decomp(
        cov_matrix, damp_percent, per_head, num_heads, device
    )
    elapsed = time.time() - start_time
    return name, eigenvalues, eigenvectors, elapsed


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
    # Get head_dim from config or compute from v_proj
    head_dim = getattr(model.config, 'head_dim', None)
    if head_dim is None:
        # Try to get from first layer's v_proj
        if hasattr(model, 'model') and hasattr(model.model, 'layers') and len(model.model.layers) > 0:
            v_proj = model.model.layers[0].self_attn.v_proj
            head_dim = v_proj.out_features // num_kv_heads
        else:
            head_dim = hidden_dim // num_heads

    # Get layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = list(model.model.layers)
    else:
        raise ValueError("Unsupported model architecture")

    nlayers = len(layers)
    logger.info(f"Processing {nlayers} layers, hidden_dim={hidden_dim}, num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")

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

    # Get intermediate_size for down_proj
    intermediate_size = model.config.intermediate_size

    # Get down_proj_blocksize from config (default 256)
    # Original ResQ uses block-wise approach for down_proj to reduce memory
    down_proj_blocksize = getattr(config, 'down_proj_blocksize', 256)
    logger.info(f"Using down_proj_blocksize={down_proj_blocksize} (intermediate_size={intermediate_size})")

    # Covariance matrices for attention and MLP inputs
    H_attn = torch.zeros((nlayers, hidden_dim, hidden_dim), device=cov_device, dtype=torch.float64)
    H_mlp = torch.zeros((nlayers, hidden_dim, hidden_dim), device=cov_device, dtype=torch.float64)

    # Per-head covariance for value projection outputs
    H_value = torch.zeros((nlayers, num_kv_heads, head_dim, head_dim), device=cov_device, dtype=torch.float64)

    # Per-head covariance for key projection outputs after RoPE (for Uc computation)
    H_key_pos = torch.zeros((nlayers, num_kv_heads, head_dim, head_dim), device=cov_device, dtype=torch.float64)

    # Covariance for down_proj input (for Ud computation)
    # Use block-wise approach: split intermediate_size into blocks of down_proj_blocksize
    # This reduces memory from [intermediate_size, intermediate_size] to [blocksize, blocksize]
    H_down_proj = torch.zeros((nlayers, down_proj_blocksize, down_proj_blocksize), device=cov_device, dtype=torch.float64)

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

        # Hook for key projection output (for key_pos after RoPE)
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'k_proj'):
            hooks.append(layer.self_attn.k_proj.register_forward_hook(
                make_hook_fn('k_output', capture_input=False)
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

        # Hook for down_proj input (for Ud computation)
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'down_proj'):
            hooks.append(layer.mlp.down_proj.register_forward_hook(
                make_hook_fn('down_proj_input', capture_input=True)
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

            # Compute key_pos covariance (key after RoPE)
            if 'k_output' in captured and 'v_output' in captured:
                try:
                    k_out = captured['k_output']
                    v_out = captured['v_output']
                    batch_seq_len = k_out.shape[1]

                    # Reshape key states: [batch, seq, num_kv_heads, head_dim]
                    key_states = k_out.view(1, batch_seq_len, num_kv_heads, head_dim).transpose(1, 2)

                    # Get position embeddings for RoPE
                    position_ids = kwargs.get('position_ids')
                    position_embeddings = kwargs.get('position_embeddings')

                    # Apply RoPE to get key_states_pos
                    if position_embeddings is not None:
                        cos, sin = position_embeddings
                    elif hasattr(layer.self_attn, 'rotary_emb'):
                        cos, sin = layer.self_attn.rotary_emb(v_out, position_ids)
                    else:
                        # Skip if no rotary embedding available
                        cos, sin = None, None

                    if cos is not None and sin is not None:
                        # Apply rotary position embedding
                        # Standard RoPE implementation
                        def rotate_half(x):
                            x1 = x[..., : x.shape[-1] // 2]
                            x2 = x[..., x.shape[-1] // 2 :]
                            return torch.cat((-x2, x1), dim=-1)

                        key_states_pos = (key_states * cos) + (rotate_half(key_states) * sin)

                        # Accumulate key_pos covariance
                        k_pos = key_states_pos.view(-1, num_kv_heads, head_dim).to(torch.float64)
                        for hd in range(num_kv_heads):
                            head_k = k_pos[:, hd, :]  # [batch * seq, head_dim]
                            H_key_pos[layer_idx, hd] += (head_k.T @ head_k).to(cov_device)
                except Exception as e:
                    logger.warning(f"Layer {layer_idx} batch {batch_idx} key_pos computation failed: {e}")

            # Compute down_proj covariance (for Ud computation)
            # Use block-wise approach: reshape to [batch, num_blocks, blocksize]
            # Then compute covariance across all blocks
            if 'down_proj_input' in captured:
                try:
                    dp_input = captured['down_proj_input']
                    # Reshape to [batch*seq, num_blocks, blocksize]
                    # intermediate_size = num_blocks * blocksize
                    x = dp_input.view(dp_input.shape[0], -1, down_proj_blocksize).to(torch.float64)
                    # Sum covariance across all blocks: [blocksize, blocksize]
                    H_down_proj[layer_idx] += torch.sum(x.mT @ x, dim=0).to(cov_device)
                except Exception as e:
                    logger.warning(f"Layer {layer_idx} batch {batch_idx} down_proj covariance failed: {e}")

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
    eigen_device = get_eigen_device()
    total_eigen_start = time.time()

    logger.info("=" * 60)
    logger.info("Starting eigenvalue decomposition...")
    logger.info(f"Total layers: {nlayers}")
    logger.info(f"Eigen device: {eigen_device}")
    logger.info(f"Matrices:")
    logger.info(f"  - attn_mlp (shared): [{hidden_dim}x{hidden_dim}]")
    logger.info(f"  - value:    [{num_kv_heads} heads x {head_dim}x{head_dim}] per layer")
    logger.info(f"  - key_pos:  [{head_dim}x{head_dim}] per layer (shared across heads)")
    logger.info(f"  - down_proj:[{down_proj_blocksize}x{down_proj_blocksize}] per layer")
    logger.info("=" * 60)

    basis_dict = {}
    eval_dict = {}

    # Normalize covariances
    normalizer = nbatches * seqlen if seqlen > 0 else 1

    # Determine number of parallel workers
    max_workers = 128 if eigen_device.type == 'cpu' else 2
    logger.info(f"Using {max_workers} parallel workers for eigendecomposition")

    # ===== Full Shared Mode =====
    # Single shared basis for attn+mlp across all layers
    logger.info("Computing shared attn_mlp basis (sum across all layers)...")

    # Sum H_attn and H_mlp across all layers and merge
    H_attn_mlp_sum = (H_attn.sum(0) + H_mlp.sum(0)) / (2 * nlayers * normalizer)

    # Eigendecomposition for shared attn_mlp
    eval_attn_mlp, evec_attn_mlp = perform_eigen_decomp(
        H_attn_mlp_sum, damp_percent=0.01, device=eigen_device
    )
    basis_dict['config'] = 'full_shared_rotation'
    basis_dict['attn_mlp'] = evec_attn_mlp
    eval_dict['config'] = 'full_shared_rotation'
    eval_dict['attn_mlp'] = eval_attn_mlp
    logger.info(f"  Shared attn_mlp basis computed: {evec_attn_mlp.shape}")

    # Per-layer eigendecompositions for value, key_pos, down_proj
    for i in tqdm(range(nlayers), desc="Per-layer basis"):
        layer_start = time.time()

        tasks = [
            ('value', H_value[i] / normalizer, True, num_kv_heads),
            ('key_pos', H_key_pos[i].sum(0) / (num_kv_heads * normalizer), False, 0),
            ('down_proj', H_down_proj[i] / normalizer, False, 0),
        ]

        results = {}
        timings = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for task_name, cov_matrix, per_head, n_heads in tasks:
                future = executor.submit(
                    perform_eigen_decomp_timed,
                    task_name, cov_matrix, 0.01, per_head, n_heads, eigen_device,
                )
                futures[future] = task_name

            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    name, eigenvalues, eigenvectors, elapsed = future.result()
                    results[name] = (eigenvalues, eigenvectors)
                    timings[name] = elapsed
                except Exception as e:
                    logger.error(f"  [Layer {i+1}] {task_name} failed: {e}")
                    raise

        if 'value' in results:
            eval_value, evec_value = results['value']
            basis_dict[f'layer.{i}.self_attn.value'] = evec_value
            eval_dict[f'layer.{i}.self_attn.value'] = eval_value

        if 'key_pos' in results:
            eval_key_pos, evec_key_pos = results['key_pos']
            basis_dict[f'layer.{i}.self_attn.key_pos'] = evec_key_pos
            eval_dict[f'layer.{i}.self_attn.key_pos'] = eval_key_pos

        if 'down_proj' in results:
            eval_down_proj, evec_down_proj = results['down_proj']
            basis_dict[f'layer.{i}.mlp.down_proj'] = evec_down_proj
            eval_dict[f'layer.{i}.mlp.down_proj'] = eval_down_proj

        cleanup_memory()

    total_eigen_elapsed = time.time() - total_eigen_start
    logger.info("=" * 60)
    logger.info(f"Eigendecomposition complete!")
    logger.info(f"Total time: {total_eigen_elapsed:.2f}s ({total_eigen_elapsed/60:.1f} min)")
    logger.info(f"Total basis matrices: {len(basis_dict)}")
    logger.info(f"Keys: {list(basis_dict.keys())[:10]}... (showing first 10)")

    return basis_dict


def generate_random_rotations(
    hidden_dim: int,
    head_dim: int,
    intermediate_dim: int = None,
    high_fraction: float = 0.125,
    seed: int = 42,
    ud_rotation_type: str = 'hadamard',
) -> Dict[str, torch.Tensor]:
    """
    Generate random orthogonal rotation matrices for ResQ.

    The rotations are block-diagonal to preserve precision groupings.
    Only two precision levels: mid (4-bit) and high (8-bit).

    Args:
        hidden_dim: Hidden dimension of the model
        head_dim: Dimension per attention head
        intermediate_dim: Intermediate dimension for down_proj (MLP)
        high_fraction: Fraction for high precision (8-bit)
        seed: Random seed for reproducibility
        ud_rotation_type: Type of rotation for Ud (down_proj):
            - 'hadamard': Use Hadamard matrix for full intermediate_dim
            - 'random': Use random orthogonal matrix for full intermediate_dim

    Returns:
        Dictionary of rotation matrices (R1_1, R1_2, R2_1, R2_2, and Hd/Hd_K or Rd)
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

    # Compute dimension splits for hidden_dim (2 regions: mid and high)
    high_dim = int(high_fraction * hidden_dim)
    mid_dim = hidden_dim - high_dim

    # Generate block-diagonal R1 for hidden dimension
    R1_1 = random_orthogonal_matrix(mid_dim)
    R1_2 = random_orthogonal_matrix(high_dim)
    rotation_dict['R1_1'] = R1_1
    rotation_dict['R1_2'] = R1_2

    # Compute dimension splits for head_dim (2 regions: mid and high)
    high_head_dim = int(high_fraction * head_dim)
    mid_head_dim = head_dim - high_head_dim

    # Generate block-diagonal R2 for head dimension
    R2_1 = random_orthogonal_matrix(mid_head_dim)
    R2_2 = random_orthogonal_matrix(high_head_dim)
    rotation_dict['R2_1'] = R2_1
    rotation_dict['R2_2'] = R2_2

    # Generate rotation for down_proj (Ud = block_diag(Pd) @ H or Rd)
    # H/Rd is applied to full intermediate_dim
    if intermediate_dim is not None:
        if ud_rotation_type == 'hadamard':
            # Get Hadamard for full intermediate dimension using get_hadK
            hadK, K = get_hadK(intermediate_dim)
            rotation_dict['Hd'] = hadK  # May be None for pure power-of-2
            rotation_dict['Hd_K'] = K
            logger.info(f"Generated Hadamard rotation for Ud: intermediate_dim={intermediate_dim}, K={K}")
        else:  # 'random'
            # Generate random orthogonal for full intermediate dimension
            # Use had_random_orthogonal for consistency with hadamard_utils
            Rd = had_random_orthogonal(intermediate_dim)
            rotation_dict['Rd'] = Rd
            logger.info(f"Generated random orthogonal rotation for Ud: intermediate_dim={intermediate_dim}")

        # Also keep block-diagonal Rd for backward compatibility (if needed elsewhere)
        # These are no longer used for Ud but may be referenced elsewhere
        high_inter_dim = int(high_fraction * intermediate_dim)
        mid_inter_dim = intermediate_dim - high_inter_dim
        Rd_1 = random_orthogonal_matrix(mid_inter_dim)
        Rd_2 = random_orthogonal_matrix(high_inter_dim)
        rotation_dict['Rd_1'] = Rd_1
        rotation_dict['Rd_2'] = Rd_2

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
