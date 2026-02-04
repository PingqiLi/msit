# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Adapted from ResQ (https://github.com/facebookresearch/resq)
"""
Main ResQ quantization processor.

This module provides the core quantization functionality:
- Applying rotation matrices to model weights
- Rearranging columns for mixed-precision layout
- Configuring quantizers for each layer
"""

import fnmatch
import logging
from typing import Dict, Optional, Any, List

import torch
import torch.nn as nn
from tqdm import tqdm

from ..utils.common import get_device, cleanup_memory
from ..utils.fuse_norm_utils import fuse_layer_norms
from ..utils.hadamard_utils import (
    matmul_hadU_cpu,
    get_hadK,
    random_orthogonal_matrix,
)
from ..components.act_quantizer import ActQuantizer, ActQuantWrapper
from ..components.weight_quantizer import (
    WeightQuantizer,
    MixedPrecisionWeightQuantizer,
    add_actquant,
)

logger = logging.getLogger(__name__)


def get_rotation_device(weight_tensor: torch.Tensor, use_npu: bool = True) -> torch.device:
    """
    Determine device for rotation operations.

    When use_npu is True, keeps operations on the weight's original device (NPU).
    When False, moves to CPU for compatibility.

    Args:
        weight_tensor: The weight tensor to get device from
        use_npu: Whether to use NPU for rotation operations

    Returns:
        Device to use for rotation operations
    """
    if not use_npu:
        return torch.device("cpu")
    return weight_tensor.device


def batch_inverse_with_fallback(matrices: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Compute batch matrix inverse with CPU fallback if NPU doesn't support it.

    Some NPU implementations may not support torch.linalg.inv. This function
    attempts the operation on the target device and falls back to CPU if needed.

    Args:
        matrices: Batch of matrices to invert [batch, n, n]
        device: Target device for computation

    Returns:
        Inverted matrices on the target device
    """
    try:
        return torch.linalg.inv(matrices.to(device))
    except RuntimeError:
        # Fallback to CPU if NPU doesn't support inverse
        return torch.linalg.inv(matrices.cpu()).to(device)


def rotate_embeddings(model: nn.Module, R1: torch.Tensor) -> None:
    """
    Rotate the embedding weights.

    Args:
        model: The transformer model
        R1: Rotation matrix
    """
    for W in [model.model.embed_tokens]:
        dtype = W.weight.data.dtype
        dev = W.weight.device
        W_ = W.weight.data.to(torch.float32)  # Keep on original device
        R1_dev = R1.to(device=dev, dtype=torch.float32)
        W.weight.data = torch.matmul(W_, R1_dev).to(dtype=dtype)


def rotate_attention_inputs(layer: nn.Module, R1: torch.Tensor) -> None:
    """
    Rotate the QKV projection input weights.

    Args:
        layer: Decoder layer
        R1: Rotation matrix
    """
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        dev = W.weight.device
        W_ = W.weight.to(torch.float32)  # Keep on original device
        R1_dev = R1.to(device=dev, dtype=torch.float32)
        W.weight.data = torch.matmul(W_, R1_dev).to(dtype=dtype)


def rotate_attention_output(layer: nn.Module, R1: torch.Tensor) -> None:
    """
    Rotate the output projection weights.

    Args:
        layer: Decoder layer
        R1: Rotation matrix
    """
    W = layer.self_attn.o_proj
    dtype = W.weight.data.dtype
    dev = W.weight.device
    W_ = W.weight.data.to(torch.float32)  # Keep on original device
    R1_T_dev = R1.T.to(device=dev, dtype=torch.float32)
    W.weight.data = torch.matmul(R1_T_dev, W_).to(dtype=dtype)

    if W.bias is not None:
        b = W.bias.data.to(torch.float32)
        W.bias.data = torch.matmul(R1_T_dev, b).to(dtype=dtype)


def rotate_mlp_input(layer: nn.Module, R1: torch.Tensor) -> None:
    """
    Rotate the MLP input weights.

    Args:
        layer: Decoder layer
        R1: Rotation matrix
    """
    mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    for W in mlp_inputs:
        dtype = W.weight.dtype
        dev = W.weight.device
        W_ = W.weight.data.to(torch.float32)  # Keep on original device
        R1_dev = R1.to(device=dev, dtype=torch.float32)
        W.weight.data = torch.matmul(W_, R1_dev).to(dtype=dtype)


def rotate_mlp_output(
    layer: nn.Module,
    R1: torch.Tensor,
) -> None:
    """
    Rotate the MLP output (down_proj) weights with R1 only (fallback, no basis).

    This is a simplified version used when no Pd basis is available.
    Only applies R1.T to the output dimension.

    Args:
        layer: Decoder layer
        R1: Rotation matrix for hidden dimension
    """
    W = layer.mlp.down_proj
    dtype = W.weight.data.dtype
    dev = W.weight.device
    W_ = W.weight.data.to(torch.float32)  # Keep on original device
    R1_T_dev = R1.T.to(device=dev, dtype=torch.float32)

    W.weight.data = torch.matmul(R1_T_dev, W_).to(dtype=dtype)

    if W.bias is not None:
        b = W.bias.data.to(torch.float32)
        W.bias.data = torch.matmul(R1_T_dev, b).to(dtype=dtype)


def rotate_mlp_output_hadamard_only(
    layer: nn.Module,
    Ua: torch.Tensor,
    hadK: Optional[torch.Tensor],
    K: int,
) -> None:
    """
    Rotate the MLP output (down_proj) weights with Hadamard transform only (skip Pd basis).

    Computes: Wd_merged = Ua.T @ Wd @ Hd

    This is used for int4_hadamard quantization where we skip the Pd eigenvector basis
    and only apply the Hadamard transform to the input dimension.

    Args:
        layer: Decoder layer
        Ua: Rotation matrix for hidden dimension (output rotation)
        hadK: Hadamard block matrix from get_hadK() (may be None for power-of-2)
        K: Block size for Hadamard factorization
    """
    W = layer.mlp.down_proj
    dtype = W.weight.data.dtype
    dev = W.weight.device

    # Step 1: Apply Ua to output dimension: W1 = Ua.T @ Wd
    W_ = W.weight.data.to(torch.float32)
    Ua_T_dev = Ua.T.to(device=dev, dtype=torch.float32)
    W_ = torch.matmul(Ua_T_dev, W_)

    # Step 2: Apply Hd using fast Hadamard (W_ @ Hd) - requires CPU
    # Note: We skip block_diag(Pd) compared to rotate_mlp_output_hadamard
    W_ = matmul_hadU_cpu(W_.cpu(), hadK, K)

    W.weight.data = W_.to(dtype=dtype).to(device=dev)

    if W.bias is not None:
        b = W.bias.data.to(torch.float32)
        W.bias.data = torch.matmul(Ua_T_dev, b).to(dtype=dtype)


def rotate_mlp_output_hadamard(
    layer: nn.Module,
    Ua: torch.Tensor,
    Pd: torch.Tensor,
    hadK: Optional[torch.Tensor],
    K: int,
    blocksize: int,
) -> None:
    """
    Rotate the MLP output (down_proj) weights using Hadamard mode.

    Computes: Wd_merged = Ua.T @ Wd @ block_diag(Pd) @ H

    Args:
        layer: Decoder layer
        Ua: Rotation matrix for hidden dimension (output rotation)
        Pd: Per-layer eigenvector matrix [blocksize, blocksize]
        hadK: Hadamard block matrix from get_hadK() (may be None for power-of-2)
        K: Block size for Hadamard factorization
        blocksize: Block size for Pd (down_proj_blocksize)
    """
    W = layer.mlp.down_proj
    dtype = W.weight.data.dtype
    dev = W.weight.device

    # Step 1: Apply Ua to output dimension on NPU: W1 = Ua.T @ Wd
    # This is the large 5120x5120 @ 5120x27392 matmul - keep on device
    W_ = W.weight.data.to(torch.float32)  # Keep on original device
    Ua_T_dev = Ua.T.to(device=dev, dtype=torch.float32)
    W_ = torch.matmul(Ua_T_dev, W_)

    intermediate_size = W_.shape[1]
    num_blocks = intermediate_size // blocksize

    # Step 2: Apply block_diag(Pd) to input dimension (on device)
    # Reshape to [hidden_dim, num_blocks, blocksize]
    W_ = W_.view(W_.shape[0], num_blocks, blocksize)
    # Apply Pd to each block (broadcast across all blocks)
    Pd_dev = Pd.to(device=dev, dtype=torch.float32)
    W_ = torch.matmul(W_, Pd_dev)
    # Reshape back to [hidden_dim, intermediate_size]
    W_ = W_.view(W_.shape[0], intermediate_size)

    # Step 3: Apply H using fast Hadamard (W_ @ H) - requires CPU
    W_ = matmul_hadU_cpu(W_.cpu(), hadK, K)

    W.weight.data = W_.to(dtype=dtype).to(device=dev)

    if W.bias is not None:
        b = W.bias.data.to(torch.float32)  # Keep on device
        W.bias.data = torch.matmul(Ua_T_dev, b).to(dtype=dtype)


def rotate_mlp_output_random(
    layer: nn.Module,
    Ua: torch.Tensor,
    Pd: torch.Tensor,
    Rd: torch.Tensor,
    blocksize: int,
) -> None:
    """
    Rotate the MLP output (down_proj) weights using random rotation mode.

    Computes: Wd_merged = Ua.T @ Wd @ block_diag(Pd) @ Rd

    Args:
        layer: Decoder layer
        Ua: Rotation matrix for hidden dimension (output rotation)
        Pd: Per-layer eigenvector matrix [blocksize, blocksize]
        Rd: Random orthogonal matrix [intermediate_size, intermediate_size]
        blocksize: Block size for Pd (down_proj_blocksize)
    """
    W = layer.mlp.down_proj
    dtype = W.weight.data.dtype
    dev = W.weight.device

    # Step 1: Apply Ua to output dimension on NPU: W1 = Ua.T @ Wd
    # This is the large 5120x5120 @ 5120x27392 matmul - keep on device
    W_ = W.weight.data.to(torch.float32)  # Keep on original device
    Ua_T_dev = Ua.T.to(device=dev, dtype=torch.float32)
    W_ = torch.matmul(Ua_T_dev, W_)

    intermediate_size = W_.shape[1]
    num_blocks = intermediate_size // blocksize

    # Step 2: Apply block_diag(Pd) to input dimension (on device)
    # Reshape to [hidden_dim, num_blocks, blocksize]
    W_ = W_.view(W_.shape[0], num_blocks, blocksize)
    # Apply Pd to each block (broadcast across all blocks)
    Pd_dev = Pd.to(device=dev, dtype=torch.float32)
    W_ = torch.matmul(W_, Pd_dev)
    # Reshape back to [hidden_dim, intermediate_size]
    W_ = W_.view(W_.shape[0], intermediate_size)

    # Step 3: Apply Rd (W_ @ Rd) - keep on device
    Rd_dev = Rd.to(device=dev, dtype=torch.float32)
    W_ = torch.matmul(W_, Rd_dev)

    W.weight.data = W_.to(dtype=dtype)

    if W.bias is not None:
        b = W.bias.data.to(torch.float32)  # Keep on device
        W.bias.data = torch.matmul(Ua_T_dev, b).to(dtype=dtype)


def rotate_head(model: nn.Module, R1: torch.Tensor) -> None:
    """
    Rotate the LM head weights.

    Args:
        model: The transformer model
        R1: Rotation matrix
    """
    W = model.lm_head
    dtype = W.weight.data.dtype
    dev = W.weight.device
    W_ = W.weight.data.to(torch.float32)  # Keep on original device
    R1_dev = R1.to(device=dev, dtype=torch.float32)
    W.weight.data = torch.matmul(W_, R1_dev).to(dtype=dtype)


def rotate_ov_proj(
    layer: nn.Module,
    num_heads: int,
    head_dim: int,
    Ub: torch.Tensor,
) -> None:
    """
    Rotate value and output projections using Ub = P @ R (random rotation).

    Ub is applied to v_proj output and absorbed into o_proj input.
    No Hadamard matrix is used.

    Args:
        layer: Decoder layer
        num_heads: Number of attention heads
        head_dim: Dimension per head
        Ub: Per-head rotation matrix [num_kv_heads, head_dim, head_dim]
    """
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj

    dtype = v_proj.weight.dtype
    dev = v_proj.weight.device

    # Get dimensions
    num_kv_heads = Ub.shape[0]
    o_proj_in_dim = o_proj.weight.shape[1]
    num_attention_heads = o_proj_in_dim // head_dim

    # v_proj: apply Ub to output dimension (batched)
    # v_proj weight shape: [num_kv_heads * head_dim, hidden_dim]
    # y_new = y @ Ub, so W_new = Ub.T @ W
    W_v = v_proj.weight.data.to(torch.float32)  # Keep on device
    W_v = W_v.view(num_kv_heads, head_dim, -1)
    # Batched matmul: Ub.T @ W_v for each head
    Ub_T = Ub.transpose(-2, -1).to(device=dev, dtype=torch.float32)
    W_v = torch.bmm(Ub_T, W_v)  # [num_kv_heads, head_dim, hidden_dim]
    v_proj.weight.data = W_v.view(-1, W_v.shape[-1]).to(dtype=dtype)

    if v_proj.bias is not None:
        b_v = v_proj.bias.data.to(torch.float32)
        b_v = b_v.view(num_kv_heads, head_dim, 1)
        b_v = torch.bmm(Ub_T, b_v)  # [num_kv_heads, head_dim, 1]
        v_proj.bias.data = b_v.view(-1).to(dtype=dtype)

    # o_proj: absorb Ub^(-1) into input dimension (batched)
    # o_proj weight shape: [hidden_dim, num_attention_heads * head_dim]
    # x_new = x @ Ub, so we need W_new = W @ Ub^(-1).T
    W_o = o_proj.weight.data.to(torch.float32)  # Keep on device
    W_o = W_o.view(W_o.shape[0], num_attention_heads, head_dim)

    # Pre-compute all inverses at once using batch inverse with fallback
    Ub_inv = batch_inverse_with_fallback(Ub.to(torch.float32), dev)
    Ub_inv_T = Ub_inv.transpose(-2, -1)  # [num_kv_heads, head_dim, head_dim]

    # For GQA: expand Ub_inv_T to match num_attention_heads
    if num_attention_heads > num_kv_heads:
        num_q_per_kv = num_attention_heads // num_kv_heads
        Ub_inv_T_expanded = Ub_inv_T.repeat_interleave(num_q_per_kv, dim=0)
    else:
        Ub_inv_T_expanded = Ub_inv_T

    # Batched matmul: W_o @ Ub_inv_T for each head
    # W_o is [hidden_dim, num_attention_heads, head_dim]
    # Ub_inv_T_expanded is [num_attention_heads, head_dim, head_dim]
    # Need to permute W_o for bmm: [num_attention_heads, hidden_dim, head_dim]
    W_o = W_o.permute(1, 0, 2)
    W_o = torch.bmm(W_o, Ub_inv_T_expanded)
    W_o = W_o.permute(1, 0, 2)  # Back to [hidden_dim, num_attention_heads, head_dim]

    o_proj.weight.data = W_o.reshape(W_o.shape[0], -1).to(dtype=dtype)


def rotate_ov_proj_rotation_only(
    layer: nn.Module,
    num_heads: int,
    head_dim: int,
    Rb: torch.Tensor,
) -> None:
    """
    Rotate value and output projections using only Rb (rotation matrix).

    Unlike rotate_ov_proj which uses Ub = Pb @ Rb, this function:
    - Applies only Rb to v_proj output (no eigenvector basis)
    - Absorbs only Rb^(-1) into o_proj input (no permutation)

    This is the "Remove Ub" mode where V is only rotated, not rearranged.

    Args:
        layer: Decoder layer
        num_heads: Number of attention heads
        head_dim: Dimension per head
        Rb: Rotation matrix [head_dim, head_dim] (shared across all heads)
    """
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj

    dtype = v_proj.weight.dtype
    dev = v_proj.weight.device

    # Get dimensions
    num_kv_heads = v_proj.out_features // head_dim
    o_proj_in_dim = o_proj.weight.shape[1]
    num_attention_heads = o_proj_in_dim // head_dim

    # v_proj: apply Rb to output dimension (batched)
    # v_proj weight shape: [num_kv_heads * head_dim, hidden_dim]
    # y_new = y @ Rb, so W_new = Rb.T @ W
    W_v = v_proj.weight.data.to(torch.float32)  # Keep on device
    W_v = W_v.view(num_kv_heads, head_dim, -1)
    # Rb is shared across all heads - expand for batched matmul
    Rb_T = Rb.T.to(device=dev, dtype=torch.float32)
    Rb_T_expanded = Rb_T.unsqueeze(0).expand(num_kv_heads, -1, -1)
    W_v = torch.bmm(Rb_T_expanded, W_v)  # [num_kv_heads, head_dim, hidden_dim]
    v_proj.weight.data = W_v.view(-1, W_v.shape[-1]).to(dtype=dtype)

    if v_proj.bias is not None:
        b_v = v_proj.bias.data.to(torch.float32)
        b_v = b_v.view(num_kv_heads, head_dim, 1)
        b_v = torch.bmm(Rb_T_expanded, b_v)  # [num_kv_heads, head_dim, 1]
        v_proj.bias.data = b_v.view(-1).to(dtype=dtype)

    # o_proj: absorb Rb^(-1) into input dimension (batched)
    # o_proj weight shape: [hidden_dim, num_attention_heads * head_dim]
    # x_new = x @ Rb, so we need W_new = W @ Rb^(-1).T
    W_o = o_proj.weight.data.to(torch.float32)  # Keep on device
    W_o = W_o.view(W_o.shape[0], num_attention_heads, head_dim)

    # Rb is orthogonal, so Rb^(-1) = Rb.T, and Rb^(-1).T = Rb
    Rb_inv_T = Rb.to(device=dev, dtype=torch.float32)  # Rb^(-1).T = (Rb.T).T = Rb
    Rb_inv_T_expanded = Rb_inv_T.unsqueeze(0).expand(num_attention_heads, -1, -1)

    # Batched matmul: W_o @ Rb_inv_T for each head
    # W_o is [hidden_dim, num_attention_heads, head_dim]
    # Need to permute W_o for bmm: [num_attention_heads, hidden_dim, head_dim]
    W_o = W_o.permute(1, 0, 2)
    W_o = torch.bmm(W_o, Rb_inv_T_expanded)
    W_o = W_o.permute(1, 0, 2)  # Back to [hidden_dim, num_attention_heads, head_dim]

    o_proj.weight.data = W_o.reshape(W_o.shape[0], -1).to(dtype=dtype)


def rearrange_o_proj(
    layer: nn.Module,
    high_fraction: float,
    head_dim: int,
    training: bool = False,
) -> None:
    """
    Rearrange o_proj columns for mixed-precision layout.

    Reorders columns so that:
    - Mid precision dimensions are at the beginning
    - High precision dimensions are at the end

    Args:
        layer: Decoder layer
        high_fraction: Fraction of dimensions at high precision (e.g., 0.125)
        head_dim: Dimension per head
        training: Whether in training mode
    """
    o_proj = layer.self_attn.o_proj

    in_dim = o_proj.weight.shape[-1]
    num_replicated_heads = in_dim // head_dim

    # Compute high_bits_length based on actual o_proj input dimension
    # This is critical for models where hidden_size != num_attention_heads * head_dim
    # Example: Qwen3-32B has hidden_size=5120, but o_proj input is 8192 (64 heads × 128 head_dim)
    high_bits_length = int(high_fraction * in_dim)
    high_length_per_head = high_bits_length // num_replicated_heads

    logger.debug(f"rearrange_o_proj: in_dim={in_dim}, num_replicated_heads={num_replicated_heads}, "
                 f"high_fraction={high_fraction}, high_bits_length={high_bits_length}, "
                 f"high_length_per_head={high_length_per_head}, head_dim={head_dim}")

    # Build column indices for rearrangement
    chunk_starts = torch.arange(0, in_dim, head_dim)

    # High precision columns (last in each head)
    high_precision_columns = torch.arange(head_dim - high_length_per_head, head_dim)
    columns_to_end = (chunk_starts.unsqueeze(1) + high_precision_columns).flatten()

    # Remaining columns (mid precision)
    all_columns = torch.arange(in_dim)
    mask = torch.ones(in_dim, dtype=torch.bool)
    mask[columns_to_end] = False
    remaining_columns = all_columns[mask]

    # New column order: [mid | high]
    new_column_order = torch.cat([remaining_columns, columns_to_end])

    # Rearrange weights
    if not training:
        Wo = o_proj.weight.data
        o_proj.weight.data = Wo[:, new_column_order]

    # Save column order for runtime rearrangement
    if training:
        permutation_matrix = torch.eye(in_dim)[:, new_column_order]
        layer.self_attn.new_column_order = permutation_matrix
    else:
        layer.self_attn.new_column_order = new_column_order


def rearrange_columns(
    model: nn.Module,
    config: Any,
    training: bool = False,
    ratio_dict: Optional[Dict[str, float]] = None,
) -> None:
    """
    Rearrange columns in all layers for mixed-precision layout.

    Args:
        model: The transformer model
        config: ResQ configuration
        training: Whether in training mode
        ratio_dict: Optional dictionary of per-layer/per-transform ratios.
                   If None, uses config.high_fraction for all layers.
    """
    # Check if remove_ub mode is enabled - skip rearrangement
    remove_ub = getattr(config, 'remove_ub', False)
    if remove_ub:
        logger.info("remove_ub=True: Skipping o_proj column rearrangement")
        return

    model_config = model.config
    num_heads = model_config.num_attention_heads
    num_kv_heads = getattr(model_config, 'num_key_value_heads', num_heads)
    model_dim = model_config.hidden_size
    # Get correct head_dim from config or v_proj
    head_dim = getattr(model_config, 'head_dim', None)
    if head_dim is None:
        if hasattr(model, 'model') and hasattr(model.model, 'layers') and len(model.model.layers) > 0:
            v_proj = model.model.layers[0].self_attn.v_proj
            head_dim = v_proj.out_features // num_kv_heads
        else:
            head_dim = model_dim // num_heads

    # Note: high_bits_length for hidden_dim is kept for reference/logging only
    # The actual o_proj high_bits_length is computed inside rearrange_o_proj
    # based on o_proj input dimension (num_attention_heads * head_dim)
    high_bits_length_hidden = int(config.high_fraction * model_dim)

    logger.debug(f"rearrange_columns: model_dim={model_dim}, head_dim={head_dim}, "
                 f"num_heads={num_heads}, num_kv_heads={num_kv_heads}, "
                 f"high_fraction={config.high_fraction}, "
                 f"high_bits_length_hidden={high_bits_length_hidden}")

    layers = list(model.model.layers)
    for idx, layer in enumerate(tqdm(layers, desc="Rearranging columns")):
        # Get per-layer ratio for Ub (value/o_proj) if available
        layer_fraction = config.high_fraction
        if ratio_dict is not None:
            ub_key = f'layer.{idx}.Ub'
            if ub_key in ratio_dict:
                layer_fraction = ratio_dict[ub_key]
                logger.debug(f"Layer {idx}: using adaptive ratio {layer_fraction:.4f}")

        rearrange_o_proj(
            layer,
            layer_fraction,  # Pass fraction (may be per-layer)
            head_dim,
            training,
        )

    cleanup_memory(verbos=False)


def apply_rotations(
    model: nn.Module,
    basis_dict: Dict[str, torch.Tensor],
    rotation_dict: Dict[str, torch.Tensor],
    config: Any,
    ratio_dict: Optional[Dict[str, float]] = None,
    mix_cfg: Optional[Dict[str, str]] = None,
) -> None:
    """
    Apply basis and rotation matrices to model weights.

    This fuses the eigenvalue basis and random rotations into the model
    weights for efficient inference.

    Args:
        model: The transformer model
        basis_dict: Dictionary of basis matrices from compute_basis
        rotation_dict: Dictionary of rotation matrices
        config: ResQ configuration
        ratio_dict: Optional dictionary of per-layer/per-transform ratios.
                   If None, uses config.high_fraction for all layers.
        mix_cfg: Optional dictionary mapping layer names/patterns to quantization types.
                 Used to skip Ud fusion for down_proj layers with non-resq types.
    """
    def should_skip_ud_fusion(layer_name: str) -> bool:
        """Check if this down_proj layer should skip Ud fusion based on mix_cfg."""
        if not mix_cfg:
            return False
        # Check exact match
        if layer_name in mix_cfg:
            return mix_cfg[layer_name].lower() != 'resq'
        # Check pattern matches
        for pattern, quant_type in mix_cfg.items():
            if fnmatch.fnmatchcase(layer_name, pattern):
                return quant_type.lower() != 'resq'
        return False

    def get_down_proj_quant_type(layer_name: str) -> str:
        """Get the quantization type for a down_proj layer from mix_cfg."""
        if not mix_cfg:
            return 'resq'
        # Check exact match
        if layer_name in mix_cfg:
            return mix_cfg[layer_name].lower()
        # Check pattern matches
        for pattern, quant_type in mix_cfg.items():
            if fnmatch.fnmatchcase(layer_name, pattern):
                return quant_type.lower()
        return 'resq'  # default

    # Check if we should skip fusion (transform-only mode)
    if getattr(config, 'should_skip_fusion', False):
        logger.info(f"output_mode='{config.output_mode}': Skipping weight fusion")
        return  # Early return - don't modify model weights

    model_config = model.config
    num_heads = model_config.num_attention_heads
    num_kv_heads = getattr(model_config, 'num_key_value_heads', num_heads)
    model_dim = model_config.hidden_size
    # Get head_dim from config or compute from v_proj
    head_dim = getattr(model_config, 'head_dim', None)
    if head_dim is None:
        # Try to get from first layer's v_proj
        if hasattr(model, 'model') and hasattr(model.model, 'layers') and len(model.model.layers) > 0:
            v_proj = model.model.layers[0].self_attn.v_proj
            head_dim = v_proj.out_features // num_kv_heads
        else:
            head_dim = model_dim // num_heads
    high_bits_length = int(config.high_fraction * model_dim)

    # Get Ud rotation type and blocksize from config
    ud_rotation_type = getattr(config, 'ud_rotation_type', 'hadamard')
    blocksize = getattr(config, 'down_proj_blocksize', 256)

    # Build composite rotation matrices
    R1_1 = rotation_dict['R1_1'].to(torch.float64)
    R1_2 = rotation_dict['R1_2'].to(torch.float64)

    assert R1_2.shape[0] == high_bits_length, \
        f"R1_2 shape {R1_2.shape[0]} != high_bits_length {high_bits_length}"

    R1 = torch.block_diag(R1_1, R1_2)
    R1_0 = rotation_dict.get('R1_0')
    if R1_0 is not None:
        R1 = torch.block_diag(R1_0.to(torch.float64), R1)

    # Get shared basis or use identity
    if 'attn_mlp' in basis_dict:
        U_attn = basis_dict['attn_mlp'].to(torch.float64)
    else:
        U_attn = torch.eye(model_dim, dtype=torch.float64)

    # Combine basis and rotation
    U_attn = torch.matmul(U_attn, R1)

    # Rotate embeddings and head
    rotate_embeddings(model, U_attn)
    rotate_head(model, U_attn)
    cleanup_memory(verbos=False)

    # Get Ud rotation matrices based on ud_rotation_type
    if ud_rotation_type == 'hadamard':
        hadK = rotation_dict.get('Hd')  # May be None for power-of-2
        K = rotation_dict.get('Hd_K', 1)
        logger.info(f"Using Hadamard rotation for Ud (K={K})")
    else:  # 'random'
        Rd = rotation_dict.get('Rd')
        logger.info(f"Using random orthogonal rotation for Ud")

    # Rotate each layer
    layers = list(model.model.layers)
    for idx, layer in enumerate(tqdm(layers, desc="Applying rotations")):
        # Rotate attention inputs
        rotate_attention_inputs(layer, U_attn)

        # Rotate value projection with per-head basis
        key = f'layer.{idx}.self_attn.value'
        if key in basis_dict:
            # Build per-head R2 rotation matrix
            R2_1 = rotation_dict['R2_1'].to(torch.float64)
            R2_2 = rotation_dict['R2_2'].to(torch.float64)
            R2 = torch.block_diag(R2_1, R2_2)
            R2_0 = rotation_dict.get('R2_0')
            if R2_0 is not None:
                R2 = torch.block_diag(R2_0.to(torch.float64), R2)

            # Check if remove_ub mode is enabled
            remove_ub = getattr(config, 'remove_ub', False)

            if remove_ub:
                # Remove Ub mode: apply only Rb (rotation) without Pb (basis)
                # V_proj and O_proj are rotated with Rb only
                rotate_ov_proj_rotation_only(layer, num_heads, head_dim, Rb=R2)
            else:
                # Standard mode: apply Ub = Pb @ Rb
                U_value = basis_dict[key].to(torch.float64)
                Ub = torch.matmul(U_value, R2)
                rotate_ov_proj(layer, num_heads, head_dim, Ub=Ub)
        # else: No rotation applied when no basis available

        # Rotate attention output
        rotate_attention_output(layer, U_attn)

        # Rotate MLP input
        rotate_mlp_input(layer, U_attn)

        # Rotate MLP output (down_proj) based on quant type and ud_rotation_type
        pd_key = f'layer.{idx}.mlp.down_proj'
        down_proj_name = f'model.layers.{idx}.mlp.down_proj'

        # Get the quantization type for this down_proj layer
        quant_type = get_down_proj_quant_type(down_proj_name)

        if quant_type == 'int4_hadamard':
            # int4_hadamard: Apply Ua.T @ W @ Hd (skip Pd basis, fuse Hd into weights)
            logger.info(f"Applying Hadamard-only rotation for {down_proj_name} (int4_hadamard)")
            rotate_mlp_output_hadamard_only(layer, U_attn, hadK, K)
        elif quant_type in ('w8a8_dynamic', 'float'):
            # w8a8_dynamic or float: Only apply Ua rotation (no Ud fusion)
            logger.info(f"Skipping Ud fusion for {down_proj_name} ({quant_type})")
            rotate_mlp_output(layer, R1=U_attn)
        elif pd_key in basis_dict:
            # resq (default): Apply full Ua.T @ W @ Pd @ Hd (or Rd)
            Pd = basis_dict[pd_key].to(torch.float64)
            if ud_rotation_type == 'hadamard':
                rotate_mlp_output_hadamard(layer, U_attn, Pd, hadK, K, blocksize)
            else:  # 'random'
                rotate_mlp_output_random(layer, U_attn, Pd, Rd, blocksize)
        else:
            # Fallback to original rotation if Pd not available
            rotate_mlp_output(layer, R1=U_attn)

    cleanup_memory(verbos=False)


def configure_quantizers(
    model: nn.Module,
    config: Any,
) -> None:
    """
    Configure quantizers for all layers.

    Sets up mixed-precision quantization parameters based on config.

    Args:
        model: The transformer model with ActQuantWrapper layers
        config: ResQ configuration
    """
    model_config = model.config
    model_dim = model_config.hidden_size
    num_heads = model_config.num_attention_heads
    head_dim = model_dim // num_heads

    high_bits_length = int(config.high_fraction * model_dim)
    high_bits_length_head = int(config.high_fraction * head_dim)

    for name, module in model.named_modules():
        if isinstance(module, ActQuantWrapper):
            # Configure input quantizer (symmetric quantization)
            module.quantizer.configure(
                bits=config.a_bits,
                groupsize=config.a_groupsize,
                sym=True,  # Always symmetric
                clip_ratio=config.a_clip_ratio,
                high_bits_length=high_bits_length,
                high_bits=config.high_bits,
                low_bits=config.low_bits,
            )

            # Configure output quantizer for attention outputs
            if 'o_proj' in name or 'v_proj' in name:
                module.out_quantizer.configure(
                    bits=config.v_bits if 'v_proj' in name else 16,
                    groupsize=-1,
                    sym=True,  # Always symmetric
                    clip_ratio=config.v_clip_ratio,
                    high_bits_length=high_bits_length_head * num_heads,
                    high_bits=config.high_bits,
                    low_bits=config.low_bits,
                )


def resq_quantize(
    model: nn.Module,
    config: Any,
    basis_dict: Optional[Dict[str, torch.Tensor]] = None,
    rotation_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> nn.Module:
    """
    Apply ResQ quantization to the model.

    This is the main entry point for ResQ quantization.

    Args:
        model: The transformer model
        config: ResQ configuration
        basis_dict: Pre-computed basis matrices (optional)
        rotation_dict: Pre-computed rotation matrices (optional)

    Returns:
        Quantized model
    """
    logger.info("Starting ResQ quantization")

    # Fuse layer norms into adjacent linear layers
    logger.info("Fusing layer norms")
    fuse_layer_norms(model)
    cleanup_memory(verbos=False)

    # Load or generate basis and rotation matrices
    if basis_dict is None and config.optimized_basis_path:
        logger.info(f"Loading basis from {config.optimized_basis_path}")
        basis_dict = torch.load(config.optimized_basis_path, map_location='cpu')

    if rotation_dict is None and config.optimized_rotation_path:
        logger.info(f"Loading rotations from {config.optimized_rotation_path}")
        rotation_dict = torch.load(config.optimized_rotation_path, map_location='cpu')

    if rotation_dict is None:
        logger.info("Generating random rotations")
        from .basis_processor import generate_random_rotations
        rotation_dict = generate_random_rotations(
            hidden_dim=model.config.hidden_size,
            head_dim=model.config.hidden_size // model.config.num_attention_heads,
            high_fraction=config.high_fraction,
            seed=config.seed,
        )

    # Apply rotations to model weights
    if basis_dict is not None and rotation_dict is not None:
        logger.info("Applying rotations to model")
        apply_rotations(model, basis_dict, rotation_dict, config)

    # Rearrange columns for mixed-precision layout
    logger.info("Rearranging columns for mixed precision")
    rearrange_columns(model, config, training=False)

    # Add activation quantization wrappers
    logger.info("Adding activation quantizers")
    add_actquant(model)

    # Configure quantizers
    logger.info("Configuring quantizers")
    configure_quantizers(model, config)

    logger.info("ResQ quantization complete")
    return model
