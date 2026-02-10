# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
Single-layer forward inference comparison between original and quantized models.

Performs a full propagated forward pass through one decoder layer for both
the original and ResQ-quantized models (using the same random input), then
compares outputs at each linear layer with proper inverse transforms applied.

Usage:
    python compare_quant_matmul.py \
        --model_path /path/to/Qwen3-32B \
        --quant_path /path/to/debug_output \
        --layer_idx 0 --seq_len 32 --seed 42
"""

import argparse
import json
import math
import os
from typing import Dict, Optional, Tuple

import torch
from safetensors import safe_open


# ---------------------------------------------------------------------------
# Utility helpers (adapted from inspect_quant_weights.py and hadamard_utils.py)
# ---------------------------------------------------------------------------

def load_safetensors(path: str) -> Dict[str, torch.Tensor]:
    """Load tensors from safetensors file or directory."""
    tensors = {}
    if os.path.isdir(path):
        files = sorted([f for f in os.listdir(path) if f.endswith('.safetensors')])
        if not files:
            raise ValueError(f"No .safetensors files found in directory: {path}")
        for filename in files:
            filepath = os.path.join(path, filename)
            with safe_open(filepath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
    else:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


def dequantize_weight(weight_int: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize integer weight using scale (symmetric quantization)."""
    return weight_int.float() * scale


def compute_error_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> Dict:
    """Compute error metrics between original and reconstructed tensors."""
    diff = original.float() - reconstructed.float()
    mse = (diff ** 2).mean().item()
    max_abs_error = diff.abs().max().item()
    rel_error = (diff.abs() / (original.float().abs() + 1e-8)).mean().item()
    signal_power = (original.float() ** 2).mean().item()
    noise_power = mse
    snr_db = 10 * math.log10(signal_power / (noise_power + 1e-10))

    # Cosine similarity
    a = original.float().reshape(-1)
    b = reconstructed.float().reshape(-1)
    cos_sim = (torch.dot(a, b) / (a.norm() * b.norm() + 1e-10)).item()

    return {
        'mse': mse,
        'cos_sim': cos_sim,
        'snr_db': snr_db,
        'max_abs_error': max_abs_error,
        'mean_rel_error': rel_error,
    }


# ---------------------------------------------------------------------------
# Hadamard transform (from hadamard_utils.py)
# ---------------------------------------------------------------------------

def is_pow2(n: int) -> bool:
    return (n & (n - 1) == 0) and (n > 0)


def hadamard_transform_cpu(u: torch.Tensor) -> torch.Tensor:
    """Fast Hadamard transform using butterfly algorithm."""
    n = u.shape[-1]
    assert is_pow2(n), f"Last dimension must be power of 2, got {n}"
    original_shape = u.shape
    x = u.reshape(-1, n)
    h = 1
    while h < n:
        x = x.view(-1, n // (2 * h), 2, h)
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        x = torch.stack([a + b, a - b], dim=2)
        x = x.view(-1, n)
        h *= 2
    return x.view(original_shape)


def matmul_hadU_cpu(X: torch.Tensor, hadK: Optional[torch.Tensor], K: int) -> torch.Tensor:
    """Apply structured Hadamard transform: X @ (hadK x H)^T."""
    n = X.shape[-1]
    if K == 1:
        return hadamard_transform_cpu(X.contiguous()) / math.sqrt(n)
    input_tensor = X.view(-1, K, n // K)
    input_tensor = hadamard_transform_cpu(input_tensor.contiguous()) / math.sqrt(n)
    input_tensor = hadK.to(input_tensor.device).to(input_tensor.dtype) @ input_tensor
    return input_tensor.reshape(X.shape)


# ---------------------------------------------------------------------------
# RMSNorm / RoPE helpers
# ---------------------------------------------------------------------------

def rmsnorm(x: torch.Tensor, weight: Optional[torch.Tensor] = None, eps: float = 1e-6) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    x_norm = x.float() / rms
    if weight is not None:
        x_norm = x_norm * weight.float()
    return x_norm


def rmsnorm_per_head(x: torch.Tensor, weight: torch.Tensor, num_heads: int, head_dim: int, eps: float = 1e-6) -> torch.Tensor:
    """Apply RMSNorm per head: x is [B, S, num_heads * head_dim]."""
    B, S, _ = x.shape
    x_heads = x.float().view(B, S, num_heads, head_dim)
    rms = torch.sqrt(torch.mean(x_heads ** 2, dim=-1, keepdim=True) + eps)
    x_norm = x_heads / rms
    # weight may be [head_dim] (shared across heads) or [num_heads * head_dim] (per-head)
    if weight.numel() == head_dim:
        x_norm = x_norm * weight.float().view(1, 1, 1, head_dim)
    else:
        x_norm = x_norm * weight.float().view(1, 1, num_heads, head_dim)
    return x_norm.view(B, S, num_heads * head_dim)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


def build_rope_cache(seq_len: int, head_dim: int, theta: float = 1_000_000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build cos/sin cache for RoPE. Returns [1, 1, seq_len, head_dim]."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    positions = torch.arange(seq_len, dtype=torch.float64)
    angles = torch.outer(positions, freqs)  # [seq_len, head_dim/2]
    # Duplicate to match full head_dim
    cos_cache = torch.cos(angles).repeat(1, 2).float()  # [seq_len, head_dim]
    sin_cache = torch.sin(angles).repeat(1, 2).float()
    return cos_cache.unsqueeze(0).unsqueeze(0), sin_cache.unsqueeze(0).unsqueeze(0)  # [1, 1, S, HD]


# ---------------------------------------------------------------------------
# GQA attention
# ---------------------------------------------------------------------------

def gqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Grouped-Query Attention with causal mask.

    Args:
        q: [B, S, num_heads * head_dim]
        k: [B, S, num_kv_heads * head_dim]
        v: [B, S, num_kv_heads * head_dim]

    Returns:
        [B, S, num_heads * head_dim]
    """
    B, S, _ = q.shape
    q = q.view(B, S, num_heads, head_dim).transpose(1, 2)       # [B, NH, S, HD]
    k = k.view(B, S, num_kv_heads, head_dim).transpose(1, 2)    # [B, NKV, S, HD]
    v = v.view(B, S, num_kv_heads, head_dim).transpose(1, 2)    # [B, NKV, S, HD]

    # GQA: expand KV heads
    heads_per_group = num_heads // num_kv_heads
    if heads_per_group > 1:
        k = k.repeat_interleave(heads_per_group, dim=1)  # [B, NH, S, HD]
        v = v.repeat_interleave(heads_per_group, dim=1)

    # Scaled dot product attention with causal mask
    scale = 1.0 / math.sqrt(head_dim)
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, NH, S, S]

    # Causal mask
    causal_mask = torch.triu(torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1)
    attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    attn_weights = torch.softmax(attn_weights.float(), dim=-1).to(q.dtype)
    attn_out = torch.matmul(attn_weights, v)  # [B, NH, S, HD]
    return attn_out.transpose(1, 2).reshape(B, S, num_heads * head_dim)


# ---------------------------------------------------------------------------
# Weight loading helpers
# ---------------------------------------------------------------------------

def load_original_layer_weights(model_path: str, layer_idx: int) -> Dict[str, torch.Tensor]:
    """Load only the specified layer's weights from the original HF model (lazy)."""
    index_path = os.path.join(model_path, 'model.safetensors.index.json')

    prefix = f"model.layers.{layer_idx}."

    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            index = json.load(f)
        weight_map = index['weight_map']

        # Find all shards containing this layer
        needed_shards = set()
        needed_keys = []
        for key, shard in weight_map.items():
            if key.startswith(prefix):
                needed_shards.add(shard)
                needed_keys.append(key)

        weights = {}
        for shard_name in needed_shards:
            shard_path = os.path.join(model_path, shard_name)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith(prefix):
                        weights[key] = f.get_tensor(key)
        return weights
    else:
        # Single file
        st_files = sorted([f for f in os.listdir(model_path) if f.endswith('.safetensors')])
        weights = {}
        for fname in st_files:
            with safe_open(os.path.join(model_path, fname), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith(prefix):
                        weights[key] = f.get_tensor(key)
        return weights


# ---------------------------------------------------------------------------
# Transform reconstruction
# ---------------------------------------------------------------------------

def reconstruct_transforms(
    transform_tensors: Dict[str, torch.Tensor],
    layer_idx: int,
) -> Dict[str, torch.Tensor]:
    """Reconstruct Ua, Ub, Ud components from saved transform tensors."""
    i = layer_idx
    result = {}

    # Ua = P_a @ R_a
    P_a = transform_tensors[f'resq.layer.{i}.P_a'].float()
    R_a = transform_tensors[f'resq.layer.{i}.R_a'].float()
    result['Ua'] = torch.matmul(P_a, R_a)

    # Ub = bmm(P_b, R_b.expand) -> [num_kv_heads, head_dim, head_dim]
    P_b_key = f'resq.layer.{i}.P_b'
    R_b_key = f'resq.layer.{i}.R_b'
    if P_b_key in transform_tensors:
        P_b = transform_tensors[P_b_key].float()  # [num_kv_heads, head_dim, head_dim]
        R_b = transform_tensors[R_b_key].float()   # [head_dim, head_dim]
        if P_b.dim() == 2:
            # Shared across heads
            result['Ub'] = torch.matmul(P_b, R_b).unsqueeze(0)
        else:
            # Per-head: expand R_b
            R_b_expanded = R_b.unsqueeze(0).expand(P_b.shape[0], -1, -1)
            result['Ub'] = torch.bmm(P_b, R_b_expanded)
    else:
        result['Ub'] = None

    # Ud: P_d + Hadamard (or random Rd)
    P_d_key = f'resq.layer.{i}.P_d'
    if P_d_key in transform_tensors:
        result['P_d'] = transform_tensors[P_d_key].float()
    else:
        result['P_d'] = None

    hadK_key = f'resq.layer.{i}.R_d_hadK'
    K_key = f'resq.layer.{i}.R_d_K'
    if hadK_key in transform_tensors:
        result['hadK'] = transform_tensors[hadK_key].float()
    else:
        result['hadK'] = None

    if K_key in transform_tensors:
        result['K'] = transform_tensors[K_key].item()
    else:
        result['K'] = 1

    # Full Ud for random mode
    Ud_key = f'resq.layer.{i}.R_d'
    if Ud_key in transform_tensors:
        result['Rd'] = transform_tensors[Ud_key].float()
    else:
        result['Rd'] = None

    return result


def load_config_from_transforms(transform_tensors: Dict[str, torch.Tensor]) -> Dict:
    """Load configuration from resq.config.* tensors."""
    cfg = {}
    cfg['num_heads'] = transform_tensors['resq.config.num_heads'].item()
    cfg['num_kv_heads'] = transform_tensors['resq.config.num_kv_heads'].item()
    cfg['head_dim'] = transform_tensors['resq.config.head_dim'].item()
    cfg['hidden_dim'] = transform_tensors['resq.config.hidden_dim'].item()
    cfg['blocksize'] = transform_tensors['resq.config.blocksize'].item()
    cfg['high_fraction'] = transform_tensors['resq.config.high_fraction'].item()
    if 'resq.config.intermediate_size' in transform_tensors:
        cfg['intermediate_size'] = transform_tensors['resq.config.intermediate_size'].item()
    # Decode ud_rotation_type
    if 'resq.config.ud_rotation_type' in transform_tensors:
        chars = transform_tensors['resq.config.ud_rotation_type'].tolist()
        cfg['ud_rotation_type'] = ''.join(chr(c) for c in chars)
    else:
        cfg['ud_rotation_type'] = 'hadamard'
    return cfg


# ---------------------------------------------------------------------------
# O_proj column rearrangement (from resq_processor.py:476-536)
# ---------------------------------------------------------------------------

def build_o_proj_column_order(in_dim: int, head_dim: int, high_fraction: float) -> torch.Tensor:
    """Build the column reordering index for o_proj mixed-precision layout."""
    num_replicated_heads = in_dim // head_dim
    high_bits_length = int(high_fraction * in_dim)
    high_length_per_head = high_bits_length // num_replicated_heads

    chunk_starts = torch.arange(0, in_dim, head_dim)
    high_precision_columns = torch.arange(head_dim - high_length_per_head, head_dim)
    columns_to_end = (chunk_starts.unsqueeze(1) + high_precision_columns).flatten()

    mask = torch.ones(in_dim, dtype=torch.bool)
    mask[columns_to_end] = False
    remaining = torch.arange(in_dim)[mask]

    return torch.cat([remaining, columns_to_end])


# ---------------------------------------------------------------------------
# Ud forward transform (apply to down_proj input in quantized path)
# ---------------------------------------------------------------------------

def apply_Ud_forward(
    x: torch.Tensor,
    P_d: Optional[torch.Tensor],
    hadK: Optional[torch.Tensor],
    K: int,
    blocksize: int,
) -> torch.Tensor:
    """
    Apply Ud transform to down_proj input: x -> x @ block_diag(P_d) @ H

    In the quantized model, the down_proj weight has Ud fused as:
        W_fused = Ua.T @ W_orig @ Ud
    where Ud = block_diag(P_d) @ H

    At inference, the input to W_fused must be: activation @ Ud
    """
    if P_d is not None:
        n = x.shape[-1]
        num_blocks = n // blocksize
        x_blocked = x.view(*x.shape[:-1], num_blocks, blocksize)
        x_blocked = torch.matmul(x_blocked, P_d.to(x.dtype))
        x = x_blocked.view(*x.shape[:-1], n)

    # Apply Hadamard
    x = matmul_hadU_cpu(x.cpu(), hadK, K)
    return x


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def get_orig_bias_or_weight(subpath: str, orig_weights: Dict[str, torch.Tensor], L: str) -> Optional[torch.Tensor]:
    """Helper to get optional weight tensor from original model."""
    key = f"{L}.{subpath}"
    if key in orig_weights:
        return orig_weights[key].float()
    return None


def compare_single_layer(
    model_path: str,
    quant_path: str,
    layer_idx: int = 0,
    seq_len: int = 32,
    seed: int = 42,
):
    print("=" * 70)
    print(f"Single-Layer Forward Comparison: Layer {layer_idx}")
    print("=" * 70)

    # ---- Load transform tensors ----
    transform_path = os.path.join(quant_path, 'resq_transforms.safetensors')
    if not os.path.exists(transform_path):
        raise FileNotFoundError(
            f"Transform file not found: {transform_path}\n"
            f"Run quantization with --output_mode debug to generate it."
        )
    print(f"Loading transforms from: {transform_path}")
    transform_tensors = load_safetensors(transform_path)

    cfg = load_config_from_transforms(transform_tensors)
    num_heads = int(cfg['num_heads'])
    num_kv_heads = int(cfg['num_kv_heads'])
    head_dim = int(cfg['head_dim'])
    hidden_dim = int(cfg['hidden_dim'])
    blocksize = int(cfg['blocksize'])
    high_fraction = cfg['high_fraction']
    ud_rotation_type = cfg['ud_rotation_type']

    print(f"Config: hidden_dim={hidden_dim}, num_heads={num_heads}, "
          f"num_kv_heads={num_kv_heads}, head_dim={head_dim}, "
          f"blocksize={blocksize}, high_fraction={high_fraction}, "
          f"ud_rotation_type={ud_rotation_type}")

    transforms = reconstruct_transforms(transform_tensors, layer_idx)
    Ua = transforms['Ua']   # [hidden_dim, hidden_dim]
    Ub = transforms['Ub']   # [num_kv_heads, head_dim, head_dim] or None

    # ---- Load quantized weights ----
    print(f"Loading quantized weights from: {quant_path}")
    quant_tensors = load_safetensors(quant_path)

    # ---- Load original weights (lazy, single layer) ----
    print(f"Loading original layer {layer_idx} weights from: {model_path}")
    orig_weights = load_original_layer_weights(model_path, layer_idx)

    # ---- Build shorthand accessors ----
    L = f"model.layers.{layer_idx}"

    def get_orig_weight(proj: str) -> torch.Tensor:
        return orig_weights[f"{L}.{proj}.weight"].float()

    def get_orig_bias(proj: str) -> Optional[torch.Tensor]:
        key = f"{L}.{proj}.bias"
        if key in orig_weights:
            return orig_weights[key].float()
        return None

    def get_quant_dequant(proj: str) -> torch.Tensor:
        """Dequantize and concatenate low + high weight parts."""
        full_name = f"{L}.{proj}"
        parts = []
        low_key = f"{full_name}.weight_low"
        high_key = f"{full_name}.weight_high"

        if low_key in quant_tensors:
            w_low = quant_tensors[low_key]
            s_low = quant_tensors[f"{full_name}.scale_low"]
            parts.append(dequantize_weight(w_low, s_low))
        if high_key in quant_tensors:
            w_high = quant_tensors[high_key]
            s_high = quant_tensors[f"{full_name}.scale_high"]
            parts.append(dequantize_weight(w_high, s_high))

        if len(parts) == 2:
            return torch.cat(parts, dim=1)
        elif len(parts) == 1:
            return parts[0]
        else:
            raise ValueError(f"No quantized weights found for {full_name}")

    def get_quant_bias(proj: str) -> Optional[torch.Tensor]:
        key = f"{L}.{proj}.bias"
        if key in quant_tensors:
            return quant_tensors[key].float()
        return None

    def get_layer_high_fraction(proj: str) -> float:
        """Get per-layer high_fraction if available (adaptive ratio)."""
        key = f"{L}.{proj}.high_fraction"
        if key in quant_tensors:
            return quant_tensors[key].item()
        return high_fraction

    # ---- Generate random input ----
    torch.manual_seed(seed)
    x = torch.randn(1, seq_len, hidden_dim, dtype=torch.float32)

    # ---- Build RoPE cache ----
    rope_theta = 1_000_000.0
    cos_cache, sin_cache = build_rope_cache(seq_len, head_dim, theta=rope_theta)
    # cos/sin: [1, 1, S, HD] -> broadcast over heads

    print(f"Input: x {list(x.shape)}, seed={seed}, rope_theta={rope_theta}")
    print()

    # ---- Norm weights ----
    gamma_input = orig_weights.get(f"{L}.input_layernorm.weight")
    gamma_post = orig_weights.get(f"{L}.post_attention_layernorm.weight")
    if gamma_input is not None:
        gamma_input = gamma_input.float()
    if gamma_post is not None:
        gamma_post = gamma_post.float()

    # ---- q_norm / k_norm weights (Qwen3 has per-head QK norm) ----
    q_norm_weight = get_orig_bias_or_weight("self_attn.q_norm.weight", orig_weights, L)
    k_norm_weight = get_orig_bias_or_weight("self_attn.k_norm.weight", orig_weights, L)

    # =====================================================================
    # ORIGINAL PATH
    # =====================================================================
    print("-" * 70)
    print("Computing ORIGINAL path...")
    print("-" * 70)

    # Input LayerNorm
    y_orig = rmsnorm(x, gamma_input)

    # Q/K/V projections
    W_q = get_orig_weight("self_attn.q_proj")
    W_k = get_orig_weight("self_attn.k_proj")
    W_v = get_orig_weight("self_attn.v_proj")
    b_q = get_orig_bias("self_attn.q_proj")
    b_k = get_orig_bias("self_attn.k_proj")
    b_v = get_orig_bias("self_attn.v_proj")

    q_orig = torch.matmul(y_orig, W_q.T)
    if b_q is not None:
        q_orig = q_orig + b_q
    k_orig = torch.matmul(y_orig, W_k.T)
    if b_k is not None:
        k_orig = k_orig + b_k
    v_orig = torch.matmul(y_orig, W_v.T)
    if b_v is not None:
        v_orig = v_orig + b_v

    # Per-head QK norm (Qwen3)
    if q_norm_weight is not None:
        q_normed_orig = rmsnorm_per_head(q_orig, q_norm_weight, num_heads, head_dim)
    else:
        q_normed_orig = q_orig
    if k_norm_weight is not None:
        k_normed_orig = rmsnorm_per_head(k_orig, k_norm_weight, num_kv_heads, head_dim)
    else:
        k_normed_orig = k_orig

    # RoPE
    # q: [B, S, num_heads * head_dim] -> [B, num_heads, S, head_dim]
    q_for_rope = q_normed_orig.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
    k_for_rope = k_normed_orig.view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    q_pos_orig = apply_rope(q_for_rope, cos_cache, sin_cache)
    k_pos_orig = apply_rope(k_for_rope, cos_cache, sin_cache)
    q_pos_orig = q_pos_orig.transpose(1, 2).reshape(1, seq_len, num_heads * head_dim)
    k_pos_orig = k_pos_orig.transpose(1, 2).reshape(1, seq_len, num_kv_heads * head_dim)

    # GQA attention
    attn_orig = gqa_attention(q_pos_orig, k_pos_orig, v_orig, num_heads, num_kv_heads, head_dim)

    # O projection
    W_o = get_orig_weight("self_attn.o_proj")
    b_o = get_orig_bias("self_attn.o_proj")
    o_orig = torch.matmul(attn_orig, W_o.T)
    if b_o is not None:
        o_orig = o_orig + b_o

    # Residual 1
    x2_orig = x + o_orig

    # Post-attention LayerNorm
    y2_orig = rmsnorm(x2_orig, gamma_post)

    # MLP
    W_gate = get_orig_weight("mlp.gate_proj")
    W_up = get_orig_weight("mlp.up_proj")
    W_down = get_orig_weight("mlp.down_proj")
    b_gate = get_orig_bias("mlp.gate_proj")
    b_up = get_orig_bias("mlp.up_proj")
    b_down = get_orig_bias("mlp.down_proj")

    gate_orig = torch.matmul(y2_orig, W_gate.T)
    if b_gate is not None:
        gate_orig = gate_orig + b_gate
    up_orig = torch.matmul(y2_orig, W_up.T)
    if b_up is not None:
        up_orig = up_orig + b_up

    mlp_inter_orig = torch.nn.functional.silu(gate_orig) * up_orig

    down_orig = torch.matmul(mlp_inter_orig, W_down.T)
    if b_down is not None:
        down_orig = down_orig + b_down

    # Residual 2
    x3_orig = x2_orig + down_orig

    # Free large original weights
    del W_q, W_k, W_v, W_o, W_gate, W_up, W_down

    # =====================================================================
    # QUANTIZED PATH
    # =====================================================================
    print("-" * 70)
    print("Computing QUANTIZED path...")
    print("-" * 70)

    # Rotate input by Ua: x_rot = x @ Ua (embedding fuses Ua into output)
    x_rot = torch.matmul(x, Ua)

    # RMSNorm with gamma=1 (gamma fused into weights)
    y_rot = rmsnorm(x_rot, weight=None)

    # Dequantized projections
    W_q_dq = get_quant_dequant("self_attn.q_proj")
    W_k_dq = get_quant_dequant("self_attn.k_proj")
    W_v_dq = get_quant_dequant("self_attn.v_proj")
    b_q_quant = get_quant_bias("self_attn.q_proj")
    b_k_quant = get_quant_bias("self_attn.k_proj")
    b_v_quant = get_quant_bias("self_attn.v_proj")

    q_rot = torch.matmul(y_rot, W_q_dq.T)
    if b_q_quant is not None:
        q_rot = q_rot + b_q_quant
    k_rot = torch.matmul(y_rot, W_k_dq.T)
    if b_k_quant is not None:
        k_rot = k_rot + b_k_quant
    v_rot = torch.matmul(y_rot, W_v_dq.T)
    if b_v_quant is not None:
        v_rot = v_rot + b_v_quant

    # Per-head QK norm (NOT fused, same weights)
    if q_norm_weight is not None:
        q_normed_rot = rmsnorm_per_head(q_rot, q_norm_weight, num_heads, head_dim)
    else:
        q_normed_rot = q_rot
    if k_norm_weight is not None:
        k_normed_rot = rmsnorm_per_head(k_rot, k_norm_weight, num_kv_heads, head_dim)
    else:
        k_normed_rot = k_rot

    # RoPE
    q_for_rope_rot = q_normed_rot.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
    k_for_rope_rot = k_normed_rot.view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    q_pos_rot = apply_rope(q_for_rope_rot, cos_cache, sin_cache)
    k_pos_rot = apply_rope(k_for_rope_rot, cos_cache, sin_cache)
    q_pos_rot = q_pos_rot.transpose(1, 2).reshape(1, seq_len, num_heads * head_dim)
    k_pos_rot = k_pos_rot.transpose(1, 2).reshape(1, seq_len, num_kv_heads * head_dim)

    # GQA attention (v_rot is in Ub-rotated space per KV head)
    attn_rot = gqa_attention(q_pos_rot, k_pos_rot, v_rot, num_heads, num_kv_heads, head_dim)

    # O projection with column rearrangement
    W_o_dq = get_quant_dequant("self_attn.o_proj")
    b_o_quant = get_quant_bias("self_attn.o_proj")

    # The o_proj weight is stored in rearranged column order [mid | high].
    # We must apply the same column permutation to the input before matmul.
    o_proj_in_dim = num_heads * head_dim
    o_proj_high_fraction = get_layer_high_fraction("self_attn.o_proj")
    o_col_order = build_o_proj_column_order(o_proj_in_dim, head_dim, o_proj_high_fraction)
    attn_rearranged = attn_rot[..., o_col_order]
    o_rot = torch.matmul(attn_rearranged, W_o_dq.T)
    if b_o_quant is not None:
        o_rot = o_rot + b_o_quant

    # Residual 1
    x2_rot = x_rot + o_rot

    # Post-attention RMSNorm with gamma=1
    y2_rot = rmsnorm(x2_rot, weight=None)

    # MLP
    W_gate_dq = get_quant_dequant("mlp.gate_proj")
    W_up_dq = get_quant_dequant("mlp.up_proj")
    W_down_dq = get_quant_dequant("mlp.down_proj")
    b_gate_quant = get_quant_bias("mlp.gate_proj")
    b_up_quant = get_quant_bias("mlp.up_proj")
    b_down_quant = get_quant_bias("mlp.down_proj")

    gate_rot = torch.matmul(y2_rot, W_gate_dq.T)
    if b_gate_quant is not None:
        gate_rot = gate_rot + b_gate_quant
    up_rot = torch.matmul(y2_rot, W_up_dq.T)
    if b_up_quant is not None:
        up_rot = up_rot + b_up_quant

    mlp_rot = torch.nn.functional.silu(gate_rot) * up_rot

    # Apply Ud transform for down_proj input
    P_d = transforms['P_d']
    hadK = transforms['hadK']
    K_val = int(transforms['K'])

    mlp_ud = apply_Ud_forward(mlp_rot, P_d, hadK, K_val, blocksize)

    down_rot = torch.matmul(mlp_ud, W_down_dq.T)
    if b_down_quant is not None:
        down_rot = down_rot + b_down_quant

    # Residual 2
    x3_rot = x2_rot + down_rot

    # =====================================================================
    # COMPARISONS (transform quantized outputs back to original space)
    # =====================================================================
    print("-" * 70)
    print("Computing comparisons...")
    print("-" * 70)

    Ua_T = Ua.T  # [hidden_dim, hidden_dim] - inverse is transpose (orthogonal)

    comparisons = []

    # q: Ua cancels on input, same output space
    comparisons.append(("q", q_orig, q_rot))
    comparisons.append(("k", k_orig, k_rot))

    # v: per-head v_rot @ Ub.T to recover original space
    if Ub is not None:
        B_, S_, _ = v_rot.shape
        v_rot_heads = v_rot.view(B_, S_, num_kv_heads, head_dim)
        # Ub: [num_kv_heads, head_dim, head_dim]
        # v_rot_heads: [B, S, num_kv_heads, head_dim]
        # v_recovered[..., h, :] = v_rot_heads[..., h, :] @ Ub[h].T
        Ub_T = Ub.transpose(-2, -1)  # [num_kv_heads, head_dim, head_dim]
        v_recovered = torch.einsum('bshd,hde->bshe', v_rot_heads, Ub_T)
        v_recovered = v_recovered.view(B_, S_, num_kv_heads * head_dim)
    else:
        v_recovered = v_rot
    comparisons.append(("v", v_orig, v_recovered))

    # attn_out: per-head attn_rot @ Ub.T (expanded to num_heads)
    if Ub is not None:
        heads_per_group = num_heads // num_kv_heads
        Ub_expanded = Ub.repeat_interleave(heads_per_group, dim=0)  # [num_heads, HD, HD]
        Ub_expanded_T = Ub_expanded.transpose(-2, -1)

        B_, S_, _ = attn_rot.shape
        attn_rot_heads = attn_rot.view(B_, S_, num_heads, head_dim)
        attn_recovered = torch.einsum('bshd,hde->bshe', attn_rot_heads, Ub_expanded_T)
        attn_recovered = attn_recovered.view(B_, S_, num_heads * head_dim)
    else:
        attn_recovered = attn_rot
    comparisons.append(("attn_out", attn_orig, attn_recovered))

    # o: o_rot @ Ua.T (Ua-rotated space -> original)
    o_recovered = torch.matmul(o_rot, Ua_T)
    comparisons.append(("o", o_orig, o_recovered))

    # x2 (residual): x2_rot @ Ua.T
    x2_recovered = torch.matmul(x2_rot, Ua_T)
    comparisons.append(("x2 (residual)", x2_orig, x2_recovered))

    # gate/up: Ua cancels, same output space
    comparisons.append(("gate", gate_orig, gate_rot))
    comparisons.append(("up", up_orig, up_rot))

    # down: down_rot @ Ua.T
    down_recovered = torch.matmul(down_rot, Ua_T)
    comparisons.append(("down", down_orig, down_recovered))

    # x3 (output): x3_rot @ Ua.T
    x3_recovered = torch.matmul(x3_rot, Ua_T)
    comparisons.append(("x3 (output)", x3_orig, x3_recovered))

    # =====================================================================
    # PRINT RESULTS
    # =====================================================================
    print()
    print(f"{'Comparison':<16} {'Shape':<22} {'MSE':>10} {'Cos Sim':>10} {'SNR(dB)':>10} {'Max Err':>10}")
    print("-" * 80)

    for name, orig_t, quant_t in comparisons:
        metrics = compute_error_metrics(orig_t, quant_t)
        shape_str = str(list(orig_t.shape))
        print(f"{name:<16} {shape_str:<22} {metrics['mse']:>10.2e} {metrics['cos_sim']:>10.5f} "
              f"{metrics['snr_db']:>10.1f} {metrics['max_abs_error']:>10.4f}")

    print("-" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Single-layer forward comparison between original and quantized models"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to original HuggingFace model directory")
    parser.add_argument("--quant_path", type=str, required=True,
                        help="Path to debug output directory")
    parser.add_argument("--layer_idx", type=int, default=0,
                        help="Layer index to compare (default: 0)")
    parser.add_argument("--seq_len", type=int, default=32,
                        help="Sequence length for random input (default: 32)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    compare_single_layer(
        model_path=args.model_path,
        quant_path=args.quant_path,
        layer_idx=args.layer_idx,
        seq_len=args.seq_len,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
