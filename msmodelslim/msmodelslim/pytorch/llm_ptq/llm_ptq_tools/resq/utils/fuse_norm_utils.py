# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Adapted from ResQ and QuaRot
"""
Fuse layer normalization into adjacent linear layers.

This is a key step in rotation-based quantization methods that
allows the normalization to be absorbed into the weight matrices.
Memory-efficient implementation that processes on CPU.
"""

import gc
import typing
import torch
from tqdm import tqdm


def cleanup_memory():
    """Clean up NPU memory."""
    gc.collect()
    try:
        import torch_npu
        if torch.npu.is_available():
            torch.npu.empty_cache()
    except ImportError:
        pass


def fuse_ln_linear(
    layernorm: torch.nn.Module,
    linear_layers: typing.Iterable[torch.nn.Linear],
) -> None:
    """
    Fuse the linear operations in LayerNorm into the adjacent linear blocks.

    This transforms: LayerNorm(x) @ W -> x @ (gamma * W)

    Memory-efficient: moves to CPU first, then converts dtype.

    Args:
        layernorm: The layer normalization module.
        linear_layers: Iterable of linear layers to fuse into.
    """
    # Get layernorm weight - move to CPU first, then convert (avoids device OOM)
    ln_weight = layernorm.weight.data.cpu().float()

    ln_bias = None
    if hasattr(layernorm, "bias") and layernorm.bias is not None:
        ln_bias = layernorm.bias.data.cpu().float()

    for linear in linear_layers:
        original_device = linear.weight.device
        linear_dtype = linear.weight.dtype

        # Move weight to CPU first, then convert to float (avoids device OOM)
        W_ = linear.weight.data.cpu().float()

        # Free device memory immediately
        linear.weight.data = torch.empty(0, device='cpu', dtype=linear_dtype)
        cleanup_memory()

        # Calculating new weight: W_new = W * gamma
        W_new = W_ * ln_weight
        del W_

        # Move back to original device and dtype
        linear.weight.data = W_new.to(dtype=linear_dtype, device=original_device)
        del W_new
        cleanup_memory()

        # Handle bias if present in layernorm
        if ln_bias is not None:
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=linear_dtype, device=original_device)
                )

            bias_ = linear.bias.data.cpu().float()
            # Need to get weight again for matmul
            W_for_bias = linear.weight.data.cpu().float()
            new_bias = bias_ + torch.matmul(W_for_bias, ln_bias)
            linear.bias.data = new_bias.to(dtype=linear_dtype, device=original_device)
            del bias_, W_for_bias, new_bias


def fuse_layer_norms(model, verbose: bool = True):
    """
    Fuse all layer normalizations in the model into adjacent linear layers.

    This function handles:
    1. Embedding mean subtraction (only needed for LayerNorm models like OPT,
       not strictly necessary for RMSNorm models like Qwen3 — see TODO below)
    2. Input layernorm -> Q, K, V projections
    3. Post-attention layernorm -> MLP up/gate projections
    4. Final layernorm -> LM head

    Memory-efficient: moves to CPU first, processes, then moves back.

    Args:
        model: The transformer model to process.
        verbose: Whether to show progress bar.
    """
    # ========== Step 1: Embedding fusion ==========
    # Subtract mean from embeddings
    if verbose:
        print("Fusing embeddings...")

    for W in [model.model.embed_tokens]:
        original_device = W.weight.device
        original_dtype = W.weight.dtype

        # Move to CPU first, then convert (avoids device OOM)
        W_ = W.weight.data.cpu().float()

        # Free device memory
        W.weight.data = torch.empty(0, device='cpu', dtype=original_dtype)
        cleanup_memory()

        # NOTE: This embedding mean subtraction is inherited from QuaRot/SliceGPT
        # but is NOT necessary for RMSNorm-based models (LLaMA, Qwen3, etc.).
        # See: https://github.com/spcl/QuaRot/issues/7
        #
        # Background:
        #   - LayerNorm (used in OPT) subtracts the mean as its first step,
        #     so fusing LayerNorm into weights requires the input to already be
        #     zero-mean. That is why QuaRot originally added this step.
        #   - RMSNorm (used in Qwen3) does NOT subtract the mean:
        #       RMSNorm(x) = x / sqrt(mean(x^2)) * gamma
        #     There is no mean-invariance property, so this subtraction is
        #     mathematically unnecessary for RMSNorm fusion.
        #
        # Why it doesn't hurt in practice:
        #   Embedding weights in these models already have near-zero row means
        #   (on the order of 1e-7 in float16), so the subtraction is almost a
        #   no-op. The precision impact is negligible.
        #
        # TODO: Remove this mean subtraction for RMSNorm models to be
        #       mathematically correct. This requires:
        #   1. Detect the norm type (RMSNorm vs LayerNorm) from the model config
        #      or by checking isinstance(layer.input_layernorm, LlamaRMSNorm).
        #   2. Only subtract embedding mean when the model uses LayerNorm
        #      (e.g., OPT), skip for RMSNorm models (e.g., LLaMA, Qwen3).
        #   3. Run a perplexity comparison (with vs without mean subtraction)
        #      on Qwen3-32B to confirm there is no regression.
        #   4. If supporting both norm types, add a `norm_type` parameter to
        #      fuse_layer_norms() or auto-detect from the model.
        W_new = W_ - W_.mean(dim=-1, keepdim=True)
        del W_

        # Move back
        W.weight.data = W_new.to(dtype=original_dtype, device=original_device)
        del W_new
        cleanup_memory()

    # ========== Step 2: Layer-by-layer fusion ==========
    layers = list(model.model.layers)

    layer_iter = tqdm(layers, desc="Fusing layer norms") if verbose else layers

    for layer in layer_iter:
        # Fuse post-attention layernorm into MLP layers
        fuse_ln_linear(
            layer.post_attention_layernorm,
            [layer.mlp.up_proj, layer.mlp.gate_proj],
        )

        # Fuse input layernorm into attention projections
        if hasattr(layer, "self_attn"):
            fuse_ln_linear(
                layer.input_layernorm,
                [
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                ],
            )
        elif hasattr(layer, "cross_attn"):
            # For models with cross attention (e.g., encoder-decoder)
            fuse_ln_linear(
                layer.input_layernorm,
                [layer.cross_attn.q_proj],
            )

        # Set layernorm weights to ones (effectively disabling them)
        layer.post_attention_layernorm.weight.data = torch.ones_like(
            layer.post_attention_layernorm.weight.data
        )
        layer.input_layernorm.weight.data = torch.ones_like(
            layer.input_layernorm.weight.data
        )

        # Clean up memory after each layer
        cleanup_memory()

    # ========== Step 3: Final layernorm fusion ==========
    if verbose:
        print("Fusing final layer norm...")

    fuse_ln_linear(
        model.model.norm,
        [model.lm_head],
    )
    model.model.norm.weight.data = torch.ones_like(model.model.norm.weight.data)

    cleanup_memory()

    if verbose:
        print("Layer norm fusion complete!")
