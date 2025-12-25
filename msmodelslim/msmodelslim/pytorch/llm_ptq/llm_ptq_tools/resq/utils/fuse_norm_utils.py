# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Adapted from ResQ and QuaRot
"""
Fuse layer normalization into adjacent linear layers.

This is a key step in rotation-based quantization methods that
allows the normalization to be absorbed into the weight matrices.
"""

import gc
import typing
import torch


def fuse_ln_linear(
    layernorm: torch.nn.Module,
    linear_layers: typing.Iterable[torch.nn.Linear],
    use_cpu: bool = True,
) -> None:
    """
    Fuse the linear operations in LayerNorm into the adjacent linear blocks.

    This transforms: LayerNorm(x) @ W -> x @ (gamma * W)

    Args:
        layernorm: The layer normalization module.
        linear_layers: Iterable of linear layers to fuse into.
        use_cpu: Whether to move tensors to CPU for computation (saves device memory).
    """
    # Get layernorm weight on CPU to save memory
    ln_weight = layernorm.weight.data
    if use_cpu:
        ln_weight = ln_weight.float().cpu()
    else:
        ln_weight = ln_weight.float()

    ln_bias = None
    if hasattr(layernorm, "bias") and layernorm.bias is not None:
        if use_cpu:
            ln_bias = layernorm.bias.data.float().cpu()
        else:
            ln_bias = layernorm.bias.data.float()

    for linear in linear_layers:
        original_device = linear.weight.device
        linear_dtype = linear.weight.dtype

        # Move weight to CPU and convert to float32 for computation
        if use_cpu:
            W_ = linear.weight.data.float().cpu()
        else:
            W_ = linear.weight.data.float()

        # Free original weight memory
        linear.weight.data = torch.empty(0, device='cpu')
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            import torch_npu
            if torch.npu.is_available():
                torch.npu.empty_cache()
        except ImportError:
            pass

        # Calculating new weight: W_new = W * gamma
        W_new = W_ * ln_weight
        del W_

        # Move back to original device and dtype
        linear.weight.data = W_new.to(dtype=linear_dtype, device=original_device)
        del W_new
        gc.collect()

        # Handle bias if present in layernorm
        if ln_bias is not None:
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=linear_dtype, device=original_device)
                )

            if use_cpu:
                bias_ = linear.bias.data.float().cpu()
                W_for_bias = linear.weight.data.float().cpu()
            else:
                bias_ = linear.bias.data.float()
                W_for_bias = linear.weight.data.float()

            new_bias = bias_ + torch.matmul(W_for_bias, ln_bias)
            linear.bias.data = new_bias.to(dtype=linear_dtype, device=original_device)
            del bias_, W_for_bias, new_bias


def fuse_layer_norms(model, use_cpu: bool = True):
    """
    Fuse all layer normalizations in the model into adjacent linear layers.

    This function handles:
    1. Embedding mean subtraction
    2. Input layernorm -> Q, K, V projections
    3. Post-attention layernorm -> MLP up/gate projections
    4. Final layernorm -> LM head

    Args:
        model: The transformer model to process.
        use_cpu: Whether to move tensors to CPU for computation (saves device memory).
                 Set to True for large models to avoid OOM.
    """
    # Embedding fusion: subtract mean from embeddings
    for W in [model.model.embed_tokens]:
        original_device = W.weight.device
        original_dtype = W.weight.dtype

        if use_cpu:
            W_ = W.weight.data.float().cpu()
        else:
            W_ = W.weight.data.float()

        # Free original memory
        W.weight.data = torch.empty(0, device='cpu')
        gc.collect()
        try:
            import torch_npu
            if torch.npu.is_available():
                torch.npu.empty_cache()
        except ImportError:
            pass

        # Subtract mean
        W_new = W_ - W_.mean(dim=-1, keepdim=True)
        del W_

        # Move back
        W.weight.data = W_new.to(dtype=original_dtype, device=original_device)
        del W_new
        gc.collect()

    layers = list(model.model.layers)

    # Fuse the linear operations in LayerNorm into the adjacent linear blocks
    for idx, layer in enumerate(layers):
        # Fuse post-attention layernorm into MLP layers
        fuse_ln_linear(
            layer.post_attention_layernorm,
            [layer.mlp.up_proj, layer.mlp.gate_proj],
            use_cpu=use_cpu,
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
                use_cpu=use_cpu,
            )
        elif hasattr(layer, "cross_attn"):
            # For models with cross attention (e.g., encoder-decoder)
            fuse_ln_linear(
                layer.input_layernorm,
                [layer.cross_attn.q_proj],
                use_cpu=use_cpu,
            )

        # Set layernorm weights to ones (effectively disabling them)
        W_norm = layer.post_attention_layernorm.weight.data
        layer.post_attention_layernorm.weight.data = torch.ones_like(W_norm)
        W_norm = layer.input_layernorm.weight.data
        layer.input_layernorm.weight.data = torch.ones_like(W_norm)

        # Clean up memory after each layer
        gc.collect()
        try:
            import torch_npu
            if torch.npu.is_available():
                torch.npu.empty_cache()
        except ImportError:
            pass

    # Fuse final layernorm into lm_head
    fuse_ln_linear(
        model.model.norm,
        [model.lm_head],
        use_cpu=use_cpu,
    )
    W_norm = model.model.norm.weight.data
    model.model.norm.weight.data = torch.ones_like(W_norm)
