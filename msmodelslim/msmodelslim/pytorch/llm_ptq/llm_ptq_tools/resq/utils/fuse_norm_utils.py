# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Adapted from ResQ and QuaRot
"""
Fuse layer normalization into adjacent linear layers.

This is a key step in rotation-based quantization methods that
allows the normalization to be absorbed into the weight matrices.
"""

import typing
import torch


def fuse_ln_linear(
    layernorm: torch.nn.Module, 
    linear_layers: typing.Iterable[torch.nn.Linear]
) -> None:
    """
    Fuse the linear operations in LayerNorm into the adjacent linear blocks.
    
    This transforms: LayerNorm(x) @ W -> x @ (gamma * W)
    
    Args:
        layernorm: The layer normalization module.
        linear_layers: Iterable of linear layers to fuse into.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight: W_new = W * gamma
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        # Handle bias if present in layernorm
        if hasattr(layernorm, "bias") and layernorm.bias is not None:
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64)
                )
            linear.bias.data = linear.bias.data.double() + torch.matmul(
                W_, layernorm.bias.double()
            )
            linear.bias.data = linear.bias.data.to(linear_dtype)


def fuse_layer_norms(model):
    """
    Fuse all layer normalizations in the model into adjacent linear layers.
    
    This function handles:
    1. Embedding mean subtraction
    2. Input layernorm -> Q, K, V projections
    3. Post-attention layernorm -> MLP up/gate projections
    4. Final layernorm -> LM head
    
    Args:
        model: The transformer model to process.
    """
    # Embedding fusion: subtract mean from embeddings
    for W in [model.model.embed_tokens]:
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

    layers = [layer for layer in model.model.layers]

    # Fuse the linear operations in LayerNorm into the adjacent linear blocks
    for layer in layers:
        # Fuse post-attention layernorm into MLP layers
        fuse_ln_linear(
            layer.post_attention_layernorm, 
            [layer.mlp.up_proj, layer.mlp.gate_proj]
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
        W_norm = layer.post_attention_layernorm.weight.data
        layer.post_attention_layernorm.weight.data = torch.ones_like(W_norm)
        W_norm = layer.input_layernorm.weight.data
        layer.input_layernorm.weight.data = torch.ones_like(W_norm)

    # Fuse final layernorm into lm_head
    fuse_ln_linear(
        model.model.norm,
        [model.lm_head],
    )
    W_norm = model.model.norm.weight.data
    model.model.norm.weight.data = torch.ones_like(W_norm)
