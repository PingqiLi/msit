# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
Model structure utilities for ResQ.

This module provides utilities for analyzing and manipulating
transformer model structures.
"""

from typing import List, Optional, Dict, Any, Union
import torch
import torch.nn as nn


def get_module_by_name(
    model: nn.Module,
    name: str,
) -> nn.Module:
    """
    Get a module by its full name path.

    Args:
        model: The root module
        name: Dot-separated path to the module (e.g., 'model.layers.0')

    Returns:
        The requested module
    """
    if not name:
        return model

    parts = name.split('.')
    module = model

    for part in parts:
        if hasattr(module, part):
            module = getattr(module, part)
        elif hasattr(module, '__getitem__'):
            module = module[int(part)]
        else:
            raise AttributeError(f"Module has no attribute '{part}'")

    return module


def set_module_by_name(
    model: nn.Module,
    name: str,
    new_module: nn.Module,
) -> None:
    """
    Set a module by its full name path.

    Args:
        model: The root module
        name: Dot-separated path to the module
        new_module: The new module to set
    """
    parts = name.split('.')
    parent = model

    for part in parts[:-1]:
        if hasattr(parent, part):
            parent = getattr(parent, part)
        elif hasattr(parent, '__getitem__'):
            parent = parent[int(part)]
        else:
            raise AttributeError(f"Module has no attribute '{part}'")

    setattr(parent, parts[-1], new_module)


def get_model_layers(model: nn.Module) -> List[nn.Module]:
    """
    Get the list of decoder layers from a transformer model.

    Args:
        model: The transformer model

    Returns:
        List of decoder layer modules
    """
    # Try common layer locations
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return list(model.model.layers)
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return list(model.transformer.h)
    elif hasattr(model, 'layers'):
        return list(model.layers)
    else:
        raise ValueError("Could not find decoder layers in model")


class ModelStructure:
    """
    Analyzes and stores transformer model structure information.

    This class identifies the locations of key components like
    attention layers, MLP layers, and normalization layers.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize with a transformer model.

        Args:
            model: The transformer model to analyze
        """
        self.model = model
        self.config = model.config
        self._analyze()

    def _analyze(self):
        """Analyze the model structure."""
        # Get basic dimensions
        self.hidden_size = self.config.hidden_size
        self.num_attention_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_heads = getattr(
            self.config, 'num_key_value_heads', self.num_attention_heads
        )
        self.intermediate_size = getattr(
            self.config, 'intermediate_size', 4 * self.hidden_size
        )
        self.num_hidden_layers = self.config.num_hidden_layers

        # Find layers
        self.layers = get_model_layers(self.model)
        self.num_layers = len(self.layers)

        # Analyze layer structure
        self._analyze_layer_structure()

    def _analyze_layer_structure(self):
        """Analyze the structure of decoder layers."""
        if not self.layers:
            return

        layer = self.layers[0]

        # Find attention components
        self.attn_name = None
        self.q_proj_name = None
        self.k_proj_name = None
        self.v_proj_name = None
        self.o_proj_name = None

        for name, module in layer.named_modules():
            if 'self_attn' in name or 'attention' in name:
                if self.attn_name is None:
                    self.attn_name = name.split('.')[0]

            module_name = name.split('.')[-1]
            if module_name in ['q_proj', 'query']:
                self.q_proj_name = name
            elif module_name in ['k_proj', 'key']:
                self.k_proj_name = name
            elif module_name in ['v_proj', 'value']:
                self.v_proj_name = name
            elif module_name in ['o_proj', 'dense', 'out_proj']:
                if 'attn' in name or 'attention' in name:
                    self.o_proj_name = name

        # Find MLP components
        self.mlp_name = None
        self.up_proj_name = None
        self.down_proj_name = None
        self.gate_proj_name = None

        for name, module in layer.named_modules():
            if 'mlp' in name or 'feed_forward' in name:
                if self.mlp_name is None:
                    self.mlp_name = name.split('.')[0]

            module_name = name.split('.')[-1]
            if module_name in ['up_proj', 'fc1', 'w1']:
                self.up_proj_name = name
            elif module_name in ['down_proj', 'fc2', 'w2']:
                self.down_proj_name = name
            elif module_name in ['gate_proj', 'w3']:
                self.gate_proj_name = name

        # Find normalization layers
        self.input_layernorm_name = None
        self.post_attention_layernorm_name = None

        for name, module in layer.named_modules():
            if isinstance(module, (nn.LayerNorm, nn.RMSNorm)) or 'norm' in name.lower():
                if 'input' in name or name == 'ln_1':
                    self.input_layernorm_name = name
                elif 'post' in name or 'ln_2' in name:
                    self.post_attention_layernorm_name = name

    def get_layer(self, idx: int) -> nn.Module:
        """Get a decoder layer by index."""
        return self.layers[idx]

    def get_layer_name(self, idx: int) -> str:
        """Get the full name of a decoder layer."""
        # Determine the layer path
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return f'model.layers.{idx}'
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return f'transformer.h.{idx}'
        else:
            return f'layers.{idx}'

    def get_attention_modules(self, layer: nn.Module) -> Dict[str, nn.Module]:
        """Get attention-related modules from a layer."""
        modules = {}

        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
            if hasattr(attn, 'q_proj'):
                modules['q_proj'] = attn.q_proj
            if hasattr(attn, 'k_proj'):
                modules['k_proj'] = attn.k_proj
            if hasattr(attn, 'v_proj'):
                modules['v_proj'] = attn.v_proj
            if hasattr(attn, 'o_proj'):
                modules['o_proj'] = attn.o_proj

        return modules

    def get_mlp_modules(self, layer: nn.Module) -> Dict[str, nn.Module]:
        """Get MLP-related modules from a layer."""
        modules = {}

        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            if hasattr(mlp, 'up_proj'):
                modules['up_proj'] = mlp.up_proj
            if hasattr(mlp, 'down_proj'):
                modules['down_proj'] = mlp.down_proj
            if hasattr(mlp, 'gate_proj'):
                modules['gate_proj'] = mlp.gate_proj

        return modules

    def get_norm_modules(self, layer: nn.Module) -> Dict[str, nn.Module]:
        """Get normalization modules from a layer."""
        modules = {}

        if hasattr(layer, 'input_layernorm'):
            modules['input_layernorm'] = layer.input_layernorm
        if hasattr(layer, 'post_attention_layernorm'):
            modules['post_attention_layernorm'] = layer.post_attention_layernorm

        return modules

    def get_all_linear_names(self) -> List[str]:
        """Get names of all linear layers in the model."""
        linear_names = []

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                linear_names.append(name)

        return linear_names

    def get_quantizable_linear_names(self) -> List[str]:
        """Get names of linear layers that should be quantized."""
        linear_names = []

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Skip LM head by default
                if 'lm_head' in name:
                    continue
                # Skip embedding layers
                if 'embed' in name:
                    continue
                linear_names.append(name)

        return linear_names

    def __repr__(self) -> str:
        return (
            f"ModelStructure(\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  num_attention_heads={self.num_attention_heads},\n"
            f"  head_dim={self.head_dim},\n"
            f"  num_key_value_heads={self.num_key_value_heads},\n"
            f"  num_hidden_layers={self.num_hidden_layers},\n"
            f"  intermediate_size={self.intermediate_size}\n"
            f")"
        )
