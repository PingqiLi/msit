# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
ResQ Calibrator for mixed-precision 4/8-bit quantization.

This module provides a calibrator that integrates ResQ with the
msmodelslim framework, following the pattern of Calibrator.
"""

import os
import gc
import functools
from collections import defaultdict
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
from tqdm import tqdm

from msmodelslim import logger as msmodelslim_logger
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save import SaverFactory

from .config import ResQConfig
from .quant_modules import LinearResQQuantizer, add_resq_quantizers
from .processors.basis_processor import compute_basis, generate_random_rotations, load_basis
from .processors.resq_processor import apply_rotations, rearrange_columns
from .utils.fuse_norm_utils import fuse_layer_norms
from .utils.common import cleanup_memory, get_device
from .gptq import GPTQ, create_gptq_quantizers, GPTQWeightQuantizer


class ResQCalibrator:
    """
    Calibrator for ResQ mixed-precision quantization.

    This calibrator:
    1. Applies eigenvalue-based rotations to model weights
    2. Rearranges columns for mixed-precision layout
    3. Replaces linear layers with ResQ quantizers
    4. Calibrates quantization parameters
    5. Saves dual weights and dual scales
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: ResQConfig,
        calib_data: List = None,
        disable_names: List[str] = None,
        basis_path: str = None,
        rotation_path: str = None,
    ):
        """
        Initialize the ResQ calibrator.

        Args:
            model: The transformer model
            cfg: ResQ configuration
            calib_data: Calibration data
            disable_names: Layer names to skip quantization
            basis_path: Path to pre-computed basis matrices
            rotation_path: Path to pre-computed rotation matrices
        """
        self.cfg = cfg
        self.logger = msmodelslim_logger
        self.calib_data = calib_data or []
        self.disable_names = disable_names or []
        self.device = get_device(cfg.dev_type, cfg.dev_id)

        # Load or compute basis and rotations
        self.basis_dict = None
        self.rotation_dict = None

        if basis_path:
            self.logger.info(f"Loading basis from {basis_path}")
            self.basis_dict = load_basis(basis_path)

        if rotation_path:
            self.logger.info(f"Loading rotations from {rotation_path}")
            self.rotation_dict = torch.load(rotation_path, map_location='cpu')

        # Apply transformations and quantization
        self.model = self._prepare_model(model)
        self.logger.info("ResQ Calibrator initialized successfully!")

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for ResQ quantization."""
        # Save original device map before any transformations
        self._original_device_map = None
        self._is_multi_device = False
        if hasattr(model, 'hf_device_map') and model.hf_device_map is not None:
            self._original_device_map = dict(model.hf_device_map)
            devices = set(str(d) for d in self._original_device_map.values())
            if len(devices) > 1:
                self._is_multi_device = True
                self.logger.info(f"Model was distributed across multiple devices: {devices}")
                self.logger.info(f"Original device_map: {self._original_device_map}")

        # Fuse layer norms
        self.logger.info("Fusing layer norms...")
        fuse_layer_norms(model)
        cleanup_memory(verbos=False)

        # Check if basis is available for full ResQ mode
        if self.basis_dict is None:
            self.logger.warning(
                "=" * 60 + "\n"
                "WARNING: No basis_path provided!\n"
                "ResQ will run in SIMPLIFIED MODE without eigenvalue-based rotation.\n"
                "This may result in suboptimal quantization quality.\n"
                "For best results, pre-compute basis using compute_basis() and provide basis_path.\n"
                "=" * 60
            )
            # In simplified mode, skip rotation and rearrangement
            # Just do uniform mixed-precision quantization
            self._use_simplified_mode = True
        else:
            self._use_simplified_mode = False

            # Generate rotations if not provided
            if self.rotation_dict is None:
                self.logger.info("Generating random rotations...")
                # Get correct head_dim from config or v_proj
                num_kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
                head_dim = getattr(model.config, 'head_dim', None)
                if head_dim is None:
                    if hasattr(model, 'model') and hasattr(model.model, 'layers') and len(model.model.layers) > 0:
                        v_proj = model.model.layers[0].self_attn.v_proj
                        head_dim = v_proj.out_features // num_kv_heads
                    else:
                        head_dim = model.config.hidden_size // model.config.num_attention_heads
                # Get intermediate_size for H/Rd generation (full dimension)
                # H and Rd are applied to the full intermediate_size, not blocksize
                intermediate_size = model.config.intermediate_size
                ud_rotation_type = self.cfg.ud_rotation_type
                self.logger.info(f"Generating rotations with ud_rotation_type={ud_rotation_type}, intermediate_size={intermediate_size}")
                self.rotation_dict = generate_random_rotations(
                    hidden_dim=model.config.hidden_size,
                    head_dim=head_dim,
                    intermediate_dim=intermediate_size,  # Use full intermediate_size for H/Rd
                    high_fraction=self.cfg.high_fraction,
                    seed=self.cfg.seed,
                    ud_rotation_type=ud_rotation_type,
                )

            # Apply rotations with basis
            self.logger.info("Applying rotations to model...")
            apply_rotations(model, self.basis_dict, self.rotation_dict, self.cfg)
            cleanup_memory(verbos=False)

            # Rearrange columns for mixed-precision layout (only when basis is available)
            self.logger.info("Rearranging columns for mixed precision...")
            rearrange_columns(model, self.cfg, training=False)
            cleanup_memory(verbos=False)

        # Replace linear layers with ResQ quantizers
        self.logger.info("Adding ResQ quantizers...")
        # Auto-detect lm_head as skip layer
        skip_names = list(self.disable_names)
        for name, _ in model.named_modules():
            if 'lm_head' in name and name not in skip_names:
                skip_names.append(name)

        model = add_resq_quantizers(
            model,
            cfg=self.cfg,
            logger=self.logger,
            high_bits=self.cfg.high_bits,
            low_bits=self.cfg.low_bits,
            high_fraction=self.cfg.high_fraction,
            skip_names=skip_names,
        )

        if self._is_multi_device:
            # For multi-device models, redistribute using accelerate
            self.logger.info("Redistributing model across multiple devices...")
            try:
                from accelerate import dispatch_model, infer_auto_device_map
                from accelerate.utils import get_balanced_memory

                # Remove any stale hooks first
                try:
                    from accelerate.hooks import remove_hook_from_module
                    for name, module in model.named_modules():
                        remove_hook_from_module(module, recurse=False)
                except Exception:
                    pass

                # Use accelerate to redistribute the model
                # First, get available devices from original device map
                available_devices = list(set(self._original_device_map.values()))
                self.logger.info(f"Available devices: {available_devices}")

                # Infer a new device map for the transformed model
                no_split_classes = ["Qwen2DecoderLayer", "Qwen3DecoderLayer", "LlamaDecoderLayer"]
                max_memory = get_balanced_memory(
                    model,
                    max_memory=None,
                    no_split_module_classes=no_split_classes,
                    dtype=model.dtype if hasattr(model, 'dtype') else torch.float16,
                )
                device_map = infer_auto_device_map(
                    model,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_classes,
                    dtype=model.dtype if hasattr(model, 'dtype') else torch.float16,
                )
                self.logger.info(f"New device_map: {device_map}")

                # Dispatch the model
                model = dispatch_model(model, device_map=device_map)
                self.logger.info("Model redistributed successfully")

                # Set input device
                if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                    self._input_device = model.model.embed_tokens.weight.device
                else:
                    # Get first device from device_map
                    first_device = list(device_map.values())[0]
                    self._input_device = torch.device(first_device) if isinstance(first_device, str) else first_device
                self.logger.info(f"Calibration data will be sent to: {self._input_device}")

            except Exception as e:
                self.logger.warning(f"Failed to redistribute model with accelerate: {e}")
                self.logger.warning("Falling back to single device mode")
                self._is_multi_device = False
                # Fall through to single-device handling

        if not self._is_multi_device:
            # For single-device models, remove accelerate hooks and move to target device
            self.logger.info(f"Moving model to device: {self.device}")

            # Remove accelerate hooks if present
            try:
                from accelerate.hooks import remove_hook_from_module
                for name, module in model.named_modules():
                    remove_hook_from_module(module, recurse=False)
                self.logger.info("Removed accelerate hooks from model")
            except Exception as e:
                self.logger.info(f"No accelerate hooks to remove or error: {e}")

            # Clear the device map
            if hasattr(model, 'hf_device_map'):
                model.hf_device_map = None
                self.logger.info("Cleared hf_device_map")

            # Move entire model to target device
            model = model.to(self.device)
            self._input_device = self.device

        return model

    @torch.no_grad()
    def run(self) -> None:
        """Run calibration on the model."""
        self.logger.info("Starting ResQ calibration...")
        self.model.eval()

        # Debug: Print model state before calibration
        self.logger.info("=" * 60)
        self.logger.info("[DEBUG] Model state before calibration:")
        self.logger.info(f"[DEBUG] self.device = {self.device}")
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            self.logger.info(f"[DEBUG] embed_tokens.weight.device = {self.model.model.embed_tokens.weight.device}")
            self.logger.info(f"[DEBUG] embed_tokens.weight.dtype = {self.model.model.embed_tokens.weight.dtype}")
        if hasattr(self.model, 'hf_device_map'):
            self.logger.info(f"[DEBUG] hf_device_map = {self.model.hf_device_map}")

        # Check first few layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            for i, layer in enumerate(self.model.model.layers[:2]):
                if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'q_proj'):
                    self.logger.info(f"[DEBUG] layer {i} q_proj.weight.device = {layer.self_attn.q_proj.weight.device}")
                break

        # Check calib_data
        if self.calib_data and len(self.calib_data) > 0:
            first_batch = self.calib_data[0]
            if isinstance(first_batch, (tuple, list)):
                self.logger.info(f"[DEBUG] calib_data[0] devices = {[t.device if hasattr(t, 'device') else type(t) for t in first_batch]}")
            elif isinstance(first_batch, dict):
                self.logger.info(f"[DEBUG] calib_data[0] devices = {[(k, v.device if hasattr(v, 'device') else type(v)) for k, v in first_batch.items()]}")
        self.logger.info("=" * 60)

        if not self.calib_data:
            self.logger.info("No calibration data provided, running data-free mode")
            self._run_datafree_mode()
        elif not self.cfg.w_rtn:
            # GPTQ mode: column-by-column quantization with Hessian
            self.logger.info("Running GPTQ mode (w_rtn=False)")
            self._run_gptq_mode()
        else:
            # RTN mode: simple round-to-nearest quantization
            self.logger.info("Running RTN mode (w_rtn=True)")
            self._run_calib_mode()

        # Disable calibration mode
        self._disable_calibration()
        self.logger.info("ResQ calibration complete!")

    def _run_calib_mode(self) -> None:
        """Run calibration with data."""
        self.logger.info(f"Running calibration with {len(self.calib_data)} samples...")

        # Use _input_device which is set based on model configuration
        # For multi-device models, this is the device of embed_tokens
        # For single-device models, this is self.device
        embed_device = self._input_device
        self.logger.info(f"Moving calibration data to device: {embed_device}")

        # Check sequence length and warn if too long for available memory
        if self.calib_data and len(self.calib_data) > 0:
            first_batch = self.calib_data[0]
            if isinstance(first_batch, (tuple, list)) and len(first_batch) > 0:
                seq_len = first_batch[0].shape[-1] if hasattr(first_batch[0], 'shape') else 0
                if seq_len > 1024:
                    self.logger.warning(
                        f"Calibration sequence length is {seq_len}. "
                        f"For large models, consider using --seq_len 512 or 1024 to reduce memory usage."
                    )

        for idx, data in enumerate(tqdm(self.calib_data, desc="Calibrating")):
            if isinstance(data, (tuple, list)):
                # Move tensors to the correct device
                data = tuple(t.to(embed_device) if isinstance(t, torch.Tensor) else t for t in data)
                # Debug first iteration
                if idx == 0:
                    self.logger.info(f"[DEBUG] First batch after .to({embed_device}):")
                    self.logger.info(f"[DEBUG] input tensors devices: {[t.device if isinstance(t, torch.Tensor) else type(t) for t in data]}")
                    self.logger.info(f"[DEBUG] input tensors shapes: {[t.shape if isinstance(t, torch.Tensor) else None for t in data]}")
                    # Check model's embed_tokens device right before forward
                    if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                        self.logger.info(f"[DEBUG] RIGHT BEFORE FORWARD: embed_tokens.weight.device = {self.model.model.embed_tokens.weight.device}")
                self.model(*data)
            elif isinstance(data, dict):
                # Move tensors to the correct device
                data = {k: v.to(embed_device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
                if idx == 0:
                    self.logger.info(f"[DEBUG] First batch after .to({embed_device}):")
                    self.logger.info(f"[DEBUG] input tensors: {[(k, v.device, v.shape) if isinstance(v, torch.Tensor) else (k, type(v)) for k, v in data.items()]}")
                self.model(**data)

            # Clear cache periodically to reduce memory fragmentation
            if (idx + 1) % 8 == 0:
                cleanup_memory(verbos=False)

    def _run_datafree_mode(self) -> None:
        """Run data-free quantization."""
        self.logger.info("Running data-free weight quantization...")

        for name, module in self.model.named_modules():
            if isinstance(module, LinearResQQuantizer):
                # Quantize weights immediately
                module.quant_weight.quantize_weight(module.weight)
                self.logger.info(f"Quantized layer: {name}")

    def _run_gptq_mode(self) -> None:
        """
        Run GPTQ-based weight quantization.

        GPTQ performs column-by-column quantization with error compensation
        using the Hessian matrix computed from calibration data.
        """
        import math

        self.logger.info("=" * 60)
        self.logger.info("Starting GPTQ quantization...")
        self.logger.info(f"  nsamples: {self.cfg.nsamples}")
        self.logger.info(f"  percdamp: {self.cfg.percdamp}")
        self.logger.info(f"  blocksize: {self.cfg.gptq_blocksize}")
        self.logger.info(f"  act_order: {self.cfg.act_order}")
        self.logger.info("=" * 60)

        # Get model layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = list(self.model.model.layers)
        else:
            raise ValueError("Unsupported model architecture for GPTQ")

        nlayers = len(layers)
        model_dim = self.model.config.hidden_size

        # Mixed precision settings
        high_fraction = self.cfg.high_fraction
        high_bits = self.cfg.high_bits
        low_bits = self.cfg.low_bits

        # Move model to CPU first
        self.model.cpu()
        cleanup_memory(verbos=False)

        # ========== Step 1: Capture first layer inputs ==========
        self.logger.info("Capturing first layer inputs...")

        embed_device = self._input_device
        if hasattr(self.model, 'model'):
            self.model.model.embed_tokens = self.model.model.embed_tokens.to(embed_device)
            if hasattr(self.model.model, 'rotary_emb'):
                self.model.model.rotary_emb = self.model.model.rotary_emb.to(embed_device)
            if hasattr(self.model.model, 'norm'):
                self.model.model.norm = self.model.model.norm.to(embed_device)

        layers[0] = layers[0].to(embed_device)

        # Capture inputs using a catcher
        dtype = next(iter(self.model.parameters())).dtype
        nsamples = min(len(self.calib_data), self.cfg.nsamples)

        # Determine sequence length from first batch
        first_batch = self.calib_data[0]
        if isinstance(first_batch, (tuple, list)):
            seq_len = first_batch[0].shape[-1] if hasattr(first_batch[0], 'shape') else 2048
        elif isinstance(first_batch, dict):
            input_ids = first_batch.get('input_ids', first_batch.get('inputs'))
            seq_len = input_ids.shape[-1] if input_ids is not None else 2048
        else:
            seq_len = 2048

        inps = torch.zeros((nsamples, seq_len, model_dim), dtype=dtype, device=embed_device)
        cache = {"i": 0, "attention_mask": None, "position_ids": None, "position_embeddings": None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache["i"]] = inp
                cache["i"] += 1
                cache["attention_mask"] = kwargs.get("attention_mask")
                cache["position_ids"] = kwargs.get("position_ids")
                cache["position_embeddings"] = kwargs.get("position_embeddings")
                raise ValueError("Catcher stop")

        # Replace first layer with catcher
        self.model.model.layers[0] = Catcher(layers[0])

        # Run calibration data to capture inputs
        for idx, data in enumerate(self.calib_data[:nsamples]):
            if cache["i"] >= nsamples:
                break
            try:
                if isinstance(data, (tuple, list)):
                    data = tuple(t.to(embed_device) if isinstance(t, torch.Tensor) else t for t in data)
                    self.model(*data)
                elif isinstance(data, dict):
                    data = {k: v.to(embed_device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
                    self.model(**data)
            except ValueError:
                pass  # Expected - catcher raises ValueError

        # Restore first layer
        self.model.model.layers[0] = layers[0]

        self.logger.info(f"Captured {cache['i']} samples")

        # Move embeddings back to CPU
        if hasattr(self.model, 'model'):
            self.model.model.embed_tokens = self.model.model.embed_tokens.cpu()
            if hasattr(self.model.model, 'rotary_emb'):
                self.model.model.rotary_emb = self.model.model.rotary_emb.cpu()

        layers[0] = layers[0].cpu()
        cleanup_memory(verbos=False)

        # ========== Step 2: Layer-by-layer GPTQ quantization ==========
        outs = torch.zeros_like(inps)
        attention_mask = cache["attention_mask"]
        position_ids = cache["position_ids"]
        position_embeddings = cache["position_embeddings"]

        # Define sequential groups of projections to quantize together
        # Following original ResQ pattern
        sequential = [
            ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
            ["self_attn.o_proj"],
            ["mlp.up_proj", "mlp.gate_proj"],
            ["mlp.down_proj"],
        ]

        for layer_idx in tqdm(range(nlayers), desc="GPTQ quantizing layers"):
            self.logger.info(f"\nLayer {layer_idx}:")
            layer = layers[layer_idx].to(embed_device)

            # Find all LinearResQQuantizer modules in this layer
            full = {}
            for name, mod in layer.named_modules():
                if isinstance(mod, LinearResQQuantizer):
                    full[name] = mod

            # Process each group of projections
            for names in sequential:
                subset = {n: full[n] for n in names if n in full}
                if not subset:
                    continue

                gptq = {}
                for name in subset:
                    self.logger.info(f"  Setting up GPTQ for {name}...", )

                    mod = subset[name]
                    # Get the underlying weight (LinearResQQuantizer stores weight)
                    weight = mod.weight

                    # Determine if mixed precision applies
                    mixed_precision = False
                    high_bits_length = 0

                    # Check if this is a down_proj with int8_down_proj
                    if self.cfg.int8_down_proj and "down_proj" in name:
                        # down_proj uses int8 only (no mixed precision)
                        mixed_precision = False
                        layer_bits = 8
                    elif "k_proj" in name or "q_proj" in name or "v_proj" in name or \
                         "up_proj" in name or "gate_proj" in name or "o_proj" in name:
                        # These projections use mixed precision
                        mixed_precision = True
                        high_bits_length = int(high_fraction * weight.shape[1])
                        layer_bits = low_bits
                    else:
                        layer_bits = low_bits

                    # Create a wrapper module for GPTQ (it expects nn.Linear interface)
                    class LinearWrapper(nn.Module):
                        def __init__(self, weight, bias):
                            super().__init__()
                            self.weight = nn.Parameter(weight.clone())
                            self.bias = nn.Parameter(bias.clone()) if bias is not None else None
                            self.in_features = weight.shape[1]
                            self.out_features = weight.shape[0]

                    wrapper = LinearWrapper(weight.data, mod.bias.data if mod.bias is not None else None)
                    wrapper = wrapper.to(embed_device)

                    gptq[name] = GPTQ(
                        wrapper,
                        mixed_precision=mixed_precision,
                        high_bits_length=high_bits_length,
                    )

                    # Configure quantizers
                    gptq[name].quantizer = GPTQWeightQuantizer(bits=layer_bits, perchannel=True, sym=self.cfg.w_sym)

                    if mixed_precision:
                        gptq[name].high_quantizer = GPTQWeightQuantizer(bits=high_bits, perchannel=True, sym=self.cfg.w_sym)

                # Register hooks to accumulate Hessian
                def make_add_batch(name):
                    def add_batch(module, inp, out):
                        gptq[name].add_batch(inp[0].data, out.data)
                    return add_batch

                handles = []
                for name in subset:
                    handles.append(subset[name].register_forward_hook(make_add_batch(name)))

                # Run forward passes to accumulate Hessian
                for j in range(nsamples):
                    kwargs = {}
                    if attention_mask is not None:
                        kwargs['attention_mask'] = attention_mask
                    if position_ids is not None:
                        kwargs['position_ids'] = position_ids
                    if position_embeddings is not None:
                        kwargs['position_embeddings'] = position_embeddings

                    outs[j] = layer(inps[j].unsqueeze(0), **kwargs)[0]

                # Remove hooks
                for h in handles:
                    h.remove()

                # Run GPTQ quantization
                for name in subset:
                    self.logger.info(f"  Running GPTQ fasterquant for {name}...")
                    gptq[name].fasterquant(
                        blocksize=self.cfg.gptq_blocksize,
                        percdamp=self.cfg.percdamp,
                        actorder=self.cfg.act_order,
                    )

                    # Copy quantized weights back to LinearResQQuantizer
                    mod = subset[name]
                    mod.weight.data = gptq[name].layer.weight.data.clone()

                    # Trigger quantization to store int8 weights
                    mod.quant_weight.quantize_weight(mod.weight)

                    gptq[name].free()

            # Run final forward through layer to get outputs for next layer
            for j in range(nsamples):
                kwargs = {}
                if attention_mask is not None:
                    kwargs['attention_mask'] = attention_mask
                if position_ids is not None:
                    kwargs['position_ids'] = position_ids
                if position_embeddings is not None:
                    kwargs['position_embeddings'] = position_embeddings

                outs[j] = layer(inps[j].unsqueeze(0), **kwargs)[0]

            # Move layer back to CPU
            layers[layer_idx] = layer.cpu()
            cleanup_memory(verbos=False)

            # Swap inputs and outputs for next layer
            inps, outs = outs, torch.zeros_like(inps)

        self.logger.info("=" * 60)
        self.logger.info("GPTQ quantization complete!")
        self.logger.info("=" * 60)

    def _disable_calibration(self) -> None:
        """Disable calibration mode on all quantizers."""
        for module in self.model.modules():
            if isinstance(module, LinearResQQuantizer):
                module.disable_calib()

    @torch.no_grad()
    def save(
        self,
        output_path: str,
        safetensors_name: str = None,
        json_name: str = None,
        save_type: List[str] = None,
        part_file_size: int = None,
    ) -> None:
        """
        Save quantized model with dual weights and dual scales.

        Args:
            output_path: Output directory
            safetensors_name: Name of safetensors file
            json_name: Name of JSON description file
            save_type: List of save types
            part_file_size: Size limit for part files (GB)
        """
        os.makedirs(output_path, exist_ok=True)

        if safetensors_name is None:
            safetensors_name = "quant_model_weight_resq.safetensors"
        if json_name is None:
            json_name = "quant_model_description_resq.json"
        if save_type is None:
            save_type = ["safe_tensor"]

        self.logger.info(f"Saving ResQ quantized model to {output_path}")

        # Collect all weights and parameters
        weight_dict = {}
        quant_description = {
            "model_quant_type": "W4A8_ResQ",
            "high_bits": self.cfg.high_bits,
            "low_bits": self.cfg.low_bits,
            "high_fraction": self.cfg.high_fraction,
            "group_size": getattr(self.cfg, 'w_groupsize', -1),
            "version": "1.0",
        }

        # Save model.embed_tokens.weight (already rotated with Ua)
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            embed_weight = self.model.model.embed_tokens.weight.data.cpu()
            weight_dict['model.embed_tokens.weight'] = embed_weight
            quant_description['model.embed_tokens.weight'] = "FLOAT"
            self.logger.info(f"Saved model.embed_tokens.weight: {embed_weight.shape}")

        # Save model.norm.weight (final layer norm)
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'norm'):
            norm_module = self.model.model.norm
            if hasattr(norm_module, 'weight') and norm_module.weight is not None:
                weight_dict['model.norm.weight'] = norm_module.weight.data.cpu()
                quant_description['model.norm.weight'] = "FLOAT"
                self.logger.info(f"Saved model.norm.weight: {norm_module.weight.shape}")

        # Debug: Print shapes of attention and MLP weights
        self.logger.info("=" * 60)
        self.logger.info("[DEBUG] Checking weight shapes (layer 0):")
        for name, module in self.model.named_modules():
            if isinstance(module, LinearResQQuantizer):
                # Print all projection layers in layer 0
                if 'layers.0' in name and any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']):
                    quant_weights = module.get_quant_weights()
                    self.logger.info(f"[DEBUG] {name} (split_dim={module.split_dim}):")
                    self.logger.info(f"  original weight shape: [{module.out_features}, {module.in_features}]")
                    if 'weight_low' in quant_weights:
                        self.logger.info(f"  weight_low shape: {quant_weights['weight_low'].shape}")
                        self.logger.info(f"  scale_low shape: {quant_weights['scale_low'].shape}")
                    if 'weight_high' in quant_weights:
                        self.logger.info(f"  weight_high shape: {quant_weights['weight_high'].shape}")
                        self.logger.info(f"  scale_high shape: {quant_weights['scale_high'].shape}")
        self.logger.info("=" * 60)

        # Save per-layer normalization weights
        for name, module in self.model.named_modules():
            # Save input_layernorm weights
            if 'input_layernorm' in name and hasattr(module, 'weight'):
                if module.weight is not None:
                    weight_dict[f"{name}.weight"] = module.weight.data.cpu()
                    quant_description[f"{name}.weight"] = "FLOAT"
                if hasattr(module, 'bias') and module.bias is not None:
                    weight_dict[f"{name}.bias"] = module.bias.data.cpu()
                    quant_description[f"{name}.bias"] = "FLOAT"

            # Save post_attention_layernorm weights
            if 'post_attention_layernorm' in name and hasattr(module, 'weight'):
                if module.weight is not None:
                    weight_dict[f"{name}.weight"] = module.weight.data.cpu()
                    quant_description[f"{name}.weight"] = "FLOAT"
                if hasattr(module, 'bias') and module.bias is not None:
                    weight_dict[f"{name}.bias"] = module.bias.data.cpu()
                    quant_description[f"{name}.bias"] = "FLOAT"

            # Save self_attn.q_norm weights (Qwen3 specific)
            if 'q_norm' in name and hasattr(module, 'weight'):
                if module.weight is not None:
                    weight_dict[f"{name}.weight"] = module.weight.data.cpu()
                    quant_description[f"{name}.weight"] = "FLOAT"

            # Save self_attn.k_norm weights (Qwen3 specific)
            if 'k_norm' in name and hasattr(module, 'weight'):
                if module.weight is not None:
                    weight_dict[f"{name}.weight"] = module.weight.data.cpu()
                    quant_description[f"{name}.weight"] = "FLOAT"

        for name, module in self.model.named_modules():
            if isinstance(module, LinearResQQuantizer):
                quant_weights = module.get_quant_weights()

                # Save low precision weights and scales
                if 'weight_low' in quant_weights:
                    weight_dict[f"{name}.weight_low"] = quant_weights['weight_low']
                    weight_dict[f"{name}.scale_low"] = quant_weights['scale_low']
                    quant_description[f"{name}.weight_low"] = "RESQ"
                    quant_description[f"{name}.scale_low"] = "RESQ"
                    # offset may be None for symmetric quantization
                    if 'offset_low' in quant_weights:
                        weight_dict[f"{name}.offset_low"] = quant_weights['offset_low']
                        quant_description[f"{name}.offset_low"] = "RESQ"

                # Save high precision weights and scales
                if 'weight_high' in quant_weights:
                    weight_dict[f"{name}.weight_high"] = quant_weights['weight_high']
                    weight_dict[f"{name}.scale_high"] = quant_weights['scale_high']
                    quant_description[f"{name}.weight_high"] = "RESQ"
                    quant_description[f"{name}.scale_high"] = "RESQ"
                    # offset may be None for symmetric quantization
                    if 'offset_high' in quant_weights:
                        weight_dict[f"{name}.offset_high"] = quant_weights['offset_high']
                        quant_description[f"{name}.offset_high"] = "RESQ"

                # Save bias if present
                if module.bias is not None:
                    weight_dict[f"{name}.bias"] = module.bias.data.cpu()
                    quant_description[f"{name}.bias"] = "RESQ"

            elif isinstance(module, nn.Linear):
                # Non-quantized linear layers (e.g., lm_head)
                weight_dict[f"{name}.weight"] = module.weight.data.cpu()
                quant_description[f"{name}.weight"] = "FLOAT"
                if module.bias is not None:
                    weight_dict[f"{name}.bias"] = module.bias.data.cpu()
                    quant_description[f"{name}.bias"] = "FLOAT"

        # Save online rotation matrices following original ResQ
        # Per-layer online projections:
        # - Uc: layer.{i}.self_attn.key_pos @ R2 (for K cache rotation after RoPE)
        # - Ud: depends on ud_rotation_type:
        #   - 'hadamard': save Pd per layer + Hd globally (runtime: act @ block_diag(Pd) @ H)
        #   - 'random': save full Ud = block_diag(Pd) @ Rd per layer (runtime: act @ Ud)
        # Note: All other U matrices (attn_mlp, value, etc.) are merged into weights
        if self.rotation_dict is not None and not self._use_simplified_mode:
            self.logger.info("Saving online rotation matrices (Uc, Ud)...")

            # Build full R2 rotation matrix for head dimension
            R2_1 = self.rotation_dict.get('R2_1')
            R2_2 = self.rotation_dict.get('R2_2')

            R2 = None
            if R2_1 is not None and R2_2 is not None:
                R2 = torch.block_diag(R2_1.to(torch.float64), R2_2.to(torch.float64))

            nlayers = self.model.config.num_hidden_layers
            intermediate_size = self.model.config.intermediate_size
            blocksize = self.cfg.down_proj_blocksize
            ud_rotation_type = self.cfg.ud_rotation_type

            # Per-layer Uc: key_pos @ R2 (for K cache rotation after RoPE)
            if self.basis_dict is not None:
                for i in range(nlayers):
                    key = f'layer.{i}.self_attn.key_pos'
                    if key in self.basis_dict:
                        U_key_pos = self.basis_dict[key].to(torch.float64)
                        if R2 is not None:
                            # U_key_pos can be per-head [num_kv_heads, head_dim, head_dim] or shared [head_dim, head_dim]
                            Uc = torch.matmul(U_key_pos, R2)
                            weight_dict[f'resq.layer.{i}.Uc'] = Uc.float().cpu()
                            quant_description[f'resq.layer.{i}.Uc'] = "FLOAT"
                        else:
                            weight_dict[f'resq.layer.{i}.Uc'] = U_key_pos.float().cpu()
                            quant_description[f'resq.layer.{i}.Uc'] = "FLOAT"
                self.logger.info(f"  Saved {nlayers} per-layer Uc matrices (key_pos @ R2)")

                # Per-layer Ud: save based on ud_rotation_type
                if ud_rotation_type == 'hadamard':
                    # Hadamard mode: save Pd per layer [blocksize, blocksize] + Hd globally
                    self.logger.info(f"  Saving Ud in Hadamard mode (Pd per layer + Hd globally)")

                    # Save Pd per layer
                    for i in range(nlayers):
                        key = f'layer.{i}.mlp.down_proj'
                        if key in self.basis_dict:
                            Pd = self.basis_dict[key].to(torch.float64)
                            weight_dict[f'resq.layer.{i}.Pd'] = Pd.float().cpu()
                            quant_description[f'resq.layer.{i}.Pd'] = "FLOAT"
                    self.logger.info(f"    Saved {nlayers} per-layer Pd matrices [{blocksize}x{blocksize}]")

                    # Save global Hadamard info
                    hadK = self.rotation_dict.get('Hd')
                    K = self.rotation_dict.get('Hd_K', 1)
                    if hadK is not None:
                        weight_dict['resq.Hd'] = hadK.float().cpu()
                        quant_description['resq.Hd'] = "FLOAT"
                        self.logger.info(f"    Saved resq.Hd [{hadK.shape[0]}x{hadK.shape[1]}]")
                    weight_dict['resq.Hd_K'] = torch.tensor(K, dtype=torch.int64)
                    quant_description['resq.Hd_K'] = "INT"
                    self.logger.info(f"    Saved resq.Hd_K = {K}")

                    # Save intermediate_size and blocksize for inference
                    weight_dict['resq.intermediate_size'] = torch.tensor(intermediate_size, dtype=torch.int64)
                    weight_dict['resq.down_proj_blocksize'] = torch.tensor(blocksize, dtype=torch.int64)
                    quant_description['resq.intermediate_size'] = "INT"
                    quant_description['resq.down_proj_blocksize'] = "INT"

                else:  # 'random' mode
                    # Random mode: compute and save full Ud = block_diag(Pd) @ Rd per layer
                    self.logger.info(f"  Saving Ud in random mode (full Ud per layer)")

                    Rd = self.rotation_dict.get('Rd')
                    if Rd is None:
                        self.logger.warning("  Rd not found in rotation_dict, skipping Ud saving")
                    else:
                        Rd = Rd.to(torch.float64)
                        num_blocks = intermediate_size // blocksize

                        for i in range(nlayers):
                            key = f'layer.{i}.mlp.down_proj'
                            if key in self.basis_dict:
                                Pd = self.basis_dict[key].to(torch.float64)

                                # Build full Ud = block_diag(Pd) @ Rd
                                # Efficient: apply Pd block-wise to Rd
                                Ud = torch.zeros(intermediate_size, intermediate_size, dtype=torch.float64)
                                for b in range(num_blocks):
                                    start_idx = b * blocksize
                                    end_idx = (b + 1) * blocksize
                                    # Ud[start:end, :] = Pd @ Rd[start:end, :]
                                    Ud[start_idx:end_idx, :] = torch.matmul(Pd, Rd[start_idx:end_idx, :])

                                weight_dict[f'resq.layer.{i}.Ud'] = Ud.float().cpu()
                                quant_description[f'resq.layer.{i}.Ud'] = "FLOAT"

                        self.logger.info(f"    Saved {nlayers} per-layer Ud matrices [{intermediate_size}x{intermediate_size}]")

            elif R2 is not None:
                # Simplified mode: save R2 as shared Uc
                weight_dict['resq.R2'] = R2.float().cpu()
                quant_description['resq.R2'] = "FLOAT"
                self.logger.info(f"  R2 shape: {R2.shape} (simplified mode)")

        # Save using SafeTensors
        if "safe_tensor" in save_type:
            from safetensors.torch import save_file

            safetensors_path = os.path.join(output_path, safetensors_name)
            save_file(weight_dict, safetensors_path)
            self.logger.info(f"Saved weights to {safetensors_path}")

        # Save JSON description
        import json
        json_path = os.path.join(output_path, json_name)
        with open(json_path, 'w') as f:
            json.dump(quant_description, f, indent=2, default=str)
        self.logger.info(f"Saved description to {json_path}")

        self.logger.info("Save complete!")


def resq_calibrate(
    model: nn.Module,
    calib_data: List,
    cfg: ResQConfig,
    output_path: str,
    disable_names: List[str] = None,
    basis_path: str = None,
    rotation_path: str = None,
) -> nn.Module:
    """
    Main entry point for ResQ calibration and saving.

    Args:
        model: The transformer model
        calib_data: Calibration data
        cfg: ResQ configuration
        output_path: Output directory for quantized model
        disable_names: Layer names to skip
        basis_path: Path to pre-computed basis
        rotation_path: Path to pre-computed rotations

    Returns:
        Quantized model
    """
    calibrator = ResQCalibrator(
        model=model,
        cfg=cfg,
        calib_data=calib_data,
        disable_names=disable_names,
        basis_path=basis_path,
        rotation_path=rotation_path,
    )

    calibrator.run()
    calibrator.save(output_path)

    return calibrator.model
