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
                self.rotation_dict = generate_random_rotations(
                    hidden_dim=model.config.hidden_size,
                    head_dim=head_dim,
                    high_fraction=self.cfg.high_fraction,
                    low_fraction=self.cfg.low_fraction,
                    seed=self.cfg.seed,
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
        else:
            self._run_calib_mode()

        # Disable calibration mode
        self._disable_calibration()
        self.logger.info("ResQ calibration complete!")

    def _run_calib_mode(self) -> None:
        """Run calibration with data."""
        self.logger.info(f"Running calibration with {len(self.calib_data)} samples...")

        # Use the configured device (self.device) for input tensors
        # Note: We use self.device instead of detecting from model.embed_tokens
        # because _prepare_model may have moved layers during transformation
        embed_device = self.device
        self.logger.info(f"Moving calibration data to device: {embed_device}")

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

    def _run_datafree_mode(self) -> None:
        """Run data-free quantization."""
        self.logger.info("Running data-free weight quantization...")

        for name, module in self.model.named_modules():
            if isinstance(module, LinearResQQuantizer):
                # Quantize weights immediately
                module.quant_weight.quantize_weight(module.weight)
                self.logger.info(f"Quantized layer: {name}")

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
        }

        for name, module in self.model.named_modules():
            if isinstance(module, LinearResQQuantizer):
                quant_weights = module.get_quant_weights()

                # Save low precision weights and scales
                if 'weight_low' in quant_weights:
                    weight_dict[f"{name}.weight_low"] = quant_weights['weight_low']
                    weight_dict[f"{name}.scale_low"] = quant_weights['scale_low']
                    weight_dict[f"{name}.offset_low"] = quant_weights['offset_low']
                    quant_description[f"{name}.weight_low"] = f"W{self.cfg.low_bits}"
                    quant_description[f"{name}.scale_low"] = "FLOAT"
                    quant_description[f"{name}.offset_low"] = "FLOAT"

                # Save high precision weights and scales
                if 'weight_high' in quant_weights:
                    weight_dict[f"{name}.weight_high"] = quant_weights['weight_high']
                    weight_dict[f"{name}.scale_high"] = quant_weights['scale_high']
                    weight_dict[f"{name}.offset_high"] = quant_weights['offset_high']
                    quant_description[f"{name}.weight_high"] = f"W{self.cfg.high_bits}"
                    quant_description[f"{name}.scale_high"] = "FLOAT"
                    quant_description[f"{name}.offset_high"] = "FLOAT"

                # Save bias if present
                if module.bias is not None:
                    weight_dict[f"{name}.bias"] = module.bias.data.cpu()
                    quant_description[f"{name}.bias"] = "FLOAT"

            elif isinstance(module, nn.Linear):
                # Non-quantized linear layers (e.g., lm_head)
                weight_dict[f"{name}.weight"] = module.weight.data.cpu()
                quant_description[f"{name}.weight"] = "FLOAT"
                if module.bias is not None:
                    weight_dict[f"{name}.bias"] = module.bias.data.cpu()
                    quant_description[f"{name}.bias"] = "FLOAT"

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
