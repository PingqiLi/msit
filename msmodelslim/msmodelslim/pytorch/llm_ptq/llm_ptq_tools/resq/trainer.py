# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Adapted from ResQ and FlatQuant patterns
"""
ResQ Trainer for quantization calibration.

This module provides the main training pipeline for ResQ quantization,
following the FlatQuant integration pattern for msmodelslim.
"""

import functools
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from .config import ResQConfig
from .processors.basis_processor import compute_basis, generate_random_rotations, load_basis
from .processors.resq_processor import resq_quantize, apply_rotations, rearrange_columns
from .models.model_utils import ModelStructure, get_model_layers
from .utils.common import get_device, cleanup_memory, set_seed

logger = logging.getLogger(__name__)

# Check for NPU availability
npu_available = False
try:
    import torch_npu
    npu_available = torch.npu.is_available()
except ImportError:
    pass


def empty_cache():
    """Empty device cache."""
    if npu_available:
        torch.npu.empty_cache()


def get_device_str(device) -> str:
    """Get device string with proper prefix."""
    if isinstance(device, int):
        device = str(device)
    if npu_available:
        if not device.startswith("npu"):
            return "npu:" + device
        return device
    else:
        return "cpu"


@dataclass
class ResQTrainingConfig:
    """Training configuration for ResQ."""
    seed: int = 42
    epochs: int = 1
    batch_size: int = 1
    deactive_amp: bool = False
    amp_dtype: str = "bfloat16"

    def __post_init__(self):
        """Set up training data type and context."""
        if self.deactive_amp:
            self.dtype = torch.float32
            self.traincast = nullcontext
        else:
            if self.amp_dtype == "bfloat16":
                self.dtype = torch.bfloat16
            elif self.amp_dtype == "float16":
                self.dtype = torch.float16
            else:
                raise ValueError(f"Invalid AMP dtype: {self.amp_dtype}")

            if npu_available:
                self.traincast = functools.partial(
                    torch.amp.autocast, device_type="npu", dtype=self.dtype
                )
            else:
                self.traincast = nullcontext


class CalibrationDataCollector:
    """Collects calibration data by running through the model."""

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize the collector.

        Args:
            model: The transformer model
            device: Device to use
        """
        self.model = model
        self.device = device
        self.layers = get_model_layers(model)
        self.first_layer_inputs = []
        self.first_layer_kwargs = []

    def collect_first_layer_inputs(self, dataloader) -> Dict[str, Any]:
        """
        Collect inputs to the first decoder layer.

        Args:
            dataloader: Calibration data loader

        Returns:
            Dictionary with layer inputs and kwargs
        """
        self.first_layer_inputs = []
        self.first_layer_kwargs = []

        class StopExecution(Exception):
            pass

        def hook_fn(module, args, kwargs):
            self.first_layer_inputs.append(tuple(
                a.cpu() if isinstance(a, torch.Tensor) else a for a in args
            ))
            self.first_layer_kwargs.append({
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in kwargs.items()
            })
            raise StopExecution

        hook = self.layers[0].register_forward_pre_hook(hook_fn, with_kwargs=True)

        self.model.to(self.device)
        self.model.eval()

        try:
            for batch in dataloader:
                try:
                    if isinstance(batch, dict):
                        batch = {
                            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()
                        }
                        with torch.no_grad():
                            self.model(**batch)
                    else:
                        batch = batch.to(self.device)
                        with torch.no_grad():
                            self.model(batch)
                except StopExecution:
                    pass
        finally:
            hook.remove()

        self.model.cpu()

        return {
            'inputs': self.first_layer_inputs,
            'kwargs': self.first_layer_kwargs,
            'nsamples': len(self.first_layer_inputs),
        }


class ResQTrainer:
    """
    Main trainer class for ResQ quantization.

    Follows the FlatQuant integration pattern for msmodelslim.
    """

    def __init__(
        self,
        model: nn.Module,
        config: ResQConfig,
        logger_instance: Optional[logging.Logger] = None,
    ):
        """
        Initialize the ResQ trainer.

        Args:
            model: The transformer model
            config: ResQ configuration
            logger_instance: Optional logger instance
        """
        self.model = model
        self.config = config
        self.logger = logger_instance or logger
        self.device = get_device()

        # Set random seed
        set_seed(config.seed)

        # Analyze model structure
        self.model_structure = ModelStructure(model)

        # Training config
        self.training_config = ResQTrainingConfig(
            seed=config.seed,
            amp_dtype=config.amp_dtype,
        )

        # Initialize collectors
        self.data_collector = CalibrationDataCollector(model, self.device)

    def compute_basis_from_data(self, dataloader) -> Dict[str, torch.Tensor]:
        """
        Compute basis matrices from calibration data.

        Args:
            dataloader: Calibration data loader

        Returns:
            Dictionary of basis matrices
        """
        self.logger.info("Computing basis matrices from calibration data")

        basis_dict = compute_basis(
            self.model,
            dataloader,
            self.config,
            device=self.device,
        )

        return basis_dict

    def load_or_compute_basis(
        self,
        dataloader=None,
    ) -> Dict[str, torch.Tensor]:
        """
        Load pre-computed basis or compute from data.

        Args:
            dataloader: Calibration data loader (required if basis not pre-computed)

        Returns:
            Dictionary of basis matrices
        """
        if self.config.optimized_basis_path:
            self.logger.info(f"Loading basis from {self.config.optimized_basis_path}")
            return load_basis(self.config.optimized_basis_path)
        elif dataloader is not None:
            return self.compute_basis_from_data(dataloader)
        else:
            self.logger.warning("No basis path or dataloader provided, using identity basis")
            return {}

    def load_or_generate_rotations(self) -> Dict[str, torch.Tensor]:
        """
        Load pre-computed rotations or generate random ones.

        Returns:
            Dictionary of rotation matrices
        """
        if self.config.optimized_rotation_path:
            self.logger.info(f"Loading rotations from {self.config.optimized_rotation_path}")
            return torch.load(self.config.optimized_rotation_path, map_location='cpu')
        else:
            self.logger.info("Generating random rotations")
            return generate_random_rotations(
                hidden_dim=self.model_structure.hidden_size,
                head_dim=self.model_structure.head_dim,
                high_fraction=self.config.high_fraction,
                low_fraction=self.config.low_fraction,
                seed=self.config.seed,
            )

    def quantize(
        self,
        dataloader=None,
        basis_dict: Optional[Dict[str, torch.Tensor]] = None,
        rotation_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> nn.Module:
        """
        Main quantization entry point.

        Args:
            dataloader: Calibration data loader
            basis_dict: Pre-computed basis matrices (optional)
            rotation_dict: Pre-computed rotation matrices (optional)

        Returns:
            Quantized model
        """
        self.logger.info("Starting ResQ quantization")
        self.logger.info(f"Config: {self.config.__dict__}")

        # Load or compute basis
        if basis_dict is None:
            basis_dict = self.load_or_compute_basis(dataloader)

        # Load or generate rotations
        if rotation_dict is None:
            rotation_dict = self.load_or_generate_rotations()

        # Apply quantization
        self.model = resq_quantize(
            self.model,
            self.config,
            basis_dict=basis_dict,
            rotation_dict=rotation_dict,
        )

        self.logger.info("ResQ quantization complete")
        return self.model

    def calibrate_layer_by_layer(
        self,
        dataloader,
    ) -> None:
        """
        Calibrate quantizers layer by layer.

        This follows the FlatQuant pattern of processing layers sequentially
        to reduce memory usage.

        Args:
            dataloader: Calibration data loader
        """
        self.logger.info("Starting layer-by-layer calibration")

        # Collect first layer inputs
        data_info = self.data_collector.collect_first_layer_inputs(dataloader)
        layer_inputs = data_info['inputs']
        layer_kwargs = data_info['kwargs']
        nsamples = data_info['nsamples']

        self.logger.info(f"Collected {nsamples} calibration samples")

        layers = get_model_layers(self.model)

        # Process each layer
        for layer_idx in tqdm(range(len(layers)), desc="Calibrating layers"):
            layer = layers[layer_idx].to(self.device)

            # Run calibration forward passes
            outputs = []
            with torch.no_grad():
                for i in range(nsamples):
                    # Move inputs to device
                    device_args = tuple(
                        a.to(self.device) if isinstance(a, torch.Tensor) else a
                        for a in layer_inputs[i]
                    )
                    device_kwargs = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in layer_kwargs[i].items()
                    }

                    with self.training_config.traincast():
                        output = layer(*device_args, **device_kwargs)
                        if isinstance(output, tuple):
                            output = output[0]
                        outputs.append(output.cpu())

            # Update layer inputs for next layer
            layer_inputs = [(out,) for out in outputs]
            layer_kwargs = layer_kwargs  # Keep original kwargs

            # Move layer back to CPU
            layer.cpu()
            empty_cache()

        self.logger.info("Layer-by-layer calibration complete")


def resq_train(
    model: nn.Module,
    dataloader,
    config: ResQConfig,
    logger_instance: Optional[logging.Logger] = None,
) -> nn.Module:
    """
    Main entry point for ResQ training/quantization.

    This follows the FlatQuant pattern: flat_quant_train.

    Args:
        model: The transformer model
        dataloader: Calibration data loader
        config: ResQ configuration
        logger_instance: Optional logger instance

    Returns:
        Quantized model
    """
    trainer = ResQTrainer(model, config, logger_instance)
    return trainer.quantize(dataloader)
