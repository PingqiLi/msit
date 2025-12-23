# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
ResQ Quantization Script for Qwen3-32B

This script performs 4/8-bit hybrid quantization using ResQ algorithm:
- High variance channels: 8-bit precision
- Low variance channels: 4-bit precision
- Saves dual weights and dual scales

Usage:
    python resq_qwen3_32b.py \
        --model_path /path/to/Qwen3-32B \
        --save_directory /path/to/output \
        --calib_file ../common/wiki.jsonl \
        --batch_size 1 \
        --high_fraction 0.125 \
        --high_bits 8 \
        --low_bits 4

Output format:
    - weight_low: 4-bit quantized weights (low variance channels)
    - weight_high: 8-bit quantized weights (high variance channels)
    - scale_low: Scale for 4-bit weights
    - scale_high: Scale for 8-bit weights
    - offset_low: Offset for 4-bit weights
    - offset_high: Offset for 8-bit weights
"""
import os
import sys
import argparse
import random
import json

import numpy as np
import torch
import torch_npu
import transformers

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..', ".."))
sys.path.append(parent_directory)

from example.common.security.path import get_valid_read_path, get_write_directory
from example.common.security.type import check_number
from example.common.utils import SafeGenerator, cmd_bool
from msmodelslim.tools.copy_config_files import copy_config_files, modify_config_json
from msmodelslim.utils.logging import set_logger_level

# Import ResQ components
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.resq import (
    ResQConfig,
    ResQCalibrator,
)


def seed_everything(seed=0) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    transformers.set_seed(seed)
    torch_npu.npu.manual_seed(seed)
    torch_npu.npu.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="ResQ Quantization for Qwen3-32B")
    parser.add_argument('--model_path', type=str, required=True,
                        help="The path of float model and tokenizer")
    parser.add_argument('--save_directory', type=str, required=True,
                        help="The path to save quant model")
    parser.add_argument('--layer_count', type=int, default=0,
                        help="Layer count when loading model (0 means all layers)")
    parser.add_argument('--calib_file', type=str, default="../common/wiki.jsonl",
                        help="The calib data for calibration")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size for calibration")
    parser.add_argument('--nsamples', type=int, default=128,
                        help="Number of calibration samples")
    parser.add_argument('--seq_len', type=int, default=2048,
                        help="Sequence length for calibration")

    # ResQ specific parameters
    parser.add_argument('--high_bits', type=int, default=8,
                        help="Bits for high precision channels (default: 8)")
    parser.add_argument('--low_bits', type=int, default=4,
                        help="Bits for low precision channels (default: 4)")
    parser.add_argument('--high_fraction', type=float, default=0.125,
                        help="Fraction of channels at high precision (default: 0.125 = 1/8)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility")

    # Device settings
    parser.add_argument('--dev_type', type=str, default='npu', choices=['npu', 'cpu'],
                        help="Device type for quantization")
    parser.add_argument('--dev_id', type=int, default=0,
                        help="Device ID")

    # Model loading
    parser.add_argument('--trust_remote_code', type=cmd_bool, default=False)

    # Basis and rotation paths (optional, for using pre-computed rotations)
    parser.add_argument('--basis_path', type=str, default=None,
                        help="Path to pre-computed basis matrices (optional)")
    parser.add_argument('--rotation_path', type=str, default=None,
                        help="Path to pre-computed rotation matrices (optional)")

    return parser.parse_args()


def custom_hook(model_config):
    """Custom hook to modify config.json for ResQ format."""
    model_config["quantize"] = "w4a8_resq"


def get_calib_dataset_batch(model_tokenizer, calib_list, batch_size, seq_len, device="npu"):
    """Prepare calibration dataset in batches."""
    calib_dataset = []
    calib_list = [calib_list[i:i + batch_size] for i in range(0, len(calib_list), batch_size)]

    for calib_data in calib_list:
        inputs = model_tokenizer(
            calib_data,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=seq_len
        ).to(device)
        calib_dataset.append(
            [value.to(device) for key, value in inputs.data.items() if isinstance(value, torch.Tensor)]
        )
    return calib_dataset


def pre_check_files(path):
    """Pre-check model path files for permissions."""
    for file in os.listdir(path):
        if not (file.endswith('.json') or file.endswith('.py')):
            continue
        _ = get_valid_read_path(os.path.join(path, file), extensions=['.json', '.py'])


def main():
    args = parse_args()
    set_logger_level("info")

    print("=" * 60)
    print("ResQ Quantization for Qwen3-32B")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Save directory: {args.save_directory}")
    print(f"High bits: {args.high_bits}, Low bits: {args.low_bits}")
    print(f"High fraction: {args.high_fraction}")
    print(f"Device: {args.dev_type}:{args.dev_id}")
    print("=" * 60)

    # Set random seed
    seed_everything(args.seed)

    # Validate paths
    model_path = args.model_path
    save_directory = get_write_directory(args.save_directory, write_mode=0o750)
    pre_check_files(model_path)
    check_number(args.batch_size, int, 1, 16, "batch_size")

    # Load model configuration
    safe_generator = SafeGenerator()
    config = safe_generator.get_config_from_pretrained(
        model_path=model_path,
        trust_remote_code=args.trust_remote_code
    )

    num_layer = config.num_hidden_layers
    if args.layer_count < 0 or args.layer_count > num_layer:
        raise ValueError(
            f"Invalid value for parameter layer_count: {args.layer_count}. "
            f"Must be between 0 and {num_layer}."
        )

    # Set layer count (0 means all layers)
    config.num_hidden_layers = args.layer_count if args.layer_count != 0 else config.num_hidden_layers
    config.use_cache = False  # Disable cache to save memory

    print(f"Number of layers: {config.num_hidden_layers}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Number of attention heads: {config.num_attention_heads}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = safe_generator.get_tokenizer_from_pretrained(
        model_path=model_path,
        config=config,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
        add_eos_token=True
    )

    # Load model
    print("Loading model...")
    model = safe_generator.get_model_from_pretrained(
        model_path=model_path,
        config=config,
        trust_remote_code=args.trust_remote_code,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation='eager'
    )

    # Load calibration data
    print("Loading calibration data...")
    if args.calib_file.endswith('.jsonl'):
        calib_dataset_path = get_valid_read_path(args.calib_file, "jsonl", is_dir=False)
        calib_prompt = []
        with open(calib_dataset_path, "r", encoding="utf-8") as file:
            lines = file.readlines()[:args.nsamples]
            for line in lines:
                calib_prompt.append(json.loads(line)['inputs_pretokenized'])
    elif args.calib_file.endswith('.json'):
        calib_dataset_path = get_valid_read_path(args.calib_file, "json", is_dir=False)
        with open(calib_dataset_path, "r", encoding="utf-8") as file:
            calib_prompt = json.load(file)[:args.nsamples]
    else:
        raise ValueError("calib_file must be a jsonl or json file")

    print(f"Loaded {len(calib_prompt)} calibration samples")

    # Prepare calibration dataset
    dataset_calib = get_calib_dataset_batch(
        tokenizer, calib_prompt, args.batch_size, args.seq_len, model.device
    )

    # Create ResQ configuration
    resq_config = ResQConfig(
        high_bits=args.high_bits,
        low_bits=args.low_bits,
        high_fraction=args.high_fraction,
        low_fraction=0.0,  # No extra low precision region
        seed=args.seed,
        dev_type=args.dev_type,
        dev_id=args.dev_id,
        rotation_granularity='full_shared',
    )

    # Disable names - typically lm_head is skipped
    disable_names = []
    for name, _ in model.named_modules():
        if 'lm_head' in name:
            disable_names.append(name)

    print(f"Layers to skip: {disable_names}")

    # Create ResQ calibrator
    print("Initializing ResQ calibrator...")
    calibrator = ResQCalibrator(
        model=model,
        cfg=resq_config,
        calib_data=dataset_calib,
        disable_names=disable_names,
        basis_path=args.basis_path,
        rotation_path=args.rotation_path,
    )

    # Run calibration
    print("Running ResQ calibration...")
    calibrator.run()

    # Save quantized model
    print("Saving quantized model...")
    calibrator.save(
        output_path=save_directory,
        json_name="quant_model_description_resq.json",
        safetensors_name="quant_model_weight_resq.safetensors",
        save_type=["safe_tensor"],
    )

    # Copy config files
    import functools
    custom_hooks = {
        'config.json': functools.partial(modify_config_json, custom_hook=custom_hook)
    }
    copy_config_files(
        input_path=model_path,
        output_path=save_directory,
        quant_config=None,  # ResQ uses its own config
        custom_hooks=custom_hooks
    )

    print("=" * 60)
    print("ResQ quantization complete!")
    print(f"Output saved to: {save_directory}")
    print("=" * 60)


if __name__ == "__main__":
    main()
