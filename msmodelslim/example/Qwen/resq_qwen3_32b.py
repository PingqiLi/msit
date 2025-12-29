# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
ResQ Quantization Script for Qwen3-32B

This script performs 4/8-bit hybrid quantization using ResQ algorithm:
- High variance channels: 8-bit precision
- Low variance channels: 4-bit precision
- Saves dual weights and dual scales
- Generates per-layer online projection matrices (Uc, Ud)

Usage:
    # Full ResQ with on-the-fly basis computation (recommended):
    python resq_qwen3_32b.py \
        --model_path /path/to/Qwen3-32B \
        --save_directory /path/to/output \
        --calib_file ../common/wiki.jsonl \
        --compute_basis \
        --save_basis_path /path/to/basis.pt

    # Using pre-computed basis:
    python resq_qwen3_32b.py \
        --model_path /path/to/Qwen3-32B \
        --save_directory /path/to/output \
        --calib_file ../common/wiki.jsonl \
        --basis_path /path/to/basis.pt

    # Simplified mode (no basis, suboptimal quality):
    python resq_qwen3_32b.py \
        --model_path /path/to/Qwen3-32B \
        --save_directory /path/to/output \
        --calib_file ../common/wiki.jsonl

Output format:
    - weight_low: 4-bit quantized weights (low variance channels)
    - weight_high: 8-bit quantized weights (high variance channels)
    - scale_low: Scale for 4-bit weights
    - scale_high: Scale for 8-bit weights
    - offset_low: Offset for 4-bit weights
    - offset_high: Offset for 8-bit weights
    - resq.layer.{i}.Uc: Per-layer online rotation for K cache (key_pos @ R2)
    - resq.layer.{i}.Ud: Per-layer online rotation for down_proj (down_proj @ Rd)

ResQ Algorithm (U = P @ R):
    - P: Eigenvector matrix from eigendecomposition of activation covariance
    - R: Block-diagonal random orthogonal rotation = block_diag(R_low, R_mid, R_high)
    - Uc = key_pos_basis @ R2 (for K cache rotation after RoPE)
    - Ud = down_proj_basis @ Rd (for down_proj input rotation)
"""
import os
import sys
import argparse
import random
import json
import gc
import shutil

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
from msmodelslim.utils.logging import set_logger_level

# Import ResQ components
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.resq import (
    ResQConfig,
    ResQCalibrator,
    compute_basis,
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

    # Basis computation options
    parser.add_argument('--compute_basis', type=cmd_bool, default=False,
                        help="Compute basis from calibration data (recommended for best quality)")
    parser.add_argument('--save_basis_path', type=str, default=None,
                        help="Path to save computed basis matrices (optional)")

    return parser.parse_args()


def custom_hook(model_config):
    """Custom hook to modify config.json for ResQ format."""
    model_config["quantize"] = "w4a8_resq"


def get_calib_dataset_batch(model_tokenizer, calib_list, batch_size, seq_len, device="npu"):
    """Prepare calibration dataset in batches."""
    print(f"[DEBUG] get_calib_dataset_batch called with device={device}")
    calib_dataset = []
    calib_list = [calib_list[i:i + batch_size] for i in range(0, len(calib_list), batch_size)]

    for idx, calib_data in enumerate(calib_list):
        inputs = model_tokenizer(
            calib_data,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=seq_len
        ).to(device)
        batch_tensors = [value.to(device) for key, value in inputs.data.items() if isinstance(value, torch.Tensor)]
        calib_dataset.append(batch_tensors)
        if idx == 0:
            # Debug print for first batch
            print(f"[DEBUG] First batch tensor devices: {[t.device for t in batch_tensors]}")
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

    # Determine mode
    if args.basis_path:
        mode = "FULL (pre-computed basis)"
    elif args.compute_basis:
        mode = "FULL (on-the-fly basis computation)"
    else:
        mode = "SIMPLIFIED (no basis - suboptimal quality)"

    print("=" * 60)
    print("ResQ Quantization for Qwen3-32B")
    print("=" * 60)
    print(f"Mode: {mode}")
    print(f"Model path: {args.model_path}")
    print(f"Save directory: {args.save_directory}")
    print(f"High bits: {args.high_bits}, Low bits: {args.low_bits}")
    print(f"High fraction: {args.high_fraction}")
    print(f"Device: {args.dev_type}:{args.dev_id}")
    if args.basis_path:
        print(f"Basis path: {args.basis_path}")
    if args.compute_basis:
        print(f"Will compute basis from calibration data")
        if args.save_basis_path:
            print(f"Will save basis to: {args.save_basis_path}")
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
    # For basis computation, load model on CPU to enable layer-by-layer processing
    # This saves device memory by only loading one layer at a time
    if args.compute_basis and not args.basis_path:
        print("Loading model on CPU for memory-efficient basis computation...")
        model = safe_generator.get_model_from_pretrained(
            model_path=model_path,
            config=config,
            trust_remote_code=args.trust_remote_code,
            device_map="cpu",  # Load on CPU for layer-by-layer processing
            torch_dtype=torch.bfloat16,
            attn_implementation='eager'
        )
        model_device = torch.device('cpu')
    else:
        model = safe_generator.get_model_from_pretrained(
            model_path=model_path,
            config=config,
            trust_remote_code=args.trust_remote_code,
            device_map="auto",
            torch_dtype="auto",
            attn_implementation='eager'
        )
        # Get the device of the embedding layer (where input_ids are sent first)
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            model_device = model.model.embed_tokens.weight.device
        elif hasattr(model, 'hf_device_map'):
            # Use hf_device_map to find the first device
            first_device = list(model.hf_device_map.values())[0]
            model_device = torch.device(first_device) if isinstance(first_device, str) else first_device
        else:
            # Fallback to explicitly use the configured device
            if args.dev_type == 'npu':
                model_device = torch.device(f'npu:{args.dev_id}')
            elif args.dev_type == 'cuda':
                model_device = torch.device(f'cuda:{args.dev_id}')
            else:
                model_device = next(model.parameters()).device
        print(f"Detected model device for calibration data: {model_device}")

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
    # For basis computation, keep data on CPU; otherwise use model device
    calib_device = 'cpu' if (args.compute_basis and not args.basis_path) else model_device
    dataset_calib = get_calib_dataset_batch(
        tokenizer, calib_prompt, args.batch_size, args.seq_len, calib_device
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
    )

    # Determine the device for layer-by-layer processing
    if args.dev_type == 'npu':
        process_device = torch.device(f'npu:{args.dev_id}')
    elif args.dev_type == 'cuda':
        process_device = torch.device(f'cuda:{args.dev_id}')
    else:
        process_device = torch.device('cpu')

    # Compute basis if requested
    basis_path = args.basis_path
    if args.compute_basis and not args.basis_path:
        print("=" * 60)
        print("Computing eigenvalue basis from calibration data...")
        print("This may take a while for large models...")
        print(f"Processing device: {process_device}")
        print("Covariance matrices stored on CPU to save memory")
        print("=" * 60)

        # Create a simple dataloader for basis computation
        class SimpleDataLoader:
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                for batch in self.data:
                    # batch is a list of tensors [input_ids, attention_mask, ...]
                    result = {'input_ids': batch[0]}
                    if len(batch) > 1 and batch[1] is not None:
                        result['attention_mask'] = batch[1]
                    yield result

            def __len__(self):
                return len(self.data)

        basis_dataloader = SimpleDataLoader(dataset_calib)

        try:
            # Compute basis with layer-by-layer processing
            # - device: where to run layer forward passes (NPU/GPU)
            # - cov_device: where to store covariance matrices (CPU to save memory)
            basis_dict = compute_basis(
                model=model,
                dataloader=basis_dataloader,
                config=resq_config,
                device=process_device,
                cov_device='cpu',  # Store covariance matrices on CPU to save NPU memory
            )

            # Save basis if path provided
            if args.save_basis_path:
                save_basis_path = args.save_basis_path
                # Handle if path is a directory (ends with / or is an existing dir)
                if save_basis_path.endswith('/') or save_basis_path.endswith('\\'):
                    save_basis_path = os.path.join(save_basis_path, "resq_basis.pt")
                elif os.path.isdir(save_basis_path):
                    save_basis_path = os.path.join(save_basis_path, "resq_basis.pt")
                save_basis_dir = os.path.dirname(save_basis_path)
                if save_basis_dir:
                    os.makedirs(save_basis_dir, exist_ok=True)
                torch.save(basis_dict, save_basis_path)
                print(f"Saved basis to: {save_basis_path}")

            # Save basis to output directory as well
            basis_output_path = os.path.join(save_directory, "resq_basis.pt")
            torch.save(basis_dict, basis_output_path)
            print(f"Saved basis to: {basis_output_path}")

            # Use the computed basis path
            basis_path = basis_output_path
            print("Basis computation complete!")

        except Exception as e:
            print(f"WARNING: Basis computation failed: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to simplified mode (no basis)")
            basis_path = None

        print("=" * 60)

        # After basis computation, reload model with device_map="auto" for calibration
        print("Reloading model for calibration...")
        del model
        gc.collect()
        try:
            torch_npu.npu.empty_cache()
        except Exception:
            pass

        model = safe_generator.get_model_from_pretrained(
            model_path=model_path,
            config=config,
            trust_remote_code=args.trust_remote_code,
            device_map="auto",
            torch_dtype="auto",
            attn_implementation='eager'
        )

        # Re-prepare calibration data for the new model device
        # Get the device of the embedding layer (where input_ids are sent first)
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            model_device = model.model.embed_tokens.weight.device
        elif hasattr(model, 'hf_device_map'):
            first_device = list(model.hf_device_map.values())[0]
            model_device = torch.device(first_device) if isinstance(first_device, str) else first_device
        else:
            # Fallback to explicitly use the configured device
            if args.dev_type == 'npu':
                model_device = torch.device(f'npu:{args.dev_id}')
            elif args.dev_type == 'cuda':
                model_device = torch.device(f'cuda:{args.dev_id}')
            else:
                try:
                    model_device = next(model.parameters()).device
                except StopIteration:
                    model_device = torch.device('cpu')
        print(f"Detected model device for calibration data (after reload): {model_device}")
        dataset_calib = get_calib_dataset_batch(
            tokenizer, calib_prompt, args.batch_size, args.seq_len, model_device
        )

    # Disable names - typically lm_head is skipped
    disable_names = []
    for name, _ in model.named_modules():
        if 'lm_head' in name:
            disable_names.append(name)

    print(f"Layers to skip: {disable_names}")

    # Debug: Check model embedding device and calibration data devices
    print("=" * 60)
    print("[DEBUG] Device verification before calibration:")
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        print(f"[DEBUG] model.model.embed_tokens.weight.device = {model.model.embed_tokens.weight.device}")
    if hasattr(model, 'hf_device_map'):
        print(f"[DEBUG] model.hf_device_map = {model.hf_device_map}")
    if dataset_calib and len(dataset_calib) > 0:
        first_batch = dataset_calib[0]
        print(f"[DEBUG] First calib batch devices: {[t.device for t in first_batch]}")
    print("=" * 60)

    # Create ResQ calibrator
    print("Initializing ResQ calibrator...")
    calibrator = ResQCalibrator(
        model=model,
        cfg=resq_config,
        calib_data=dataset_calib,
        disable_names=disable_names,
        basis_path=basis_path,  # Use computed or provided basis_path
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

    # Copy config files manually (avoid using quant_config parameter)
    for file in os.listdir(model_path):
        if file.endswith('.json') or file.endswith('.py'):
            src_path = os.path.join(model_path, file)
            dst_path = os.path.join(save_directory, file)
            if file == 'config.json':
                # Modify config.json to add quantize field
                with open(src_path, 'r', encoding='utf-8') as f:
                    model_config = json.load(f)
                model_config['quantize'] = 'W4A8_ResQ'
                with open(dst_path, 'w', encoding='utf-8') as f:
                    json.dump(model_config, f, indent=2, ensure_ascii=False)
            else:
                shutil.copy2(src_path, dst_path)

    print("=" * 60)
    print("ResQ quantization complete!")
    print(f"Output saved to: {save_directory}")
    print("")
    print("Output files include:")
    print("  - quant_model_weight_resq.safetensors (quantized weights)")
    print("  - quant_model_description_resq.json (quantization metadata)")
    print("  - resq_basis.pt (if basis was computed)")
    print("")
    print("Per-layer online projection matrices (U = P @ R):")
    print(f"  - resq.layer.{{0..{config.num_hidden_layers-1}}}.Uc: K cache rotation (key_pos @ R2)")
    print(f"  - resq.layer.{{0..{config.num_hidden_layers-1}}}.Ud: down_proj rotation (down_proj @ Rd)")
    print("=" * 60)


if __name__ == "__main__":
    main()
