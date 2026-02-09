# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
ResQ Quantization Script for Qwen3-32B

This script performs 4/8-bit hybrid quantization using ResQ algorithm:
- High variance channels: 8-bit precision
- Low variance channels: 4-bit precision
- Saves dual weights and dual scales
- Generates per-layer online projection matrices (Uc, Ud)

Usage:
    # Mode 1: Fixed ratio (current behavior, backward compatible)
    python resq_qwen3_32b.py \
        --model_path /path/to/Qwen3-32B \
        --save_directory /path/to/output \
        --high_fraction 0.125 \
        --compute_basis

    # Mode 2: Hybrid adaptive (recommended for best quality)
    python resq_qwen3_32b.py \
        --model_path /path/to/Qwen3-32B \
        --save_directory /path/to/output \
        --adaptive_ratio_mode hybrid \
        --compute_basis \
        --compute_kurtosis

    # Mode 3: Single algorithm (e.g., CEV only)
    python resq_qwen3_32b.py \
        --model_path /path/to/Qwen3-32B \
        --save_directory /path/to/output \
        --adaptive_ratio_mode cev \
        --cev_target_variance 0.95 \
        --compute_basis

    # Mode 4: Per-transform algorithm override
    python resq_qwen3_32b.py \
        --model_path /path/to/Qwen3-32B \
        --save_directory /path/to/output \
        --adaptive_ratio_mode hybrid \
        --transform_algorithms '{"Ua": "cev", "Ub": "kurtosis", "Ud": "hessian"}' \
        --compute_basis \
        --compute_kurtosis

    # Mode 5: Use pre-computed ratios
    python resq_qwen3_32b.py \
        --model_path /path/to/Qwen3-32B \
        --save_directory /path/to/output \
        --adaptive_ratio_path /path/to/resq_adaptive_ratios.json \
        --basis_path /path/to/basis.pt

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
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.resq.processors.basis_processor import save_basis


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

    # Output mode: determines what outputs to generate
    parser.add_argument('--output_mode', type=str, default='fused',
                        choices=['fused', 'transforms_only', 'debug'],
                        help="Output mode: 'fused' (default) saves fused weights + online transforms, "
                             "'transforms_only' saves decomposed P/R matrices without fusion, "
                             "'debug' saves both fused weights and decomposed transforms")
    # Backward compatibility: keep old argument but mark deprecated
    parser.add_argument('--save_transforms_only', type=cmd_bool, default=None,
                        help="[DEPRECATED] Use --output_mode='transforms_only' instead")

    # Remove Ub mode - use rotation only for value projection
    parser.add_argument('--remove_ub', type=cmd_bool, default=False,
                        help="Remove Ub (eigenvector basis) for V_proj transformation. "
                             "When enabled, V_proj uses only Rb rotation (no basis/permutation), "
                             "and rearrange_o_proj is skipped.")

    # Adaptive ratio parameters
    parser.add_argument('--adaptive_ratio_mode', type=str, default='fixed',
                        choices=['fixed', 'hessian', 'kurtosis', 'cev', 'hybrid'],
                        help="Ratio mode: 'fixed' uses high_fraction, others compute adaptively")
    parser.add_argument('--adaptive_min_ratio', type=float, default=0.0625,
                        help="Minimum ratio bound for adaptive mode (default: 1/16)")
    parser.add_argument('--adaptive_max_ratio', type=float, default=0.25,
                        help="Maximum ratio bound for adaptive mode (default: 1/4)")
    parser.add_argument('--cev_target_variance', type=float, default=0.95,
                        help="Target variance for CEV algorithm (default: 0.95)")
    parser.add_argument('--adaptive_alignment', type=int, default=512,
                        help="Hardware alignment boundary (default: 512)")
    parser.add_argument('--transform_algorithms', type=str, default=None,
                        help="Per-transform algorithm override as JSON, e.g., '{\"Ua\": \"cev\"}'")
    parser.add_argument('--ub_head_aggregation', type=str, default='max',
                        choices=['max', 'mean'],
                        help="Aggregation method for Ub across heads (default: 'max')")
    parser.add_argument('--adaptive_ratio_path', type=str, default=None,
                        help="Path to pre-computed adaptive ratios JSON")
    parser.add_argument('--compute_kurtosis', type=cmd_bool, default=False,
                        help="Compute kurtosis during basis computation (enables kurtosis algorithm)")
    parser.add_argument('--down_proj_ratio_threshold', type=float, default=None,
                        help="Ratio threshold for down_proj quant type selection. "
                             "Layers with ratio < threshold use int4_hadamard, >= threshold use w8a8_dynamic. "
                             "Default: midpoint of (adaptive_min_ratio + adaptive_max_ratio) / 2")
    parser.add_argument('--mix_cfg', type=str, default=None,
                        help="JSON dict mapping layer name patterns to quant types: "
                             "'resq', 'w8a8_dynamic', 'int4_hadamard', or 'float'. "
                             "Example: '{\"*.mlp.down_proj\": \"w8a8_dynamic\"}'")

    # NPU memory configuration
    parser.add_argument('--max_memory_per_device', type=str, default=None,
                        help="Max memory per NPU device for model loading (e.g., '55GiB'). "
                             "Used to tell accelerate actual NPU memory since it can't auto-detect it. "
                             "If not set, tries torch.npu.get_device_properties() auto-detection.")
    parser.add_argument('--disable_cpu_offload', type=cmd_bool, default=True,
                        help="Prevent accelerate from offloading layers to CPU (default: True). "
                             "Set to False if you want to allow CPU offloading.")

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


def fix_meta_norm_params(model, model_path, model_device):
    """Fix Qwen3 q_norm/k_norm meta tensors left by device_map='auto'."""
    meta_norm_params = []
    for name, param in model.named_parameters():
        if param.device.type == 'meta' and ('norm' in name.lower()):
            meta_norm_params.append(name)

    if not meta_norm_params:
        return

    print(f"Found {len(meta_norm_params)} meta norm parameters, loading from pretrained...")
    from safetensors.torch import load_file
    import glob as glob_module

    weight_files = sorted(glob_module.glob(os.path.join(model_path, "*.safetensors")))
    loaded_count = 0

    for wf in weight_files:
        if not meta_norm_params:
            break
        state_dict = load_file(wf, device='cpu')
        for name in list(meta_norm_params):
            if name in state_dict:
                parts = name.split('.')
                module = model
                for part in parts[:-1]:
                    module = getattr(module, part)
                param_name = parts[-1]

                new_param = torch.nn.Parameter(
                    state_dict[name].to(model_device),
                    requires_grad=False
                )
                setattr(module, param_name, new_param)
                meta_norm_params.remove(name)
                loaded_count += 1
                print(f"  Loaded {name}: {new_param.shape}")
        del state_dict

    if meta_norm_params:
        print(f"WARNING: Could not load {len(meta_norm_params)} params: {meta_norm_params}")
    else:
        print(f"Successfully loaded {loaded_count} meta norm parameters")


def build_npu_max_memory(max_memory_per_device=None, disable_cpu_offload=True):
    """Build max_memory dict for NPU devices.

    accelerate can't auto-detect NPU memory, causing it to underestimate available
    memory and offload layers to CPU. This function provides the actual memory info.

    Args:
        max_memory_per_device: Manual memory specification (e.g., "55GiB").
            If None, tries torch.npu.get_device_properties() auto-detection.
        disable_cpu_offload: If True, sets CPU memory to "0GiB" to prevent offloading.

    Returns:
        max_memory dict mapping device indices to memory limits, or None if
        no NPU devices are available.
    """
    num_npus = torch.npu.device_count()
    if num_npus == 0:
        return None

    max_memory = {}

    if max_memory_per_device:
        # Use the user-specified value for all NPUs
        for i in range(num_npus):
            max_memory[i] = max_memory_per_device
        print(f"Using manual max_memory: {max_memory_per_device} x {num_npus} NPUs")
    else:
        # Try auto-detection via torch.npu.get_device_properties()
        try:
            for i in range(num_npus):
                props = torch.npu.get_device_properties(i)
                total_mem = props.total_memory
                # Use 85% to leave headroom for runtime allocations
                usable_mem = int(total_mem * 0.85)
                max_memory[i] = usable_mem
                print(f"NPU {i}: {total_mem / (1024**3):.1f} GiB total, "
                      f"using {usable_mem / (1024**3):.1f} GiB (85%)")
        except (AttributeError, RuntimeError) as e:
            print(f"WARNING: Could not auto-detect NPU memory ({e}). "
                  f"Consider passing --max_memory_per_device (e.g., '55GiB').")
            return None

    if disable_cpu_offload:
        max_memory["cpu"] = "0GiB"

    return max_memory


def main():
    args = parse_args()
    set_logger_level("info")

    # Handle deprecated argument
    if args.save_transforms_only is not None:
        import warnings
        warnings.warn(
            "--save_transforms_only is deprecated, use --output_mode='transforms_only' instead",
            DeprecationWarning
        )
        if args.save_transforms_only:
            args.output_mode = 'transforms_only'

    # Determine mode description
    if args.output_mode == 'transforms_only':
        mode = "TRANSFORM-ONLY (save decomposed P/R matrices without fusion)"
    elif args.output_mode == 'debug':
        mode = "DEBUG (save both fused weights and decomposed transforms)"
    elif args.basis_path:
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
    print("")
    print("Precision Configuration:")
    print(f"  High bits: {args.high_bits}, Low bits: {args.low_bits}")
    if args.adaptive_ratio_mode != 'fixed':
        print(f"  Ratio mode: {args.adaptive_ratio_mode} (adaptive)")
        print(f"  Min ratio: {args.adaptive_min_ratio}, Max ratio: {args.adaptive_max_ratio}")
        print(f"  Alignment: {args.adaptive_alignment}")
        # Determine per-transform algorithms display
        if args.transform_algorithms:
            print(f"  Per-transform algorithms: {args.transform_algorithms}")
        else:
            alg = args.adaptive_ratio_mode
            print(f"  Per-transform algorithms: Ua={alg}, Ub={alg}, Uc={alg}, Ud={alg}")
        print(f"  Ub head aggregation: {args.ub_head_aggregation}")
        if args.compute_kurtosis:
            print(f"  Kurtosis computation: enabled")
        if args.adaptive_ratio_path:
            print(f"  Pre-computed ratios: {args.adaptive_ratio_path}")
        # Show down_proj ratio threshold
        threshold = args.down_proj_ratio_threshold
        if threshold is None:
            threshold = (args.adaptive_min_ratio + args.adaptive_max_ratio) / 2
            print(f"  Down proj ratio threshold: {threshold:.4f} (default)")
        else:
            print(f"  Down proj ratio threshold: {threshold:.4f}")
    else:
        print(f"  High fraction: {args.high_fraction} (fixed)")
        print(f"  Down proj: all layers use w8a8_dynamic (fixed mode)")

    print(f"  Device: {args.dev_type}:{args.dev_id}")
    print("")
    if args.basis_path:
        print(f"Basis path: {args.basis_path}")
    if args.compute_basis:
        print(f"Will compute basis from calibration data")
        if args.save_basis_path:
            print(f"Will save basis to: {args.save_basis_path}")
    if args.output_mode == 'transforms_only':
        print(f"Transform-only mode: Will save decomposed P/R matrices without fusion")
    elif args.output_mode == 'debug':
        print(f"Debug mode: Will save both fused weights and decomposed P/R transforms")
    if args.remove_ub:
        print(f"Remove Ub mode: V_proj uses Rb rotation only (no Pb basis)")
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
            attn_implementation='flash_attention_2'
        )
        model_device = torch.device('cpu')
    else:
        # Build max_memory to tell accelerate actual NPU memory
        # (accelerate can't auto-detect NPU memory and may offload layers to CPU)
        npu_max_memory = build_npu_max_memory(
            max_memory_per_device=args.max_memory_per_device,
            disable_cpu_offload=args.disable_cpu_offload,
        )

        load_kwargs = dict(
            model_path=model_path,
            config=config,
            trust_remote_code=args.trust_remote_code,
            device_map="auto",
            torch_dtype="auto",
            attn_implementation='eager',
        )
        if npu_max_memory is not None:
            load_kwargs["max_memory"] = npu_max_memory

        model = safe_generator.get_model_from_pretrained(**load_kwargs)

        # Verify no layers ended up on CPU
        if hasattr(model, 'hf_device_map'):
            cpu_layers = [k for k, v in model.hf_device_map.items() if v == 'cpu']
            if cpu_layers:
                print(f"WARNING: {len(cpu_layers)} layers mapped to CPU: {cpu_layers[:5]}...")
                print("Consider passing --max_memory_per_device to specify NPU memory.")

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

        # Fix meta norm parameters for Qwen3
        fix_meta_norm_params(model, model_path, model_device)

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

    # Parse transform_algorithms JSON if provided
    transform_algorithms = {}
    if args.transform_algorithms:
        transform_algorithms = json.loads(args.transform_algorithms)

    # Parse mix_cfg JSON if provided
    mix_cfg = {
            #    "*.self_attn.o_proj": "w8a8_dynamic", 
               "*.mlp.down_proj": "w8a8_dynamic"}
    if args.mix_cfg:
        mix_cfg = json.loads(args.mix_cfg)

    if mix_cfg:
        print(f"  Mix config: {mix_cfg}")
    
    # Auto-enable compute_kurtosis for algorithms that need it
    needs_kurtosis = args.adaptive_ratio_mode in ['kurtosis', 'hybrid']
    if needs_kurtosis and not args.compute_kurtosis:
        print(f"Note: Auto-enabling kurtosis computation for {args.adaptive_ratio_mode} mode")
        args.compute_kurtosis = True

    # Create ResQ configuration
    resq_config = ResQConfig(
        high_bits=args.high_bits,
        low_bits=args.low_bits,
        high_fraction=args.high_fraction,
        low_fraction=0.0,  # No extra low precision region
        seed=args.seed,
        dev_type=args.dev_type,
        dev_id=args.dev_id,
        output_mode=args.output_mode,
        remove_ub=args.remove_ub,
        # Adaptive ratio configuration
        adaptive_ratio=(args.adaptive_ratio_mode != 'fixed'),
        adaptive_algorithm=args.adaptive_ratio_mode if args.adaptive_ratio_mode != 'fixed' else 'hybrid',
        adaptive_min_ratio=args.adaptive_min_ratio,
        adaptive_max_ratio=args.adaptive_max_ratio,
        cev_target_variance=args.cev_target_variance,
        adaptive_alignment=args.adaptive_alignment,
        transform_algorithms=transform_algorithms,
        ub_head_aggregation=args.ub_head_aggregation,
        adaptive_ratio_path=args.adaptive_ratio_path,
        compute_kurtosis=args.compute_kurtosis,
        down_proj_ratio_threshold=args.down_proj_ratio_threshold,
        mix_cfg=mix_cfg,
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
    # Initialize adaptive ratio data (will be populated if compute_kurtosis is enabled)
    eval_dict = None
    kurtosis_dict = None

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
            # - compute_kurtosis: enables kurtosis computation for adaptive ratio
            basis_result = compute_basis(
                model=model,
                dataloader=basis_dataloader,
                config=resq_config,
                device=process_device,
                cov_device='cpu',  # Store covariance matrices on CPU to save NPU memory
                compute_kurtosis=args.compute_kurtosis,
            )

            # Handle return value based on adaptive mode and compute_kurtosis
            # - compute_kurtosis=True: returns (basis_dict, eval_dict, kurtosis_dict)
            # - adaptive_ratio in [hessian, cev], compute_kurtosis=False: returns (basis_dict, eval_dict)
            # - adaptive_ratio=False, compute_kurtosis=False: returns basis_dict
            if args.compute_kurtosis:
                basis_dict, eval_dict, kurtosis_dict = basis_result
                print(f"Kurtosis computation complete: {len(kurtosis_dict)} entries")
            elif args.adaptive_ratio_mode in ['hessian', 'cev']:
                basis_dict, eval_dict = basis_result
                kurtosis_dict = None
            else:
                basis_dict = basis_result

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
                save_basis(basis_dict, save_basis_path, eval_dict, kurtosis_dict)
                print(f"Saved basis to: {save_basis_path}")

            # Save basis to output directory as well
            basis_output_path = os.path.join(save_directory, "resq_basis.pt")
            save_basis(basis_dict, basis_output_path, eval_dict, kurtosis_dict)
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

        # Build max_memory for reload (same as initial load)
        npu_max_memory = build_npu_max_memory(
            max_memory_per_device=args.max_memory_per_device,
            disable_cpu_offload=args.disable_cpu_offload,
        )

        reload_kwargs = dict(
            model_path=model_path,
            config=config,
            trust_remote_code=args.trust_remote_code,
            device_map="auto",
            torch_dtype="auto",
            attn_implementation='eager',
        )
        if npu_max_memory is not None:
            reload_kwargs["max_memory"] = npu_max_memory

        model = safe_generator.get_model_from_pretrained(**reload_kwargs)

        # Verify no layers ended up on CPU
        if hasattr(model, 'hf_device_map'):
            cpu_layers = [k for k, v in model.hf_device_map.items() if v == 'cpu']
            if cpu_layers:
                print(f"WARNING: {len(cpu_layers)} layers mapped to CPU: {cpu_layers[:5]}...")
                print("Consider passing --max_memory_per_device to specify NPU memory.")

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

        # Fix meta norm parameters for Qwen3
        fix_meta_norm_params(model, model_path, model_device)

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
        eval_dict = eval_dict,
        kurtosis_dict = kurtosis_dict
    )

    # Set eval_dict and kurtosis_dict if computed during basis computation
    # These are used by the calibrator's adaptive ratio computation
    if args.adaptive_ratio_mode != 'fixed' and args.compute_basis and not args.basis_path:
        if eval_dict is not None:
            # calibrator.eval_dict = eval_dict
            print(f"Set eval_dict with {len(eval_dict)} entries")
        if kurtosis_dict is not None:
            # calibrator.kurtosis_dict = kurtosis_dict
            print(f"Set kurtosis_dict with {len(kurtosis_dict)} entries")

    # Run calibration
    print("Running ResQ calibration...")
    calibrator.run()

    # Save quantized model
    print("Saving quantized model...")
    calibrator.save(
        output_path=save_directory,
        json_name="quant_model_description.json",
        safetensors_name="model.safetensors",
        save_type=["safe_tensor"],
    )

    # Copy config files manually (avoid using quant_config parameter)
    # Exclude model.safetensors.index.json - it will be generated by calibrator.save()
    EXCLUDED_JSON_FILES = {'model.safetensors.index.json'}
    for file in os.listdir(model_path):
        if file.endswith('.json') or file.endswith('.py'):
            if file in EXCLUDED_JSON_FILES:
                continue
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
    if args.output_mode == 'transforms_only':
        print("Output files include (transform-only mode):")
        print("  - resq_transforms.safetensors (decomposed P and R matrices)")
        print("  - resq_transforms_meta.json (metadata)")
        print("")
        print(f"Per-layer decomposed transform matrices (U = P @ R) for {config.num_hidden_layers} layers:")
        print(f"  - resq.layer.{{i}}.P_a, R_a: attn/mlp input rotation")
        print(f"  - resq.layer.{{i}}.P_b, R_b: v_proj output rotation (per-head)")
        print(f"  - resq.layer.{{i}}.P_c, R_c: q/k_proj output rotation (post-RoPE)")
        print(f"  - resq.layer.{{i}}.P_d, R_d: down_proj input rotation")
    elif args.output_mode == 'debug':
        print("Output files include (debug mode - both fused and transforms):")
        print("  - quant_model_weight_resq.safetensors (quantized weights + online transforms)")
        print("  - quant_model_description_resq.json (quantization metadata)")
        print("  - resq_transforms.safetensors (decomposed P and R matrices)")
        print("  - resq_transforms_meta.json (transform metadata)")
        print("  - resq_basis.pt (if basis was computed)")
        print("")
        print("Fused mode includes online projection matrices (U = P @ R):")
        print(f"  - resq.layer.{{0..{config.num_hidden_layers-1}}}.Uc: K cache rotation (key_pos @ R2)")
        print(f"  - resq.layer.{{0..{config.num_hidden_layers-1}}}.Ud/Pd+Hd: down_proj rotation")
        print("")
        print("Transform-only mode includes decomposed P/R matrices for verification:")
        print(f"  - resq.layer.{{i}}.P_a, R_a, P_b, R_b, P_c, R_c, P_d, R_d")
    else:  # fused mode (default)
        print("Output files include:")
        print("  - quant_model_weight_resq.safetensors (quantized weights + online transforms)")
        print("  - quant_model_description_resq.json (quantization metadata)")
        print("  - resq_basis.pt (if basis was computed)")
        print("")
        print("Per-layer online projection matrices (U = P @ R):")
        print(f"  - resq.layer.{{0..{config.num_hidden_layers-1}}}.Uc: K cache rotation (key_pos @ R2)")
        print(f"  - resq.layer.{{0..{config.num_hidden_layers-1}}}.Ud/Pd+Hd: down_proj rotation")
    print("=" * 60)


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_norm_weight_is_one.py

用法：
  python check_norm_weight_is_one.py /path/to/model.safetensors
  python check_norm_weight_is_one.py /path/to/model.safetensors --only model.layers.0.input_layernorm.weight
  python check_norm_weight_is_one.py /path/to/model.safetensors --pattern "input_layernorm.weight"
  python check_norm_weight_is_one.py /path/to/model.safetensors --atol 1e-6 --rtol 1e-6
  python check_norm_weight_is_one.py /path/to/model.safetensors --exclude-qk-norm  # 排除 q_norm/k_norm

说明：
  - 会检查所有匹配到的 norm weight key 是否接近全 1
  - 默认模式：匹配包含 norm/layernorm/rmsnorm 且以 .weight 结尾的 key
  - 对于 Qwen3 等带 QK-Norm 的模型，建议使用 --exclude-qk-norm 排除 q_norm/k_norm
    （这些 norm 无法融合到权重中，保持原始训练值是正确的）
"""

import argparse
import re
import sys
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("safetensors_path", help="量化后的 .safetensors 文件路径")
    ap.add_argument("--only", default=None, help="只检查某一个具体 key（完全匹配）")
    ap.add_argument("--pattern", default=None, help="自定义正则/子串匹配 key（例如 'input_layernorm.weight'）")
    ap.add_argument("--atol", type=float, default=1e-5, help="绝对容差")
    ap.add_argument("--rtol", type=float, default=0.0, help="相对容差")
    ap.add_argument("--max_report", type=int, default=20, help="最多报告多少个失败 key")
    ap.add_argument("--exclude-qk-norm", action="store_true",
                    help="排除 q_norm 和 k_norm（Qwen3 等模型的 QK-Norm 无法融合，保持原值是正确的）")
    args = ap.parse_args()

    try:
        from safetensors.torch import safe_open
        import torch
    except Exception as e:
        print("ERROR: 需要安装 safetensors 和 torch：pip install safetensors torch", file=sys.stderr)
        raise e

    # 默认筛选规则：常见 norm weight 命名
    default_re = re.compile(r".*(rmsnorm|layernorm|norm).*\.weight$", re.IGNORECASE)

    def key_selected(k: str) -> bool:
        # 排除 QK-Norm（q_norm/k_norm）- 这些 norm 无法融合到权重中
        if getattr(args, 'exclude_qk_norm', False) and ('q_norm' in k or 'k_norm' in k):
            return False

        if args.only is not None:
            return k == args.only
        if args.pattern is not None:
            # 支持子串或正则：若 pattern 看起来像正则就用 re.search，否则也能当子串用
            try:
                return re.search(args.pattern, k) is not None
            except re.error:
                return args.pattern in k
        return default_re.match(k) is not None

    selected = []
    failed = []

    with safe_open(args.safetensors_path, framework="pt") as f:
        keys = list(f.keys())

        for k in keys:
            if not key_selected(k):
                continue

            selected.append(k)
            t = f.get_tensor(k)  # torch tensor
            original_dtype = t.dtype

            # 统一转 float32 做比较（bfloat16/float16 等都能转）
            tf = t.float().numpy()
            ones = np.ones_like(tf, dtype=np.float32)

            ok = np.allclose(tf, ones, rtol=args.rtol, atol=args.atol)
            if not ok:
                # 统计偏差
                abs_err = np.abs(tf - 1.0)
                max_err = float(abs_err.max()) if abs_err.size else 0.0
                mean_err = float(abs_err.mean()) if abs_err.size else 0.0
                # 取一些样本值
                sample = tf.reshape(-1)[:8].tolist() if tf.size else []
                failed.append((k, original_dtype, t.shape, max_err, mean_err, sample))

    if not selected:
        print("WARN: 未匹配到任何 norm.weight key。你可以用 --pattern 指定匹配规则，或用 --only 指定具体 key。")
        print("      例如：--pattern 'input_layernorm.weight'  或  --only 'model.layers.0.input_layernorm.weight'")
        return

    print(f"Checked file: {args.safetensors_path}")
    print(f"Matched keys: {len(selected)}")

    if not failed:
        print("PASS: 所有匹配到的 norm.weight 都是 1（在给定容差内）。")
        return

    print(f"FAIL: {len(failed)} 个 key 的 norm.weight 不是全 1（显示前 {min(args.max_report, len(failed))} 个）：")
    for i, (k, dtype, shape, max_err, mean_err, sample) in enumerate(failed[:args.max_report], 1):
        print(f"[{i}] {k}")
        print(f"    dtype={dtype}, shape={shape}")
        print(f"    max|x-1|={max_err:.6g}, mean|x-1|={mean_err:.6g}")
        print(f"    sample(first 8)={sample}")

    # 非 0 退出码方便 CI
    sys.exit(2)

if __name__ == "__main__":
    main()
