# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
Script to inspect saved ResQ quantized weights and compare to original model.

Usage:
    python inspect_quant_weights.py --quant_path /path/to/quant_model_weight_resq.safetensors \
                                    --model_path /path/to/original_model \
                                    [--layer_idx 0] [--verbose]
"""

import argparse
import os
from typing import Dict, Optional

import torch
from safetensors import safe_open


def load_safetensors(path: str) -> Dict[str, torch.Tensor]:
    """Load tensors from safetensors file or directory."""
    tensors = {}

    if os.path.isdir(path):
        # Load all safetensors files in directory
        files = sorted([f for f in os.listdir(path) if f.endswith('.safetensors')])
        if not files:
            raise ValueError(f"No .safetensors files found in directory: {path}")

        for filename in files:
            filepath = os.path.join(path, filename)
            with safe_open(filepath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
    else:
        # Load single file
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    return tensors


def print_tensor_info(name: str, tensor: torch.Tensor, num_values: int = 5):
    """Print information about a tensor."""
    print(f"  {name}:")
    print(f"    dtype: {tensor.dtype}")
    print(f"    shape: {list(tensor.shape)}")
    print(f"    min: {tensor.min().item():.6f}, max: {tensor.max().item():.6f}")

    # Print some sample values
    flat = tensor.flatten()
    if len(flat) > num_values:
        sample_indices = torch.linspace(0, len(flat) - 1, num_values).long()
        sample_values = flat[sample_indices]
    else:
        sample_values = flat
    print(f"    sample values: {[f'{v.item():.6f}' for v in sample_values]}")


def dequantize_weight(weight_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize int8 weight using scale (symmetric quantization)."""
    return weight_int8.float() * scale


def compute_error_metrics(original: torch.Tensor, reconstructed: torch.Tensor):
    """Compute error metrics between original and reconstructed tensors."""
    diff = original - reconstructed

    # Mean Squared Error
    mse = (diff ** 2).mean().item()

    # Max Absolute Error
    max_abs_error = diff.abs().max().item()

    # Relative Error (normalized by original weight magnitude)
    rel_error = (diff.abs() / (original.abs() + 1e-8)).mean().item()

    # Signal-to-Noise Ratio (in dB)
    signal_power = (original ** 2).mean().item()
    noise_power = (diff ** 2).mean().item()
    snr_db = 10 * torch.log10(torch.tensor(signal_power / (noise_power + 1e-10))).item()

    return {
        'mse': mse,
        'rmse': mse ** 0.5,
        'max_abs_error': max_abs_error,
        'mean_rel_error': rel_error,
        'snr_db': snr_db,
    }


def inspect_quant_weights(
    quant_path: str,
    model_path: Optional[str] = None,
    layer_idx: int = 0,
    verbose: bool = False,
):
    """
    Inspect quantized weights and optionally compare to original model.

    Args:
        quant_path: Path to quantized safetensors file
        model_path: Path to original HuggingFace model (optional)
        layer_idx: Which layer to inspect in detail
        verbose: Print all tensors (not just summary)
    """
    print("=" * 80)
    print("ResQ Quantized Weights Inspector")
    print("=" * 80)
    print(f"Loading quantized weights from: {quant_path}")

    # Load quantized weights
    quant_tensors = load_safetensors(quant_path)

    print(f"\nTotal tensors: {len(quant_tensors)}")

    # Categorize tensors
    weight_low = {}
    weight_high = {}
    scale_low = {}
    scale_high = {}
    rotation_matrices = {}
    other_tensors = {}

    for name, tensor in quant_tensors.items():
        if '.weight_low' in name:
            weight_low[name] = tensor
        elif '.weight_high' in name:
            weight_high[name] = tensor
        elif '.scale_low' in name:
            scale_low[name] = tensor
        elif '.scale_high' in name:
            scale_high[name] = tensor
        elif name.startswith('resq.'):
            rotation_matrices[name] = tensor
        else:
            other_tensors[name] = tensor

    print(f"\n--- Tensor Categories ---")
    print(f"  weight_low tensors: {len(weight_low)}")
    print(f"  weight_high tensors: {len(weight_high)}")
    print(f"  scale_low tensors: {len(scale_low)}")
    print(f"  scale_high tensors: {len(scale_high)}")
    print(f"  rotation matrices: {len(rotation_matrices)}")
    print(f"  other tensors: {len(other_tensors)}")

    # Print summary of weight dtypes
    if weight_low:
        first_low = next(iter(weight_low.values()))
        print(f"\n--- Weight Storage ---")
        print(f"  weight_low dtype: {first_low.dtype} (expected: int8 for 4-bit)")
    if weight_high:
        first_high = next(iter(weight_high.values()))
        print(f"  weight_high dtype: {first_high.dtype} (expected: int8 for 8-bit)")

    # Print rotation matrices info
    if rotation_matrices:
        print(f"\n--- Rotation Matrices ---")
        for name, tensor in sorted(rotation_matrices.items()):
            print(f"  {name}: shape={list(tensor.shape)}, dtype={tensor.dtype}")

    # Print other tensors (embed, norm, etc.)
    if other_tensors and verbose:
        print(f"\n--- Other Tensors ---")
        for name, tensor in sorted(other_tensors.items()):
            print(f"  {name}: shape={list(tensor.shape)}, dtype={tensor.dtype}")

    # Detailed inspection of specified layer
    print(f"\n{'=' * 80}")
    print(f"Detailed Inspection: Layer {layer_idx}")
    print("=" * 80)

    layer_prefix = f"model.layers.{layer_idx}."
    projections = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                   'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']

    for proj in projections:
        full_prefix = layer_prefix + proj

        w_low_key = f"{full_prefix}.weight_low"
        w_high_key = f"{full_prefix}.weight_high"
        s_low_key = f"{full_prefix}.scale_low"
        s_high_key = f"{full_prefix}.scale_high"

        if w_low_key in quant_tensors or w_high_key in quant_tensors:
            print(f"\n--- {proj} ---")

            if w_low_key in quant_tensors:
                w_low = quant_tensors[w_low_key]
                s_low = quant_tensors.get(s_low_key)
                print(f"  Low precision (4-bit):")
                print(f"    weight shape: {list(w_low.shape)}, dtype: {w_low.dtype}")
                print(f"    weight range: [{w_low.min().item()}, {w_low.max().item()}] (expected: [-8, 7] for 4-bit)")
                if s_low is not None:
                    print(f"    scale shape: {list(s_low.shape)}, dtype: {s_low.dtype}")
                    print(f"    scale range: [{s_low.min().item():.6f}, {s_low.max().item():.6f}]")

            if w_high_key in quant_tensors:
                w_high = quant_tensors[w_high_key]
                s_high = quant_tensors.get(s_high_key)
                print(f"  High precision (8-bit):")
                print(f"    weight shape: {list(w_high.shape)}, dtype: {w_high.dtype}")
                print(f"    weight range: [{w_high.min().item()}, {w_high.max().item()}] (expected: [-128, 127] for 8-bit)")
                if s_high is not None:
                    print(f"    scale shape: {list(s_high.shape)}, dtype: {s_high.dtype}")
                    print(f"    scale range: [{s_high.min().item():.6f}, {s_high.max().item():.6f}]")

            # Compute total dimensions
            total_cols = 0
            if w_low_key in quant_tensors:
                total_cols += quant_tensors[w_low_key].shape[1]
            if w_high_key in quant_tensors:
                total_cols += quant_tensors[w_high_key].shape[1]

            low_cols = quant_tensors[w_low_key].shape[1] if w_low_key in quant_tensors else 0
            high_cols = quant_tensors[w_high_key].shape[1] if w_high_key in quant_tensors else 0

            print(f"  Split: {low_cols} (4-bit) + {high_cols} (8-bit) = {total_cols} total")
            if total_cols > 0:
                print(f"  High fraction: {high_cols / total_cols:.3f}")

    # Compare with original model if provided
    if model_path:
        print(f"\n{'=' * 80}")
        print("Comparison with Original Model")
        print("=" * 80)
        print(f"Loading original model from: {model_path}")

        try:
            from transformers import AutoModelForCausalLM

            original_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
            )

            print("\nComparing layer 0 projections...")

            for proj in projections:
                full_name = f"model.layers.{layer_idx}.{proj}"

                # Get original weight
                try:
                    parts = proj.split('.')
                    module = original_model.model.layers[layer_idx]
                    for part in parts:
                        module = getattr(module, part)
                    original_weight = module.weight.data.float()
                except Exception as e:
                    print(f"  {proj}: Could not get original weight: {e}")
                    continue

                # Get quantized weights and reconstruct
                w_low_key = f"model.layers.{layer_idx}.{proj}.weight_low"
                w_high_key = f"model.layers.{layer_idx}.{proj}.weight_high"
                s_low_key = f"model.layers.{layer_idx}.{proj}.scale_low"
                s_high_key = f"model.layers.{layer_idx}.{proj}.scale_high"

                if w_low_key not in quant_tensors and w_high_key not in quant_tensors:
                    print(f"  {proj}: No quantized weights found")
                    continue

                # Reconstruct full weight
                reconstructed_parts = []

                if w_low_key in quant_tensors:
                    w_low = quant_tensors[w_low_key]
                    s_low = quant_tensors[s_low_key]
                    recon_low = dequantize_weight(w_low, s_low)
                    reconstructed_parts.append(recon_low)

                if w_high_key in quant_tensors:
                    w_high = quant_tensors[w_high_key]
                    s_high = quant_tensors[s_high_key]
                    recon_high = dequantize_weight(w_high, s_high)
                    reconstructed_parts.append(recon_high)

                if len(reconstructed_parts) == 2:
                    # Concatenate along columns (dim=1)
                    reconstructed = torch.cat(reconstructed_parts, dim=1)
                else:
                    reconstructed = reconstructed_parts[0]

                # Check shape match
                if original_weight.shape != reconstructed.shape:
                    print(f"  {proj}: Shape mismatch! Original {list(original_weight.shape)} vs Reconstructed {list(reconstructed.shape)}")
                    print(f"    NOTE: This is expected if rotations were applied to the original weights.")
                    print(f"    The quantized weights include rotation transforms.")
                    continue

                # Compute error metrics
                metrics = compute_error_metrics(original_weight, reconstructed)

                print(f"\n  {proj}:")
                print(f"    Original shape: {list(original_weight.shape)}")
                print(f"    Reconstructed shape: {list(reconstructed.shape)}")
                print(f"    MSE: {metrics['mse']:.2e}")
                print(f"    RMSE: {metrics['rmse']:.6f}")
                print(f"    Max Abs Error: {metrics['max_abs_error']:.6f}")
                print(f"    Mean Rel Error: {metrics['mean_rel_error']:.4f}")
                print(f"    SNR: {metrics['snr_db']:.2f} dB")

                # Print sample comparison
                if verbose:
                    print(f"    Sample values (first 5 elements of first row):")
                    print(f"      Original:     {original_weight[0, :5].tolist()}")
                    print(f"      Reconstructed: {reconstructed[0, :5].tolist()}")

            del original_model

        except ImportError:
            print("Error: transformers library not found. Install with: pip install transformers")
        except Exception as e:
            print(f"Error loading original model: {e}")

    print(f"\n{'=' * 80}")
    print("Inspection Complete")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Inspect ResQ quantized weights")
    parser.add_argument("--quant_path", type=str, required=True,
                        help="Path to quantized safetensors file")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to original HuggingFace model for comparison")
    parser.add_argument("--layer_idx", type=int, default=0,
                        help="Layer index to inspect in detail (default: 0)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information for all tensors")

    args = parser.parse_args()

    if not os.path.exists(args.quant_path):
        print(f"Error: Path not found: {args.quant_path}")
        return

    if os.path.isdir(args.quant_path):
        files = [f for f in os.listdir(args.quant_path) if f.endswith('.safetensors')]
        if not files:
            print(f"Error: No .safetensors files found in directory: {args.quant_path}")
            return

    inspect_quant_weights(
        quant_path=args.quant_path,
        model_path=args.model_path,
        layer_idx=args.layer_idx,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
