#!/usr/bin/env python3
"""Plot per-layer adaptive high-precision ratios from a ResQ quantization log.

Reads a log file produced by resq_qwen3_32b.py and plots per-layer ratios
for each projection (Ua, Ub, Uc, Ud) as a line chart.

Usage:
    python plot_adaptive_ratios.py --log_file /path/to/quantization.log
    python plot_adaptive_ratios.py --log_file /path/to/quantization.log --output ratios.png
    python plot_adaptive_ratios.py --json_file /path/to/resq_adaptive_ratios.json
"""
import argparse
import json
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt


def parse_ratios_from_log(log_path):
    """Parse adaptive ratio lines from a log file.

    Expected log format (output by ResQCalibrator):
        Computed adaptive ratios:
          Ua: 0.1250
          layer.0.Ub: 0.1200
          layer.0.Uc: 0.1100
          layer.0.Ud: 0.1050
          layer.1.Ub: 0.1300
          ...

    Returns:
        dict: {proj_name: {layer_idx: ratio}} e.g. {"Ub": {0: 0.12, 1: 0.13, ...}}
        Also returns Ua ratio separately (shared across layers).
    """
    # Pattern: "layer.{i}.{proj}: {ratio}" or "Ua: {ratio}"
    layer_pattern = re.compile(r'layer\.(\d+)\.(\w+):\s*([0-9.]+)')
    ua_pattern = re.compile(r'(?<!\w)Ua:\s*([0-9.]+)')

    ratios = defaultdict(dict)  # proj_name -> {layer_idx: ratio}
    ua_ratio = None

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Check for per-layer ratio
            m = layer_pattern.search(line)
            if m:
                layer_idx = int(m.group(1))
                proj = m.group(2)
                ratio = float(m.group(3))
                ratios[proj][layer_idx] = ratio
                continue

            # Check for Ua (shared, not per-layer)
            m = ua_pattern.search(line)
            if m:
                ua_ratio = float(m.group(1))

    return dict(ratios), ua_ratio


def parse_ratios_from_json(json_path):
    """Parse adaptive ratios from the saved JSON file.

    Expected JSON format (output by save_adaptive_ratios):
        {
          "per_layer_ratios": {
            "Ua": 0.125,
            "layer.0.Ub": 0.12,
            "layer.0.Uc": 0.11,
            ...
          },
          ...
        }

    Returns:
        Same format as parse_ratios_from_log.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    raw_ratios = data.get('per_layer_ratios', data)

    layer_pattern = re.compile(r'^layer\.(\d+)\.(\w+)$')
    ratios = defaultdict(dict)
    ua_ratio = None

    for key, value in raw_ratios.items():
        m = layer_pattern.match(key)
        if m:
            layer_idx = int(m.group(1))
            proj = m.group(2)
            ratios[proj][layer_idx] = float(value)
        elif key == 'Ua':
            ua_ratio = float(value)

    return dict(ratios), ua_ratio


def plot_ratios(ratios, ua_ratio, output_path=None, title=None):
    """Plot per-layer ratios as a line chart.

    Args:
        ratios: {proj_name: {layer_idx: ratio}}
        ua_ratio: Ua ratio (shared, drawn as horizontal line) or None
        output_path: Save figure to this path (show interactively if None)
        title: Custom title
    """
    if not ratios and ua_ratio is None:
        print("No ratio data found.", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(14, 5))

    # Color and marker settings per projection
    style_map = {
        'Ub': dict(color='#e74c3c', marker='o', label='Ub (V proj)'),
        'Uc': dict(color='#3498db', marker='s', label='Uc (K proj)'),
        'Ud': dict(color='#2ecc71', marker='^', label='Ud (down_proj)'),
    }

    # Plot per-layer projections
    for proj in sorted(ratios.keys()):
        layer_data = ratios[proj]
        layers = sorted(layer_data.keys())
        values = [layer_data[l] for l in layers]

        style = style_map.get(proj, dict(marker='D', label=proj))
        ax.plot(layers, values, linewidth=1.5, markersize=3, **style)

    # Plot Ua as horizontal dashed line (shared across all layers)
    if ua_ratio is not None:
        all_layers = set()
        for proj_data in ratios.values():
            all_layers.update(proj_data.keys())
        if all_layers:
            x_min, x_max = min(all_layers), max(all_layers)
            ax.hlines(ua_ratio, x_min, x_max, colors='#9b59b6',
                      linestyles='dashed', linewidth=1.5, label=f'Ua (shared) = {ua_ratio:.4f}')

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('High-precision Ratio', fontsize=12)
    ax.set_title(title or 'Per-layer Adaptive High-precision Ratio', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Integer x-ticks
    all_layers = set()
    for proj_data in ratios.values():
        all_layers.update(proj_data.keys())
    if all_layers:
        n_layers = max(all_layers) + 1
        if n_layers <= 32:
            ax.set_xticks(range(n_layers))
        else:
            ax.set_xticks(range(0, n_layers, max(1, n_layers // 16)))

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Plot per-layer adaptive ratios from ResQ quantization log or JSON')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--log_file', type=str, help='Path to quantization log file')
    group.add_argument('--json_file', type=str, help='Path to resq_adaptive_ratios.json')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output image path (e.g., ratios.png). Shows interactively if omitted.')
    parser.add_argument('--title', type=str, default=None, help='Custom chart title')
    args = parser.parse_args()

    if args.log_file:
        ratios, ua_ratio = parse_ratios_from_log(args.log_file)
    else:
        ratios, ua_ratio = parse_ratios_from_json(args.json_file)

    # Summary
    for proj in sorted(ratios.keys()):
        vals = list(ratios[proj].values())
        print(f"{proj}: {len(vals)} layers, range [{min(vals):.4f}, {max(vals):.4f}]")
    if ua_ratio is not None:
        print(f"Ua (shared): {ua_ratio:.4f}")

    plot_ratios(ratios, ua_ratio, output_path=args.output, title=args.title)


if __name__ == '__main__':
    main()
