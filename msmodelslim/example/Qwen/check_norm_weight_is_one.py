#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_norm_weight_is_one.py

用法：
  python check_norm_weight_is_one.py /path/to/model.safetensors
  python check_norm_weight_is_one.py /path/to/model.safetensors --only model.layers.0.input_layernorm.weight
  python check_norm_weight_is_one.py /path/to/model.safetensors --pattern "input_layernorm.weight"
  python check_norm_weight_is_one.py /path/to/model.safetensors --atol 1e-6 --rtol 1e-6

说明：
  - 会检查所有匹配到的 norm weight key 是否接近全 1
  - 默认模式：匹配包含 norm/layernorm/rmsnorm 且以 .weight 结尾的 key
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
