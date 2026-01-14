# Pangu 7B 量化脚本使用指南

本目录包含针对 Pangu 7B 模型（OpenPangu-Embedded-7B）的量化脚本。这些脚本基于 `msmodelslim` 开发，专门用于在昇腾 (Ascend) NPU 上进行 W8A8 和 W8A16 量化。

## 文件列表

- `quant_pangu_w8a8.py`: Pangu 7B W8A8 量化脚本。
- `quant_pangu_w8a16.py`: Pangu 7B W8A16 量化脚本。

## 功能特性

- **完整文件迁移**：脚本会自动将原始权重目录中的所有非权重文件（如 `.py` 建模文件、`tokenizer` 相关文件等）复制到目标目录，确保量化后的模型目录结构完整，可直接用于推理。
- **自动层选择**：默认跳过 `down_proj` 层以保持精度，这是针对类 Llama/Qwen 架构的优化策略。
- **离群值抑制 (Anti-Outlier)**：集成了 `msmodelslim` 的离群值平滑处理，提升量化后的精度。

## 环境要求

- 昇腾 NPU 环境（如 910B）
- 已安装 `msmodelslim`
- 已安装 `torch` 和 `torch_npu`

## 使用方法
```bash
# W8A8Dynamic量化
python3 quant_pangu.py \
    --model_path /path/to/openPangu-Embedded-7B-V1.1 \
    --save_directory /path/to/openPangu-Embedded-7B-V1.1-W8A8D \
    --is_dynamic True \
    --w_bit 8 \
    --a_bit 8
```

```bash
# W8A16量化
python3 quant_pangu.py \
    --model_path /path/to/openPangu-Embedded-7B-V1.1 \
    --save_directory /path/to/openPangu-Embedded-7B-V1.1-W8A16 \
    --w_bit 8 \
    --a_bit 16
```

## 关键参数说明

- `--model_path`: 原始 Pangu 7B 模型的本地目录。
- `--save_directory`: 量化后模型的保存路径。
- `--calib_file`: 校准数据集路径。通常使用 `.jsonl` 格式。
- `--device_type`: 必须设置为 `npu` 以进行实际量化和离群值分析。
- `--trust_remote_code`: 脚本内部默认开启 `True`，以支持 Pangu 的自定义架构文件。

## 注意事项

- 量化过程中会进行离群值分析，建议在 NPU 上运行以保证速度。
- 脚本会自动修改 `config.json` 中的 `quantization_config` 和 `torch_dtype`，并生成 `quant_model_description.json` 供 vLLM-Ascend 推理时使用。
