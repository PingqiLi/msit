# -*- coding: utf-8 -*-
import os
import sys
import argparse
import torch

# Add sys.path for direct source usage
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..', ".."))
sys.path.append(parent_directory)

from msmodelslim.core.QAL import QDType, QScope
from msmodelslim.core.runner.dp_layer_wise_runner import DPLayerWiseRunner
from msmodelslim.quant.processor import QuaRotProcessorConfig
from msmodelslim.quant.processor.quant.autoround import AutoroundProcessorConfig, QuantStrategyConfig
from msmodelslim.quant.quantizer.base import QConfig
from msmodelslim.quant.quantizer.linear import LinearQConfig
from msmodelslim.utils.logging import set_logger_level

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3 Quantization Script")
    parser.add_argument('--model_path', type=str, default="/workspace/weights/Qwen3-30B",
                        help="Path to the float model")
    parser.add_argument('--save_path', type=str, default="/workspace/weights/Qwen3-30B-W4A4-OfflineQuaRot",
                        help="Path to save the quantized model")
    parser.add_argument('--calib_file', type=str, default=None,
                        help="Path to the calibration dataset file (jsonl)")
    return parser.parse_args()

def get_calib_dataset(calib_path):
    if not os.path.exists(calib_path):
        print(f"Warning: Calibration file not found at {calib_path}, using dummy data.")
        return ["Hello world"] * 128

    calib_data = []
    try:
        import json
        with open(calib_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    text = item.get('text') or item.get('content') or item.get('input')
                    if text:
                        calib_data.append(text)
                except:
                    pass
    except Exception as e:
        print(f"Error loading calibration data: {e}, using dummy data.")

    if not calib_data:
        print("Warning: Calibration data is empty, using dummy data.")
        return ["Hello world"] * 128
        
    return calib_data[:128]

if __name__ == "__main__":
    set_logger_level("info")
    args = parse_args()

    model_path = args.model_path
    save_path = args.save_path

    # Auto-detect calibration file path if not provided
    if args.calib_file:
        calib_path = args.calib_file
    else:
        # Try to locate mix_calib.jsonl relative to this script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Structure: msit/msmodelslim/qwen3_quantization/quarot_offline/script.py
        # Target: msit/msmodelslim/lab_calib/mix_calib.jsonl
        calib_path = os.path.abspath(os.path.join(current_dir, "../../lab_calib/mix_calib.jsonl"))

    print(f"Using calibration file: {calib_path}")
    calib_data = get_calib_dataset(calib_path)

    print(f"正在加载模型适配器，路径: {model_path}")
    from msmodelslim.model.qwen3_moe.model_adapter import Qwen3MoeModelAdapter
    adapter = Qwen3MoeModelAdapter(model_type="Qwen3-30B", model_path=model_path)

    # ==========================================
    # 2. 量化配置定义 (Quantization Config Definitions)
    # ==========================================

    # W4A4 Config (Default for Experts)
    w4a4_config = LinearQConfig(
        weight=QConfig(dtype=QDType.INT4, scope=QScope.PER_GROUP, symmetric=True, method='minmax', ext={'group_size': 128}),
        act=QConfig(dtype=QDType.INT4, scope=QScope.PER_TOKEN, symmetric=True, method='minmax')
    )
    
    # W8A8 Config (For Attention & specified experts)
    w8a8_config = LinearQConfig(
        weight=QConfig(dtype=QDType.INT8, scope=QScope.PER_CHANNEL, symmetric=True, method='minmax'),
        act=QConfig(dtype=QDType.INT8, scope=QScope.PER_TOKEN, symmetric=True, method='minmax')
    )
    
    # Float Config (For MoE Gates)
    float_config = LinearQConfig(
        weight=QConfig(dtype=QDType.FLOAT, scope=QScope.PER_TENSOR, symmetric=True, method='minmax'),
        act=QConfig(dtype=QDType.FLOAT, scope=QScope.PER_TENSOR, symmetric=True, method='minmax')
    )

    # ==========================================
    # 3. 各种 Processor 配置
    # ==========================================
    
    # 3.1 IterSmooth (Pre)
    from msmodelslim.quant.processor.anti_outlier import IterSmoothProcessorConfig
    iter_smooth_1 = IterSmoothProcessorConfig(
        alpha=0.9, scale_min=1e-5, symmetric=False,
        enable_subgraph_type=["ov", "up-down"]
    )

    # 3.2 QuaRot
    quarot_config = QuaRotProcessorConfig(
        online=False
    )

    # 3.3 IterSmooth (Post)
    iter_smooth_2 = IterSmoothProcessorConfig(
        alpha=0.9, scale_min=1e-5, symmetric=False,
        enable_subgraph_type=["norm-linear"]
    )
    
    # 3.4 AutoRound Strategies
    autoround_strategies = []
    
    # 1. Attention层: W8A8
    autoround_strategies.append(QuantStrategyConfig(
        qconfig=w8a8_config, 
        include=["*self_attn*"] 
    ))
    
    # 2. MoE Gate: Float (BF16)
    autoround_strategies.append(QuantStrategyConfig(
        qconfig=float_config, 
        include=["*mlp.gate*"]
    ))
    
    # 3. 最后两层 Experts (Layer 46, 47): W8A8
    autoround_strategies.append(QuantStrategyConfig(
        qconfig=w8a8_config,
        include=[
            "*layers.46.mlp.experts*", 
            "*layers.47.mlp.experts*"
        ]
    ))

    # 4. 默认策略: Experts 使用 W4A4 (除了最后两层)
    # Put this LAST as a fallback for all other layers
    autoround_strategies.append(QuantStrategyConfig(
        qconfig=w4a4_config, 
        include=["*"] 
    ))

    # 3.5 AutoRound Config
    autoround_config = AutoroundProcessorConfig(
        type="autoround_quant",
        iters=400,
        enable_minmax_tuning=True,
        enable_round_tuning=True,
        strategies=autoround_strategies
    )

    # ==========================================
    # 4. 执行量化
    # ==========================================
    runner = DPLayerWiseRunner(adapter=adapter, backend='hccl')
    runner.add_processor(iter_smooth_1)
    runner.add_processor(quarot_config)
    runner.add_processor(iter_smooth_2)
    runner.add_processor(autoround_config)

    # 添加 Saver Processor
    from msmodelslim.app.quant_service.modelslim_v1.save.ascendv1 import AscendV1Config
    save_config = AscendV1Config(
        save_directory=save_path,
        part_file_size=4
    )
    runner.add_processor(save_config)
    
    runner.run(calib_data=calib_data, device_indices=[0])
    
    print("Quantization Finished!")
