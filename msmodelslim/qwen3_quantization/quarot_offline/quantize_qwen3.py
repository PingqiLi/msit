# -*- coding: utf-8 -*-
import sys
import os
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
msmodelslim_path = os.path.abspath(os.path.join(current_dir, "../../"))
if msmodelslim_path not in sys.path:
    sys.path.insert(0, msmodelslim_path)

from msmodelslim.core.QAL import QDType, QScope
from msmodelslim.core.runner.dp_layer_wise_runner import DPLayerWiseRunner

from msmodelslim.quant.processor.quant.autoround import AutoroundProcessorConfig, QuantStrategyConfig
from msmodelslim.quant.processor.quarot import QuaRotProcessorConfig
from msmodelslim.quant.quantizer.base import QConfig
from msmodelslim.quant.quantizer.linear import LinearQConfig

def main():
    # ==========================================
    # 1. 基础配置
    # ==========================================
    # 模型路径
    model_path = "/workspace/weights/Qwen3-30B"
    # 输出路径
    save_path = "/workspace/weights/Qwen3-30B-W4A4-OfflineQuaRot"
    
    # 校准数据路径
    calib_path = os.path.join(msmodelslim_path, "lab_calib/mix_calib.jsonl")
    if not os.path.exists(calib_path):
        print(f"Warning: Calibration file not found at {calib_path}, using dummy data.")
        calib_data = ["Hello world"] * 10
    else:
        # 简单读取jsonl文件的一列作为校准数据，这里假设是 list of strings format
        # 如果是复杂jsonl，需根据实际Key修改读取逻辑
        import json
        calib_data = []
        with open(calib_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    # 尝试常见的key
                    text = item.get('text') or item.get('content') or item.get('input')
                    if text:
                        calib_data.append(text)
                except:
                    pass
        # 限制校准数据量
        calib_data = calib_data[:128]

    print(f"正在加载模型适配器，路径: {model_path}")
    from msmodelslim.model.qwen3_moe.model_adapter import Qwen3MoeModelAdapter
    adapter = Qwen3MoeModelAdapter(model_type="Qwen3-30B", model_path=model_path)

    # ==========================================
    # 2. 量化配置定义
    # ==========================================
    
    # W4A4 Config (Default for Experts)
    w4a4_config = LinearQConfig(
        weight=QConfig(dtype=QDType.INT4, scope=QScope.PER_GROUP, symmetric=True, method='minmax', ext={'group_size': 128}),
        act=QConfig(dtype=QDType.INT4, scope=QScope.PER_TOKEN, symmetric=True, method='minmax')
    )
    
    # W8A8 Config (For Attention and Last Experts)
    w8a8_config = LinearQConfig(
        weight=QConfig(dtype=QDType.INT8, scope=QScope.PER_CHANNEL, symmetric=True, method='minmax'),
        act=QConfig(dtype=QDType.INT8, scope=QScope.PER_TOKEN, symmetric=True, method='minmax')
    )
    
    # Float Config (For MoE Gate) - Use Float/BF16
    float_config = LinearQConfig(
        weight=QConfig(dtype=QDType.FLOAT, scope=QScope.PER_TENSOR, symmetric=True, method='minmax'),
        act=QConfig(dtype=QDType.FLOAT, scope=QScope.PER_TENSOR, symmetric=True, method='minmax')
    )

    # ==========================================
    # 3. 策略配置 (Layers Strategy)
    # ==========================================
    strategies = []
    
    # 1. 默认策略: Experts 使用 W4A4 (除了最后两层)
    strategies.append(QuantStrategyConfig(qconfig=w4a4_config, include=["*"]))

    # 2. Attention层: W8A8
    strategies.append(QuantStrategyConfig(
        qconfig=w8a8_config, 
        include=["*.self_attn"] # 匹配所有 self_attn 模块
    ))
    
    # 3. MoE Gate: Float (BF16)
    strategies.append(QuantStrategyConfig(
        qconfig=float_config, 
        include=["*.mlp.gate"]
    ))
    
    # 4. 最后两层 Experts (Layer 46, 47): W8A8
    strategies.append(QuantStrategyConfig(
        qconfig=w8a8_config,
        include=[
            "*layers.46.mlp.experts", 
            "*layers.47.mlp.experts"
        ]
    ))

    # ==========================================
    # 4. 算法流程配置
    # ==========================================
    
    # 3.1 Iterative Smooth (1)
    from msmodelslim.quant.processor.anti_outlier import IterSmoothProcessorConfig
    iter_smooth_1 = IterSmoothProcessorConfig(
        alpha=0.9, scale_min=1e-5, symmetric=False,
        enable_subgraph_type=["ov", "up-down"]
    )

    # 3.2 Quarot (Online -> Offline)
    quarot_config = QuaRotProcessorConfig(
        online=False, block_size=-1, max_tp_size=4,
        down_proj_online_layers=[]
    )

    # 3.3 Iterative Smooth (2)
    iter_smooth_2 = IterSmoothProcessorConfig(
        alpha=0.9, scale_min=1e-5, symmetric=False,
        enable_subgraph_type=["norm-linear"]
    )
    
    # 3.4 AutoRound
    autoround_config = AutoroundProcessorConfig(
        iters=2,
        enable_minmax_tuning=True,
        enable_round_tuning=True,
        strategies=strategies # 使用自定义策略
    )

    # ==========================================
    # 5. 执行量化
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

if __name__ == "__main__":
    main()
