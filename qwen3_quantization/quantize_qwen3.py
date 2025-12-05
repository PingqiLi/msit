# -*- coding: utf-8 -*-
import sys
import os
import torch

# 确保 msmodelslim 在 PYTHONPATH 中
# 如果脚本在 msit/qwen3_quantization 下，我们需要将 msit/msmodelslim 加入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
msmodelslim_path = os.path.abspath(os.path.join(current_dir, "../msmodelslim"))
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
    # 模型路径 (请修改为您实际的模型路径)
    model_path = "/workspace/weights/Qwen3-30B"
    # 输出路径
    save_path = "./qwen3_w4a4_output"
    
    # 校准数据 (这里使用示例数据，实际请加载真实数据集)
    calib_data = ["Hello world", "This is a test prompt for calibration."] * 10

    print(f"正在加载模型适配器，路径: {model_path}")
    # 初始化 Adapter
    # 使用 Qwen3MoeModelAdapter
    from msmodelslim.model.qwen3_moe.model_adapter import Qwen3MoeModelAdapter
    adapter = Qwen3MoeModelAdapter(model_type="Qwen3-30B", model_path=model_path)

    # ==========================================
    # 2. 量化配置 (W4A4)
    # ==========================================
    # 权重配置: INT4, Per-Group (group_size=128), Symmetric, MinMax
    weight_config = QConfig(
        dtype=QDType.INT4,
        scope=QScope.PER_GROUP,
        symmetric=True,
        method='minmax',
        ext={'group_size': 128}
    )
    
    # 激活配置: INT4, Per-Token, Symmetric, MinMax
    act_config = QConfig(
        dtype=QDType.INT4,
        scope=QScope.PER_TOKEN,
        symmetric=True,
        method='minmax'
    )
    
    linear_qconfig = LinearQConfig(weight=weight_config, act=act_config)

    # ==========================================
    # 3. 算法配置 (LAOS Pipeline: IterSmooth -> Quarot -> IterSmooth -> AutoRound)
    # ==========================================
    
    # 3.1 Iterative Smooth (第一阶段)
    from msmodelslim.quant.processor.anti_outlier import IterSmoothProcessorConfig
    iter_smooth_1 = IterSmoothProcessorConfig(
        alpha=0.9,
        scale_min=1e-5,
        symmetric=False,
        enable_subgraph_type=["ov", "up-down"]
    )

    # 3.2 Quarot 旋转 (处理异常值)
    quarot_config = QuaRotProcessorConfig(
        online=True,
        block_size=32,
        max_tp_size=4,
        down_proj_online_layers=[1,3,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
    )

    # 3.3 Iterative Smooth (第二阶段)
    iter_smooth_2 = IterSmoothProcessorConfig(
        alpha=0.9,
        scale_min=1e-5,
        symmetric=False,
        enable_subgraph_type=["norm-linear"]
    )
    
    # 3.4 AutoRound 量化 (优化权重)
    autoround_config = AutoroundProcessorConfig(
        iters=400,
        enable_minmax_tuning=True,
        enable_round_tuning=True,
        strategies=[
            QuantStrategyConfig(
                qconfig=linear_qconfig,
                include=["*"]  # 应用于所有层
            )
        ]
    )

    # ==========================================
    # 4. 执行量化
    # ==========================================
    print("初始化 Runner...")
    # backend='hccl' 用于 NPU 分布式环境，单卡也可使用
    runner = DPLayerWiseRunner(adapter=adapter, backend='hccl')
    
    print("添加量化处理器...")
    # 注意顺序：IterSmooth -> Quarot -> IterSmooth -> AutoRound
    runner.add_processor(iter_smooth_1)
    runner.add_processor(quarot_config)
    runner.add_processor(iter_smooth_2)
    runner.add_processor(autoround_config)
    
    print("开始运行量化流程...")
    # device_indices=[0] 指定使用第 0 号卡
    runner.run(calib_data=calib_data, device_indices=[0])
    
    print("量化完成！")
    
    # ==========================================
    # 5. 保存模型 (可选)
    # ==========================================
    # runner.run() 可能会自动保存，或者你可以手动保存 adapter.model
    # adapter.model.save_pretrained(save_path)
    # adapter.tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    main()
