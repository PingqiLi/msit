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

from msmodelslim.core import QDType, QScope
from msmodelslim.core.runner.dp_layer_wise_runner import DPLayerWiseRunner
from msmodelslim.model.qwen3.model_adapter import Qwen3ModelAdapter
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
    # model_type="qwen3" 确保加载正确的配置
    adapter = Qwen3ModelAdapter(model_type="qwen3", model_path=model_path)

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
    # 3. 算法配置 (Quarot + AutoRound)
    # ==========================================
    
    # 配置 Quarot (旋转)
    # online=True: 开启在线旋转 (对 W4A4 必须)
    # max_tp_size=1: 单卡推理设为 1。如果是多卡 TP，请设为相应的 TP 数 (如 2, 4, 8)
    quarot_config = QuaRotProcessorConfig(
        online=True,
        block_size=-1,
        max_tp_size=1 
    )
    
    # 配置 AutoRound (自适应舍入)
    # iters: 迭代次数，推荐 200
    autoround_config = AutoroundProcessorConfig(
        iters=200,
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
    # 注意顺序：先 Quarot 旋转，再 AutoRound 量化
    runner.add_processor(quarot_config)
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
