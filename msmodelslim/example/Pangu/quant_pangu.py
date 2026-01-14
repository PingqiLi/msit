# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
import os
import argparse
import sys
import torch
import torch.nn.functional as F
import shutil
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..', ".."))
sys.path.append(parent_directory)

from example.common.utils import cmd_bool, SafeGenerator
from example.common.security.path import get_valid_read_path, get_write_directory
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig

import transformers.modeling_attn_mask_utils as masking_utils
# Patch for transformers compatibility: older model code may look for transformers.masking_utils
# which was moved to transformers.modeling_attn_mask_utils in newer versions.
try:
    import transformers.modeling_attn_mask_utils as masking_utils
    # Handle both correct name and potential typos in model code
    if not hasattr(masking_utils, "create_causal_mask"):
        def create_causal_mask(config, input_embeds, attention_mask, cache_position, past_key_values, position_ids=None):
            from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
            return _prepare_4d_causal_attention_mask(
                attention_mask,
                (input_embeds.shape[0], input_embeds.shape[1]),
                input_embeds,
                past_key_values_length=past_key_values.get_seq_length() if past_key_values is not None else 0,
                sliding_window=getattr(config, "sliding_window", None),
            )
        masking_utils.create_causal_mask = create_causal_mask
    
    # Always ensure create_casul_mask (typo version) exists in the patched module
    if not hasattr(masking_utils, "create_casul_mask"):
        masking_utils.create_casul_mask = masking_utils.create_causal_mask
    
    sys.modules["transformers.masking_utils"] = masking_utils
except (ImportError, AttributeError):
    pass

CPU = "cpu"
NPU = "npu"


def get_pangu_disable_names(num_layers: int) -> list:
    disable_names = []
    # Add sensitive layers here if needed, e.g., for i in range(num_layers): disable_names.append(f"model.layers.{i}.mlp.down_proj")
    return disable_names


def copy_all_files(model_dir, dest_dir):
    """Copy all files except weight files to the destination directory."""
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, mode=0o750, exist_ok=True)
    
    # Weight file extensions to skip
    weight_exts = (".bin", ".safetensors", ".pt", ".pth", ".ckpt", ".md")
    black_list = ["model.safetensors.index.json"]
    
    for filename in os.listdir(model_dir):
        src_path = os.path.join(model_dir, filename)
        if os.path.isfile(src_path):
            # Skip weight files
            if filename.endswith(weight_exts) or filename in black_list:
                continue
            # Skip files that will be generated/modified
            if filename in ["config.json", "quant_model_description.json"]:
                continue
                
            dest_path = os.path.join(dest_dir, filename)
            if not os.path.exists(dest_path):
                shutil.copy2(src_path, dest_path)
                os.chmod(dest_path, 0o640)
        elif os.path.isdir(src_path):
            if filename in ["__pycache__", ".git"]:
                continue
            dest_path = os.path.join(dest_dir, filename)
            if not os.path.exists(dest_path):
                shutil.copytree(src_path, dest_path, dirs_exist_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="model and tokenizer path")
    parser.add_argument('--save_directory', type=str, required=True, help="directory to save quantized model")
    parser.add_argument('--mindie_format', action="store_true", help="Compatible with quantization formats required by MindIE")
    parser.add_argument('--calib_file', type=str, default='../common/teacher_qualification.jsonl', help="calibration data file")
    parser.add_argument('--w_bit', type=int, default=8)
    parser.add_argument('--a_bit', type=int, default=8)
    parser.add_argument('--use_kvcache_quant', type=cmd_bool, default=False)
    parser.add_argument('--device_type', type=str, choices=[CPU, NPU], default=NPU)
    parser.add_argument('--trust_remote_code', type=cmd_bool, default=True)
    parser.add_argument('--anti_method', type=str, default='m3', help="Optional anti-outlier method (e.g., m3)")
    parser.add_argument('--act_method', type=int, default=1, help="1: MinMax, 2: Histogram, 3: Auto")
    parser.add_argument('--open_outlier', type=cmd_bool, default=True)
    parser.add_argument('--is_dynamic', type=cmd_bool, default=False)
    parser.add_argument('--is_lowbit', type=cmd_bool, default=False)
    parser.add_argument('--group_size', type=int, default=64)
    parser.add_argument('--fraction', type=float, default=0.01)
    parser.add_argument('--co_sparse', type=cmd_bool, default=False)
    parser.add_argument('--disable_level', type=str, default='L0')
    parser.add_argument('--do_smooth', type=cmd_bool, default=False)
    parser.add_argument('--use_sigma', type=cmd_bool, default=False)
    parser.add_argument('--use_reduce_quant', type=cmd_bool, default=False)
    parser.add_argument('--sigma_factor', type=float, default=3.0)
    parser.add_argument('--w_sym', type=cmd_bool, default=True)
    parser.add_argument('--use_fa_quant', type=cmd_bool, default=False)
    parser.add_argument('--pdmix', type=cmd_bool, default=False)
    parser.add_argument('--disable_last_linear', type=cmd_bool, default=True)
    parser.add_argument('--w_method', type=str, default='min_max')
    parser.add_argument('--part_file_size', type=int, default=None)
    args = parser.parse_args()

    # Check paths
    args.model_path = get_valid_read_path(args.model_path, is_dir=True, check_user_stat=True)
    args.save_directory = get_write_directory(args.save_directory, write_mode=0o750)

    # 1. 加载模型
    # 使用 Auto 系列工具加载，它们能正确处理模型文件内的相对导入
    sys.path.insert(0, args.model_path)
    
    device_map = CPU if args.device_type == CPU else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
        torch_dtype="auto",
        local_files_only=True
    ).eval()

    config = AutoConfig.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
        use_fast=False
    )

    # 2. 设置回退层
    disable_names = get_pangu_disable_names(config.num_hidden_layers)

    # 3. 加载校准集
    checker = SafeGenerator()
    calib_texts = checker.load_jsonl(args.calib_file)
    calib_data = []
    # 使用少量数据进行校准
    for text in calib_texts[:32]:
        inputs = tokenizer(text, return_tensors='pt', padding=True).to(args.device_type)
        calib_data.append([inputs.data['input_ids'], inputs.data['attention_mask']])

    # 4. 异常值抑制 (参照 rerference_pangu_vl.py)
    anti_config = AntiOutlierConfig(
        w_bit=args.w_bit,
        a_bit=args.a_bit,
        anti_method=args.anti_method,
        dev_type=args.device_type,
        dev_id=0 if args.device_type == CPU else model.device.index,
        disable_anti_names=disable_names
    )
    anti_outlier = AntiOutlier(model, calib_data=calib_data, cfg=anti_config)
    anti_outlier.process()

    # 5. 模型量化
    quant_config = QuantConfig(
        w_bit=args.w_bit,
        a_bit=args.a_bit,
        w_sym=True,
        use_kvcache_quant=args.use_kvcache_quant,  # 根据要求禁用 KV cache
        disable_names=disable_names,
        dev_type=args.device_type,
        dev_id=0 if args.device_type == CPU else model.device.index,
        act_method=args.act_method,
        mm_tensor=False,
        open_outlier=args.open_outlier,
        is_dynamic=args.is_dynamic,
        is_lowbit=args.is_lowbit,
        group_size=args.group_size
    )
    calibrator = Calibrator(model, quant_config, calib_data=calib_data, disable_level='L0')
    calibrator.run()

    # 6. 保存权重
    save_type = "safe_tensor" if args.mindie_format else "ascendV1"
    calibrator.save(args.save_directory, save_type=[save_type], part_file_size=args.part_file_size)

    # 7. 保存其他配置文件
    quant_type = quant_config.model_quant_type.lower()
    auto_config = checker.get_config_from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    checker.modify_config(args.model_path, args.save_directory, auto_config.torch_dtype, quant_type, args)
    
    # 复制所有外部文件（建模代码等）
    copy_all_files(args.model_path, args.save_directory)
