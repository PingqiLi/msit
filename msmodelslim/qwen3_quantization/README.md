# Qwen3 W4A4 量化工具

本目录包含了针对 Qwen3-A3B-30B 模型进行 W4A4 (4-bit 权重, 4-bit 激活) 量化的脚本和配置文件。
使用了 **Quarot** (旋转) 和 **AutoRound** (自适应舍入) 算法来保证量化精度。

## 目录结构

*   `quantize_qwen3.py`: Python 量化脚本，直接调用 msmodelslim API。
*   `qwen3_w4a4_config.yaml`: 一键量化配置文件，可配合 msmodelslim CLI 使用。

## 使用方法

### 前置条件

确保您已安装 Ascend CANN 软件栈，并且 NPU 环境可用。
本工具依赖同级目录下的 `msmodelslim` 源码。

**重要：因为您可能需要修改 `msmodelslim` 源码，建议按以下步骤安装：**

```bash
cd ../msmodelslim
bash install.sh
cd ../qwen3_quantization
```

*注意：`install.sh` 会从 CANN 路径复制依赖的 `.so` 文件并重新安装 `msmodelslim`。如果您之后修改了源码（例如修复 Bug），请再次运行 `bash install.sh` 或 `pip install .` 以更新系统环境中的包。*

### 方法一：使用 Python 脚本 (推荐)

直接运行 Python 脚本，适合需要自定义逻辑或调试的场景。

1.  打开 `quantize_qwen3.py`，修改 `model_path` 为您的模型实际路径。
2.  (可选) 修改 `calib_data` 加载真实的校准数据集。
3.  运行脚本：

```bash
python quantize_qwen3.py
```

### 方法二：使用 CLI + 配置文件

如果您更喜欢使用命令行工具，可以使用 `qwen3_w4a4_config.yaml`。

1.  修改 `qwen3_w4a4_config.yaml` 中的 `dataset` 字段，指向您的校准数据文件 (json 格式)。
2.  运行以下命令：

```bash
# 运行 CLI
msmodelslim quant\
    --model_type qwen3 \
    --model_path /workspace/weights/Qwen3-30B \
    --save_path /workspace/weights/Qwen3-30B-W4A4-OfflineQuaRot-test \
    --config_path qwen3_w4a4_config.yaml \
    --device npu:0 \
    --trust_remote_code True
```

## 如何推送到 GitHub

如果您想将此本地仓库推送到您的 GitHub 账户，请按照以下步骤操作：

1.  **在 GitHub 上创建新仓库**：
    *   登录 GitHub，点击右上角的 "+" -> "New repository"。
    *   输入仓库名称 (例如 `msit-qwen3`)，点击 "Create repository"。

2.  **添加远程仓库地址**：
    在终端中，进入 `msit` 根目录，执行以下命令 (将 URL 替换为您刚才创建的仓库地址)：

    ```bash
    # 移除旧的 remote (如果有)
    git remote remove origin
    
    # 添加新的 remote
    git remote add origin https://github.com/您的用户名/msit-qwen3.git
    ```

3.  **推送到 GitHub**：

    ```bash
    # 推送 qwen3 分支
    git push -u origin qwen3
    ```
