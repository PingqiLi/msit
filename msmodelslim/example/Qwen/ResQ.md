# ResQ 量化技术文档 — Qwen3-32B

## 1. 概述

ResQ (Residual Quantization) 是一种 4/8-bit 混合精度量化方法。核心思路：

- 通过对激活值协方差矩阵做特征分解，得到 **特征向量矩阵 P**（按方差从小到大排序）
- 组合 **块对角随机正交旋转 R**，将 `U = P @ R` 融合进权重
- 高方差通道 → 8-bit 精度（保留精度）；低方差通道 → 4-bit 精度（节省显存）
- 推理时需要两组在线变换矩阵：**Uc**（K cache 旋转）和 **Ud**（down_proj 输入旋转）

原始论文：[ResQ (Facebook Research)](https://github.com/facebookresearch/resq)

---

## 2. Qwen3-32B 模型架构

所有维度来自 `model.config`，在代码中多处验证一致：

| 参数 | 值 | 说明 |
|------|------|------|
| `hidden_size` | 5120 | 隐藏层维度 |
| `num_attention_heads` | 64 | Query 头数 |
| `num_key_value_heads` | 8 | KV 头数 (GQA) |
| `head_dim` | 128 | 每头维度（显式配置，非 hidden_size / num_heads） |
| `intermediate_size` | 25600 | MLP 中间维度 (= 100 × 256) |
| `num_hidden_layers` | 64 | Transformer 层数 |
| `down_proj_blocksize` | 256 | down_proj 协方差分块大小（默认） |

### 权重维度表

| 投影层 | 权重形状 `[out, in]` | 说明 |
|--------|---------------------|------|
| `q_proj` | [8192, 5120] | 64 heads × 128 head_dim, hidden_size |
| `k_proj` | [1024, 5120] | 8 KV heads × 128 head_dim, hidden_size |
| `v_proj` | [1024, 5120] | 8 KV heads × 128 head_dim, hidden_size |
| `o_proj` | [5120, 8192] | hidden_size, 64 heads × 128 head_dim |
| `gate_proj` | [25600, 5120] | intermediate_size, hidden_size |
| `up_proj` | [25600, 5120] | intermediate_size, hidden_size |
| `down_proj` | [5120, 25600] | hidden_size, intermediate_size |

注意：Qwen3-32B 使用 GQA (Grouped Query Attention)，64 个 Query 头共享 8 个 KV 头，每个 KV 头对应 `64/8 = 8` 个 Query 头。

---

## 3. 文件结构

```
msmodelslim/
├── example/Qwen/
│   ├── resq_qwen3_32b.py           # 入口脚本 (928 行)
│   └── ResQ.md                      # 本文档
└── msmodelslim/pytorch/llm_ptq/llm_ptq_tools/resq/
    ├── __init__.py
    ├── config.py                    # ResQConfig 配置类
    ├── calibrator.py                # ResQCalibrator 校准器 (1658 行)
    ├── gptq.py                      # GPTQ 实现 (383 行)
    ├── quant_modules.py             # 量化模块 (663 行)
    ├── processors/
    │   ├── basis_processor.py       # 基向量计算 (946 行)
    │   ├── resq_processor.py        # 旋转融合 & 列重排 (891 行)
    │   └── adaptive_ratio.py        # 自适应比率算法 (984 行)
    └── utils/
        ├── hadamard_utils.py        # Hadamard 变换工具 (710 行)
        ├── fuse_norm_utils.py       # LayerNorm 融合 (208 行)
        └── common.py                # 通用工具函数
```

### 文件依赖关系

```
resq_qwen3_32b.py
  ├── ResQConfig (config.py)
  ├── compute_basis (processors/basis_processor.py)
  │     └── hadamard_utils.py (get_hadK, random_orthogonal_matrix)
  └── ResQCalibrator (calibrator.py)
        ├── fuse_layer_norms (utils/fuse_norm_utils.py)
        ├── apply_rotations (processors/resq_processor.py)
        │     └── hadamard_utils.py (matmul_hadU_cpu)
        ├── rearrange_columns (processors/resq_processor.py)
        ├── add_resq_quantizers (quant_modules.py)
        ├── AdaptiveRatioComputer (processors/adaptive_ratio.py)
        └── GPTQ (gptq.py)
```

---

## 4. 完整算法流程

### 概览

```
Float Model
    ↓
[Step 1] LayerNorm 融合
    ↓
[Step 2] 校准数据 → 激活收集 → 协方差矩阵
    ↓
[Step 3] 特征分解 → P (特征向量), λ (特征值)
    ↓
[Step 4] 随机旋转生成 → R1, R2, Hd/Rd
    ↓
[Step 5] 旋转融合进权重 → U = P @ R
    ↓
[Step 6] 列重排 → [4-bit | 8-bit] 布局
    ↓
[Step 7] 量化器替换 → LinearResQQuantizer / LinearW8A8DynamicQuantizer
    ↓
[Step 8] 校准 (RTN / GPTQ / data-free)
    ↓
[Step 9] 保存量化模型
```

---

### Step 1: LayerNorm 融合

**文件**: `fuse_norm_utils.py` — `fuse_layer_norms()`

将 LayerNorm 的线性缩放参数 γ 吸收到相邻线性层的权重中：

```
LayerNorm(x) @ W  →  x @ (γ * W)
```

**三步操作**：

1. **Embedding 均值减除**（`fuse_norm_utils.py:146`）
   ```python
   W_new = W_ - W_.mean(dim=-1, keepdim=True)
   ```
   > **注意**：此步对 RMSNorm 模型（Qwen3）数学上非必需。RMSNorm 不减均值，只做
   > `x / sqrt(mean(x²)) * γ`。实践中 embedding 权重行均值接近零（~1e-7），影响可忽略。
   > 参见代码中 TODO 注释（`fuse_norm_utils.py:136-145`）。

2. **逐层融合**（`fuse_norm_utils.py:159-192`）
   - `input_layernorm` → `q_proj, k_proj, v_proj`
   - `post_attention_layernorm` → `up_proj, gate_proj`
   - 融合后将 norm 权重设为全 1

3. **最终 norm 融合**（`fuse_norm_utils.py:198-202`）
   - `model.norm` → `lm_head`

**内存优化**：所有矩阵乘法在 CPU 上执行以避免设备 OOM。每层处理后调用 `cleanup_memory()`。

---

### Step 2: 激活收集与协方差矩阵计算

**文件**: `basis_processor.py` — `compute_basis()`

#### 2.1 InputCatcher 捕获首层输入

```python
class InputCatcher(nn.Module):
    def forward(self, inp, **kwargs):
        self.inputs.append(inp.cpu())
        self.attention_masks.append(kwargs.get('attention_mask'))
        self.position_ids_list.append(kwargs.get('position_ids'))
        self.position_embeddings_list.append(kwargs.get('position_embeddings'))
        raise ValueError("Catcher stop")
```

替换 `model.model.layers[0]`，运行所有校准数据，捕获：`inputs`, `attention_mask`, `position_ids`, `position_embeddings`。

#### 2.2 协方差矩阵初始化

5 组协方差矩阵，存储在 CPU（`cov_device='cpu'`）：

| 矩阵 | 形状 | 说明 |
|-------|------|------|
| `H_attn` | `[64, 5120, 5120]` | 注意力输入 (q_proj 输入) |
| `H_mlp` | `[64, 5120, 5120]` | MLP 输入 (up_proj 输入) |
| `H_value` | `[64, 8, 128, 128]` | V 投影输出 (per KV head) |
| `H_key_pos` | `[64, 8, 128, 128]` | K 投影输出 (RoPE 后, per KV head) |
| `H_down_proj` | `[64, 256, 256]` | down_proj 输入 (分块, 非全维度) |

#### 2.3 Hook 收集与协方差累积

逐层处理（`basis_processor.py:398-580`）：

```python
for layer_idx in range(nlayers):
    layer = layers[layer_idx].to(device)  # 移到 NPU
    # 注册 hooks
    hooks = [
        q_proj.register_forward_hook(capture 'attn_input'),
        k_proj.register_forward_hook(capture 'k_output'),
        v_proj.register_forward_hook(capture 'v_output'),
        up_proj.register_forward_hook(capture 'mlp_input'),
        down_proj.register_forward_hook(capture 'down_proj_input'),
    ]
    # 逐 batch forward，累积协方差
    for batch in batches:
        out = layer(inp, **kwargs)
        H_attn[layer_idx] += X_attn.T @ X_attn
        H_mlp[layer_idx] += X_mlp.T @ X_mlp
        H_value[layer_idx, head] += X_v.T @ X_v   # per head
        H_key_pos[layer_idx, head] += X_k_pos.T @ X_k_pos  # post-RoPE
        # down_proj: block-wise
        X_dp = dp_input.view(batch, -1, blocksize)
        H_down_proj[layer_idx] += sum(X_dp.mT @ X_dp, dim=0)
    # 移回 CPU
    layers[layer_idx] = layer.cpu()
    inps, outs = outs, [None] * nbatches  # 输出变为下层输入
```

**关键设计**：
- **down_proj 分块协方差**：将 25600 维 reshape 为 `[batch, 100, 256]`，对 100 个 block 的协方差求和，得到 `[256, 256]`。避免存储 `[25600, 25600]`。
- **Key post-RoPE**：对 k_proj 输出应用 RoPE 后再计算协方差，因为 Uc 需要在 RoPE 之后生效。

#### 2.4 Kurtosis 累积（可选）

当 `compute_kurtosis=True` 时，使用 streaming 方法累积每个激活类型的：
- `sum`, `sum_sq`, `sum_4th`, `count`

用于后续自适应比率计算。

---

### Step 3: 特征分解

**文件**: `basis_processor.py` — `perform_eigen_decomp()`

#### 3.1 归一化

```python
normalizer = nbatches * seqlen
H_attn_mlp_sum = (H_attn.sum(0) + H_mlp.sum(0)) / (2 * nlayers * normalizer)
```

#### 3.2 共享基向量 (Pa)

**全局共享**：将所有层的 H_attn 和 H_mlp 求和后取平均，做单次特征分解：

```python
H_attn_mlp_sum = (H_attn.sum(0) + H_mlp.sum(0)) / (2 * nlayers * normalizer)
eval_attn_mlp, evec_attn_mlp = torch.linalg.eigh(H + dampening)
```

产出一个共享的 Pa `[5120, 5120]`，用于所有层的 attention/MLP 输入。

#### 3.3 Per-layer 基向量

对每层独立计算 3 组基向量（`basis_processor.py:633-678`）：

| 基向量 | 协方差输入 | per_head | 产出形状 |
|--------|-----------|----------|----------|
| Pb (value) | `H_value[i] / normalizer` | Yes, 8 heads | `[8, 128, 128]` |
| Pc (key_pos) | `H_key_pos[i].sum(0) / (8 * normalizer)` | No (合并 8 heads) | `[128, 128]` |
| Pd (down_proj) | `H_down_proj[i] / normalizer` | No | `[256, 256]` |

#### 3.4 并行化

使用 `ThreadPoolExecutor` 对每层的 3 组特征分解并行执行（`max_workers=128` on CPU）。

#### 3.5 Dampening

```python
damp = 0.01 * torch.mean(torch.diag(H))
H[diag] += damp
```

---

### Step 4: 随机旋转生成

**文件**: `basis_processor.py` — `generate_random_rotations()`

#### R1: hidden_dim 旋转

```python
high_dim = int(0.125 * 5120) = 640
mid_dim = 5120 - 640 = 4480

R1 = block_diag(R1_1[4480×4480], R1_2[640×640])
```

#### R2: head_dim 旋转

```python
high_head_dim = int(0.125 * 128) = 16
mid_head_dim = 128 - 16 = 112

R2 = block_diag(R2_1[112×112], R2_2[16×16])
```

#### Hd/Rd: intermediate_dim 旋转

根据 `ud_rotation_type` 配置：

- **`hadamard`（默认）**：调用 `get_hadK(25600)`
  - 25600 = 100 × 256，`get_hadK` 返回预计算的 `hadK100` 矩阵 `[100, 100]`（`hadamard_utils.py` 中的 `get_hadK100()` 函数）
  - 运行时 Hadamard 变换通过 butterfly 算法分解为 `hadK @ (n/K 个独立 DFT)` 形式

- **`random`**：`Rd = random_orthogonal_matrix(25600)` — QR 分解生成正交矩阵

所有随机矩阵使用 `torch.float64` 精度生成，通过 QR 分解确保正交性。

---

### Step 5: 旋转融合进权重

**文件**: `resq_processor.py` — `apply_rotations()`

#### 5.1 构建 Ua

```python
# basis_processor 产出 Pa [5120, 5120]
Ua = Pa @ R1  # [5120, 5120]
```

#### 5.2 融合操作

以下操作在 `apply_rotations()` 中逐层执行（`resq_processor.py:604-776`）：

| 操作 | 公式 | 函数 |
|------|------|------|
| Embedding | `W_emb = W_emb @ Ua` | `rotate_embeddings()` |
| LM Head | `W_lm = W_lm @ Ua` | `rotate_head()` |
| Q/K/V proj | `W_qkv = W_qkv @ Ua` | `rotate_attention_inputs()` |
| V/O proj | Ub = Pb @ R2 → `W_v = Ub.T @ W_v` (per head), `W_o = W_o @ Ub_inv.T` | `rotate_ov_proj()` |
| O proj 输出 | `W_o = Ua.T @ W_o` | `rotate_attention_output()` |
| Up/Gate proj | `W_up/gate = W_up/gate @ Ua` | `rotate_mlp_input()` |
| Down proj | 见下文 | 见下文 |

#### 5.3 Down proj 旋转（3 种模式）

根据 `mix_cfg` 中该层的 `quant_type` 决定：

**`resq`（默认）**：
```python
W_d = Ua.T @ W_d @ block_diag(Pd) @ Hd   # Hadamard 模式
W_d = Ua.T @ W_d @ block_diag(Pd) @ Rd   # Random 模式
```
函数：`rotate_mlp_output_hadamard()` / `rotate_mlp_output_random()`

**`int4_hadamard`**：
```python
W_d = Ua.T @ W_d @ Hd    # 跳过 Pd
```
函数：`rotate_mlp_output_hadamard_only()`

**`w8a8_dynamic` / `float`**：
```python
W_d = Ua.T @ W_d          # 跳过整个 Ud
```
函数：`rotate_mlp_output()`

#### 5.4 V/O proj 旋转详解

`rotate_ov_proj()`（`resq_processor.py:334-404`）：

```python
# V proj: W_v_new = Ub.T @ W_v (per KV head, batched)
W_v = W_v.view(num_kv_heads, head_dim, hidden_dim)
W_v = torch.bmm(Ub.transpose(-2, -1), W_v)

# O proj: W_o_new = W_o @ Ub_inv.T
# GQA: expand Ub_inv_T for num_attention_heads
Ub_inv = torch.linalg.inv(Ub)      # [8, 128, 128]
Ub_inv_T = Ub_inv.transpose(-2, -1)
Ub_inv_T_expanded = Ub_inv_T.repeat_interleave(8, dim=0)  # [64, 128, 128]
W_o = W_o.view(5120, 64, 128)
W_o = torch.bmm(W_o.permute(1,0,2), Ub_inv_T_expanded).permute(1,0,2)
```

#### 5.5 Remove Ub 模式

当 `config.remove_ub=True` 时，V/O proj 只用 Rb（不乘 Pb），列重排也跳过：

```python
rotate_ov_proj_rotation_only(layer, num_heads, head_dim, Rb=R2)
```

---

### Step 6: 列重排

**文件**: `resq_processor.py` — `rearrange_columns()` / `rearrange_o_proj()`

**仅对 o_proj 的输入维度**（8192 = 64 heads × 128 head_dim）进行重排。

对每个 head 的 `head_dim=128` 维度：
- 最后 `high_length_per_head = high_bits_length // 64` 列是高精度列
- 重排为 `[mid_cols | high_cols]` 的新列顺序

```python
# o_proj input: 8192 = 64 * 128
high_bits_length = int(0.125 * 8192) = 1024
high_length_per_head = 1024 // 64 = 16

# 每头最后 16 列移到末尾
new_order = [mid_columns | high_columns]
o_proj.weight.data = W[:, new_order]
```

当 `remove_ub=True` 时跳过此步。

---

### Step 7: 量化器替换

**文件**: `quant_modules.py` — `add_resq_quantizers()`

将 `nn.Linear` 替换为量化模块：

| 量化类型 | 替换为 | 适用层 |
|----------|--------|--------|
| `resq` | `LinearResQQuantizer` | q/k/v/o/gate/up_proj（默认） |
| `w8a8_dynamic` | `LinearW8A8DynamicQuantizer` | down_proj（固定模式下所有层） |
| `int4_hadamard` | `LinearResQQuantizer(high_fraction=0)` | down_proj（自适应模式低比率层） |
| `float` | 保持 `nn.Linear` | 不量化 |

#### LinearResQQuantizer

- **权重量化** (`ResQWeightQuantizer`)：沿 `split_dim=1`（输入维度）分割
  - `weight[:, :low_dim]` → 4-bit 对称量化 → `int8` 存储
  - `weight[:, low_dim:]` → 8-bit 对称量化 → `int8` 存储
  - 量化公式：`q = round(w / scale)`, `scale = max_abs / (2^(bits-1) - 1)`

- **激活量化** (`ResQActQuantizer`)：per-token 动态混合精度
  - `x[..., :low_dim]` → 4-bit `fake_quantize`
  - `x[..., low_dim:]` → 8-bit `fake_quantize`

#### Projection → Transform 映射

```python
# add_resq_quantizers() 中的映射逻辑
q_proj, k_proj, v_proj, gate_proj, up_proj  →  Ua ratio
o_proj                                       →  Ub ratio (per-layer)
down_proj                                    →  Ud ratio (per-layer)
```

---

### Step 8: 校准

**文件**: `calibrator.py` — `run()`

三种模式：

#### 8.1 RTN 模式 (`w_rtn=True`, 默认)

```python
for batch in calib_data:
    model(batch)  # forward pass → 触发量化器收集 min/max → 计算 scale
```

#### 8.2 GPTQ 模式 (`w_rtn=False`)

**文件**: `calibrator.py:739-1054`, `gptq.py`

逐层 Hessian-based 量化：

1. **捕获首层输入**：用 Catcher 收集 `nsamples` 个输入
2. **逐层处理**（`calibrator.py:886-1050`）：

```python
sequential_groups = [
    ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
    ["self_attn.o_proj"],
    ["mlp.up_proj", "mlp.gate_proj"],
    ["mlp.down_proj"],
]

for layer_idx in range(nlayers):
    layer = layers[layer_idx].to(device)
    for group in sequential_groups:
        # (a) 注册 hooks 累积 Hessian: H += X.T @ X
        # (b) 禁用量化（避免 forward 时删除 weight）
        # (c) 运行 nsamples forward passes
        # (d) fasterquant(): 列逐列量化 + 误差补偿
        # (e) 复制量化权重回 LinearResQQuantizer
    # forward 获取输出作为下层输入
    layers[layer_idx] = layer.cpu()
```

3. **GPTQ fasterquant()**（`gptq.py:178-332`）：

```python
# Cholesky 分解得到 Hinv
H += percdamp * mean(diag(H)) * I
Hinv = cholesky_inverse(H)

# 按 blocksize (128) 分块
for i1 in range(0, columns, blocksize):
    for col in range(blocksize):
        # 选择量化器（混合精度）
        if col_idx >= high_dim:
            q = high_quantizer.quantize(w)  # 8-bit
        else:
            q = quantizer.quantize(w)       # 4-bit
        # 误差补偿
        err = (w - q) / Hinv[i, i]
        W[:, i:] -= err @ Hinv[i, i:]
```

4. **Cholesky CPU 回退**：NPU 可能不支持 `torch.linalg.cholesky`，代码中有完整的 CPU fallback 逻辑（`gptq.py:246-279`）。

#### 8.3 Data-free 模式

无校准数据时直接对权重做量化：

```python
for module in model.modules():
    if isinstance(module, LinearResQQuantizer):
        module.quant_weight.quantize_weight(module.weight)
```

---

### Step 9: 保存

**文件**: `calibrator.py` — `save()`

#### 9.1 Output Mode

| 模式 | 保存内容 |
|------|----------|
| `fused`（默认） | 量化权重 + 在线变换矩阵 |
| `transforms_only` | 仅分解的 P/R 矩阵（不修改权重） |
| `debug` | 两者都保存 |

#### 9.2 保存内容详解

见下文 [输出文件格式](#9-输出文件格式) 章节。

---

## 5. 变换矩阵详解

### 四组变换矩阵

| 矩阵 | 公式 | 维度 | 共享/Per-layer | 融合 vs 在线 |
|-------|------|------|---------------|-------------|
| **Ua** | `Pa @ R1` | `[5120, 5120]` | 全局共享 | 融合进权重 |
| **Ub** | `Pb @ R2` | `[8, 128, 128]` | Per-layer | 融合进权重 |
| **Uc** | `Pc @ R2` | `[8, 128, 128]` 或 `[128, 128]` | Per-layer | **保存用于在线推理** |
| **Ud** | `block_diag(Pd) @ Hd` 或 `block_diag(Pd) @ Rd` | `[25600, 25600]` | Per-layer | **保存用于在线推理** |

### Ua 详解

- **Pa** `[5120, 5120]`：所有层 H_attn + H_mlp 的全局共享特征向量
- **R1** = `block_diag(R1_1[4480, 4480], R1_2[640, 640])`
- 融合到：embed_tokens, q/k/v_proj (input side), o_proj (output side), up/gate_proj (input side), down_proj (output side), lm_head

### Ub 详解

- **Pb** `[8, 128, 128]`：per-layer per-head value 特征向量
- **R2** = `block_diag(R2_1[112, 112], R2_2[16, 16])`
- 融合到：v_proj (output side), o_proj (input side, GQA expand)

### Uc 详解

- **Pc** `[128, 128]`：per-layer key_pos 特征向量（post-RoPE, 合并 8 heads）
- **R2**：同 Ub 使用的 R2
- **保存为** `resq.layer.{i}.Uc`，推理时应用于 K cache（RoPE 之后）

### Ud 详解

- **Pd** `[256, 256]`：per-layer down_proj 特征向量（blocksize 级别）
- **Hd**：Hadamard 矩阵（`get_hadK(25600)` → `hadK100[100, 100]`, K=100）
- 或 **Rd** `[25600, 25600]`：随机正交矩阵

**Hadamard 模式保存**：`resq.layer.{i}.Pd` + 全局 `resq.Hd`
**Random 模式保存**：`resq.layer.{i}.Ud`（完整 25600×25600 矩阵）

---

## 6. 自适应比率系统

**文件**: `adaptive_ratio.py`

### 概述

自适应比率系统替代固定 `high_fraction=0.125`，根据每层的量化难度自动确定 4-bit / 8-bit 比例。

### 4 种算法

#### 6.1 Hessian Trace Algorithm

```python
normalized_trace = sum(eigenvalues) / dim
# Log-scale 模式（默认）
log_trace = log(1 + normalized_trace)
log_ref = log(1 + reference_trace)    # 全局平均（需要 two-pass）
relative = log_trace / (2 * log_ref)
ratio = min_ratio + (max_ratio - min_ratio) * clamp(relative, 0, 1)
```

#### 6.2 Kurtosis Algorithm

```python
# 自适应阈值模式（默认）
threshold_low = percentile(all_kurtosis, 10)
threshold_high = percentile(all_kurtosis, 90)
# 线性插值
t = (kurtosis - threshold_low) / (threshold_high - threshold_low)
ratio = min_ratio + t * (max_ratio - min_ratio)
```

#### 6.3 CEV (Cumulative Explained Variance) Algorithm

```python
sorted_evals = sort_descending(eigenvalues)
cum_var = cumsum(sorted_evals) / total_variance
high_dim = first index where cum_var >= 0.95
ratio = clamp(high_dim / total_dim, min_ratio, max_ratio)
```

#### 6.4 Hybrid Algorithm

```python
ratio = 0.4 * hessian_ratio + 0.3 * kurtosis_ratio + 0.3 * cev_ratio
```

### Two-pass 机制

当 `hessian_log_scale=True` 或 `kurtosis_adaptive_thresholds=True`（均为默认）时，使用两轮计算：

1. **Pass 1**：收集所有层/变换的 trace history 和 kurtosis values
2. **计算全局参考值**：global `reference_trace`（均值），adaptive kurtosis thresholds（percentile）
3. **Pass 2**：用校准后的参考值重新计算最终比率

### Per-transform 比率

每个变换独立计算比率：

| 变换 | 特征值来源 | 维度 | 对齐 |
|------|-----------|------|------|
| Ua | `attn_mlp` per-layer traces | 5120 | 512 |
| Ub | `value` per-head (aggregated max/mean) | 128 | 512 (>128, 退化为无对齐) |
| Uc | `key_pos` per-layer | 128 | 512 (>128, 退化为无对齐) |
| Ud | `down_proj` per-layer | 25600 (对齐级别) | 512 |

### Down proj 量化类型决策

基于 per-layer Ud ratio 和阈值（`calibrator.py:569-624`）：

```python
threshold = (adaptive_min_ratio + adaptive_max_ratio) / 2  # 默认
if ratio < threshold:
    mix_cfg[layer] = 'int4_hadamard'
else:
    mix_cfg[layer] = 'w8a8_dynamic'
```

### 已知问题：比率集中在 0.2 附近

| 原因 | 算法 | 效果 |
|------|------|------|
| Sigmoid 饱和 | Hessian | 所有层 → max_ratio |
| Kurtosis 范围窄 | Kurtosis | 仅使用 ~30% 比率范围 |
| 频谱集中 | CEV | 聚集在 min_ratio 附近 |
| 加权平均压缩 | Hybrid | 进一步压缩分布 |
| 512 对齐 on 5120 | Ua | 仅有 0.1 和 0.2 两个有效值 |

---

## 7. 量化模块详解

**文件**: `quant_modules.py`

### 7.1 symmetric_quantize_to_int8

```python
def symmetric_quantize_to_int8(tensor, bits, quant_dim=1):
    n = 2 ** (bits - 1) - 1         # 7 for 4-bit, 127 for 8-bit
    abs_max = tensor.abs().max(dim=quant_dim, keepdim=True)[0]
    scale = abs_max / n
    q = round(tensor / scale)
    q = clamp(q, -(n+1), n)         # [-8, 7] for 4-bit, [-128, 127] for 8-bit
    return q.to(int8), scale
```

### 7.2 ResQWeightQuantizer

- `split_dim=1`：沿输入维度分割（所有投影层统一使用 dim=1）
- `weight[:, :low_dim]` → `symmetric_quantize_to_int8(bits=4)` → `low_weight (int8)`, `low_weight_scale`
- `weight[:, low_dim:]` → `symmetric_quantize_to_int8(bits=8)` → `high_weight (int8)`, `high_weight_scale`
- `quantize_weight()` 返回拼接的 dequantized weight 用于推理

### 7.3 ResQActQuantizer

Per-token 动态混合精度激活量化：

```python
x_low = x[..., :low_dim]   # 4-bit fake_quantize
x_high = x[..., low_dim:]  # 8-bit fake_quantize
return cat([x_low, x_high], dim=-1)
```

### 7.4 LinearResQQuantizer

组合 `ResQWeightQuantizer` + `ResQActQuantizer`：

```python
def forward(self, x):
    x = self.quant_input(x)           # 激活量化
    weight = self.quant_weight(self.weight)  # 权重量化
    return F.linear(x, weight, self.bias)

def get_quant_weights(self):
    return {
        'weight_low': int8_tensor, 'scale_low': float_tensor,
        'weight_high': int8_tensor, 'scale_high': float_tensor,
        'high_fraction': actual_fraction,
    }
```

### 7.5 LinearW8A8DynamicQuantizer

Per-channel int8 对称权重量化（用于 `w8a8_dynamic` 层）：

```python
def get_quant_weights(self):
    return {
        'weight': int8_tensor,           # [out_features, in_features]
        'weight_scale': float_tensor,    # [out_features, 1]
        'weight_offset': zeros_int8,     # [out_features, 1] (对称量化为零)
    }
```

---

## 8. GPTQ 集成

**文件**: `gptq.py`

### GPTQ 类

```python
class GPTQ:
    def add_batch(self, inp, out=None):
        # H += X.T @ X (Hessian 累积)

    def fasterquant(self, blocksize=128, percdamp=0.01, actorder=False):
        # 1. Dampening: H += percdamp * mean(diag(H)) * I
        # 2. Cholesky: Hinv = cholesky(cholesky_inverse(H))
        # 3. Block-wise 列量化 + 误差补偿
```

### 混合精度支持

```python
# fasterquant() 中根据列位置选择量化器
if col_idx >= high_dim:
    q = self.high_quantizer.quantize(w)  # 8-bit
else:
    q = self.quantizer.quantize(w)       # 4-bit
```

### 内存管理

- Cholesky 在 NPU 不支持时自动 fallback 到 CPU（`gptq.py:246-279`）
- 逐层处理后将层移回 CPU
- 将非 Parameter 的 plain tensor attrs 也显式移到 CPU（`calibrator.py:1041-1045`）

---

## 9. 输出文件格式

### 文件列表

| 文件 | 说明 |
|------|------|
| `model.safetensors` | 量化权重 + 在线变换矩阵 |
| `quant_model_description.json` | 量化元数据（每个张量的类型标注） |
| `model.safetensors.index.json` | 张量索引 |
| `config.json` | 模型配置 + `"quantize": "W4A8_ResQ"` |
| `resq_basis.pt` | 计算的基向量（如果 `--compute_basis`） |
| `resq_adaptive_ratios.json` | 自适应比率结果（如果使用自适应模式） |

### 张量命名规范

#### ResQ 量化层 (LinearResQQuantizer)

```
model.layers.{i}.self_attn.q_proj.weight_low     # int8, [8192, low_dim]
model.layers.{i}.self_attn.q_proj.scale_low       # float32, [8192, 1]
model.layers.{i}.self_attn.q_proj.weight_high     # int8, [8192, high_dim]
model.layers.{i}.self_attn.q_proj.scale_high      # float32, [8192, 1]
model.layers.{i}.self_attn.q_proj.high_fraction   # float32 scalar
```

同样适用于 `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`。

#### W8A8 Dynamic 层 (LinearW8A8DynamicQuantizer)

```
model.layers.{i}.mlp.down_proj.weight             # int8, [5120, 25600]
model.layers.{i}.mlp.down_proj.weight_scale        # float32, [5120, 1]
model.layers.{i}.mlp.down_proj.weight_offset        # int8, [5120, 1] (全零)
```

#### 非量化层 (FLOAT)

```
model.embed_tokens.weight                           # 已旋转 (Ua)
model.norm.weight                                   # 全 1 (融合后)
model.layers.{i}.input_layernorm.weight             # 全 1 (融合后)
model.layers.{i}.post_attention_layernorm.weight    # 全 1 (融合后)
model.layers.{i}.self_attn.q_norm.weight            # Qwen3 特有
model.layers.{i}.self_attn.k_norm.weight            # Qwen3 特有
lm_head.weight                                      # 已旋转 (Ua)
```

#### 在线变换矩阵

**Hadamard 模式** (`ud_rotation_type='hadamard'`)：
```
resq.layer.{i}.Uc        # float32, [8, 128, 128] 或 [128, 128]
resq.layer.{i}.Pd        # float32, [256, 256] (仅 resq/int4_hadamard 层)
resq.Hd                  # float32, [100, 100] (全局共享)
```

**Random 模式** (`ud_rotation_type='random'`)：
```
resq.layer.{i}.Uc        # float32
resq.layer.{i}.Ud        # float32, [25600, 25600] (完整矩阵)
```

### quant_model_description.json 格式

```json
{
  "model_quant_type": "ResQ",
  "high_bits": 8,
  "low_bits": 4,
  "high_fraction": 0.125,
  "group_size": -1,
  "version": "1.0",
  "model.embed_tokens.weight": "FLOAT",
  "model.layers.0.self_attn.q_proj.weight_low": "RESQ",
  "model.layers.0.self_attn.q_proj.scale_low": "RESQ",
  "model.layers.0.mlp.down_proj.weight": "W8A8_DYNAMIC",
  "resq.layer.0.Uc": "FLOAT",
  ...
}
```

---

## 10. 命令行参数完整参考

### 基础参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_path` | str | **必需** | 浮点模型路径 |
| `--save_directory` | str | **必需** | 输出目录 |
| `--layer_count` | int | 0 | 量化层数 (0=全部) |
| `--calib_file` | str | `../common/wiki.jsonl` | 校准数据文件 |
| `--batch_size` | int | 1 | 校准批大小 |
| `--nsamples` | int | 128 | 校准样本数 |
| `--seq_len` | int | 2048 | 序列长度 |
| `--seed` | int | 42 | 随机种子 |

### 精度参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--high_bits` | int | 8 | 高精度位数 |
| `--low_bits` | int | 4 | 低精度位数 |
| `--high_fraction` | float | 0.125 | 高精度通道比例 |

### 设备参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dev_type` | str | `npu` | 设备类型 (`npu`, `cpu`) |
| `--dev_id` | int | 0 | 设备 ID |
| `--max_memory_per_device` | str | None | NPU 最大显存 (e.g., `55GiB`) |
| `--disable_cpu_offload` | bool | True | 阻止 accelerate 将层卸载到 CPU |

### Basis 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--compute_basis` | bool | False | 从校准数据计算 basis |
| `--basis_path` | str | None | 预计算 basis 路径 |
| `--save_basis_path` | str | None | 保存 basis 的路径 |
| `--rotation_path` | str | None | 预计算旋转矩阵路径 |

### 输出模式

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--output_mode` | str | `fused` | `fused`, `transforms_only`, `debug` |
| `--remove_ub` | bool | False | V_proj 仅用 Rb（不用 Pb） |

### 自适应比率参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--adaptive_ratio_mode` | str | `fixed` | `fixed`, `hessian`, `kurtosis`, `cev`, `hybrid` |
| `--adaptive_min_ratio` | float | 0.0625 | 最小比率 (1/16) |
| `--adaptive_max_ratio` | float | 0.25 | 最大比率 (1/4) |
| `--adaptive_alignment` | int | 512 | 硬件对齐边界 |
| `--cev_target_variance` | float | 0.95 | CEV 目标方差 |
| `--transform_algorithms` | str | None | Per-transform 算法 JSON |
| `--ub_head_aggregation` | str | `max` | Ub 跨头聚合 (`max`, `mean`) |
| `--adaptive_ratio_path` | str | None | 预计算比率 JSON 路径 |
| `--compute_kurtosis` | bool | False | 计算 kurtosis（`kurtosis`/`hybrid` 模式自动启用） |
| `--down_proj_ratio_threshold` | float | None | down_proj 量化类型阈值（默认: min+max 中点） |
| `--hessian_log_scale` | bool | True | Hessian 使用 log-scale |
| `--kurtosis_adaptive_thresholds` | bool | True | Kurtosis 自适应阈值 |
| `--kurtosis_percentile_low` | float | 10.0 | 低 percentile |
| `--kurtosis_percentile_high` | float | 90.0 | 高 percentile |

### Mix Config

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--mix_cfg` | str | None | JSON dict 指定每层量化类型 |

### GPTQ 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--w_rtn` | bool | True | RTN 模式 (False 启用 GPTQ) |
| `--percdamp` | float | 0.01 | GPTQ dampening |
| `--act_order` | bool | False | GPTQ activation ordering |
| `--gptq_blocksize` | int | 128 | GPTQ block size |
| `--gptq_device` | str | None | GPTQ 处理设备 |

### 其他

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--trust_remote_code` | bool | False | 信任远程代码 |

---

## 11. 使用示例

### 固定比率模式（默认）

```bash
python msmodelslim/example/Qwen/resq_qwen3_32b.py \
    --model_path /path/to/Qwen3-32B \
    --save_directory /path/to/output \
    --high_fraction 0.125 \
    --compute_basis
```

### 混合自适应模式（推荐）

```bash
python msmodelslim/example/Qwen/resq_qwen3_32b.py \
    --model_path /path/to/Qwen3-32B \
    --save_directory /path/to/output \
    --adaptive_ratio_mode hybrid \
    --compute_basis \
    --compute_kurtosis
```

### GPTQ 模式

```bash
python msmodelslim/example/Qwen/resq_qwen3_32b.py \
    --model_path /path/to/Qwen3-32B \
    --save_directory /path/to/output \
    --compute_basis \
    --w_rtn False \
    --percdamp 0.01 \
    --gptq_blocksize 128
```

### Per-transform 算法覆盖

```bash
python msmodelslim/example/Qwen/resq_qwen3_32b.py \
    --model_path /path/to/Qwen3-32B \
    --save_directory /path/to/output \
    --adaptive_ratio_mode hybrid \
    --transform_algorithms '{"Ua": "cev", "Ub": "kurtosis", "Ud": "hessian"}' \
    --compute_basis \
    --compute_kurtosis
```

### 使用预计算 basis

```bash
python msmodelslim/example/Qwen/resq_qwen3_32b.py \
    --model_path /path/to/Qwen3-32B \
    --save_directory /path/to/output \
    --basis_path /path/to/resq_basis.pt
```

### Transform-only 模式

```bash
python msmodelslim/example/Qwen/resq_qwen3_32b.py \
    --model_path /path/to/Qwen3-32B \
    --save_directory /path/to/output \
    --output_mode transforms_only \
    --compute_basis
```

### Mix Config 用法

```bash
python msmodelslim/example/Qwen/resq_qwen3_32b.py \
    --model_path /path/to/Qwen3-32B \
    --save_directory /path/to/output \
    --compute_basis \
    --mix_cfg '{"*.mlp.down_proj": "w8a8_dynamic", "*.self_attn.o_proj": "float"}'
```

---

## 12. 内存优化策略

| 策略 | 位置 | 效果 |
|------|------|------|
| **逐层 basis 计算** | `basis_processor.py` | 每次只在 NPU 上放一层 |
| **CPU 协方差存储** | `cov_device='cpu'` | 避免 5 组大矩阵占用 NPU 显存 |
| **分块 down_proj 协方差** | `H_down_proj [64, 256, 256]` | 避免 `[25600, 25600]` |
| **Streaming kurtosis** | Welford's algorithm | 不存储全部激活，流式计算统计量 |
| **模型重载** | `resq_qwen3_32b.py:750-810` | Basis 计算后释放 CPU 模型，重新加载到 NPU |
| **GPTQ 逐层处理** | `calibrator.py:886` | 每层处理后移回 CPU，显式清理 plain tensor attrs |
| **LayerNorm 融合 CPU 执行** | `fuse_norm_utils.py` | 权重移到 CPU 做乘法，避免设备 OOM |
| **Cholesky CPU fallback** | `gptq.py:246-279` | NPU 不支持 Cholesky 时自动回退 |

---

## 13. 多设备支持

### NPU 显存自动检测

`build_npu_max_memory()`（`resq_qwen3_32b.py:333-378`）：

```python
props = torch.npu.get_device_properties(i)
usable_mem = int(props.total_memory * 0.85)  # 85% 利用率
```

### accelerate dispatch_model

模型重载后用 `device_map="auto"` 自动分布。代码检测 CPU 溢出并采取不同策略（`calibrator.py:289-350`）：

- **无 CPU 溢出**：正常使用 `dispatch_model`
- **有 CPU 溢出**：跳过 `dispatch_model`，改用轻量级 `AlignDevicesHook(offload=False)` 做激活路由，避免权重变为 meta tensor

### Meta tensor 处理

`fix_meta_norm_params()`（`resq_qwen3_32b.py:288-330`）：对 `device_map="auto"` 遗留的 meta 态 `q_norm`/`k_norm` 参数，从 safetensors 文件重新加载。

---

## 14. ResQConfig 完整参数表

**文件**: `config.py`

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `high_bits` | int | 8 | 高精度位数 |
| `low_bits` | int | 4 | 低精度位数 |
| `high_fraction` | float | 0.125 | 高精度比例 |
| `seed` | int | 0 | 随机种子 |
| `dev_type` | str | `npu` | 设备类型 |
| `dev_id` | int | 0 | 设备 ID |
| `mix_cfg` | dict | `{}` | per-layer 量化类型 (fnmatch patterns) |
| `adaptive_ratio` | bool | False | 启用自适应比率 |
| `adaptive_algorithm` | str | `hybrid` | 算法: `hessian`, `kurtosis`, `cev`, `hybrid` |
| `adaptive_min_ratio` | float | 0.0625 | 最小比率 |
| `adaptive_max_ratio` | float | 0.25 | 最大比率 |
| `cev_target_variance` | float | 0.95 | CEV 目标方差 |
| `adaptive_alignment` | int | 512 | 对齐边界 |
| `transform_algorithms` | dict | `{}` | per-transform 算法覆盖 |
| `ub_head_aggregation` | str | `max` | Ub 跨头聚合方法 |
| `compute_kurtosis` | bool | False | 计算 kurtosis |
| `hessian_log_scale` | bool | True | Hessian log-scale |
| `kurtosis_adaptive_thresholds` | bool | True | Kurtosis 自适应阈值 |
| `kurtosis_percentile_low` | float | 10.0 | 低 percentile |
| `kurtosis_percentile_high` | float | 90.0 | 高 percentile |
| `down_proj_ratio_threshold` | float | None | down_proj 量化类型阈值 |
| `remove_ub` | bool | False | V_proj 仅用 Rb |
| `use_npu_rotation` | bool | True | NPU 上做旋转 matmul |
| `w_rtn` | bool | True | RTN (True) / GPTQ (False) |
| `percdamp` | float | 0.01 | GPTQ dampening |
| `act_order` | bool | False | GPTQ activation ordering |
| `gptq_blocksize` | int | 128 | GPTQ block size |
| `gptq_device` | str | None | GPTQ 处理设备 |
| `ud_rotation_type` | str | `hadamard` | Ud 旋转类型: `hadamard` / `random` |
| `down_proj_blocksize` | int | 256 | Pd 块大小 |
| `output_mode` | str | `fused` | 输出模式 |
| `a_bits` | int | 4 | 激活量化位数 |
| `w_bits` | int | 4 | 权重量化位数 |
| `w_sym` | bool | True | 对称量化 |
| `is_dynamic` | bool | True | 动态激活量化 |
| `fp32_had` | bool | True | fp32 Hadamard |
| `amp_dtype` | str | `bfloat16` | 数据类型 |

---

## 15. 已知问题与改进方向

### 15.1 自适应比率集中问题

所有四种自适应算法（hessian, kurtosis, cev, hybrid）产出的比率聚集在 0.15-0.20 附近，无法有效区分简单层和困难层。

**建议修复**：
1. **Percentile rank normalization**：对所有层的原始比率做 rank，然后 remap 到 [min, max]
2. **Log-scale Hessian**（已实现，`hessian_log_scale=True`）
3. **自适应 kurtosis 阈值**（已实现，`kurtosis_adaptive_thresholds=True`）
4. **更宽的比率范围**：从 [1/16, 1/4] 扩展到 [1/32, 1/2]
5. **更小的对齐粒度**：从 512 减到 128 或 256

### 15.2 Embedding 均值减除

对 RMSNorm 模型数学上非必需，应检测 norm 类型后条件跳过。参见 `fuse_norm_utils.py:136-145` 中的 TODO。
