# ResQ Quantization Script Analysis Report

## Overview

The script [resq_qwen3_32b.py](msmodelslim/example/Qwen/resq_qwen3_32b.py) performs 4/8-bit hybrid quantization on Qwen3-32B using the ResQ algorithm. The core idea is:
- **High variance channels** → 8-bit precision (preserve accuracy)
- **Low variance channels** → 4-bit precision (reduce memory)
- Eigenvalue-based rotation matrices (Uc, Ud) are generated per-layer for online inference

---

## Step-by-Step Analysis

### Step 1: Configuration and Model Loading (Lines 93-288)

**Classes/Functions:**
- `parse_args()` (line 93-141) - Parses command-line arguments
- `ResQConfig` from [config.py](msmodelslim/pytorch/llm_ptq/llm_ptq_tools/resq/config.py)

**Key Parameters:**
```python
ResQConfig(
    high_bits=8,        # Precision for high-variance channels
    low_bits=4,         # Precision for low-variance channels
    high_fraction=0.125 # 12.5% channels use 8-bit (1/8)
)
```

**Flow:**
1. Load model configuration via `SafeGenerator.get_config_from_pretrained()` (line 219)
2. Load tokenizer (line 241)
3. Load model - on CPU if computing basis, or with `device_map="auto"` otherwise (lines 253-288)

---

### Step 2: Calibration Data Preparation (Lines 291-313)

**Function:** `get_calib_dataset_batch()` (line 149-168)

```python
def get_calib_dataset_batch(model_tokenizer, calib_list, batch_size, seq_len, device):
    # Tokenize calibration prompts into batches
    inputs = model_tokenizer(calib_data, return_tensors='pt', ...)
    batch_tensors = [value.to(device) for key, value in inputs.data.items()]
    return calib_dataset  # List of [input_ids, attention_mask] tensors
```

**Data source:** JSONL file (e.g., `wiki.jsonl`) containing text prompts for calibration.

---

### Step 3: Activation Collection (Basis Computation)

**Location:** [basis_processor.py](msmodelslim/pytorch/llm_ptq/llm_ptq_tools/resq/processors/basis_processor.py)

**Main Function:** `compute_basis()`

**Basis Mode:** Full shared mode - computes a single shared basis for attention+MLP across all layers, which provides the best balance of quality and efficiency.

**How Activations Are Collected:**

1. **InputCatcher Class** - Captures first layer inputs:
```python
class InputCatcher(nn.Module):
    def forward(self, inp, **kwargs):
        self.inputs.append(inp.cpu())  # Store activations
        self.attention_masks.append(kwargs.get('attention_mask'))
        self.position_ids.append(kwargs.get('position_ids'))
        raise ValueError("Catcher stop")  # Stop forward pass
```

2. **Covariance Matrix Initialization**:
```python
H_attn = torch.zeros((nlayers, hidden_dim, hidden_dim), dtype=torch.float64)   # Attention input
H_mlp = torch.zeros((nlayers, hidden_dim, hidden_dim), dtype=torch.float64)    # MLP input
H_value = torch.zeros((nlayers, num_kv_heads, head_dim, head_dim), ...)        # V projection
H_key_pos = torch.zeros((nlayers, num_kv_heads, head_dim, head_dim), ...)      # K after RoPE
H_down_proj = torch.zeros((nlayers, down_proj_blocksize, down_proj_blocksize)) # down_proj input
```

3. **Hook-Based Collection**:
   - Uses `register_forward_hook()` to capture activations at specific layers
   - Accumulates outer products: `H += X.T @ X` (covariance estimation)

**Script Invocation:**
```python
basis_dict = compute_basis(
    model=model,
    dataloader=basis_dataloader,
    config=resq_config,
    device=process_device,      # NPU/GPU for forward passes
    cov_device='cpu',           # CPU for covariance storage (memory saving)
)
```

---

### Step 4: PCA / Eigendecomposition

**Location:** [basis_processor.py](msmodelslim/pytorch/llm_ptq/llm_ptq_tools/resq/processors/basis_processor.py)

**Function:** `perform_eigen_decomp()`

```python
def perform_eigen_decomp(
    cov_matrix: torch.Tensor,
    damp_percent: float = 0.01,
    per_head: bool = False,
    num_heads: int = 0,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Eigenvalue decomposition on covariance matrix.
    Returns: (eigenvalues, eigenvectors) sorted ascending
    """
    # Add damping for numerical stability
    diag_mean = torch.mean(torch.diag(H))
    H.diagonal().add_(damp_percent * diag_mean)

    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(H)

    # Sort by eigenvalues (ascending - low variance first)
    sorted_indices = torch.argsort(eigenvalues)
    return eigenvalues[sorted_indices], eigenvectors[:, sorted_indices]
```

**Purpose:** Eigenvectors define the rotation basis (P). Channels are sorted by variance so that:
- Low eigenvalue → Low variance → 4-bit precision (beginning of tensor)
- High eigenvalue → High variance → 8-bit precision (end of tensor)

---

### Step 5: Random Rotation Generation

**Location:** [basis_processor.py](msmodelslim/pytorch/llm_ptq/llm_ptq_tools/resq/processors/basis_processor.py)

**Function:** `generate_random_rotations()`

```python
# Block-diagonal random orthogonal matrices:
R1 = block_diag(R1_0, R1_1, R1_2)  # Hidden dimension: (low, mid, high)
R2 = block_diag(R2_0, R2_1, R2_2)  # Head dimension
Rd = block_diag(Rd_0, Rd_1, Rd_2)  # Intermediate dimension (MLP)
```

**Why?** Random rotations within each precision tier spread quantization error evenly, preventing systematic bias.

---

### Step 6: Column Reordering

**Location:** [resq_processor.py](msmodelslim/pytorch/llm_ptq/llm_ptq_tools/resq/processors/resq_processor.py)

**Functions:**
- `rearrange_columns()` - Main entry point
- `rearrange_o_proj()` - O projection specific

```python
def rearrange_o_proj(layer, high_bits_length, low_bits_length, head_dim, ...):
    """
    Reorder columns for mixed-precision layout.
    New column order: [low_precision | middle | high_precision]
    """
    # Build index mapping
    columns_to_beginning = low_variance_indices      # 4-bit region
    columns_to_end = high_variance_indices           # 8-bit region
    remaining_columns = middle_indices               # Standard precision

    new_column_order = torch.cat([columns_to_beginning, remaining_columns, columns_to_end])

    # Apply reordering to weight matrix
    o_proj.weight.data = Wo[:, new_column_order]

    # Store order for runtime use
    layer.self_attn.new_column_order = new_column_order
```

**Effect:** After reordering, the weight matrix has a predictable structure:
```
|--- 4-bit region ---|--- standard ---|--- 8-bit region ---|
```

---

### Step 7: Rotation Matrix Fusion

**Location:** [resq_processor.py](msmodelslim/pytorch/llm_ptq/llm_ptq_tools/resq/processors/resq_processor.py)

**Function:** `apply_rotations()`

The composite rotation `U = P @ R` (basis @ random rotation) is fused into model weights:

```python
def apply_rotations(model, basis_dict, rotation_dict, config):
    """Fuse U = P @ R into model weights."""

    # Build composite rotation
    R1 = torch.block_diag(R1_0, R1_1, R1_2)
    U_attn = torch.matmul(basis_attn, R1)  # U = P @ R

    # Apply to each layer
    for layer in model.layers:
        rotate_embeddings(model, U_attn)           # Token embeddings
        rotate_attention_inputs(layer, U_attn)     # Q, K, V projections
        rotate_attention_output(layer, U_attn)     # O projection
        rotate_mlp_input(layer, U_attn)            # gate_proj, up_proj
        rotate_mlp_output(layer, U_attn, R4)       # down_proj
        rotate_ov_proj(layer, U2)                  # Per-head V rotation
```

**Sub-functions:**
- `rotate_embeddings()` - `embed_tokens.weight @ U.T`
- `rotate_attention_inputs()` - `qkv_proj.weight = U @ W`
- `rotate_mlp_output()` - Includes Hadamard transform

**Hadamard Transform** (optional, for outlier suppression):
```python
def rotate_mlp_output(layer, R1, R4, no_had=False):
    if not no_had:
        had_K, K = get_hadK(W_.shape[-1])
        W_ = matmul_hadU_cpu(W_, K * torch.linalg.inv(R4).t(), K)
```

---

### Step 8: ResQ Calibrator Initialization and Run

**Location:** Main script

```python
# Create calibrator
calibrator = ResQCalibrator(
    model=model,
    cfg=resq_config,
    calib_data=dataset_calib,
    disable_names=disable_names,  # Skip lm_head
    basis_path=basis_path,
    rotation_path=args.rotation_path,
)

# Run calibration
calibrator.run()
```

**ResQCalibrator Class:** [calibrator.py](msmodelslim/pytorch/llm_ptq/llm_ptq_tools/resq/calibrator.py)

The `run()` method orchestrates:
1. Model preparation (`_prepare_model()`)
2. Layer norm fusion
3. Rotation application
4. Column rearrangement
5. Quantizer insertion
6. Calibration data forward passes
7. Scale/offset computation

---

### Step 9: Weight Saving with Specific Names

**Location:** [calibrator.py](msmodelslim/pytorch/llm_ptq/llm_ptq_tools/resq/calibrator.py) - `save()` method

**Script Invocation:**
```python
calibrator.save(
    output_path=save_directory,
    json_name="quant_model_description_resq.json",
    safetensors_name="quant_model_weight_resq.safetensors",
    save_type=["safe_tensor"],
)
```

**Saved Tensors:**

1. **Floating-point weights:**
   - `model.embed_tokens.weight` (rotated)
   - `model.norm.weight` (final layer norm)

2. **Dual quantized weights per linear layer**:
```python
weight_dict[f"{name}.weight_low"] = quant_weights['weight_low']   # 4-bit
weight_dict[f"{name}.scale_low"] = quant_weights['scale_low']
weight_dict[f"{name}.offset_low"] = quant_weights['offset_low']

weight_dict[f"{name}.weight_high"] = quant_weights['weight_high']  # 8-bit
weight_dict[f"{name}.scale_high"] = quant_weights['scale_high']
weight_dict[f"{name}.offset_high"] = quant_weights['offset_high']
```

3. **Per-layer online rotation matrices**:
```python
# Uc: K cache rotation (applied after RoPE during inference)
weight_dict[f'resq.layer.{i}.Uc'] = Uc.float().cpu()

# Ud: down_proj input rotation
weight_dict[f'resq.layer.{i}.Ud'] = Ud.float().cpu()
```

**Description JSON:**
```python
quant_description = {
    "model_quant_type": "W4A8_ResQ",
    "high_bits": 8,
    "low_bits": 4,
    "high_fraction": 0.125,
    "group_size": -1,
    "version": "1.0",
}
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT: Float Model                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ [1] Calibration Data Loading                                    │
│     get_calib_dataset_batch() → List of tokenized batches       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ [2] Activation Collection (if compute_basis=True)               │
│     InputCatcher hooks → Capture X at each layer                │
│     H_attn, H_mlp, H_value, H_key_pos, H_down_proj += X.T @ X   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ [3] PCA / Eigendecomposition                                    │
│     perform_eigen_decomp(H) → (eigenvalues, eigenvectors=P)     │
│     Sort by eigenvalue: low variance → high variance            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ [4] Random Rotation Generation                                  │
│     R = block_diag(R_low, R_mid, R_high)                        │
│     R1 (hidden), R2 (head), Rd (intermediate)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ [5] Rotation Fusion into Weights                                │
│     U = P @ R                                                   │
│     embed_tokens, qkv_proj, o_proj, mlp @ U                     │
│     + Optional Hadamard transform                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ [6] Column Reordering                                           │
│     rearrange_columns() → [4-bit | mid | 8-bit] layout          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ [7] Quantizer Replacement                                       │
│     nn.Linear → LinearResQQuantizer                             │
│     Dual weight storage (weight_low, weight_high)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ [8] Calibration Run                                             │
│     Forward passes → Collect min/max for scale computation      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ [9] Save Model                                                  │
│     ├── quant_model_weight_resq.safetensors                     │
│     │   ├── {layer}.weight_low / scale_low / offset_low (4-bit) │
│     │   ├── {layer}.weight_high / scale_high / offset_high(8-bit│
│     │   ├── resq.layer.{i}.Uc (K cache rotation)                │
│     │   └── resq.layer.{i}.Ud (down_proj rotation)              │
│     ├── quant_model_description_resq.json                       │
│     ├── resq_basis.pt (optional)                                │
│     └── config.json (quantize: "W4A8_ResQ")                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key File Summary

| Component | File Path |
|-----------|-----------|
| Main script | `example/Qwen/resq_qwen3_32b.py` |
| Config class | `pytorch/llm_ptq/llm_ptq_tools/resq/config.py` |
| Calibrator | `pytorch/llm_ptq/llm_ptq_tools/resq/calibrator.py` |
| Basis computation | `pytorch/llm_ptq/llm_ptq_tools/resq/processors/basis_processor.py` |
| Rotation/reorder | `pytorch/llm_ptq/llm_ptq_tools/resq/processors/resq_processor.py` |
| Weight quantizer | `pytorch/llm_ptq/llm_ptq_tools/resq/quant_modules.py` |
| Activation quantizer | `pytorch/llm_ptq/llm_ptq_tools/resq/components/act_quantizer.py` |
| Hadamard utils | `pytorch/llm_ptq/llm_ptq_tools/resq/utils/hadamard_utils.py` |

---

## Output Format Summary

**Safetensors file contains:**
- `model.embed_tokens.weight` - Rotated embeddings (float)
- `model.layers.{i}.self_attn.q_proj.weight_low/high` - Dual precision Q
- `model.layers.{i}.self_attn.k_proj.weight_low/high` - Dual precision K
- `model.layers.{i}.self_attn.v_proj.weight_low/high` - Dual precision V
- `model.layers.{i}.self_attn.o_proj.weight_low/high` - Dual precision O
- `model.layers.{i}.mlp.gate_proj.weight_low/high` - Dual precision gate
- `model.layers.{i}.mlp.up_proj.weight_low/high` - Dual precision up
- `model.layers.{i}.mlp.down_proj.weight_low/high` - Dual precision down
- `resq.layer.{i}.Uc` - Per-layer K cache rotation matrix
- `resq.layer.{i}.Ud` - Per-layer down_proj rotation matrix
