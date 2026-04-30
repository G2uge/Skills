---
name: convert-gpu-to-xpu-yaml
description: |
  Convert GPU YAML training configuration files to XPU YAML configuration files.
  Automatically perform field mapping, parameter transformation, and structural adjustments based on predefined mapping rules.
keywords: gpu, xpu, yaml, convert, paddleformer, config, transform
---

# GPU to XPU YAML Configuration Converter

This Skill guides the Agent to convert GPU training configuration YAML files to XPU-compatible configurations.

## Input Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `gpu_yaml_path` | Yes | Path to GPU YAML configuration file |
| `output_path` | No | Output XPU YAML file path, defaults to adding `_xpu` suffix to original filename |
| `reference_yaml` | No | Reference XPU YAML path for filling in missing fields |
| `model_path` | No | **Override** the `model_name_or_path` in configuration file |
| `train_dataset_path` | No | **Override** training dataset path (highest priority) |
| `eval_dataset_path` | No | **Override** evaluation dataset path (highest priority) |
| `dataset_dir` | No | **Dataset directory path**, Skill extracts filenames from GPU YAML and constructs full paths (medium priority) |
| `dataset_path` | No | **Batch override** all dataset path prefixes (lowest priority) |

## Execution Flow Overview

```
Input: gpu_yaml_path
  │
  ▼
┌─────────────────────────────┐
│  Step 1: Read & Parse GPU    │
│  - Load YAML file            │
│  - Extract all fields/values │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Step 2: Apply Conversion    │
│  - Field mapping             │
│  - Parameter transformation  │
│  - Structural adjustments    │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Step 3: Fill Default Values │
│  - Check required fields     │
│  - Use reference config      │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Step 4: Generate XPU Config │
│  - Write file                │
│  - Generate conversion report│
└─────────────────────────────┘
               │
               ▼
Output: XPU YAML file path + conversion report
```

---

## Step 1: Read and Parse GPU Configuration

**Execute**:
```bash
cat {gpu_yaml_path}
```

**Extract Information**:
- All top-level fields and values
- Nested structures (e.g., `freeze_config`, `optimizers`, etc.)
- Key training parameters: `stage`, `model_name_or_path`, `per_device_train_batch_size`, `gradient_accumulation_steps`, etc.

---

## Step 2: Apply Conversion Rules

### 2.1 Field Mapping Rules

| GPU Field | XPU Processing | Description |
|-----------|----------------|-------------|
| (no device) | **Add** `device: xpu` | Required for XPU execution |
| `_attn_implementation` | **Rename+Comment** `# attn_impl: {original_value}` | Field renaming |
| `backend: nccl` | **Replace Value** `backend: bkcl` | Communication backend replacement |
| `distributed_backend: nccl` | **Replace Value** `distributed_backend: bkcl` | Communication backend replacement |
| `cuda*` | **Comment or Delete** | GPU-specific fields |

**Execution Steps**:
1. Check if `_attn_implementation` field exists
   - If yes: Create new line `# attn_impl: {original_value}`, delete original line
2. Add `device: xpu` at end of file
3. Replace all `nccl` with `bkcl`

### 2.2 Parameter Transformation Rules

| Parameter | GPU Typical Value | XPU Suggested Value | Adjustment Logic |
|-----------|-------------------|---------------------|------------------|
| `gradient_accumulation_steps` | 16 | 8 | **Halve** (based on memory) |
| `packing` | true | false | **Disable** (memory management) |
| `benchmark` | true | false | **Disable** (stability) |

**Execution Steps**:
1. Find `gradient_accumulation_steps` field
   - If 16 → change to 8
   - If 8 → keep 8 (or change to 4)
   - Rule: typically half of GPU value
2. Find `packing` field, change to `false`
3. Find `benchmark` field, change to `false`

### 2.3 Structural Adjustment Rules

**Fields to Comment Out** (XPU may not support):

| Field | Processing |
|-------|------------|
| `_attn_implementation` | Rename to `attn_impl` and comment |
| `moe_deep_gemm` | Comment `# moe_deep_gemm: {original_value}` |
| `fuse_attention_qkv` | Comment `# fuse_attention_qkv: {original_value}` |
| `fuse_attention_ffn` | Comment `# fuse_attention_ffn: {original_value}` |
| `fuse_rms_norm` | Comment `# fuse_rms_norm: {original_value}` |
| `tp_delay_scale_loss` | Comment `# tp_delay_scale_loss: {original_value}` |

**Fields to Add**:

```yaml
# XPU Required Fields
device: xpu

# Optional Optimization (if using Pipeline Parallel)
pp_delay_scale_loss: true
```

**Execution Steps**:
1. Iterate all fields, if matching above list, add `# ` comment
2. Add `device: xpu` at end of file
3. If using PP, add `pp_delay_scale_loss: true`

### 2.4 Training Step Parameter Preservation Rules

**Principle**: The following training control parameters must **remain consistent** with the original GPU configuration and **must not be modified**:

| Parameter | Description | Processing Rule |
|-----------|-------------|-----------------|
| `max_steps` | Maximum training steps | **Keep original value**, do not modify |
| `eval_steps` | Evaluation steps | **Keep original value**, do not modify |
| `save_steps` | Checkpoint save steps | **Keep original value**, do not modify |
| `num_train_epochs` | Training epochs | **Keep original value**, do not modify |

**Execution Steps**:
1. Read the above parameter values from GPU YAML
2. Write directly to XPU YAML **without any transformation**
3. **Do not override these fields even if `reference_yaml` is provided**
4. **Priority**: GPU original value > value in reference_yaml

**Example**:
```yaml
# GPU Original Config
max_steps: 100
eval_steps: 200
save_steps: 100

# XPU Generated Config (Keep exactly the same)
max_steps: 100
eval_steps: 200
save_steps: 100
```

---

## Step 2.5: Apply Path Overrides (Optional)

If user provides path override parameters, replace corresponding paths in GPU configuration.

### 2.5.1 Path Override Rules

| Override Parameter | Target Fields | Description | Priority |
|-------------------|---------------|-------------|----------|
| `model_path` | `model_name_or_path` | Model path override | High |
| `train_dataset_path` | `train_dataset_path` / `train_data_path` | Training dataset path direct override | Highest |
| `eval_dataset_path` | `eval_dataset_path` / `eval_data_path` / `validation_dataset_path` | Evaluation dataset path direct override | Highest |
| `dataset_dir` | `train_dataset_path` / `eval_dataset_path` | Dataset directory, auto-extracts filenames from GPU YAML | Medium |
| `dataset_path` | All dataset paths | Batch override prefix | Lowest |

### 2.5.2 Execution Steps

1. **Check `model_path` parameter**
   - If provided: Replace `model_name_or_path: {model_path}`
   - Mark as `[overridden]` in conversion report

2. **Check `train_dataset_path` parameter (highest priority)**
   - If provided: Find and replace fields in order:
     - `train_dataset_path`
     - `train_data_path`
     - `train_data`
   - Mark as `[overridden]` in conversion report

3. **Check `eval_dataset_path` parameter (highest priority)**
   - If provided: Find and replace fields in order:
     - `eval_dataset_path`
     - `eval_data_path`
     - `validation_dataset_path`
     - `val_dataset_path`
   - Mark as `[overridden]` in conversion report

4. **Check `dataset_dir` parameter (used when specific paths not provided)**
   - Only effective when `train_dataset_path` and `eval_dataset_path` are not provided
   - Extract training filename from GPU YAML:
     ```bash
     train_file=$(grep -E "^train_dataset_path:|^train_data_path:" {gpu_yaml_path} | head -1 | awk -F'/' '{print $NF}' | tr -d ' ')
     train_file=${train_file:-train.jsonl}  # default: train.jsonl
     ```
   - Extract evaluation filename from GPU YAML:
     ```bash
     eval_file=$(grep -E "^eval_dataset_path:|^eval_data_path:|^validation_dataset_path:|^val_dataset_path:" {gpu_yaml_path} | head -1 | awk -F'/' '{print $NF}' | tr -d ' ')
     eval_file=${eval_file:-val.jsonl}  # default: val.jsonl
     ```
   - Construct full paths:
     ```yaml
     train_dataset_path: "${dataset_dir}/${train_file}"
     eval_dataset_path: "${dataset_dir}/${eval_file}"
     ```
   - Mark as `[concatenated]` in conversion report

5. **Check `dataset_path` parameter (batch override, lowest priority)**
   - Only effective when none of the above path parameters are provided
   - Replace prefix of all dataset paths with `dataset_path` value

### 2.5.3 Override Examples

**Scenario 1**: Direct path override (using `train_dataset_path`/`eval_dataset_path`)

```yaml
# GPU Original Config
model_name_or_path: /root/paddlejob/share-storage/old-model-path/Qwen3-VL-30B-A3B
train_dataset_path: /root/paddlejob/share-storage/old-data-path/train.jsonl
eval_dataset_path: /root/paddlejob/share-storage/old-data-path/val.jsonl
```

**User Input Parameters**:
- `model_path: /root/paddlejob/zhangxiao_dev/data/Qwen3-VL-30B-A3B-Thinking`
- `train_dataset_path: /root/paddlejob/Gruge/data/coco_grounding/train.jsonl`
- `eval_dataset_path: /root/paddlejob/Gruge/data/coco_grounding/val2.jsonl`

**Converted XPU Config**:
```yaml
model_name_or_path: /root/paddlejob/zhangxiao_dev/data/Qwen3-VL-30B-A3B-Thinking
train_dataset_path: /root/paddlejob/Gruge/data/coco_grounding/train.jsonl
eval_dataset_path: /root/paddlejob/Gruge/data/coco_grounding/val2.jsonl
```

---

**Scenario 2**: Using dataset directory concatenation (using `dataset_dir`)

```yaml
# GPU Original Config
model_name_or_path: /root/old-path/Qwen3-VL-30B-A3B
train_dataset_path: /root/old-data/train.jsonl
eval_dataset_path: /root/old-data/val.jsonl
```

**User Input Parameters**:
- `model_path: /root/paddlejob/zhangxiao_dev/data/Qwen3-VL-30B-A3B-Thinking`
- `dataset_dir: /root/paddlejob/tmp/datasets`

**Concatenation Process**:
1. Extract filenames from GPU YAML: `train.jsonl` and `val.jsonl`
2. Construct full paths:
   - `train_dataset_path: /root/paddlejob/tmp/datasets/train.jsonl`
   - `eval_dataset_path: /root/paddlejob/tmp/datasets/val.jsonl`

**Converted XPU Config**:
```yaml
model_name_or_path: /root/paddlejob/zhangxiao_dev/data/Qwen3-VL-30B-A3B-Thinking
train_dataset_path: /root/paddlejob/tmp/datasets/train.jsonl
eval_dataset_path: /root/paddlejob/tmp/datasets/val.jsonl
```

---

## Step 3: Fill Default Values

Check for required fields, if missing, fill from reference config or default values:

**XPU Core Required Fields**:
```yaml
device: xpu              # Added in Step 2
bf16: true               # Add if not in original config
amp_master_grad: true    # Add if not in original config
```

**Communication Config** (if distributed fields exist in config):
```yaml
bkcl_timeout: 1000
bkcl_socket_ifname: "eth0"
bkcl_enable_xdr: 1
```

**Execution Steps**:
1. Check if required fields exist
2. If missing, fill from `reference_yaml` or default values
3. **Do not override** existing values, only fill missing fields

---

## Step 4: Generate XPU Configuration

**Determine Output Path**:
- If `output_path` provided, use it
- Otherwise: `{original_filename}_xpu.yaml`, e.g., `train_gpu.yaml` → `train_gpu_xpu.yaml`

**Write File**:
```bash
cat > {output_path} << 'EOF'
{Converted YAML content}
EOF
```

**Generate Conversion Report**:

```markdown
📋 GPU → XPU Configuration Conversion Report
============================================

Source: {gpu_yaml_path}
Target: {output_path}

🔧 Applied Conversion Rules:

[Field Mapping]
  ✓ Added device: xpu
  ✓ Commented _attn_implementation → # attn_impl: flashmask
  ✓ backend: nccl → bkcl

[Parameter Transformation]
  ✓ gradient_accumulation_steps: 16 → 8
  ✓ packing: true → false
  ✓ benchmark: true → false

[Structural Adjustment]
  ✓ Commented moe_deep_gemm
  ✓ Commented fuse_attention_qkv
  ✓ Commented fuse_attention_ffn

[Path Overrides]
  ✓ model_name_or_path: /root/share-storage/.../Qwen3-VL-30B → /root/zhangxiao_dev/data/Qwen3-VL-30B-Thinking
  ✓ train_dataset_path: /root/share-storage/.../train.jsonl → /root/Gruge/data/coco_grounding/train.jsonl
  ✓ eval_dataset_path: /root/share-storage/.../val.jsonl → /root/Gruge/data/coco_grounding/val2.jsonl

[Default Value Filling]
  ✓ bf16: true
  ✓ amp_master_grad: true

📊 Key Parameter Comparison:
  Parameter                      GPU Value                               XPU Value
  -----------------------------  --------------------------------------  --------------------------------------
  device                         (none)                                  xpu
  model_name_or_path             /root/share-storage/.../Qwen3-VL-30B    /root/zhangxiao_dev/data/Qwen3-VL-30B  [overridden]
  train_dataset_path             /root/share-storage/.../train.jsonl     /root/tmp/datasets/train.jsonl         [concatenated]
  eval_dataset_path              /root/share-storage/.../val.jsonl       /root/tmp/datasets/val.jsonl           [concatenated]
  gradient_accumulation_steps    16                                      8
  packing                        true                                    false
  benchmark                      true                                    false

✅ Conversion complete, config saved to: {output_path}

⚠️ Notes:
  1. Verify XPU environment is correctly configured
  2. Recommend using smaller max_steps for initial testing
  3. If OOM occurs, further reduce batch_size or increase gradient_accumulation_steps
```

---

## Execution Examples

### Example 1: Basic Conversion

**User Input**:
- `gpu_yaml_path = "/data/configs/qwen3vl_gpu.yaml"`

**Original GPU Config (excerpt)**:
```yaml
model_name_or_path: Qwen/Qwen3-VL-30B-A3B
stage: sft
_attn_implementation: flashmask
gradient_accumulation_steps: 16
packing: true
benchmark: true
moe_deep_gemm: true
bf16: true
```

**Agent Execution**:

1. **Field Mapping**:
   - `_attn_implementation: flashmask` → `# attn_impl: flashmask`
   - Add `device: xpu`

2. **Parameter Transformation**:
   - `gradient_accumulation_steps: 16` → `8`
   - `packing: true` → `false`
   - `benchmark: true` → `false`

3. **Structural Adjustment**:
   - `moe_deep_gemm: true` → `# moe_deep_gemm: true`

**Generated XPU Config**:
```yaml
model_name_or_path: Qwen/Qwen3-VL-30B-A3B
stage: sft
# attn_impl: flashmask
gradient_accumulation_steps: 8
packing: false
benchmark: false
# moe_deep_gemm: true
bf16: true
device: xpu
```

### Example 2: Using Reference Config for Filling

**User Input**:
- `gpu_yaml_path = "/data/configs/llama3_gpu.yaml"`
- `reference_yaml = "/templates/xpu_reference.yaml"`

**Agent Execution**:
- Execute standard conversion
- Discover GPU config missing `amp_master_grad` field
- Fill from `reference_yaml`: `amp_master_grad: true`

---

## Conversion Rules Quick Reference

| GPU Config | XPU Conversion | Description |
|------------|----------------|-------------|
| (no device) | `device: xpu` | Add required |
| `_attn_implementation` | `# attn_impl` | Rename+Comment |
| `nccl` | `bkcl` | Backend replacement |
| `gradient_accumulation_steps: 16` | `gradient_accumulation_steps: 8` | Halve |
| `packing: true` | `packing: false` | Disable |
| `benchmark: true` | `benchmark: false` | Disable |
| `moe_deep_gemm` | `# moe_deep_gemm` | Comment |
| `fuse_attention_*` | `# fuse_attention_*` | Comment |
| `bf16: true` | `bf16: true` | Keep unchanged |

---

## Notes

1. **No Override Principle**: Only fill missing fields, do not override existing values in GPU config
2. **Step-by-Step Validation**: Recommend checking YAML syntax correctness after each conversion step
3. **Flexible Adjustment**: `gradient_accumulation_steps` can be fine-tuned based on actual XPU memory
4. **Fusion Operators**: Commented fusion operators can be gradually enabled based on XPU support
5. **Backup Original**: Recommend backing up original GPU config before conversion
