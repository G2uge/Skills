---
name: convert-gpu-to-xpu-yaml
description: |
  将 GPU YAML 训练配置文件转换为 XPU YAML 配置文件。
  基于预定义的映射规则，自动执行字段映射、参数变换和结构调整。
keywords: gpu, xpu, yaml, convert, paddleformer, config, 转换, GPU, XPU
---

> **执行约束**：执行者必须严格遵循本 skill 定义的调用顺序，不得擅自添加前置检查或跳过逻辑。子 skill 内部自行处理安装/更新判断。

# GPU 到 XPU YAML 配置转换器

本 Skill 指导 Agent 将 GPU 训练配置 YAML 文件转换为适用于 XPU 的配置。

## 输入参数

| 参数 | 必需 | 说明 |
|------|------|------|
| `gpu_yaml_path` | 是 | GPU YAML 配置文件路径 |
| `output_path` | 否 | 输出 XPU YAML 文件路径，默认在原文件旁添加 `_xpu` 后缀 |
| `reference_yaml` | 否 | 参考 XPU YAML 路径，用于补全缺失字段 |
| `model_path` | 否 | **覆盖**配置文件中 `model_name_or_path` 的路径 |
| `train_dataset_path` | 否 | **覆盖**训练数据集路径（优先级最高） |
| `eval_dataset_path` | 否 | **覆盖**评估数据集路径（优先级最高） |
| `dataset_dir` | 否 | **数据集目录路径**，Skill 从 GPU YAML 提取文件名与此目录拼接（优先级中） |
| `dataset_path` | 否 | **批量覆盖**所有数据集路径前缀（优先级最低） |

## 执行流程概览

```
输入: gpu_yaml_path
  │
  ▼
┌─────────────────────────┐
│  步骤1: 读取并解析GPU配置 │
│  - 加载YAML文件          │
│  - 提取所有字段和值       │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  步骤2: 应用转换规则      │
│  - 字段映射              │
│  - 参数变换              │
│  - 结构调整              │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  步骤3: 补全默认值        │
│  - 检查必需字段           │
│  - 使用参考配置补全       │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  步骤4: 生成XPU配置       │
│  - 写入文件              │
│  - 生成转换报告           │
└─────────────────────────┘
           │
           ▼
输出: XPU YAML 文件路径 + 转换报告
```

---

## 步骤1: 读取并解析 GPU 配置

**执行**：
```bash
cat {gpu_yaml_path}
```

**提取信息**：
- 所有顶层字段和值
- 嵌套结构（如 `freeze_config`, `optimizers` 等）
- 关键训练参数：`stage`, `model_name_or_path`, `per_device_train_batch_size`, `gradient_accumulation_steps` 等

---

## 步骤2: 应用转换规则

### 2.1 字段映射规则

| GPU 字段 | XPU 处理 | 说明 |
|---------|---------|------|
| （无 device） | **新增** `device: xpu` | XPU 运行必需 |
| `_attn_implementation` | **重命名+注释** `# attn_impl: {原值}` | 字段重命名 |
| `backend: nccl` | **值替换** `backend: bkcl` | 通信后端替换 |
| `distributed_backend: nccl` | **值替换** `distributed_backend: bkcl` | 通信后端替换 |
| `cuda*` | **注释或删除** | GPU 特有字段 |

**执行步骤**：
1. 检查是否包含 `_attn_implementation` 字段
   - 如果有：创建新行 `# attn_impl: {原值}`，删除原行
2. 在文件末尾添加 `device: xpu`
3. 将所有 `nccl` 替换为 `bkcl`

### 2.2 参数变换规则

| 参数 | GPU 典型值 | XPU 建议值 | 调整逻辑 |
|------|-----------|-----------|---------|
| `gradient_accumulation_steps` | 16 | 8 | **减半**（根据显存） |
| `packing` | true | false | **关闭**（内存管理） |
| `benchmark` | true | false | **关闭**（稳定性） |

**执行步骤**：
1. 找到 `gradient_accumulation_steps` 字段
   - 如果是 16 → 改为 8
   - 如果是 8 → 保持 8（或改为 4）
   - 规则：通常为 GPU 值的一半
2. 找到 `packing` 字段，改为 `false`
3. 找到 `benchmark` 字段，改为 `false`

### 2.3 结构调整规则

**需要注释掉的字段**（XPU 可能不支持）：

| 字段 | 处理方式 |
|------|---------|
| `_attn_implementation` | 重命名为 `attn_impl` 并注释 |
| `moe_deep_gemm` | 注释 `# moe_deep_gemm: {原值}` |
| `fuse_attention_qkv` | 注释 `# fuse_attention_qkv: {原值}` |
| `fuse_attention_ffn` | 注释 `# fuse_attention_ffn: {原值}` |
| `fuse_rms_norm` | 注释 `# fuse_rms_norm: {原值}` |
| `tp_delay_scale_loss` | 注释 `# tp_delay_scale_loss: {原值}` |

**新增字段**：

```yaml
# XPU 必需字段
device: xpu

# 可选优化字段（如果使用 Pipeline Parallel）
pp_delay_scale_loss: true
```

**执行步骤**：
1. 遍历所有字段，如果匹配上述列表，添加 `# ` 注释
2. 在文件末尾添加 `device: xpu`
3. 如果使用 PP，添加 `pp_delay_scale_loss: true`

### 2.4 训练步骤参数保持规则

**原则**：以下训练控制参数必须与 GPU 原配置**保持一致**，**禁止修改**：

| 参数 | 说明 | 处理规则 |
|------|------|---------|
| `max_steps` | 最大训练步数 | **保持原值**，禁止修改 |
| `eval_steps` | 评估步数 | **保持原值**，禁止修改 |
| `save_steps` | 保存检查点步数 | **保持原值**，禁止修改 |
| `num_train_epochs` | 训练轮数 | **保持原值**，禁止修改 |

**执行步骤**：
1. 从 GPU YAML 读取上述参数值
2. 直接原样写入 XPU YAML，**不做任何变换**
3. **即使传入了 `reference_yaml`，也不覆盖这些字段**
4. **优先级**：GPU 原值 > reference_yaml 中的值

**示例**：
```yaml
# GPU 原配置
max_steps: 100
eval_steps: 200
save_steps: 100

# XPU 生成配置（保持完全一致）
max_steps: 100
eval_steps: 200
save_steps: 100
```

---

## 步骤2.5: 应用路径覆盖（可选）

如果用户提供了路径覆盖参数，用其替换 GPU 配置中的对应路径。

### 2.5.1 路径覆盖规则

| 覆盖参数 | 目标字段 | 说明 | 优先级 |
|----------|----------|------|--------|
| `model_path` | `model_name_or_path` | 模型路径覆盖 | 高 |
| `train_dataset_path` | `train_dataset_path` / `train_data_path` | 训练数据路径直接覆盖 | 最高 |
| `eval_dataset_path` | `eval_dataset_path` / `eval_data_path` / `validation_dataset_path` | 评估数据路径直接覆盖 | 最高 |
| `dataset_dir` | `train_dataset_path` / `eval_dataset_path` | 数据集目录，自动提取 GPU YAML 文件名拼接 | 中 |
| `dataset_path` | 所有数据集路径 | 批量覆盖前缀 | 最低 |

### 2.5.2 执行步骤

1. **检查 `model_path` 参数**
   - 如果传入：替换 `model_name_or_path: {model_path}`
   - 在转换报告中标记 `[覆盖]`

2. **检查 `train_dataset_path` 参数（最高优先级）**
   - 如果传入：按顺序查找并替换以下字段：
     - `train_dataset_path`
     - `train_data_path`
     - `train_data`
   - 在转换报告中标记 `[覆盖]`

3. **检查 `eval_dataset_path` 参数（最高优先级）**
   - 如果传入：按顺序查找并替换以下字段：
     - `eval_dataset_path`
     - `eval_data_path`
     - `validation_dataset_path`
     - `val_dataset_path`
   - 在转换报告中标记 `[覆盖]`

4. **检查 `dataset_dir` 参数（当未传具体路径时使用）**
   - 仅当未传入 `train_dataset_path` 或 `eval_dataset_path` 时生效
   - 从 GPU YAML 中提取训练集文件名：
     ```bash
     train_file=$(grep -E "^train_dataset_path:|^train_data_path:" {gpu_yaml_path} | head -1 | awk -F'/' '{print $NF}' | tr -d ' ')
     train_file=${train_file:-train.jsonl}  # 默认值为 train.jsonl
     ```
   - 从 GPU YAML 中提取评估集文件名：
     ```bash
     eval_file=$(grep -E "^eval_dataset_path:|^eval_data_path:|^validation_dataset_path:|^val_dataset_path:" {gpu_yaml_path} | head -1 | awk -F'/' '{print $NF}' | tr -d ' ')
     eval_file=${eval_file:-val.jsonl}  # 默认值为 val.jsonl
     ```
   - 拼接完整路径：
     ```yaml
     train_dataset_path: "${dataset_dir}/${train_file}"
     eval_dataset_path: "${dataset_dir}/${eval_file}"
     ```
   - 在转换报告中标记 `[拼接]`

5. **检查 `dataset_path` 参数（批量覆盖，最低优先级）**
   - 仅当未传入以上任何路径参数时生效
   - 将所有数据集路径的前缀替换为 `dataset_path` 值

### 2.5.3 覆盖示例

**场景1**：直接覆盖路径（使用 `train_dataset_path`/`eval_dataset_path`）

```yaml
# GPU 原配置
model_name_or_path: /root/paddlejob/share-storage/old-model-path/Qwen3-VL-30B-A3B
train_dataset_path: /root/paddlejob/share-storage/old-data-path/train.jsonl
eval_dataset_path: /root/paddlejob/share-storage/old-data-path/val.jsonl
```

**用户传入参数**：
- `model_path: /root/paddlejob/zhangxiao_dev/data/Qwen3-VL-30B-A3B-Thinking`
- `train_dataset_path: /root/paddlejob/Gruge/data/coco_grounding/train.jsonl`
- `eval_dataset_path: /root/paddlejob/Gruge/data/coco_grounding/val2.jsonl`

**转换后 XPU 配置**：
```yaml
model_name_or_path: /root/paddlejob/zhangxiao_dev/data/Qwen3-VL-30B-A3B-Thinking
train_dataset_path: /root/paddlejob/Gruge/data/coco_grounding/train.jsonl
eval_dataset_path: /root/paddlejob/Gruge/data/coco_grounding/val2.jsonl
```

---

**场景2**：使用数据集目录拼接（使用 `dataset_dir`）

```yaml
# GPU 原配置
model_name_or_path: /root/old-path/Qwen3-VL-30B-A3B
train_dataset_path: /root/old-data/train.jsonl
eval_dataset_path: /root/old-data/val.jsonl
```

**用户传入参数**：
- `model_path: /root/paddlejob/zhangxiao_dev/data/Qwen3-VL-30B-A3B-Thinking`
- `dataset_dir: /root/paddlejob/tmp/datasets`

**执行拼接**：
1. 从 GPU YAML 提取文件名：`train.jsonl` 和 `val.jsonl`
2. 拼接完整路径：
   - `train_dataset_path: /root/paddlejob/tmp/datasets/train.jsonl`
   - `eval_dataset_path: /root/paddlejob/tmp/datasets/val.jsonl`

**转换后 XPU 配置**：
```yaml
model_name_or_path: /root/paddlejob/zhangxiao_dev/data/Qwen3-VL-30B-A3B-Thinking
train_dataset_path: /root/paddlejob/tmp/datasets/train.jsonl
eval_dataset_path: /root/paddlejob/tmp/datasets/val.jsonl
```

---

## 步骤3: 补全默认值

检查是否包含以下必需字段，如缺失则从参考配置或默认值补全：

**XPU 核心必需字段**：
```yaml
device: xpu              # 步骤2已添加
bf16: true               # 如果原配置没有，添加
amp_master_grad: true    # 如果原配置没有，添加
```

**通信配置**（如果配置中有分布式相关字段）：
```yaml
bkcl_timeout: 1000
bkcl_socket_ifname: "eth0"
bkcl_enable_xdr: 1
```

**执行步骤**：
1. 检查必需字段是否存在
2. 如缺失，根据 `reference_yaml` 或默认值补全
3. **不覆盖**已有值，仅补充缺失字段

---

## 步骤4: 生成 XPU 配置

**确定输出路径**：
- 如果提供了 `output_path`，使用该路径
- 否则：`{原文件名}_xpu.yaml`，如 `train_gpu.yaml` → `train_gpu_xpu.yaml`

**写入文件**：
```bash
cat > {output_path} << 'EOF'
{转换后的 YAML 内容}
EOF
```

**生成转换报告**：

```markdown
📋 GPU → XPU 配置转换报告
========================

源配置: {gpu_yaml_path}
目标配置: {output_path}

🔧 应用的转换规则:

[字段映射]
  ✓ 添加 device: xpu
  ✓ 注释 _attn_implementation → # attn_impl: flashmask
  ✓ backend: nccl → bkcl

[参数变换]
  ✓ gradient_accumulation_steps: 16 → 8
  ✓ packing: true → false
  ✓ benchmark: true → false

[结构调整]
  ✓ 注释 moe_deep_gemm
  ✓ 注释 fuse_attention_qkv
  ✓ 注释 fuse_attention_ffn

[路径覆盖]
  ✓ model_name_or_path: /root/share-storage/.../Qwen3-VL-30B → /root/zhangxiao_dev/data/Qwen3-VL-30B-Thinking
  ✓ train_dataset_path: /root/share-storage/.../train.jsonl → /root/Gruge/data/coco_grounding/train.jsonl
  ✓ eval_dataset_path: /root/share-storage/.../val.jsonl → /root/Gruge/data/coco_grounding/val2.jsonl

[默认值补全]
  ✓ bf16: true
  ✓ amp_master_grad: true

📊 关键参数对比:
  参数                           GPU值                                   XPU值
  -----------------------------  --------------------------------------  --------------------------------------
  device                         (无)                                    xpu
  model_name_or_path             /root/share-storage/.../Qwen3-VL-30B    /root/zhangxiao_dev/data/Qwen3-VL-30B  [覆盖]
  train_dataset_path             /root/share-storage/.../train.jsonl     /root/tmp/datasets/train.jsonl         [拼接]
  eval_dataset_path              /root/share-storage/.../val.jsonl       /root/tmp/datasets/val.jsonl           [拼接]
  gradient_accumulation_steps    16                                      8
  packing                        true                                    false
  benchmark                      true                                    false

✅ 转换完成，配置已保存至: {output_path}

⚠️ 注意事项:
  1. 请验证 XPU 环境是否正确配置
  2. 建议首次运行使用较小 max_steps 测试
  3. 如遇到 OOM，可进一步减小 batch_size 或增大 gradient_accumulation_steps
```

---

## 执行示例

### 示例1: 基础转换

**用户输入**:
- `gpu_yaml_path = "/data/configs/qwen3vl_gpu.yaml"`

**GPU 原配置（节选）**:
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

**Agent 执行转换**:

1. **字段映射**:
   - `_attn_implementation: flashmask` → `# attn_impl: flashmask`
   - 添加 `device: xpu`

2. **参数变换**:
   - `gradient_accumulation_steps: 16` → `8`
   - `packing: true` → `false`
   - `benchmark: true` → `false`

3. **结构调整**:
   - `moe_deep_gemm: true` → `# moe_deep_gemm: true`

**生成的 XPU 配置**:
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

### 示例2: 使用参考配置补全

**用户输入**:
- `gpu_yaml_path = "/data/configs/llama3_gpu.yaml"`
- `reference_yaml = "/templates/xpu_reference.yaml"`

**Agent 执行**:
- 执行标准转换
- 发现 GPU 配置缺少 `amp_master_grad` 字段
- 从 `reference_yaml` 补全：`amp_master_grad: true`

---

## 转换规则速查表

| GPU 配置 | XPU 转换 | 说明 |
|----------|----------|------|
| （无 device） | `device: xpu` | 新增必需 |
| `_attn_implementation` | `# attn_impl` | 重命名+注释 |
| `nccl` | `bkcl` | 后端替换 |
| `gradient_accumulation_steps: 16` | `gradient_accumulation_steps: 8` | 减半 |
| `packing: true` | `packing: false` | 关闭 |
| `benchmark: true` | `benchmark: false` | 关闭 |
| `moe_deep_gemm` | `# moe_deep_gemm` | 注释 |
| `fuse_attention_*` | `# fuse_attention_*` | 注释 |
| `bf16: true` | `bf16: true` | 保持不变 |

---

## 注意事项

1. **不覆盖原则**: 仅补充缺失字段，不覆盖 GPU 配置中已存在的值
2. **分步验证**: 每步转换后建议检查 YAML 语法正确性
3. **灵活调整**: `gradient_accumulation_steps` 可根据实际 XPU 显存微调
4. **融合算子**: 注释掉的融合算子可根据 XPU 支持情况逐步启用
5. **备份原文件**: 转换前建议备份原 GPU 配置
