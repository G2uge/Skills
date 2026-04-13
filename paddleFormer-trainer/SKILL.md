---
name: paddleFormer-trainer
description: |
  基于GPU配置自动生成XPU训练配置并启动PaddleFormer模型训练。
  支持智能YAML检索、GPU→XPU配置自动转换、错误检测与自修复等高级功能。
  Use when users:
  (1) want to train a model using PaddleFormer on XPU
  (2) need to convert GPU config to XPU config automatically
  (3) ask about model training with auto-repair capabilities
  (4) mention converting GPU YAML to XPU YAML
  Keywords: paddleformer, train, xpu, gpu, convert, yaml, config, 训练, 转换
---

# PaddleFormer XPU 训练配置自动生成器

## 核心能力

这个 Skill 实现从 GPU 配置到 XPU 配置的自动化转换与训练启动，包含以下核心能力：

1. **智能YAML检索** - 基于语义相似性自动检索最匹配的GPU训练配置
2. **智能配置转换** - 基于实际配置对比总结的映射规则，将 GPU YAML 转换为 XPU YAML
3. **启动脚本生成** - 基于模板自动构建 XPU 训练启动脚本
4. **错误自修复** - 训练失败时自动检测、分类并修复错误
5. **参考配置支撑** - 使用预置的可运行 XPU 配置作为修复基准

## 工作流程

```
模型识别 → 智能YAML检索 → GPU→XPU配置转换 → 启动执行 → 自修复迭代
```

### 阶段1：Agent 驱动的 GPU YAML 检索与选择

**目标**：通过 Agent 自主完成 PaddleFormers 仓库定位和最优 GPU YAML 模板选择

> **设计原则**：从"代码驱动"转向"Agent 驱动"
> - 不再依赖固定的评分规则和硬编码逻辑
> - 由 Agent 基于语义理解自主决策
> - Skill 提供基础数据和能力，Agent 完成智能匹配

#### 1.1 仓库定位（Agent 调用 Skill）

Agent 调用 Skill 提供的仓库定位能力，自动识别 PaddleFormers 仓库位置。

**定位策略（分层优先）**：

| 层级 | 定位方式 | 说明 |
|------|----------|------|
| Layer 1 | Python运行环境 | 通过 `import paddleformers` 获取模块路径，支持 editable install |
| Layer 2 | 常见开发目录 | 当前目录、用户主目录、环境变量路径、固定路径 |
| Layer 3 | 全局扩展搜索 | 受限的全局搜索（限制深度为3层） |

**Agent 执行流程**：
```
1. 调用 Skill 的仓库定位功能
2. 获取检索过程报告，理解搜索路径和结果
3. 如果定位失败，根据报告提示用户或扩展搜索
```

**示例输出**：
```
开始分层检索 PaddleFormers 仓库...
  Layer 1: 搜索Python运行环境...
    ✓ 从paddleformers模块定位: /root/paddlejob/Gruge/PaddleFormers
✓ Layer 1 成功: 在Python环境中找到
```

#### 1.2 候选 YAML 检索（Agent 自主完成）

Agent 基于任务需求（模型名称、任务类型等）在仓库中检索候选 GPU YAML 文件。

**Agent 任务**：
```
1. 解析模型名称，提取关键特征：
   - 模型系列（Qwen3、Llama、DeepSeek等）
   - 参数规模（8B、30B、A3B等）
   - 任务类型（sft、pretrain、instruct等）
   - 结构特征（VL、MoE、Text等）

2. 扫描候选 YAML 文件列表

3. 对每个候选文件，提取并理解其语义信息
```

**候选文件数据结构示例**：
```yaml
model_name: "Qwen3-VL-30B-A3B"
features:
  family: "qwen3"
  variant: "vl"
  size: "30B"
  structure: ["moe", "vl"]
  task_type: ""

candidates:
  - file_path: "/path/to/qwen3vl_sft.yaml"
    file_name: "qwen3vl_sft.yaml"
    relative_path: "examples/config/qwen3vl/sft.yaml"
    model_name_or_path: "Qwen/Qwen3-VL-30B-A3B"
    stage: "sft"
    device: "gpu"
    path_hints:
      model_family_from_path: ["qwen3vl"]
      task_from_path: ["sft"]
      device_from_path: "gpu"
    params:
      per_device_train_batch_size: 1
      gradient_accumulation_steps: 8
      # ... 其他参数
```

#### 1.3 最优模板选择（Agent 语义决策）

Agent 基于上下文语义进行综合判断，选择最合适的 YAML 文件作为模板。

**选择维度（Agent 自主分析）**：

| 维度 | 分析要点 | 示例 |
|------|----------|------|
| **模型结构匹配** | 模型系列、参数规模、结构特征（VL/MoE） | Qwen3-VL-30B 优先匹配 qwen3vl 配置 |
| **任务类型匹配** | 训练阶段（sft/pretrain/instruct） | instruct 任务优先选 instruct.yaml |
| **路径语义** | 目录结构暗示的用途 | `examples/config/qwen3vl/sft/` 暗示这是 qwen3vl 的 SFT 配置 |
| **配置完整性** | 必需字段是否齐全 | 检查 model_name_or_path、stage、batch_size 等 |
| **设备适配性** | 是否为 GPU 配置 | 优先选择 device: gpu 或未标注 XPU 的配置 |

**Agent 决策流程**：
```
1. 理解目标模型特征
   - 从模型名称提取：family, size, variant, task

2. 分析每个候选 YAML 的语义
   - 读取文件内容，理解配置意图
   - 分析文件路径，获取上下文线索
   - 对比模型特征与配置特征

3. 综合判断并排序
   - 高优先级：模型系列完全匹配 + 任务类型匹配
   - 中优先级：模型系列匹配 + 配置完整
   - 低优先级：仅部分特征匹配

4. 选择最优模板
   - 如果存在明显最佳匹配，选择该文件
   - 如果多个候选相近，列出供用户确认或选择最完整的
   - 如果没有匹配，考虑放宽条件或进入回退流程
```

**决策示例**：
```
目标模型: Qwen3-VL-30B-A3B-Instruct

候选分析:
  1. qwen3vl_sft.yaml
     - 模型系列匹配: ✓ (qwen3vl)
     - 任务类型: SFT（接近 Instruct）
     - 配置完整性: 完整
     - 推荐度: ★★★★☆

  2. qwen3vl_pretrain.yaml
     - 模型系列匹配: ✓ (qwen3vl)
     - 任务类型: Pretrain（差距较大）
     - 配置完整性: 完整
     - 推荐度: ★★★☆☆

  3. ernie45vl_sft.yaml
     - 模型系列匹配: ✗ (不同系列)
     - 推荐度: ★★☆☆☆

Agent 决策: 选择 qwen3vl_sft.yaml 作为最佳模板
理由: 模型系列完全匹配，任务类型相近（SFT 与 Instruct 都是微调任务），配置完整
```

**重要原则**：
- Agent 应解释选择理由，保持透明
- 当不确定时，可以列出 Top-3 候选供用户确认
- 避免完全依赖文件名，应深入理解配置内容

### 阶段1.4：Agent 使用 Skill 的具体步骤

**Agent 调用 Skill 完成仓库定位和 YAML 检索的完整流程**：

```
步骤1: 初始化检索器（Agent 驱动模式）
  - 调用: GPUYamlFinder(auto_search=False)
  - 说明: 不自动搜索，由 Agent 控制搜索流程

步骤2: 执行分层搜索（逐层决策）

  2.1 Layer 1: 环境感知层
    - 调用: finder.search_layer1_environment()
    - 分析返回的 found_paths 和 environment_info
    - 决策:
      ├─ 如果找到有效路径 → 使用 finder.set_paddleformers_root(path) 设置
      └─ 如果未找到 → 继续 Layer 2

  2.2 Layer 2: 工作空间层
    - 调用: finder.search_layer2_workspace()
    - 分析当前工作空间和常见目录
    - 决策:
      ├─ 如果找到有效路径 → 使用 finder.set_paddleformers_root(path) 设置
      └─ 如果未找到 → 继续 Layer 3

  2.3 Layer 3: 扩展搜索层（可选）
    - 调用: finder.search_layer3_extended(max_depth=3)
    - 权衡搜索时间与成功率
    - 决策:
      ├─ 如果找到有效路径 → 使用 finder.set_paddleformers_root(path) 设置
      └─ 如果未找到 → 提示用户手动指定路径

  或者使用便捷方法:
    - 调用: finder.execute_layered_search(stop_on_first=True)
    - 自动按优先级执行三层搜索，返回第一个找到的路径

步骤3: 获取搜索上下文（用于分析和报告）
  - 调用: finder.get_search_context()
  - 获取完整的环境信息和搜索结果
  - 用于生成搜索报告和决策说明

步骤4: 获取候选 YAML 列表
  - 调用: finder.find_candidate_yamls(model_name)
  - 获取返回数据结构:
    {
      "model_name": "目标模型名称",
      "features": {提取的模型特征},
      "candidates": [{YAML文件详细信息列表}],
      "paddleformers_root": "仓库路径",
      "search_report": "搜索过程报告"
    }

步骤5: 分析每个候选 YAML
  对每个 candidate in candidates:
    - 读取 file_name, relative_path 理解文件位置
    - 查看 model_name_or_path 字段，判断模型匹配度
    - 查看 stage 字段，判断任务类型匹配度
    - 分析 path_hints 中的语义信息
    - 查看 params 中的关键训练参数

步骤6: 基于语义理解做决策
  综合以下维度评估:
    ✓ 模型系列是否匹配 (qwen3, llama, deepseek等)
    ✓ 结构特征是否匹配 (VL, MoE, Text等)
    ✓ 任务类型是否匹配或相近 (sft/pretrain/instruct)
    ✓ 配置是否完整 (关键字段是否齐全)
    ✓ 路径语义是否暗示正确用途

步骤7: 选择最优模板
  - 如果存在明显最佳匹配，选择该文件
  - 如果多个候选相近，列出Top-3并说明理由
  - 如果没有匹配，考虑放宽条件或进入回退流程
```

**Agent 决策报告示例**：
```
📋 Agent YAML 选择报告
========================
目标模型: Qwen3-VL-30B-A3B-Instruct

模型特征分析:
  - 系列: qwen3
  - 结构: VL + MoE (A3B激活参数)
  - 任务: Instruct (微调)

候选 YAML 分析:

  [候选1] examples/config/qwen3vl/sft.yaml
    ✓ 模型系列匹配: file_name 包含 qwen3vl
    ✓ 任务类型匹配: stage=SFT, 与 Instruct 同为微调任务
    ✓ 结构匹配: 配置适用于 VL 模型
    ✓ 路径语义: examples/config/qwen3vl/ 明确指向 qwen3vl 配置
    ✓ 配置完整性: model_name_or_path, stage, batch_size 等字段齐全
    → 推荐度: ★★★★★ (首选)

  [候选2] examples/config/qwen3vl/pretrain.yaml
    ✓ 模型系列匹配: qwen3vl
    ✗ 任务类型不匹配: stage=Pretrain, 与 Instruct 差距较大
    → 推荐度: ★★★☆☆

Agent 决策: 选择 examples/config/qwen3vl/sft.yaml
决策理由:
  1. 模型系列完全匹配 (qwen3vl)
  2. 任务类型相近 (SFT 与 Instruct 都是微调)
  3. 配置适用于 VL+MoE 结构
  4. 配置字段完整，可直接使用
```

### 阶段2：Agent 驱动的 GPU → XPU 配置转换（核心）

**目标**：由 Agent 基于 GPU YAML 和转换规则，动态生成 XPU YAML

> **设计原则**：从"代码驱动"转向"Agent 驱动"
> - 移除硬编码的转换逻辑（`_apply_mapping_rules`等）
> - Agent 根据 SKILL.md 规则和转换上下文，自主决策每个字段的转换方式
> - Skill 提供原始配置、规则定义、参考配置，Agent 执行转换

#### 2.0 Agent 驱动的配置转换流程

```
开始配置转换
    ↓
获取转换上下文（Skill 提供）
    - GPU 原始配置
    - 模型特征分析
    - 转换规则定义
    - 参考配置
    ↓
Agent 基于规则逐字段分析
    - 哪些字段需要映射/重命名？
    - 哪些参数需要调整？
    - 哪些字段需要注释/删除？
    - 哪些字段需要新增？
    ↓
Agent 执行转换决策
    - 应用字段映射规则
    - 应用参数变换规则
    - 应用结构调整规则
    - 使用参考配置补全缺失字段
    ↓
生成 XPU YAML 文件
    ↓
验证配置完整性
```

#### 2.1 获取转换上下文

Agent 调用 `config_generator.get_conversion_context()` 获取完整上下文：

```python
# 步骤1: 获取转换上下文
context = generator.get_conversion_context(
    gpu_config_path="/path/to/gpu_config.yaml",
    model_name="Qwen3-VL-30B-A3B-Thinking"
)

# 上下文数据结构
{
  "source": {
    "gpu_config_path": "...",
    "gpu_config": {原始GPU配置字典},
    "model_name": "Qwen3-VL-30B-A3B-Thinking",
    "model_features": {
      "family": "qwen3",
      "variant": "vl",
      "structure": ["moe", "vl"],
      "task_type": "thinking",
      ...
    }
  },
  "rules": {
    "field_mappings": {...},
    "parameter_transformations": {...},
    "structural_adjustments": {...},
    "pattern_rules": {...},
    "model_specific": {...}
  },
  "reference": {
    "default_values": {...},
    "reference_config": {...}
  },
  "validation": {
    "validation_rules": {...}
  }
}
```

#### 2.2 Agent 执行配置转换

Agent 基于转换上下文，按以下步骤执行转换：

**步骤1: 字段映射（Field Mapping）**

```
规则来源: context["rules"]["field_mappings"]

执行逻辑:
1. 遍历 GPU 配置中的所有字段
2. 检查是否需要映射:
   - source_key → target_key
   - source_value → target_value
3. 应用映射规则:

示例:
  GPU: _attn_implementation: flashmask
  XPU: # attn_impl: flashmask  (注释掉，字段重命名)

  GPU: (无 device 字段)
  XPU: device: xpu  (新增必需字段)

  GPU: backend: nccl
  XPU: backend: bkcl  (值替换)
```

**步骤2: 参数变换（Parameter Transformation）**

```
规则来源: context["rules"]["parameter_transformations"]

执行逻辑:
1. 识别需要调整的参数
2. 根据变换类型执行调整:

硬件相关调整:
  gradient_accumulation_steps: 16 → 8
  原因: XPU 显存特性，通常为 GPU 值的一半

布尔值切换:
  packing: true → false
  benchmark: true → false
  原因: XPU 内存管理和稳定性考虑

精度保持:
  bf16: true → true  (保持不变，XPU 支持 bf16)
```

**步骤3: 结构调整（Structural Adjustment）**

```
规则来源: context["rules"]["structural_adjustments"]

执行逻辑:
1. 注释掉 XPU 不支持的字段:
   - _attn_implementation  →  # _attn_implementation
   - moe_deep_gemm: true  →  # moe_deep_gemm: true
   - fuse_attention_qkv  →  # fuse_attention_qkv
   - fuse_attention_ffn  →  # fuse_attention_ffn
   - fuse_rms_norm  →  # fuse_rms_norm

2. 新增 XPU 必需字段:
   - device: xpu
   - pp_delay_scale_loss: true (可选优化)

3. 移除 GPU 特有字段（如果存在）:
   - cuda_*
   - nccl_*
```

**步骤4: 应用模型特定规则**

```
规则来源: context["rules"]["model_specific"]

根据模型特征匹配的规则:

Qwen3-VL-30B-A3B 匹配规则 "Qwen3-VL.*":
  必需字段:
    freeze_config: "freeze_vision freeze_aligner"
    template: "qwen3_vl"
    moe_grouped_gemm: true
    use_expert_parallel: true

  XPU 调整:
    gradient_accumulation_steps: 8
    packing: false
    benchmark: false
    device: "xpu"
```

**步骤5: 默认值补全**

```
参考来源: context["reference"]["default_values"]

补全缺失的必需字段（不覆盖已有值）:
  - device: xpu
  - bf16: true
  - amp_master_grad: true
  - bkcl_timeout: 1000
  - bkcl_socket_ifname: "eth0"
```

**步骤6: 输出路径统一**

```
基于 output_dir 派生路径:
  output_dir: ./checkpoints/train_xxx
  logging_dir: ./checkpoints/train_xxx/vdl_log
  dataset_output_dir: ./checkpoints/train_xxx/dataset_cache
```

#### 2.3 保存转换结果

Agent 完成转换后，调用 Skill 保存配置：

```python
# Agent 生成的 XPU 配置
xpu_config = {...}  # Agent 根据规则转换后的配置

# 保存配置
config_path, generation_method = generator.save_xpu_config(
    xpu_config=xpu_config,
    model_name="Qwen3-VL-30B-A3B-Thinking",
    output_dir="./checkpoints",
    gpu_config_path="/path/to/gpu_config.yaml",
    generation_report="Agent 转换报告..."
)
```

#### 2.4 Agent 转换报告示例

```
📋 Agent GPU→XPU 配置转换报告
================================

源配置: PaddleFormers/tests/config/benchmark/config/sft/Qwen3-VL-30B-A3B-Instruct.yaml
目标模型: Qwen3-VL-30B-A3B-Thinking

模型特征:
  - 系列: qwen3
  - 结构: VL + MoE
  - 任务: Thinking (微调)

转换详情:

[字段映射]
  ✓ _attn_implementation → # attn_impl (注释，flashmask 可能不支持)
  ✓ (新增) device: xpu

[参数变换]
  ✓ gradient_accumulation_steps: 16 → 8 (XPU显存调整)
  ✓ packing: true → false (内存管理)
  ✓ benchmark: true → false (稳定性)

[结构调整]
  ✓ 注释 moe_deep_gemm (XPU可能不支持)
  ✓ 注释 fuse_attention_qkv (需验证)
  ✓ 注释 fuse_attention_ffn (需验证)
  ✓ 注释 fuse_rms_norm (需验证)

[模型特定规则]
  ✓ 应用 Qwen3-VL 规则
  ✓ freeze_config: freeze_vision freeze_aligner
  ✓ template: qwen3_vl
  ✓ use_expert_parallel: true

[默认值补全]
  ✓ bf16: true
  ✓ amp_master_grad: true
  ✓ bkcl_timeout: 1000wen jian

输出路径:
  - XPU YAML: ./checkpoints/train_Qwen3_VL_30B_A3B_Thinking_xpu_20250115.yaml
  - output_dir: ./checkpoints/train_Qwen3_VL_30B_A3B_Thinking_xpu_20250115
  - logging_dir: ./checkpoints/train_Qwen3_VL_30B_A3B_Thinking_xpu_20250115/vdl_log

✅ 转换完成，配置已保存
```

#### 配置生成来源优先级

**优先级1：GPU YAML 配置（强制优先）**
- **强制要求**：在 PaddleFormers 仓库中查找目标模型的 GPU YAML 文件
- 以 GPU 配置为基础模板进行 GPU→XPU 转换
- 保留原配置中的模型结构、训练参数和数据配置
- 转换后使用参考 YAML 仅进行参数修正与适配（如设备类型、算子配置等）
- **转换方式**：Agent 基于规则自主执行，非硬编码

**优先级2：XPU 参考配置（回退方案，需显式允许）**
- **仅在以下情况触发**：
  - 在 PaddleFormers 仓库中**无法找到**对应 GPU YAML
  - 用户**显式设置** `allow_fallback=True`
- 使用 `reference_configs/xpu_reference.yaml` 作为基础模板
- 结合模型信息推断生成配置
- **必须在输出中明确说明使用了回退方案**

#### 正确的配置生成流程

```
开始生成 XPU 配置
    ↓
在 PaddleFormers 仓库中查找 GPU YAML
    ↓
┌──────────────────┐
│ 是否找到 GPU YAML? │
└──────────────────┘
    ↓ 是                      ↓ 否
使用 GPU YAML 作为基础        检查 allow_fallback 参数
应用 GPU→XPU 映射规则         ↓ True          ↓ False
使用参考配置修正参数        使用参考配置      报错退出
    ↓                        生成配置
标记为"基于GPU配置转换"      标记为"回退方案"
    ↓                          ↓
输出 XPU YAML 文件并说明生成方式
```

#### 强制模式 vs 回退模式

**强制模式（默认）**：
```python
# 找不到 GPU YAML 时报错退出
launcher.prepare_training("Qwen3-VL-30B", allow_fallback=False)
```

**回退模式（需显式声明）**：
```python
# 找不到 GPU YAML 时使用参考配置
launcher.prepare_training("CustomModel", allow_fallback=True)
```

**错误提示示例**（强制模式下未找到 GPU YAML）：
```
❌ 错误：未找到 Qwen3-VL-30B-A3B 的 GPU YAML 配置

📋 搜索过程:
  开始分层检索 PaddleFormers 仓库...
  ✓ Layer 1 成功: 在Python环境中找到 - /root/paddlejob/PaddleFormers
  查找 GPU YAML: Qwen3-VL-30B-A3B
    模型特征: {'family': 'qwen3', 'size': '30B', ...}
    扫描到 3 个候选文件
      qwen3_vl_8b_sft.yaml: 得分 45
      qwen3_vl_30b_pretrain.yaml: 得分 55
      qwen3_vl_30b_instruct.yaml: 得分 58
  ⚠ 最佳匹配得分过低 (58)，可能不是有效匹配

💡 建议操作:
   1. 确认模型名称拼写正确
   2. 检查 PaddleFormers 仓库路径
   3. 手动指定GPU配置路径
   4. 如需使用参考配置生成，请设置 allow_fallback=True
```

#### 2.1 转换规则体系

基于 **Qwen3-VL-30B-A3B** GPU→XPU 实际配置对比分析，总结出以下规则体系：

##### 2.1.1 字段映射规则（直接替换）

| 规则 | GPU字段 | XPU字段 | 处理方式 |
|------|---------|---------|----------|
| 设备声明 | （省略，默认cuda） | `device: xpu` | **新增必需** |
| Attention实现 | `_attn_implementation` | `attn_impl` | **重命名+注释** |
| 通信后端 | `nccl` | `bkcl` | **值替换** |
| 设备类型 | `cuda` | `xpu` | **值替换** |

**AI 执行指令**：
```
字段映射执行步骤：
1. 检查GPU配置是否包含 _attn_implementation 字段
   - 如果有：在XPU配置中重命名为 attn_impl 并注释掉
   - 原因：XPU对flashmask支持可能需要特殊版本

2. 添加 device: xpu 字段到文件末尾
   - 这是XPU运行的必需字段

3. 如果出现 backend/distributed_backend 字段且值为 nccl
   - 替换为 bkcl
```

##### 2.1.2 参数变换规则（值调整）

基于实际配置对比，以下参数需要调整：

| 参数 | GPU值 | XPU值 | 调整逻辑 |
|------|-------|-------|----------|
| `gradient_accumulation_steps` | 16 | 8 | **减半**，根据XPU内存调整 |
| `packing` | true | false | **关闭**，简化XPU内存管理 |
| `benchmark` | true | false | **关闭**，确保XPU稳定性 |
| `max_steps` | 500 | 100 | **任务相关**，根据任务调整 |
| `save_steps` | 500 | 100 | **同步调整**，与max_steps一致 |

**AI 执行指令**：
```
参数变换执行步骤：
1. 调整梯度累积步数：
   - 如果 GPU 配置中 gradient_accumulation_steps = 16
   - XPU 配置设置为 8（经验值，可根据显存微调）
   - 规则：通常为GPU值的一半，保持有效batch size

2. 关闭数据打包：
   - 设置 packing: false
   - 原因：XPU内存管理策略不同

3. 关闭benchmark模式：
   - 设置 benchmark: false
   - 原因：确保XPU运行稳定性

4. 训练步数相关参数（max_steps, save_steps）：
   - 这些参数主要与训练任务相关，非硬件差异
   - 如果是相同任务，保持与GPU一致
   - 如果是不同任务，根据任务类型调整
```

##### 2.1.3 结构调整规则（增删改）

**需要注释掉的字段**（XPU可能不支持）：

| 字段 | GPU状态 | XPU处理 | 原因 |
|------|---------|---------|------|
| `_attn_implementation` | 有值 | **注释掉** | flashmask支持问题 |
| `moe_deep_gemm` | true | **注释掉** | Deep GEMM可能是GPU特有 |
| `fuse_attention_qkv` | （无） | **保持注释** | 融合算子需验证 |
| `fuse_attention_ffn` | （无） | **保持注释** | 融合算子需验证 |
| `fuse_rms_norm` | （无） | **保持注释** | 融合算子需验证 |

**新增字段**：

| 字段 | 值 | 位置 | 说明 |
|------|-----|------|------|
| `device` | `xpu` | 文件末尾 | XPU必需 |
| `pp_delay_scale_loss` | `true` | PP配置段 | XPU优化（可选） |

**AI 执行指令**：
```
结构调整执行步骤：
1. 注释掉以下字段（如果存在）：
   - _attn_implementation
   - moe_deep_gemm
   - fuse_attention_qkv
   - fuse_attention_ffn
   - fuse_rms_norm
   - tp_delay_scale_loss

   注释格式：
   原：moe_deep_gemm: true
   改：# moe_deep_gemm: true

2. 添加XPU必需字段：
   - 在文件末尾添加 device: xpu

3. 可选添加XPU优化字段：
   - 在PP配置段添加 pp_delay_scale_loss: true
```

##### 2.1.4 参考配置使用原则

**核心原则**：参考配置仅用于**修正和补全**，不直接作为生成基础

**使用场景**：
1. **参数修正**：GPU→XPU转换后，使用参考配置修正XPU特有参数
2. **缺失补全**：当GPU配置缺少必需字段时，从参考配置补全
3. **回退生成**：当找不到GPU配置时，基于参考配置生成（需明确说明）

**补全的默认值（仅用于缺失字段）**：

```yaml
# XPU核心必需字段（如GPU配置中不存在）
device: xpu
bf16: true
amp_master_grad: true

# 通信配置（缺失时补全）
bkcl_timeout: 1000
bkcl_socket_ifname: "eth0"
bkcl_enable_xdr: 1

# 并行策略（基于硬件补全）
tensor_model_parallel_size: 1
expert_model_parallel_size: 8
use_expert_parallel: true
```

**注意**：参考配置中的参数**不覆盖**GPU配置中已存在的值，仅补充缺失字段。

##### 2.1.5 模式级规则抽象

**模式1：后端统一替换模式**
```
所有通信后端统一替换：
- nccl → bkcl
- cuda → xpu
应用范围：backend, distributed_backend, device 字段
```

**模式2：融合算子处理模式**
```
GPU融合算子在XPU上的处理策略：
- 字段名匹配 fuse_*
- 字段名匹配 *_fusion
处理：注释掉，根据测试结果逐步启用
原因：融合算子需要XPU特定实现支持
```

**模式3：内存优化模式**
```
根据硬件内存自动调整：
- gradient_accumulation_steps：通常为GPU值的一半
- packing：建议关闭（false）
- benchmark：建议关闭（false）
```

**模式4：精度模式**
```
XPU默认使用bf16，与GPU保持一致：
- 如果GPU使用bf16：保留
- 如果GPU使用fp16：转换为bf16
- fp16_opt_level：可保留但可能不生效
```

#### 2.2 转换执行流程

AI 执行配置转换的完整流程：

```
步骤1: 预处理
  - 读取GPU YAML文件
  - 解析所有字段和值
  - 识别模型类型（如 Qwen3-VL）

步骤2: 应用字段映射规则
  - 重命名 _attn_implementation → attn_impl（并注释）
  - 添加 device: xpu
  - 替换 nccl → bkcl（如果出现）

步骤3: 应用参数变换规则
  - gradient_accumulation_steps: 16 → 8
  - packing: true → false
  - benchmark: true → false
  - 根据任务调整 max_steps（如需要）

步骤4: 应用结构调整规则
  - 注释掉 moe_deep_gemm
  - 注释掉 fuse_* 相关字段
  - 可选添加 pp_delay_scale_loss

步骤5: 默认值补全
  - 检查必需字段是否存在
  - 使用默认值补全缺失字段

步骤6: 验证和输出
  - 验证YAML语法
  - 生成转换报告
  - 保存XPU YAML文件
```

### 阶段3：XPU 启动脚本生成

**目标**：解析预置的 XPU shell 模板并填充动态参数

**动态参数映射**：

| 参数 | 说明 | 如何指定 |
|------|------|----------|
| `{{MODEL_NAME}}` | 模型名称 | 从模型配置自动提取 |
| `{{MODEL_PATH}}` | 模型路径 | 从模型配置自动提取 |
| `{{CONFIG_FILE}}` | XPU YAML 配置文件路径 | 自动生成 |
| `{{OUTPUT_DIR}}` | 训练输出目录 | 从模型配置或默认生成 |
| `{{NUM_XPUS}}` | XPU 设备数量 | 自动检测或用户指定 |
| `{{XPU_DEVICES}}` | XPU 设备列表 | 自动检测或用户指定 |
| `{{PYTHON_ENV_PATH}}` | **Python 虚拟环境路径** | **必需：用户必须指定** |

**关于 PYTHON_ENV_PATH**：

这是启动脚本中**唯一需要用户手动指定**的路径参数，用于激活 Paddle/PaddleFormers 运行环境。

**指定方式**：
```bash
# 方式1: 在环境变量中设置
export PYTHON_ENV_PATH="/root/paddlejob/zhangxiao_dev/qwen_env"

# 方式2: 在配置生成时指定
# AI 在生成配置时会询问用户环境路径
```

**验证逻辑**：
脚本会自动检查 `${PYTHON_ENV_PATH}/bin/activate` 是否存在，如果不存在会报错并退出。

**示例**：
```bash
# 用户环境路径示例
/root/paddlejob/zhangxiao_dev/qwen_env
/workspace/paddle_env
/home/user/paddle_xpu_env
```

### 阶段4：训练任务启动

**执行流程**：
1. 填充启动脚本模板，替换所有动态参数
2. 保存启动脚本到输出目录
3. 向用户展示配置摘要：
   ```
   � 训练配置摘要
   =================
   模型名称: {model_name}

   🔧 配置转换:
      - 基准配置: {gpu_yaml_path}
      - 转换规则: Qwen3-VL系列规则
      - 主要调整:
        * gradient_accumulation_steps: 16 → 8
        * packing: true → false
        * device: xpu (新增)

   � 配置文件:
      - XPU YAML: {xpu_yaml_path}
      - 启动脚本: {launch_script_path}

   📊 训练参数:
      - Stage: {stage}
      - Max Steps: {max_steps}
      - Batch Size: {per_device_train_batch_size}
      - Gradient Accumulation: {gradient_accumulation_steps}

   💾 输出路径（统一基于 output_dir）:
      - 模型权重/检查点: {output_dir}/
      - 训练日志: {output_dir}/vdl_log/
      - 数据集缓存: {output_dir}/dataset_cache/

   �️ XPU 配置:
      - 设备数量: {num_xpus}
      - 设备列表: {xpu_devices}
      - Python环境: {python_env_path}

   ⚠️  重要提示:
      - 训练前请确认 Python 环境路径正确: {python_env_path}
      - 启动时会自动激活该环境的 Paddle/PaddleFormers
   ```
4. **自主总结工作流并介绍工作流程**: 在生成所有训练相关文件后，Agent 应主动总结本次生成过程的工作流，包括：PaddleFormers 仓库定位方式、GPU YAML 选择理由、GPU→XPU 配置转换的关键调整、输出文件结构说明，帮助用户理解整个配置生成过程。
5. **判断用户意图并执行**
   - 如果用户指令中明确说明"训练"（而非"生成"配置），直接启动训练
   - 如果用户仅要求生成配置或意图不明确，明确询问用户是否立即启动训练
   - 根据用户确认或指令意图执行训练启动

### 阶段5：错误检测与自修复机制

**错误分类与处理策略**：

| 错误类型 | 示例 | 处理方式 |
|---------|------|---------|
| 参数缺失 | `KeyError: 'device'` | 从参考配置补全 |
| 配置不合法 | `Invalid device` | 修正为合法值 |
| 显存不足 | `XPU OOM` | 减小 batch_size，增大 accumulation |
| 通信超时 | `BKCL timeout` | 增加 timeout 值 |
| 算子不支持 | `Operator not supported` | 注释掉相关配置 |

**自修复迭代流程**：
```
执行训练 → 检测错误 → 分类错误 → 尝试修复 → 重新生成配置 → 重试执行
                ↓
         达到最大重试次数(3次) → 输出完整报错 → 提供优化建议
```

## 配置文件结构

```
paddle-trainer2/
├── SKILL.md                          # 本文件
├── templates/
│   ├── xpu_train.sh.template         # XPU 启动脚本模板
│   └── mapping_rules.yaml            # GPU→XPU 映射规则定义
├── scripts/
│   ├── config_generator.py           # 配置生成器核心
│   ├── gpu_yaml_finder.py            # GPU YAML 检索器
│   ├── error_handler.py              # 错误检测与修复
│   └── train_launcher.py             # 训练启动器
└── reference_configs/
    └── xpu_reference.yaml            # XPU 参考配置
```

## 使用示例

### 示例1：基于GPU配置转换（推荐方式）

**场景**：在 PaddleFormers 仓库中找到对应 GPU 配置，以此为基础生成 XPU 配置

**用户**: "将 Qwen3-VL-30B-A3B-Instruct 的 GPU 配置转换为 XPU 配置"

**AI 执行流程**：

1. **在 PaddleFormers 仓库中查找 GPU 配置**：
   ```
   ✓ 找到 GPU 配置: PaddleFormers/qwen3_vl/configs/qwen3_vl_8b_sft.yaml
   ```

2. **读取 GPU 配置**：
   ```yaml
   # GPU 配置关键字段
   _attn_implementation: flashmask
   stage: VL-SFT
   gradient_accumulation_steps: 16
   packing: true
   benchmark: true
   moe_deep_gemm: true
   max_steps: 500
   ```

3. **应用转换规则**：
   ```yaml
   # XPU 配置转换过程
   # 1. 字段映射
   device: xpu                              # 新增
   # attn_impl: flashmask                   # 重命名并注释

   # 2. 参数变换
   gradient_accumulation_steps: 8           # 16 → 8
   packing: false                           # true → false
   benchmark: false                         # true → false
   max_steps: 500                           # 保持（相同任务）

   # 3. 结构调整
   # moe_deep_gemm: true                    # 注释掉
   freeze_config: freeze_vision freeze_aligner  # 保留
   ```

4. **生成 XPU 配置**：
   ```yaml
   # 生成的 XPU YAML（节选）
   ### model
   model_name_or_path: /path/to/Qwen3-VL-30B-A3B

   ### finetuning
   stage: VL-SFT
   gradient_accumulation_steps: 8
   packing: false
   benchmark: false
   # moe_deep_gemm: true

   freeze_config: freeze_vision freeze_aligner
   device: xpu
   ```

5. **展示转换报告**：
   ```
   📋 配置转换报告
   =================
   模型名称: Qwen3-VL-30B-A3B-Thinking

   🔧 生成方式:
      ✅ 基于GPU配置转换（推荐方式）
         - 源GPU配置: PaddleFormers/qwen3_vl/configs/qwen3_vl_8b_sft.yaml
         - 应用GPU→XPU映射规则进行转换
         - 使用参考配置补全缺失字段（仅补充，不覆盖）
         - 这是推荐的生成方式，配置更可靠

   🔧 应用规则:
      ✓ 添加 device: xpu
      ✓ 注释 _attn_implementation
      ✓ 调整 gradient_accumulation_steps: 16 → 8
      ✓ 关闭 packing: true → false
      ✓ 关闭 benchmark: true → false
      ✓ 注释 moe_deep_gemm

   📊 关键参数:
      - Stage: VL-SFT
      - Max Steps: 500
      - Batch Size: 1 × 8 = 8 (per_device × accumulation)

   ✅ 验证结果: 配置完整，可以启动训练
   ```

---

### 示例2：基于参考配置生成（回退方式）

**场景**：在 PaddleFormers 仓库中**未找到**对应 GPU 配置，使用参考配置作为基础

**用户**: "训练 CustomModel-7B 模型，但在 PaddleFormers 中找不到对应配置"

**AI 执行流程**：

1. **在 PaddleFormers 仓库中查找 GPU 配置**：
   ```
   ✗ 未找到 CustomModel-7B 的 GPU 配置
   ✓ 使用 XPU 参考配置作为基础（回退方案）
   ```

2. **基于参考配置生成**：
   ```yaml
   # 使用 reference_configs/xpu_reference.yaml 作为基础
   # 应用模型特定规则（如果有匹配）
   # 推断生成配置
   ```

3. **展示转换报告（明确说明回退）**：
   ```
   📋 配置转换报告
   =================
   模型名称: CustomModel-7B

   ⚠️ 生成方式:
      基于XPU参考配置生成（回退方案）:
        - 未找到模型 CustomModel-7B 的GPU YAML配置
        - 使用 reference_configs/xpu_reference.yaml 作为基础
        - 应用了模型特定规则: 无
        - 所有参数基于参考配置推断，建议人工复核关键参数

   ⚠️ 重要提示:
      此配置基于参考配置推断生成，可能存在以下问题：
      1. 模型结构可能与实际不符
      2. 训练参数可能需要调整
      3. 建议人工检查以下关键字段：
         - model_name_or_path
         - num_hidden_layers
         - hidden_size
         - num_attention_heads

   🔧 应用规则:
      ✓ 基于参考配置初始化
      ✓ 应用默认XPU参数

   📊 关键参数:
      - 训练阶段: 请根据实际需求设置
      - Batch Size: 1 × 8 = 8 (per_device × accumulation)
      - 建议: 首次训练请使用较小max_steps验证配置正确性

   ⚠️ 验证结果: 配置已生成，建议人工复核后再启动训练
   ```

## 路径配置机制

模型路径和数据集路径支持**动态配置**，允许用户在运行时显式指定，未提供时使用默认路径。

### 路径优先级规则

| 优先级 | 来源 | 说明 |
|--------|------|------|
| 1 | 用户提供 | 用户显式指定的路径（最高优先级） |
| 2 | 默认路径 | 基于当前工作目录的默认路径 |

### 默认路径结构

基于当前工作目录（`./`）构建的默认路径：

```
./
├── weights/                           # 模型权重目录（自动创建）
│   └── {model_name}/
├── datasets/                          # 数据集目录（不自动创建）
│   ├── train.jsonl                   # 训练数据集
│   └── eval.jsonl                    # 评估数据集
└── checkpoints/                       # 训练输出根目录（可自定义）
    └── train_{model_name}_xpu_{timestamp}/   # 具体训练任务目录
        ├── config.yaml               # 训练配置文件
        ├── train_xpu.sh              # 启动脚本
        ├── checkpoints/              # 模型检查点
        ├── vdl_log/                  # VisualDL训练日志
        └── dataset_cache/            # 数据集预处理缓存
```

### 路径配置示例

**方式1：使用默认路径**
```bash
# 不指定路径，使用默认
模型路径: ./weights/Qwen3-VL-30B-A3B-Thinking
数据集路径: ./datasets/train.jsonl
```

**方式2：用户自定义路径**
```bash
# 指定自定义路径
模型路径: /data/models/Qwen3-VL-30B/
数据集路径: /data/coco_grounding/train.jsonl
```

### 输出路径统一配置机制

**所有训练输出文件统一存放在 `output_dir` 指定的目录下**：

```
{output_dir}/                           # 用户指定的输出根目录
├── config.yaml                         # 训练配置文件
├── train_xpu.sh                        # 启动脚本
├── checkpoints/                        # 模型检查点目录
│   ├── checkpoint-100/
│   ├── checkpoint-200/
│   └── latest/
├── vdl_log/                            # VisualDL 日志目录
│   └── events.out.tfevents.xxx
└── dataset_cache/                      # 数据集预处理缓存
    └── processed_xxx.arrow
```

**路径统一配置规则**：
1. `output_dir`: 用户指定的输出根目录（默认：`./checkpoints/`）
2. `logging_dir`: 自动派生为 `{output_dir}/vdl_log/`
3. `dataset_output_dir`: 自动派生为 `{output_dir}/dataset_cache/`
4. 所有路径自动创建，无需手动干预

### 配置摘要中的路径展示

生成配置时会展示最终使用的路径：
```
📁 路径配置:
   - 模型权重: ./weights/Qwen3-VL-30B-A3B-Thinking
   - 训练数据: ./datasets/train.jsonl
   - 评估数据: ./datasets/eval.jsonl
   - 输出目录: ./checkpoints/train_qwen3vl_xpu_20250115/
     ├── checkpoints/        ← 模型权重保存位置
     ├── vdl_log/           ← 训练日志保存位置
     └── dataset_cache/     ← 数据集缓存位置

⚠️  路径检查:
   ✓ weights目录已创建
   ⚠ datasets目录不存在，请确保数据集文件存在
   ✓ 输出目录结构已创建（checkpoints/vdl_log/dataset_cache）
```

## 关键参数说明

### XPU 环境变量

| 变量 | 默认值 | 说明 |
|------|-------|------|
| BKCL_TIMEOUT | 1000 | 通信超时（毫秒） |
| BKCL_SOCKET_IFNAME | eth0 | 网络接口 |
| BKCL_ENABLE_XDR | 1 | 启用 XDR 加速 |
| XPU_VISIBLE_DEVICES | 0,1,2,3,4,5,6,7 | 可见设备列表 |

### 转换规则速查表

| GPU 配置 | XPU 转换 | 说明 |
|----------|----------|------|
| （无device字段） | `device: xpu` | 必需新增 |
| `_attn_implementation: flashmask` | `# attn_impl: flashmask` | 重命名+注释 |
| `gradient_accumulation_steps: 16` | `gradient_accumulation_steps: 8` | 减半调整 |
| `packing: true` | `packing: false` | 关闭 |
| `benchmark: true` | `benchmark: false` | 关闭 |
| `moe_deep_gemm: true` | `# moe_deep_gemm: true` | 注释 |
| `nccl` | `bkcl` | 后端替换 |
| `bf16: true` | `bf16: true` | 保持不变 |

## 注意事项

### 配置生成来源（重要）

**基于GPU配置转换（推荐方式）**：
- 优先在 PaddleFormers 仓库中查找对应 GPU YAML 配置
- 以 GPU 配置为基础进行 GPU→XPU 转换
- 保留原配置的模型结构、训练参数和数据配置
- 这是最可靠的生成方式

**基于参考配置生成（回退方式）**：
- 仅在找不到 GPU 配置时触发
- 使用 `reference_configs/xpu_reference.yaml` 作为基础
- 所有参数基于参考配置推断
- **必须在输出中明确说明**
- **建议人工复核关键参数后再启动训练**

### 其他注意事项

1. **PaddleFormers 仓库定位**：系统采用分层优先策略自动定位仓库，优先从 Python 环境获取准确路径。如定位失败，可使用 `--paddleformers-root` 参数显式指定
2. **配置来源**：当前映射规则基于 **Qwen3-VL-30B-A3B** 实际配置对比分析得出
3. **模型差异**：不同模型系列（如 DeepSeek、Llama）可能有特殊规则，参考 `mapping_rules.yaml` 中的 `model_specific_rules`
4. **融合算子**：XPU 对融合算子的支持需要验证，建议先注释掉，测试通过后逐步启用
5. **内存调整**：`gradient_accumulation_steps` 可根据实际 XPU 显存大小微调
6. **任务适配**：`max_steps`、`save_steps` 等参数主要与训练任务相关，相同任务可保持一致
7. **参考配置使用**：参考配置仅用于参数修正与补全，不直接作为生成基础。使用回退方案生成的配置需人工复核

## 参考文件

- **映射规则详细定义**：`templates/mapping_rules.yaml`
- **XPU 参考配置**：`reference_configs/xpu_reference.yaml`
- **分析来源**：
  - GPU配置：`Qwen3-VL-30B-A3B-Instruct.yaml`（PaddleFormers）
  - XPU配置：`Qwen3-VL-30B-A3B-Thinking.yaml`（已验证）


## Agent 驱动的错误检测与修复

### 设计原则

错误检测与修复模块已从"代码驱动"重构为"Agent 驱动"架构：

- **错误检测**: Skill 提供基于正则的错误信号检测，Agent 结合上下文进行语义分析
- **修复决策**: 不由预定义规则决定，由 Agent 基于上下文推理自主决策
- **显式暴露**: 不可修复的错误必须完整返回，不得静默处理或掩盖
- **安全约束**: 修复执行前有严格的验证机制，防止危险操作

### 错误处理流程

```
训练执行 → 错误信号检测 → Agent 分析决策 → 修复执行/显式返回
                ↓
        达到最大重试次数 → 输出完整报错 → 提供人工处理建议
```

### Agent 错误处理执行步骤

#### 步骤1: 获取错误上下文

Agent 调用 `error_handler.get_error_context()` 获取完整上下文：

```python
# 获取错误处理上下文
context = handler.get_error_context(
    log_content=error_log,
    config_path="./checkpoints/train_xpu.yaml",
    model_name="Qwen3-VL-30B-A3B-Thinking"
)

# 上下文数据结构
{
  "error_detection": {
    "signals_detected": true,
    "signals": [
      {
        "error_type": "out_of_memory",
        "matched_text": "XPU OOM",
        "extracted_params": []
      }
    ],
    "log_snippet": "..."
  },
  "current_state": {
    "config_path": "...",
    "current_config": {...},
    "relevant_params": {...},
    "repair_attempt_count": 0,
    "max_repair_attempts": 3
  },
  "rules": {
    "error_handling_rules": {...},
    "reference_config_keys": [...]
  },
  "available_repair_actions": {...},
  "reference_values": {...}
}
```

#### 步骤2: 分析错误与评估可修复性

Agent 基于上下文进行语义分析：

**分析维度**:
1. **错误类型识别**: 从信号中提取错误类型和严重程度
2. **上下文理解**: 结合当前配置和模型信息
3. **修复尝试评估**: 检查是否达到最大修复次数
4. **可修复性判断**: 根据规则判断是否可以/应该修复

**不可修复错误类型**（必须显式返回）:
- `runtime_error`: 段错误、断言失败等，通常与配置无关
- `operator_not_supported`（核心算子）: 核心算子不支持会导致模型无法运行
- `unknown`: 未知错误类型，需要人工分析

#### 步骤3: 选择修复策略

如果 Agent 判断错误可修复，基于规则选择策略：

| 错误类型 | 推荐策略 | Agent 决策要点 |
|---------|---------|---------------|
| `missing_parameter` | FROM_REFERENCE | 参数是否必需、是否与模型兼容 |
| `invalid_config` | USE_DEFAULT | 原值无效原因、默认值适用性 |
| `out_of_memory` | REDUCE_BATCH_SIZE | batch size 调整对收敛的影响 |
| `communication_timeout` | INCREASE_TIMEOUT | 是否已接近最大值、是否需要人工介入 |
| `network_interface` | AUTO_DETECT_INTERFACE | 可用接口列表、当前配置是否正确 |
| `operator_not_supported` | COMMENT_OUT_OPERATOR | 算子是否为必需核心算子 |

#### 步骤4: 生成修复计划

Agent 生成标准化的修复计划：

```python
repair_plan = {
    "should_repair": true,
    "strategy": "REDUCE_BATCH_SIZE",
    "config_changes": {
        "gradient_accumulation_steps": 16
    },
    "comment_out_fields": [],
    "reasoning": "XPU OOM 错误，当前 batch_size=1，需增加 accumulation_steps",
    "confidence": "high"
}
```

**修复计划验证**:
- 调用 `validate_repair_plan()` 验证计划合法性
- 检查危险操作（删除文件、执行命令等）
- 验证参数类型和范围

#### 步骤5: 执行修复

如果验证通过，调用 `apply_repair()` 执行修复：

```python
success, new_path, details = handler.apply_repair(
    config_path="./checkpoints/train_xpu.yaml",
    repair_plan=repair_plan,
    error_context=context
)
```

**执行保障**:
- 自动备份原配置
- 失败时自动回滚
- 详细记录修复日志

### 不可修复错误处理原则

对于无法可靠修复的错误，Agent 必须遵循**显式暴露原则**：

```python
# Agent 判断不可修复
repair_plan = {
    "should_repair": false,
    "reason": "RuntimeError 通常与配置无关，可能是环境或代码 bug，建议人工分析",
    "confidence": "high"
}

# 执行结果
result = {
    "status": "not_repaired",
    "error_type": "runtime_error",
    "error_message": "原始错误日志...",
    "reason": "Agent 判断不应执行修复: RuntimeError 通常与配置无关",
    "recommendation": "建议检查环境配置和代码版本"
}
```

**必须显式暴露的错误**:
- 运行时错误（段错误、断言失败）
- 核心算子不支持
- 未知错误类型
- 已达到最大修复次数仍失败

### Agent 决策报告示例

#### 示例1: 可修复错误（OOM）

```
📋 Agent 错误处理报告
========================
错误信号检测:
  - 类型: out_of_memory
  - 匹配文本: "XPU OOM"
  - 严重程度: critical

当前状态分析:
  - 模型: Qwen3-VL-30B-A3B-Thinking
  - 当前 batch_size: 1
  - 当前 accumulation_steps: 8
  - 修复尝试次数: 0/3

可修复性评估: ✅ 可修复
  - 错误类型属于可修复类别
  - 未达到最大修复次数
  - 有明确的修复策略

修复策略选择: REDUCE_BATCH_SIZE
  - 当前 batch_size 已为 1，无法减小
  - 选择增加 gradient_accumulation_steps
  - 从 8 增加到 16

修复计划:
  should_repair: true
  strategy: REDUCE_BATCH_SIZE
  config_changes:
    gradient_accumulation_steps: 16
  confidence: high

执行结果: ✅ 修复成功
  - 配置已备份: train_xpu.yaml.backup.20250115_143022
  - 修改已应用
  - 建议重新启动训练
```

#### 示例2: 不可修复错误（RuntimeError）

```
📋 Agent 错误处理报告
========================
错误信号检测:
  - 类型: runtime_error
  - 匹配文本: "RuntimeError: CUDA error"
  - 严重程度: critical

当前状态分析:
  - 模型: Qwen3-VL-30B-A3B-Thinking
  - 修复尝试次数: 0/3

可修复性评估: ❌ 不可修复
  - RuntimeError 通常与配置无关
  - 可能是环境、依赖或代码 bug
  - 自动修复可能掩盖根本问题

Agent 决策: 不执行修复
  reason: |
    RuntimeError 通常与配置无关，可能是环境或代码 bug。
    自动修复可能引入新的问题或掩盖根本原因。
    建议人工分析完整错误日志。

显式返回错误信息:
  status: not_repaired
  error_type: runtime_error
  error_message: "RuntimeError: CUDA error: device-side assert triggered"
  reason: "Agent 判断不应执行修复"
  recommendation: |
    1. 检查环境配置是否正确
    2. 确认 Paddle/PaddleFormers 版本兼容性
    3. 检查输入数据是否合法
    4. 查看完整堆栈信息定位问题
```

### 错误处理规则速查

#### 错误类型与推荐策略

| 错误类型 | 严重程度 | 推荐策略 | 可自动修复 |
|---------|---------|---------|-----------|
| `missing_parameter` | high | FROM_REFERENCE | ✅ |
| `invalid_config` | high | USE_DEFAULT | ✅ |
| `out_of_memory` | critical | REDUCE_BATCH_SIZE | ✅ |
| `communication_timeout` | medium | INCREASE_TIMEOUT | ✅ |
| `network_interface` | medium | AUTO_DETECT_INTERFACE | ✅ |
| `operator_not_supported` | high | COMMENT_OUT_OPERATOR | ⚠️ 需评估 |
| `runtime_error` | critical | CANNOT_FIX | ❌ |
| `unknown` | unknown | CANNOT_FIX | ❌ |

#### 修复安全约束

| 约束项 | 限制值 | 说明 |
|-------|-------|------|
| 最大修复次数 | 3 | 防止无限循环修复 |
| 最小 batch_size | 1 | 不得小于 1 |
| 最大 accumulation_steps | 64 | 防止过大影响收敛 |
| 最大 timeout | 5000ms | BKCL 超时上限 |
| 必需备份 | true | 每次修复必须备份 |
| 失败回滚 | true | 保存失败时自动恢复 |

**禁止操作**:
- 删除配置文件
- 修改模型权重文件
- 执行系统命令
- 修改环境变量

### 与其他模块的关系

```
┌─────────────────┐
│  训练启动器      │
│ train_launcher  │
└────────┬────────┘
         │ 启动训练
         ▼
┌─────────────────┐     错误日志      ┌─────────────────┐
│  训练进程        │ ───────────────→ │  错误处理器      │
│  (XPU训练)      │                  │ error_handler   │
└────────┬────────┘                  └────────┬────────┘
         │                                    │
         │ 正常完成                            │ get_error_context()
         │                                    ▼
         │                           ┌─────────────────┐
         │                           │   Agent 决策     │
         │                           │  分析/决策/修复  │
         │                           └────────┬────────┘
         │                                    │
         │                                    │ apply_repair()
         │                                    │ 或 显式返回
         │                                    ▼
         │                           ┌─────────────────┐
         └────────────────────────→ │   修复结果       │
                                     │ 成功/失败/不可修复│
                                     └─────────────────┘
```
