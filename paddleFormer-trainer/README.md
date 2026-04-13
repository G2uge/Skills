# PaddleFormer XPU 训练配置自动生成器

[![Agent-Driven](https://img.shields.io/badge/Architecture-Agent--Driven-blue)](SKILL.md)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)

采用 **Agent 驱动架构** 的 PaddleFormer XPU 训练配置管理系统。Skill 提供基础能力与上下文，AI Agent 基于语义理解自主完成关键决策。

---

## 目录

- [核心功能](#核心功能)
- [适用场景](#适用场景)
- [工作流程](#工作流程)
- [输入参数](#输入参数)
- [输出结果](#输出结果)
- [使用示例](#使用示例)
- [项目结构](#项目结构)
- [设计原则](#设计原则)

---

## 核心功能

### 1. 智能 GPU YAML 检索
- **三层分层搜索**自动定位 PaddleFormers 仓库
- **语义匹配**基于模型特征（系列、规模、任务类型）检索候选配置
- **Agent 自主决策**选择最优 GPU YAML 模板

### 2. GPU→XPU 智能配置转换
- **字段映射**: 自动处理字段重命名、新增、注释（如 `_attn_implementation` → 注释）
- **参数变换**: 根据 XPU 特性调整参数（如 `gradient_accumulation_steps: 16 → 8`）
- **结构调整**: 添加 XPU 必需字段、注释不支持的算子
- **模型特定规则**: 支持 Qwen3-VL、DeepSeek MoE 等复杂结构

### 3. 启动脚本生成
- 基于模板自动生成 XPU 训练启动脚本
- 自动填充模型路径、配置文件路径、设备列表等参数
- 输出脚本已赋予执行权限

### 4. 安全错误处理
- **错误信号检测**: 正则匹配识别常见错误类型
- **Agent 驱动修复**: 基于上下文分析决策是否修复及如何修复
- **显式暴露原则**: 不可修复错误（如 RuntimeError）必须完整返回，不得静默处理
- **安全保障**: 自动备份、失败回滚、最大 3 次修复尝试

### 5. 统一输出管理
- 所有输出文件统一存放在 `output_dir` 下
- 自动创建子目录：`checkpoints/`、`vdl_log/`、`dataset_cache/`、`paddleformers_dist_log/`

---

## 适用场景

| 场景 | 说明 |
|------|------|
| **XPU 训练启动** | 用户想要在 XPU 上训练 PaddleFormer 模型 |
| **GPU→XPU 配置转换** | 用户已有 GPU 配置，需要转换为 XPU 兼容配置 |
| **配置自动生成** | 用户没有现成配置，需要基于参考配置生成 |
| **错误自动修复** | 训练遇到配置错误，需要自动检测和修复 |

---

## 工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│  阶段1: PaddleFormers 仓库定位                                    │
│  └── 分层搜索 (Python环境 → 工作空间 → 扩展搜索)                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  阶段2: GPU YAML 检索与选择 (Agent 驱动)                          │
│  └── 提取模型特征 → 扫描候选 YAML → Agent 语义分析选择最优配置        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  阶段3: GPU→XPU 配置转换 (Agent 驱动)                             │
│  └── 字段映射 → 参数变换 → 结构调整 → 默认值补全                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  阶段4: 启动脚本生成与工作流总结                                    │
│  └── 填充模板生成脚本 → 自主总结工作流 → 判断用户意图启动训练          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  阶段5: 错误检测与修复 (Agent 驱动)                                │
│  └── 错误信号检测 → Agent 分析决策 → 修复执行/显式返回               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 输入参数

### 主要输入参数

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `model_name` | string | **是** | 模型名称，如 `"Qwen3-VL-30B-A3B-Thinking"` |
| `model_name_or_path` | string | 否 | 模型权重路径。若未提供，默认使用 `"./weights/{model_name}"` |
| `train_dataset_path` | string | 否 | 训练数据集路径。若未提供，默认使用 `"./datasets/train.jsonl"` |
| `gpu_yaml_path` | string | 否 | 指定的 GPU YAML 配置文件路径。若未提供，Agent 将自动检索选择 |
| `output_dir` | string | 否 | 训练输出根目录，默认 `"./output"` |
| `paddleformers_root` | string | 否 | PaddleFormers 仓库根目录。若未提供，将自动分层搜索 |
| `allow_fallback` | boolean | 否 | 找不到 GPU YAML 时是否允许使用参考配置生成，默认 `false` |
| `custom_params` | dict | 否 | 自定义参数，覆盖自动生成的配置值 |
| `max_attempts` | int | 否 | 最大修复尝试次数，默认 `3` |

### 参数详细说明

#### `model_name`
- **用途**: 标识要训练的模型，用于检索匹配的 GPU YAML 和生成配置
- **格式**: 模型名称应包含系列、规模等信息，如 `"Qwen3-VL-30B-A3B-Thinking"`
- **示例**: `"Qwen3-VL-30B-A3B-Instruct"`, `"DeepSeek-V2-Chat"`

#### `model_name_or_path`
- **用途**: 指定模型权重文件的路径
- **默认**: `"./weights/{model_name}"`
- **注意**: 若路径不存在，会自动创建目录。建议训练前将模型权重文件放入该目录
- **示例**: `"/data/models/Qwen3-VL-30B-A3B"`, `"./weights/my_model"`

#### `train_dataset_path`
- **用途**: 指定训练数据集文件的路径
- **默认**: `"./datasets/train.jsonl"`
- **注意**: 数据集文件需为 JSONL 格式，若文件不存在会发出警告
- **示例**: `"/data/coco_grounding/train.jsonl"`, `"./datasets/train.jsonl"`

#### `gpu_yaml_path`
- **用途**: 指定基准 GPU 配置文件路径
- **注意**: 若提供，将直接使用该配置作为转换基础；若未提供，Agent 将自动检索选择
- **推荐**: 让 Agent 自动检索选择，以确保选择最适合的配置

#### `output_dir`
- **用途**: 指定所有训练输出文件的存放目录
- **默认**: `"./output"`
- **结构**: 将自动创建子目录 `checkpoints/`、`vdl_log/`、`dataset_cache/`、`paddleformers_dist_log/`

#### `allow_fallback`
- **用途**: 控制找不到 GPU YAML 时的行为
- **取值**:
  - `false`（默认）: 报错退出，提示未找到 GPU YAML
  - `true`: 使用 `reference_configs/xpu_reference.yaml` 作为基础生成配置
- **注意**: 回退方案生成的配置需人工复核关键参数

#### `custom_params`
- **用途**: 自定义参数字典，用于覆盖自动生成的配置值
- **常见用法**: 指定模型路径、数据集路径、训练参数等
- **示例**:
  ```python
  custom_params = {
      "model_name_or_path": "/data/models/Qwen3-VL-30B",
      "train_dataset_path": "/data/coco/train.jsonl",
      "max_steps": 1000,
      "learning_rate": "5.0e-5"
  }
  ```

#### `max_attempts`
- **用途**: 训练启动失败时的最大自动修复尝试次数
- **默认**: `3`
- **说明**: 超过此次数仍未成功将停止尝试并返回错误信息

---

## 输出结果

### 输出文件结构

```
{output_dir}/                           # 用户指定的输出根目录
├── train_{model_name}_xpu_{timestamp}.yaml   # XPU 训练配置文件
├── run_train_{model_name}_xpu.sh             # 训练启动脚本
├── checkpoints/                        # 模型检查点目录
│   ├── checkpoint-100/
│   ├── checkpoint-200/
│   └── latest/
├── vdl_log/                            # VisualDL 训练日志
│   └── events.out.tfevents.xxx
├── dataset_cache/                      # 数据集预处理缓存
└── paddleformers_dist_log/             # 分布式训练日志
    ├── workerlog.0
    ├── workerlog.1
    └── ...
```

### 输出文件说明

| 文件/目录 | 类型 | 说明 |
|-----------|------|------|
| `train_{model}_xpu_{timestamp}.yaml` | 配置文件 | XPU 训练配置文件，包含所有训练参数 |
| `run_train_{model}_xpu.sh` | 启动脚本 | 可直接执行的 Bash 脚本，用于启动训练 |
| `checkpoints/` | 目录 | 模型权重保存目录，按训练步数组织 |
| `vdl_log/` | 目录 | VisualDL 日志，用于训练过程可视化 |
| `dataset_cache/` | 目录 | 数据集预处理缓存 |
| `paddleformers_dist_log/` | 目录 | 分布式训练日志，包含各 worker 的输出 |

### 配置生成来源标识

生成的 XPU 配置文件会包含元数据，标识生成方式：

```yaml
_meta:
  generated_at: "20250115_143022"
  model_name: "Qwen3-VL-30B-A3B-Thinking"
  source_gpu_config: "/path/to/gpu.yaml"
  generator_version: "2.0-agent-driven"
```

---

## 使用示例

### 示例1: 完整训练启动（推荐）

用户明确说明"训练"模型，Agent 将自动完成所有步骤并直接启动训练：

```python
from scripts.train_launcher import TrainLauncher

launcher = TrainLauncher(output_base_dir="./output")

# 自动完成：定位仓库 → 检索 YAML → 转换配置 → 生成脚本 → 启动训练
result = launcher.run_with_repair(
    model_name="Qwen3-VL-30B-A3B-Thinking",
    max_attempts=3
)

if result["success"]:
    print(f"✅ 训练启动成功！")
    print(f"  PID: {result['pid']}")
    print(f"  输出目录: {result['output_dir']}")
    print(f"  监控日志: tail -f {result['output_dir']}/paddleformers_dist_log/workerlog.0")
else:
    print(f"❌ 训练启动失败: {result['error']}")
```

### 示例2: 仅生成配置（不启动训练）

用户仅要求生成配置，Agent 将询问是否启动训练：

```python
from scripts.train_launcher import TrainLauncher

launcher = TrainLauncher(output_base_dir="./output")

# 生成配置和脚本
xpu_yaml, script_path, config_info, candidate_data = launcher.prepare_training(
    model_name="Qwen3-VL-30B-A3B-Thinking"
)

# 展示配置摘要
launcher.print_summary(config_info)

print(f"\n✓ 配置已生成")
print(f"  XPU YAML: {xpu_yaml}")
print(f"  启动脚本: {script_path}")
print(f"\n手动启动命令: bash {script_path}")
```

### 示例3: 命令行使用

```bash
# 仅准备配置（不启动训练）
python scripts/train_launcher.py Qwen3-VL-30B-A3B-Thinking --dry-run

# 启动训练（支持自动修复，最多3次尝试）
python scripts/train_launcher.py Qwen3-VL-30B-A3B-Thinking --max-attempts 3

# 指定 PaddleFormers 仓库路径
python scripts/train_launcher.py Qwen3-VL-30B-A3B-Thinking \
    --paddleformers-root /path/to/PaddleFormers

# 指定输出目录
python scripts/train_launcher.py Qwen3-VL-30B-A3B-Thinking \
    --output-dir ./my_training_output
```

### 示例4: Agent 驱动配置转换（细粒度控制）

```python
from scripts.gpu_yaml_finder import GPUYamlFinder
from scripts.config_generator import XPUConfigGenerator

# Step 1: 定位 PaddleFormers 仓库
finder = GPUYamlFinder()
paddleformers_root = finder.execute_layered_search()
print(f"✓ 定位仓库: {paddleformers_root}")

# Step 2: 获取候选 YAML（Agent 分析选择）
candidate_data = finder.find_candidate_yamls("Qwen3-VL-30B-A3B-Thinking")
# Agent 基于语义分析选择最优配置...
selected_gpu_yaml = "PaddleFormers/configs/qwen3vl_sft.yaml"

# Step 3: 获取转换上下文
generator = XPUConfigGenerator()
context = generator.get_conversion_context(
    gpu_config_path=selected_gpu_yaml,
    model_name="Qwen3-VL-30B-A3B-Thinking"
)

# Step 4: Agent 基于上下文执行转换决策
xpu_config = {
    # Agent 根据规则生成的配置...
    "device": "xpu",
    "gradient_accumulation_steps": 8,
    # ...
}

# Step 5: 保存配置
config_path, _ = generator.save_xpu_config(
    xpu_config=xpu_config,
    model_name="Qwen3-VL-30B-A3B-Thinking",
    output_dir="./output"
)
print(f"✓ 配置已保存: {config_path}")
```

---

## 项目结构

```
paddle-trainer2/
├── SKILL.md                    # Agent 执行指南（详细规则说明）
├── README.md                   # 本文件
├── scripts/                    # Python 脚本
│   ├── gpu_yaml_finder.py      # GPU YAML 检索器（Agent 驱动）
│   ├── config_generator.py     # 配置生成器（Agent 驱动）
│   ├── error_handler.py        # 错误处理器（Agent 驱动）
│   └── train_launcher.py       # 训练启动器（整合以上模块）
├── templates/                  # 模板与规则
│   ├── mapping_rules.yaml      # GPU→XPU 映射规则与错误处理规则
│   └── xpu_train.sh.template   # 启动脚本模板
├── reference_configs/          # 参考配置
│   └── xpu_reference.yaml      # XPU 参考配置
└── output/                     # 训练输出目录（可配置）
    ├── checkpoints/            # 模型检查点
    ├── vdl_log/                # VisualDL 训练日志
    ├── dataset_cache/          # 数据集预处理缓存
    └── paddleformers_dist_log/ # 分布式训练日志
```

---

## 设计原则

### Agent 驱动原则
- **Skill 提供上下文**: 提供原始数据、规则和参考信息
- **Agent 执行决策**: 基于语义理解自主判断和选择
- **可解释输出**: 决策过程可追溯，结果可解释

### 显式暴露原则
对于以下错误类型，Agent **不得**执行修复，必须显式返回：
- `runtime_error`: 段错误、断言失败（通常与配置无关）
- `operator_not_supported`（核心算子）: 核心功能缺失
- `unknown`: 未知错误类型

### 安全约束
- 最大修复次数: 3 次（防止无限循环）
- 必需备份: 每次修复自动备份
- 失败回滚: 保存失败时自动恢复
- 禁止操作: 删除文件、执行系统命令、修改环境变量

### 配置生成来源优先级

**优先级1: GPU YAML 配置（强制优先）**
- 在 PaddleFormers 仓库中查找目标模型的 GPU YAML 文件
- 以 GPU 配置为基础模板进行 GPU→XPU 转换
- 这是推荐的生成方式

**优先级2: XPU 参考配置（回退方案，需显式允许）**
- 仅在找不到 GPU YAML 且 `allow_fallback=True` 时触发
- 使用 `reference_configs/xpu_reference.yaml` 作为基础
- 必须在输出中明确说明使用了回退方案

---

## 更多信息

- **详细文档**: 参见 [SKILL.md](SKILL.md)，包含：
  - Agent 执行指南
  - GPU YAML 检索与选择详细流程
  - GPU→XPU 配置转换规则
  - 错误检测与修复机制
  - 详细规则速查表

---

## 许可

Copyright © 2024 PaddleFormer Team
