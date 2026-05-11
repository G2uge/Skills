---
name: run-paddle-xpu-model-training
description: Run Paddle-based model training tasks in Ubuntu environments, especially for XPU / Kunlun hardware. Applies to any PaddleFormers-based model training task, including but not limited to LLM architectures. This version includes mandatory skill invocation mechanism for SubAgents.
---

> **执行约束**：执行者必须严格遵循本 skill 定义的调用顺序，不得擅自添加前置检查或跳过逻辑。子 skill 内部自行处理安装/更新判断。
>
> **SubAgent 强制调用约束**：所有 SubAgent 必须通过 Skill 工具强制调用指定 Skill，不得直接执行技术操作。
>
> **动态路径约束**：本 skill 使用相对路径 `{SKILL_ROOT}` 引用其他 skill，子 Agent 必须根据当前 skill 文件位置动态计算路径，禁止写死绝对路径。
>
> **Step 执行方式约束**：本 Skill 定义的所有 Step（Step 0~Step 5）必须通过**创建 SubAgent**的方式调用执行，禁止直接在主 Agent 中执行任何 Step 的具体逻辑。主 Agent 仅负责任务编排和结果汇总。

# 一、总体目标

给定以下输入信息：

* 模型名称
* 模型结构 GPU YAML 配置
* 模型路径
* 模型训练数据集

完成该模型在**XPU 硬件环境下的训练适配工作**。

完整执行流程如下：

> **输入检查 → 工作空间初始化 → 模型环境搭建 → 模型配置处理 → 模型执行 → 验证 → 记录**

**说明：**

* 当前单轮任务仅处理 **一个模型** 的 XPU 训练适配工作。
* 整个流程由主 Agent 编排，SubAgent 分工执行。
* 所有工作目录和路径固定，通过主 Agent 强记忆管理。

---

# 二、特殊要求传递机制

## 2.1 设计目标

允许主 Agent 在标准流程基础上，向特定 SubAgent 传递**额外约束、覆盖参数或特殊指令**。

## 2.2 特殊要求传递方式

### 传递位置

特殊要求通过 `special_requirements` 字段在创建 SubAgent 时传递：

```yaml
subagent_invocation:
  params_required: {...}       # 必需动态参数
  params_optional: {...}       # 可选参数
  special_requirements:        # ★ 特殊要求（新增）
    description: "要求描述"
    constraints: [...]         # 额外约束
    overrides: {...}           # 参数覆盖
    priority: "normal"         # 优先级
```

### 特殊要求结构

```yaml
special_requirements:
  # 描述：为什么需要特殊处理
  description: "使用清华镜像源加速下载，因为当前环境访问默认源很慢"

  # 约束列表：额外的限制条件
  constraints:
    - "必须使用国内镜像源"
    - "安装完成后验证网络连通性"
    - "如镜像源失败，尝试备用源"

  # 参数覆盖：覆盖 Skill 定义的默认值
  overrides:
    http_proxy: "http://mirrors.tuna.tsinghua.edu.cn"
    pip_index_url: "https://pypi.tuna.tsinghua.edu.cn/simple"
    timeout: 600          # 覆盖默认的 300

  # 优先级：影响特殊要求的强制执行程度
  priority: "high"        # low/normal/high/critical

  # 回退策略：特殊要求无法满足时的处理方式
  fallback:
    action: "report_and_wait"   # continue/report_and_wait/abort
    message: "镜像源配置失败，请检查网络"
```

## 2.3 特殊要求 vs Skill 规范的关系

### 优先级规则

```text
特殊要求（critical） > Skill 强制约束 > 特殊要求（high） >
Skill 默认配置 > 特殊要求（normal/low）
```

### 冲突处理

| 场景 | 处理方式 |
|-----|---------|
| 特殊要求违反 Skill 强制约束 | 子 Agent 必须拒绝，返回冲突说明 |
| 特殊要求覆盖 Skill 默认值 | 允许，子 Agent 记录覆盖项 |
| 多个特殊要求冲突 | 按 priority 排序，高优先级优先 |

## 2.4 子 Agent 处理流程

```yaml
subagent_execution_with_requirements:
  step1: "接收参数和 special_requirements"
  step2: "验证 special_requirements 与 Skill 约束的兼容性"
  step3: "合并配置：Skill 默认值 ← overrides 覆盖"
  step4: "执行 Skill，同时遵守 constraints"
  step5: "如无法满足 requirements，按 fallback 策略处理"
  step6: "返回结果，包含 requirements 执行状态"
```

---

# 三、动态路径解析机制

## SKILL_ROOT 变量定义

**{SKILL_ROOT}**：当前 skill 文件所在目录的**父目录**（即 skills 根目录）

### 目录结构示例

```text
{SKILL_ROOT}/                              # skills/ 根目录
├── model-traning-workflow2/               # 当前 skill 所在目录
│   └── SKILL2.md                          # 当前文件
├── setup-paddleformer-xpu-env/            # 目标 skill
│   └── SKILL.md
├── convert-gpu-to-xpu-yaml/               # 目标 skill
│   └── SKILL.md
├── generate-xpu-launch-script/            # 目标 skill
│   └── SKILL.md
└── run-xpu-training-with-monitor/         # 目标 skill
    └── SKILL.md
```

### 路径计算方法

**主 Agent 职责**：
```yaml
master_agent_path_resolution:
  description: "主 Agent 传递当前 skill 路径给子 Agent"
  params_to_pass:
    CURRENT_SKILL_PATH: "当前 skill 文件的绝对路径"
    # 示例：/home/user/projects/skills/model-traning-workflow2/SKILL2.md
```

**子 Agent 计算方法**：
```python
# 伪代码示例
import os

def calculate_skill_root(current_skill_path):
    """
    计算 SKILL_ROOT
    current_skill_path: 当前 skill 文件的绝对路径
    """
    # 获取父目录（model-traning-workflow2/）
    parent_dir = os.path.dirname(current_skill_path)
    # 再获取父目录的父目录（skills/）
    skill_root = os.path.dirname(parent_dir)
    return skill_root

# 示例
current_path = "/home/user/projects/skills/model-traning-workflow2/SKILL2.md"
skill_root = calculate_skill_root(current_path)
# 结果：/home/user/projects/skills

# 目标 skill 路径
target_skill = f"{skill_root}/setup-paddleformer-xpu-env/SKILL.md"
# 结果：/home/user/projects/skills/setup-paddleformer-xpu-env/SKILL.md
```

### 路径变量对照表

| 变量 | 说明 | 示例值 |
|------|------|--------|
| `{CURRENT_SKILL_PATH}` | 当前 skill 文件绝对路径 | `/path/to/skills/model-traning-workflow2/SKILL2.md` |
| `{SKILL_ROOT}` | skills 根目录 | `/path/to/skills` |
| `{SKILL_ROOT}/setup-paddleformer-xpu-env/SKILL.md` | Step 1 目标 skill | `/path/to/skills/setup-paddleformer-xpu-env/SKILL.md` |
| `{SKILL_ROOT}/convert-gpu-to-xpu-yaml/SKILL.md` | Step 2 目标 skill 1 | `/path/to/skills/convert-gpu-to-xpu-yaml/SKILL.md` |
| `{SKILL_ROOT}/generate-xpu-launch-script/SKILL.md` | Step 2 目标 skill 2 | `/path/to/skills/generate-xpu-launch-script/SKILL.md` |
| `{SKILL_ROOT}/prune-model-layers/SKILL.md` | Step 2.5 目标 skill | `/path/to/skills/prune-model-layers/SKILL.md` |
| `{SKILL_ROOT}/run-xpu-training-with-monitor/SKILL.md` | Step 3 目标 skill | `/path/to/skills/run-xpu-training-with-monitor/SKILL.md` |
| `{SKILL_ROOT}/orchestrate-xpu-validation/SKILL.md` | Step 5 目标 skill | `/path/to/skills/orchestrate-xpu-validation/SKILL.md` |

---

# 四、角色分工

---

## 2.1 主 Agent（任务编排者）

主 Agent 负责：

### 核心职责

1. 统一负责任务流程编排
2. 将任务分发至对应 SubAgent
3. 在各 SubAgent 间传递上下文与中间结果
4. 汇总所有步骤执行结果
5. 输出最终任务报告

### 职责边界

> 主 Agent **不直接执行任何技术操作**。

---

## 2.2 SubAgent（子任务执行者）

每个 SubAgent：

### 原则

> **仅负责执行单一职责任务**

即：

* 一个 SubAgent 只处理一个 Step 的工作内容。

### 通信约束

> SubAgent 之间禁止直接通信。

所有上下文传递均通过：

> **主 Agent 中转**

---

## 2.3 SubAgent 强制调用规范

### 核心原则

> **子 Agent 必须通过 Skill 工具强制调用指定 Skill，不得直接执行技术操作**

### 调用机制

```text
主 Agent 创建 SubAgent
    ↓
SubAgent 读取 Skill 文件（获取执行规范）
    ↓
SubAgent 调用 Skill 工具（强制执行）
    ↓
返回结果给主 Agent
```

### 强制调用声明格式

每个 SubAgent 必须在其 Prompt 中包含：

```yaml
subagent_mandate:
  role: "SubAgent-N"
  step: "Step N: 步骤名称"

  # 强制调用声明
  skill_invocation:
    required: true
    skill_name: "<skill-name>"
    skill_path: "{SKILL_ROOT}/<skill-name>/SKILL.md"
    # 注意：{SKILL_ROOT} 是当前 skill 所在目录的父目录（即 skills/ 目录）
    # 子 Agent 需要使用当前 skill 文件位置推算 SKILL_ROOT

  # 上下文传递策略
  context_passing:
    strategy: "minimal"  # 最小化传递
    params_from_master:  # 主 Agent 必须传递的动态参数
      - param1
      - param2
    params_from_skill:   # 子 Agent 从 Skill 读取的静态参数
      - 所有执行步骤
      - 默认值
      - 约束条件
```

### Skill 根目录说明

**SKILL_ROOT 定义**：当前 skill 文件所在目录的**父目录**

```text
目录结构示例：
{SKILL_ROOT}/                           # skills/ 目录
├── model-traning-workflow2/
│   └── SKILL2.md    <-- 当前 skill 文件位置
├── setup-paddleformer-xpu-env/
│   └── SKILL.md
├── convert-gpu-to-xpu-yaml/
│   └── SKILL.md
└── ...
```

**子 Agent 定位 SKILL_ROOT 的方法**：
1. 获取当前 skill 文件路径（由主 Agent 传递或从上下文获取）
2. 取该路径的父目录的父目录，即为 SKILL_ROOT
3. 示例：若当前 skill 在 `/path/to/skills/model-traning-workflow2/SKILL2.md`
4. 则 SKILL_ROOT = `/path/to/skills`

### SubAgent 标准执行流程

子 Agent 被创建后，必须按以下流程执行：

```yaml
subagent_execution_flow:
  step1_read_skill:
    action: "使用 Read 工具读取 SKILL.md"
    purpose: "获取完整执行规范"

  step2_extract_params:
    action: "从 Skill 提取 inputs 定义"
    purpose: "了解需要哪些参数"

  step3_invoke_skill:
    action: "使用 Skill 工具调用"
    command: 'skill: "<skill-name>"'

  step4_return_result:
    action: "按 Skill 定义格式返回结果"
```

### 禁止行为

- ❌ 子 Agent 直接执行 bash 命令完成技术操作
- ❌ 子 Agent 忽略 Skill 定义，自行决定执行逻辑
- ❌ 主 Agent 传递完整 Skill 内容给子 Agent
- ❌ 主 Agent 传递执行步骤给子 Agent

---

# 五、整体交互流程

---

```text
主Agent                                                     SubAgents
│                                                              │
│
│  ───────────── 启动 SubAgent-1 ────────────────>             │
│                         (Step 1: 模型环境搭建)                │
││<────────── 返回环境搭建结果 / 若失败立即终止 ───────────────│
│
│
│  ───────────── 启动 SubAgent-2 ────────────────>             │
│                         (Step 2: 模型配置处理)                │
││<────────── 返回配置处理结果 ───────────────────────────────│
│
│
│  ───────────── 启动 SubAgent-2.5 ──────────────>             │
│                         (Step 2.5: 层配置控制)                │
││<────────── 返回层配置结果（pruned/full）───────────────────│
│
│
│  ───────────── 启动 SubAgent-3 ────────────────>             │
│                         (Step 3: 模型执行)                    │
││<────────── 返回执行结果 ───────────────────────────────────│
│
│
│  ───────────── 启动 SubAgent-4 ────────────────>             │
│                         (Step 4: XPU 训练问题修复)            │
││<────────── 返回修复结果 ───────────────────────────────────│
│
│
│  ───────────── 启动 SubAgent-5 ────────────────>             │
│                         (Step 5: 验证)                        │
││<────────── 返回验证结果 ───────────────────────────────────│
```

---

## 循环机制

Step3～Step5 存在循环关系，分为两个阶段：

### 阶段一：减层快速验证（pruned phase）

```text
模型执行（减层）
   ↓
失败 → 问题修复 → 重新执行（保持减层）
   ↓
成功 → 触发恢复全量 → 进入阶段二
```

### 阶段二：全量正式训练（full phase）

```text
模型执行（全量）
   ↓
失败 → 问题修复 → 重新执行（保持全量）
   ↓
成功 → 进入 Step 5 验证
```

### 恢复触发条件

当 `current_phase == "pruned"` 且 Step 3 成功时：
- 主 Agent 标记 `need_restore = true`
- 重新启动 SubAgent-2.5（action=restore）恢复全量配置
- 恢复后 `current_phase` 更新为 `"full"`
- 重新启动 SubAgent-3 执行全量训练

直到：

> **全量训练成功并通过 Step 5 验证后结束流程**

---

# 六、目录结构与输入定义

---

## 4.1 固定目录结构

所有工作目录固定如下：

```yaml
workspace:
  BASE_DIR: "/root/paddlejob/tmp"
  VENV_DIR: "/root/paddlejob/tmp/paddle"
  VENV_ACTIVATE: "/root/paddlejob/tmp/paddle/bin/activate"
  DATASETS_DIR: "/root/paddlejob/tmp/datasets"
  OUTPUT_DIR: "/root/paddlejob/tmp/output"
  REPOS_DIR: "/root/paddlejob/tmp/repos"
```

目录说明：
- `paddle/` - Python 虚拟环境目录
- `datasets/` - 用户输入文件副本（GPU yaml、训练数据等）
- `output/` - 所有输出文件（XPU yaml、启动脚本、检查点、日志）
- `repos/` - git clone 的代码仓库（PaddleFleet、PaddleFormers）

---

## 4.2 系统输入

用户必须提供以下全部内容：

```yaml
模型名称:
  示例: qwen_vl_30B
  说明: 模型标识名称

GPU_YAML_PATH:
  示例: /root/paddlejob/zhangxiao_dev/configs/qwen3vl_gpu.yaml
  说明: 用户指定的 GPU YAML 配置文件原始路径

模型路径:
  示例: /root/paddlejob/zhangxiao_dev/data/Qwen3-VL-30B-A3B-Thinking
  说明: 模型权重文件路径

数据集源路径:
  示例: /root/paddlejob/zhangxiao_dev/data/coco_grounding
  说明: 训练数据集源目录路径
```

---

# 七、执行步骤定义

---

# Step 0：输入完整性检查与主 Agent 记忆（主 Agent）

---

## 目标
在执行流程前，校验所有必需输入是否由用户明确提供完整，并建立主 Agent 强记忆。

## 检查项

用户必须显式提供以下全部内容：
* **模型名称** - 模型标识名称
* **GPU_YAML_PATH** - GPU YAML 配置文件原始路径
* **模型路径** - 模型权重文件路径
* **数据集源路径** - 训练数据集源目录路径

## 主 Agent 强记忆

Step 0 完成后，主 Agent 必须持久化以下上下文：

```yaml
session_memory:
  user_inputs:
    model_name: " "<用户提供的模型名称>"
    gpu_yaml_path: " "<用户提供的GPU yaml路径>"
    model_path: " "<用户提供的模型路径>"
    dataset_source_path: " "<用户提供的数据集路径>"

  workspace:
    BASE_DIR: "/root/paddlejob/tmp"
    VENV_DIR: "/root/paddlejob/tmp/paddle"
    VENV_ACTIVATE: "/root/paddlejob/tmp/paddle/bin/activate"
    DATASETS_DIR: "/root/paddlejob/tmp/datasets"
    OUTPUT_DIR: "/root/paddlejob/tmp/output"
    REPOS_DIR: "/root/paddlejob/tmp/repos"

  # 动态计算路径（基于 dataset_source_path 的 basename）
  computed_paths:
    dataset_folder_name: " "<dataset_source_path 的文件夹名>"
    dataset_workspace_path: "/root/paddlejob/tmp/datasets/<文件夹名>"
    train_dataset_path: " "<将原始 train_dataset_path 前缀替换为 dataset_workspace_path>"
    eval_dataset_path: " "<将原始 eval_dataset_path 前缀替换为 dataset_workspace_path>"

  generated_files:
    xpu_yaml: "/root/paddlejob/tmp/output/xpu_config.yaml"
    launch_script: "/root/paddlejob/tmp/output/train_xpu.sh"

  # 减层控制状态（用于两阶段训练：减层冒烟 → 恢复全量）
  layer_pruning:
    enabled: true               # 开关，默认启用。可通过 special_requirements 传入 false 关闭
    mode: "extreme_fast"        # 减层模式：extreme_fast | fast | balanced
    component: "auto"           # 作用范围：auto | all | {具体字段路径}
    current_phase: null         # 当前阶段：null | pruned | full
    is_restored: false          # 是否已完成从 pruned 到 full 的恢复
    need_restore: false         # 标记：减层跑通后需要触发恢复

  # ★ YAML 修改记录（由 Step 3 训练过程产生，用于 Step 2.5 恢复时参考）
  yaml_fixes:
    # 结构示例（由 SubAgent-3 在训练过程中动态填充）
    # - field: "recompute_num_layers"
    #   from: 11
    #   to: 7
    #   reason: "减层联动：recompute_num_layers <= chunk_size"
    #   restore_on_full: true      # 全量恢复时是否需要恢复该字段
    # - field: "pipeline_model_parallel_size"
    #   from: 4
    #   to: 1
    #   reason: "环境限制：world_size 与并行度不匹配"
    #   restore_on_full: false     # 环境限制类修改，全量时保持
```

## 校验约束
* 所有必填项必须由用户显式提供；
* 禁止使用默认值代替；
* 禁止根据上下文推测缺失内容；
* 禁止自动补全缺失参数；
* 禁止在信息不完整时继续执行流程。

## 异常处理

若任一信息缺失：

> 立即终止流程并提示用户补充。

## 返回格式

```json
{
  "输入检查": "Success | Fail",
  "failure_summary": "<失败原因说明>",
  "session_memory": {
    "user_inputs": { ... },
    "workspace": { ... },
    "generated_files": { ... }
  }
}
```

---

# Step 1：工作空间初始化与模型环境搭建

---

## 目标
1. 创建工作空间目录结构
2. 复制用户输入文件到 datasets 目录
3. 依次安装 Paddle-XPU、PaddleFleet、PaddleFormer

## 执行者

> SubAgent-1

---

### 强制调用声明

```yaml
skill_invocation:
  required: true
  skill_name: "setup-paddleformer-xpu-env"
  skill_path: "{SKILL_ROOT}/setup-paddleformer-xpu-env/SKILL.md"
```

### 最小上下文传递

主 Agent 创建 SubAgent-1 时，**仅传递**：

```yaml
# 动态参数（必需）
params_required:
  BASE_DIR: "/root/paddlejob/tmp"                    # 工作基础目录
  VENV_DIR: "/root/paddlejob/tmp/paddle"             # 虚拟环境目录
  REPOS_DIR: "/root/paddlejob/tmp/repos"             # 代码仓库存放目录
  VENV_ACTIVATE: "/root/paddlejob/tmp/paddle/bin/activate"  # 激活脚本路径

# 可选参数（使用 Skill 默认值）
params_optional: []

# 特殊要求（主 Agent 对当前 SubAgent 的额外约束）
special_requirements:
  description: "主 Agent 对本 Step 的特殊要求说明"
  constraints: []      # 额外约束条件
  overrides: {}        # 覆盖 Skill 默认值的参数
  priority: "normal"   # 优先级：low/normal/high/critical
```

**子 Agent 自行从 Skill 获取**：
- 代理配置（http_proxy 等）
- 子 Skill 调用顺序（install-paddle-xpu → install-paddlefleet → install-paddleformers）
- 验证脚本
- 返回格式

### SubAgent-1 执行流程

1. **确定 SKILL_ROOT**：根据当前 skill 文件位置推算父目录
2. **读取 Skill**：`Read {SKILL_ROOT}/setup-paddleformer-xpu-env/SKILL.md`
3. **提取规范**：从 Skill 的 `Execution flow` 章节获取执行步骤
4. **调用 Skill**：`skill: "setup-paddleformer-xpu-env"`
5. **返回结果**：按 Skill 定义的 JSON 格式返回

---

## 执行内容

### 1.1 创建工作空间目录

```bash
mkdir -p /root/paddlejob/tmp/{paddle,datasets,output,repos}
```

### 1.2 复制用户输入文件

```bash
# 复制 GPU yaml 文件
cp "${gpu_yaml_path}" /root/paddlejob/tmp/datasets/gpu_config.yaml

# 复制训练数据集（保留原始文件夹名称）
cp -r "${dataset_source_path}" /root/paddlejob/tmp/datasets/
```

### 1.3 安装 Paddle-XPU、PaddleFleet、PaddleFormer 环境

**调用 Skill**：`setup-paddleformer-xpu-env`

**传递参数**：
```yaml
inputs:
  BASE_DIR: "/root/paddlejob/tmp"
  VENV_DIR: "/root/paddlejob/tmp/paddle"
  REPOS_DIR: "/root/paddlejob/tmp/repos"
  VENV_ACTIVATE: "/root/paddlejob/tmp/paddle/bin/activate"
```

**内部执行逻辑**：
1. 配置代理（固定）：
   ```bash
   export http_proxy=http://agent.baidu.com:8891
   export https_proxy=$http_proxy
   export no_proxy=localhost,bj.bcebos.com,su.bcebos.com,paddle-ci.gz.bcebos.com
   ```

2. 检查并 git clone Paddle 源码到 `${REPOS_DIR}/Paddle`，存在则 git pull

3. 在 `${REPOS_DIR}/Paddle/build` 编译安装 PaddlePaddle-XPU 到 `${VENV_DIR}`

4. 检查并 git clone PaddleFleet 到 `${REPOS_DIR}/PaddleFleet`，存在则 git pull

5. 在 `${REPOS_DIR}/PaddleFleet` 安装 PaddleFleet

6. 检查并 git clone PaddleFormers 到 `${REPOS_DIR}/PaddleFormers`，存在则 git pull

7. 在 `${REPOS_DIR}/PaddleFormers` 安装 PaddleFormer

8. 验证环境并输出完成信息

**代理配置说明**：
- 使用固定代理：`http://agent.baidu.com:8891`
- no_proxy 包含：`localhost,bj.bcebos.com,su.bcebos.com,paddle-ci.gz.bcebos.com`

---

## 返回格式

```json
{
  "环境搭建": "Success | Fail",
  "工作空间": {
    "BASE_DIR": "/root/paddlejob/tmp",
    "VENV_DIR": "/root/paddlejob/tmp/paddle",
    "DATASETS_DIR": "/root/paddlejob/tmp/datasets",
    "OUTPUT_DIR": "/root/paddlejob/tmp/output",
    "REPOS_DIR": "/root/paddlejob/tmp/repos"
  },
  "已复制文件": [
    "/root/paddlejob/tmp/datasets/gpu_config.yaml",
    "/root/paddlejob/tmp/datasets/..."
  ],
  "failure_summary": "<失败原因说明>"
}
```

---

# Step 2：模型配置处理

---

## 执行者

> SubAgent-2

---

### 强制调用声明

```yaml
skill_invocation:
  required: true
  skills:
    - name: "convert-gpu-to-xpu-yaml"
      path: "{SKILL_ROOT}/convert-gpu-to-xpu-yaml/SKILL.md"
      order: 1
    - name: "generate-xpu-launch-script"
      path: "{SKILL_ROOT}/generate-xpu-launch-script/SKILL.md"
      order: 2
      dependency: "convert-gpu-to-xpu-yaml 的输出文件"
```

### 最小上下文传递

主 Agent 创建 SubAgent-2 时，**仅传递**：

```yaml
# 动态参数（必需）
params_required:
  gpu_yaml_path: "/root/paddlejob/tmp/datasets/gpu_config.yaml"
  output_path: "/root/paddlejob/tmp/output/xpu_config.yaml"
  python_env_path: "/root/paddlejob/tmp/paddle"
  output_dir: "/root/paddlejob/tmp/output"
  model_name: "<模型名称>"
  # 路径覆盖参数（用于覆盖 GPU YAML 中的硬编码路径）
  model_path: "<用户提供的模型路径>"
  dataset_dir: "/root/paddlejob/tmp/datasets" 

# 可选参数（使用 Skill 默认值）
params_optional:
  reference_yaml: null
  num_xpus: 8
  xpu_devices: "0,1,2,3,4,5,6,7"
  # 数据集路径：主 Agent 已将原始路径映射到工作空间路径
  train_dataset_path: "/root/paddlejob/tmp/datasets/<dataset_folder_name>/train.jsonl"
  eval_dataset_path: "/root/paddlejob/tmp/datasets/<dataset_folder_name>/val2.jsonl"

# 特殊要求（主 Agent 对当前 SubAgent 的额外约束）
special_requirements:
  description: "主 Agent 对本 Step 的特殊要求说明"
  constraints: []
  overrides: {}
  priority: "normal"
```

**子 Agent 自行从 Skills 获取**：
- GPU→XPU 转换规则（字段映射、参数变换、结构调整）
- 启动脚本模板
- 依赖处理逻辑

### SubAgent-2 执行流程

1. **确定 SKILL_ROOT**：根据当前 skill 文件位置推算父目录
2. **读取 Skill 1**：`Read {SKILL_ROOT}/convert-gpu-to-xpu-yaml/SKILL.md`
3. **调用 Skill 1**：`skill: "convert-gpu-to-xpu-yaml"` → 生成 xpu_config.yaml
4. **读取 Skill 2**：`Read {SKILL_ROOT}/generate-xpu-launch-script/SKILL.md`
5. **调用 Skill 2**：`skill: "generate-xpu-launch-script"`（使用第一个 Skill 的输出作为输入）
6. **返回结果**：包含两个生成文件路径的 JSON

---

## 执行内容

### 2.1 GPU 配置转换为 XPU 配置

**调用 Skill**：`convert-gpu-to-xpu-yaml`

**输入参数**：
```yaml
inputs:
  gpu_yaml_path: "/root/paddlejob/tmp/datasets/gpu_config.yaml"
  output_path: "/root/paddlejob/tmp/output/xpu_config.yaml"
  reference_yaml: null  # 可选，如有参考配置则传入
  # 路径覆盖参数（覆盖 GPU YAML 中的硬编码路径，使用工作空间内的映射路径）
  model_path: "/root/paddlejob/zhangxiao_dev/data/Qwen3-VL-30B-A3B-Thinking"
  train_dataset_path: "/root/paddlejob/tmp/datasets/coco_grounding/train.jsonl"
  eval_dataset_path: "/root/paddlejob/tmp/datasets/coco_grounding/val2.jsonl"
```

**输出**：
- 生成的 XPU YAML 文件路径：`/root/paddlejob/tmp/output/xpu_config.yaml`

### 2.2 生成 XPU 启动脚本

**调用 Skill**：`generate-xpu-launch-script`

**输入参数**：
```yaml
inputs:
  config_file: "/root/paddlejob/tmp/output/xpu_config.yaml"  # 上一步的输出
  python_env_path: "/root/paddlejob/tmp/paddle"
  output_dir: "/root/paddlejob/tmp/output"
  model_name: "<从主Agent记忆获取>"
  num_xpus: 8
  xpu_devices: "0,1,2,3,4,5,6,7"
```

**输出**：
- 生成的启动脚本路径：`/root/paddlejob/tmp/output/train_xpu.sh`

**依赖说明**：
- `generate-xpu-launch-script` 依赖 `convert-gpu-to-xpu-yaml` 的输出文件

---

## 异常处理

若任一 skill 调用失败：

> 终止流程并提示用户修复输入。

---

## 返回格式

```json
{
  "模型配置处理": "Success | Fail",
  "生成文件": {
    "xpu_yaml": "/root/paddlejob/tmp/output/xpu_config.yaml",
    "launch_script": "/root/paddlejob/tmp/output/train_xpu.sh"
  },
  "模型运行环境": "单机 | 多机",
  "failure_summary": "<失败原因说明>"
}
```

---

# Step 2.5：层配置控制

---

## 执行者

> SubAgent-2.5

---

### 强制调用声明

```yaml
skill_invocation:
  required: true
  skill_name: "prune-model-layers"
  skill_path: "{SKILL_ROOT}/prune-model-layers/SKILL.md"
```

### 最小上下文传递

主 Agent 创建 SubAgent-2.5 时，**仅传递**：

```yaml
# 动态参数（必需）
params_required:
  model_path: "<用户提供的模型路径>"
  action: "prune | restore"     # 主 Agent 告知本次操作类型
  mode: "extreme_fast"          # action=prune 时生效
  component: "auto"

# 可选参数
params_optional:
  backup_path: null             # action=restore 时，如已知备份路径可传入

  # ★ YAML 联动恢复参数（仅在 action=restore 时生效）
  xpu_yaml_path: null           # XPU YAML 配置文件路径，恢复时同步修复 YAML
  yaml_restore_overrides: {}    # YAML 恢复字段映射，如 {recompute_num_layers: 11}
  yaml_keep_as_is: []           # YAML 中保持不变的字段列表（环境限制类修改）

# 特殊要求
special_requirements:
  description: "主 Agent 对本 Step 的特殊要求说明"
  constraints: []
  overrides: {}
  priority: "normal"
```

### SubAgent-2.5 执行流程

1. **确定 SKILL_ROOT**：根据当前 skill 文件位置推算父目录
2. **读取 Skill**：`Read {SKILL_ROOT}/prune-model-layers/SKILL.md`
3. **根据 action 调用 Skill**：
   - `action == "prune"`：调用 `prune-model-layers --mode extreme_fast`
   - `action == "restore"`：调用 `prune-model-layers --mode full`
4. **返回结果**：按 Skill 定义的 JSON 格式返回

### 主 Agent 调度逻辑

```yaml
layer_pruning_gate:
  logic:
    - if: "layer_pruning.enabled == false"
      action: "跳过 SubAgent-2.5，直接进入 Step 3"

    - elif: "current_phase == 'pruned' AND need_restore == true"
      action: "启动 SubAgent-2.5，action=restore"
      params:
        # ★ 传递 YAML 恢复参数（来自 Step 3 记录的 yaml_fixes_history）
        xpu_yaml_path: "/root/paddlejob/tmp/output/xpu_config.yaml"
        yaml_restore_overrides: "<来自 Step 3 的 yaml_fixes 记录（仅减层联动修改）>"
        yaml_keep_as_is: "<环境限制类字段列表，保持当前值不变>"
      next: "恢复后 current_phase='full'，is_restored=true，重新进入 Step 3"

    - elif: "current_phase == null"
      action: "启动 SubAgent-2.5，action=prune"
      next: "设置 current_phase='pruned'，进入 Step 3"

    - else:
      action: "current_phase 已确定且无需恢复，跳过 SubAgent-2.5，直接进入 Step 3"
```

---

## 返回格式

```json
{
  "层配置处理": "Success | Fail",
  "action": "prune | restore",
  "current_phase": "pruned | full",
  "backup_path": "config.json.bak.xxx",
  "prune_result": {
    "original_layers": 36,
    "target_layers": 6,
    "components_modified": 1
  },
  "yaml_restore": {
    "status": "Success | Fail | NotRequired",
    "yaml_path": "/root/paddlejob/tmp/output/xpu_config.yaml",
    "fields_restored": [
      {
        "field": "recompute_num_layers",
        "from": 7,
        "to": 11,
        "reason": "减层联动恢复"
      }
    ],
    "fields_kept": [
      {
        "field": "pipeline_model_parallel_size",
        "value": 1,
        "reason": "环境限制：8 卡 world_size 不匹配 PP=4"
      }
    ]
  },
  "failure_summary": "<失败原因>"
}
```

---

# Step 3：模型执行

---

## 执行者

> SubAgent-3

---

### 强制调用声明

```yaml
skill_invocation:
  required: true
  skill_name: "run-xpu-training-with-monitor"
  skill_path: "{SKILL_ROOT}/run-xpu-training-with-monitor/SKILL.md"
```

### 文件修改限制

> **⚠️ 强制限制**：本 Step 仅允许修改以下文件类型，**禁止修改任何其他类型文件**：
> - ✅ YAML 配置文件（`.yaml`, `.yml`）
> - ✅ Shell 启动脚本（`.sh`）

**遇到以下情况的处理方式**：
- 若错误涉及修改 Python 源码、模型权重、数据文件等 → **禁止自行修复**
- 若错误涉及修改 Paddle/PaddleFormers 框架源码 → **禁止自行修复**
- 若错误需要安装额外系统依赖 → **禁止自行安装**

**正确处理流程**：
```yaml
error_handling:
  allowed_file_types: [".yaml", "yml", ".sh"]

  when_error_requires_source_modification:
    action: "report_to_master"  # 禁止自行修复
    message_format: |
      错误类型: <error_type>
      错误信息: <error_message>
      涉及文件: <file_path>
      建议操作: <suggested_fix>

  when_yaml_error:
    action: "auto_fix"  # 仅 YAML 错误可自动修复
    max_retries: 3
```

### 最小上下文传递

主 Agent 创建 SubAgent-3 时，**仅传递**：

```yaml
# 动态参数（必需）
params_required:
  launch_script: "/root/paddlejob/tmp/output/train_xpu.sh"
  config_file: "/root/paddlejob/tmp/output/xpu_config.yaml"
  python_env_path: "/root/paddlejob/tmp/paddle"
  output_dir: "/root/paddlejob/tmp/output"
  training_phase: "pruned | full"    # 来自 SubAgent-2.5，标记当前训练阶段
  is_restored_run: false             # 是否为恢复全量后的重新运行

# 可选参数（使用 Skill 默认值）
params_optional:
  log_file: "/root/paddlejob/tmp/output/paddleformers_dist_log/workerlog.0"
  timeout: 300
  max_retries: 3
  stuck_timeout: 60

# 特殊要求（主 Agent 对当前 SubAgent 的额外约束）
special_requirements:
  description: "主 Agent 对本 Step 的特殊要求说明"
  constraints: []
  overrides: {}
  priority: "normal"
```

**子 Agent 自行从 Skill 获取**：
- 环境预检查步骤
- 训练监控逻辑（loss 检测、错误检测、阻塞检测）
- YAML 错误自动修复规则
- 资源清理流程
- 状态判定优先级

### SubAgent-3 执行流程

1. **确定 SKILL_ROOT**：根据当前 skill 文件位置推算父目录
2. **读取 Skill**：`Read {SKILL_ROOT}/run-xpu-training-with-monitor/SKILL.md`
3. **理解监控逻辑**：重点理解 `步骤2: 监控训练状态` 的状态判定规则
4. **调用 Skill**：`skill: "run-xpu-training-with-monitor"`
5. **处理结果**：
   - 成功 → 返回训练详情
   - YAML 错误 → 按 Skill 定义修复后重试
   - 其他错误 → 按 Skill 定义返回失败
6. **返回结果**：按 Skill 定义的 JSON 格式返回

---

## 执行逻辑

**调用 Skill**：`run-xpu-training-with-monitor`

**输入参数**：
```yaml
inputs:
  launch_script: "/root/paddlejob/tmp/output/train_xpu.sh"  # Step 2 生成
  config_file: "/root/paddlejob/tmp/output/xpu_config.yaml"  # Step 2 生成
  python_env_path: "/root/paddlejob/tmp/paddle"
  output_dir: "/root/paddlejob/tmp/output"  # 训练输出目录，用于定位日志
  log_file: "/root/paddlejob/tmp/output/paddleformers_dist_log/workerlog.0"  # 日志路径（基于 output_dir）
  timeout: 300
  max_retries: 3
  stuck_timeout: 60
```

**功能说明**：
- 执行启动脚本并实时监控训练状态
- 以检测到 loss 输出为训练成功标志
- **自动识别并修复 YAML 配置错误**：若检测到 YAML 错误，自动修改 `/root/paddlejob/tmp/output/xpu_config.yaml` 并重试
- **其他错误直接抛出**：OOM、运行时错误等不可修复错误直接返回失败
- **⚠️ 重要限制**：仅允许修改 YAML 和 Shell 脚本文件，禁止修改 Python 源码、模型文件等
- **★ YAML 修改记录**：所有 YAML 修改必须记录在 `yaml_fixes` 中，标注 `restore_on_full` 标记（true=全量恢复时需恢复，false=环境限制类保持）

**阶段切换说明**：
- 当 `training_phase == "pruned"` 且训练成功时，返回中设置 `trigger_restore: true`，主 Agent 据此触发恢复全量
- 当 `training_phase == "full"` 且训练成功时，正常进入 Step 5 精度验证

---

## 异常处理

### 可自动修复的错误
- YAML 配置格式错误 → 自动修复并重试
- 启动脚本参数错误 → 自动修复并重试

### 需上报主 Agent 的错误（禁止自行修复）

以下错误涉及需要主 Agent / 用户决策的问题，SubAgent **禁止自行修复**，必须返回 `escalation: {required: true}`：

- 涉及修改 Python 源码的错误
- 涉及修改 Paddle/PaddleFormers 框架的错误
- 需要修改模型权重文件的错误
- 需要安装系统级依赖的错误
- **数据集格式与模型/模板不兼容**（`dataset_incompatibility`）
  - 典型表现：大量 "preprocess data error"、"empty messages" WARNING（1000+ 条）
  - 原因：视觉数据用文本模型训练、模板与数据类型不匹配
  - 需决策：换数据集、换模型、或改模板
- **模型架构不被当前框架支持**（`model_unsupported`）
  - 典型表现："model architecture not supported"
  - 需决策：更换模型或升级框架版本
- **训练长时间（>5分钟）无有效进展**（`pseudo_active`）
  - 典型表现：日志持续增长但无 loss/global_step，或反复输出同一初始化信息
  - 原因：数据预处理死循环、迭代器卡住、模板与数据不匹配
  - 需决策：分析日志判断根因，可能需要更换数据集或调整配置
- **模板与数据类型不匹配**
  - 典型表现：用 `qwen3` 纯文本模板训练 VLM 视觉数据
  - 需决策：更换 template 或更换数据集

**上报格式**（必须包含 escalation 字段）：
```json
{
  "模型运行状态": "Fail",
  "错误类型": "source_modification_required",
  "不可修复原因": "错误涉及修改 Paddle 框架源码",
  "涉及文件": "/path/to/paddle/source.py",
  "建议操作": "请检查 Paddle 版本兼容性",
  "failure_summary": "<详细错误信息>",
  "escalation": {
    "required": true,
    "escalation_reason": "dataset_incompatibility",
    "subagent_can_fix": false,
    "detected_patterns": ["<检测到的异常模式>"],
    "root_cause_analysis": "<根因分析>",
    "suggested_actions": ["<建议操作1>", "<建议操作2>"],
    "requires_human_decision": true
  }
}
```

### 通用处理原则
若运行报错且不可修复：

> 立即终止执行并返回错误信息。若判断为 SubAgent 无法修复的问题，**必须**返回包含 `escalation.required: true` 的 JSON，不得自行尝试修复。

---

## 返回格式

```json
{
  "模型运行状态": "Success | Fail",
  "training_phase": "pruned | full",
  "trigger_restore": false,
  "训练详情": {
    "launch_script": "/root/paddlejob/tmp/output/train_xpu.sh",
    "config_file": "/root/paddlejob/tmp/output/xpu_config.yaml",
    "output_dir": "/root/paddlejob/tmp/output/checkpoints",
    "retry_count": 0,
    "yaml_fixed": false,
    "yaml_fixes": [
      {
        "field": "pipeline_model_parallel_size",
        "from": 4,
        "to": 1,
        "reason": "环境限制：world_size 与并行度不匹配",
        "restore_on_full": false
      },
      {
        "field": "recompute_num_layers",
        "from": 11,
        "to": 7,
        "reason": "减层联动：recompute_num_layers <= chunk_size",
        "restore_on_full": true
      },
      {
        "field": "using_sonic_moe",
        "from": true,
        "to": false,
        "reason": "环境限制：sonicmoe 算子未编译",
        "restore_on_full": false
      },
      {
        "field": "apply_rope_fusion",
        "from": true,
        "to": false,
        "reason": "环境限制：fused_rope 不支持 partial_rotary",
        "restore_on_full": false
      }
    ]
  },
  "failure_summary": "<运行失败原因>",
  "escalation": {
    "required": false,
    "escalation_reason": "",
    "subagent_can_fix": false,
    "detected_patterns": [],
    "root_cause_analysis": "",
    "suggested_actions": [],
    "requires_human_decision": false
  }
}
```

**escalation 字段说明**：
- 当 `required: true` 时，表示 SubAgent 检测到无法自行修复的问题，必须上报主 Agent
- 主 Agent 收到 `escalation.required: true` 后，应：
  1. 读取 `root_cause_analysis` 了解根因
  2. 参考 `suggested_actions` 进行决策
  3. 根据 `detected_patterns` 确认问题现象
  4. 执行修复后，决定是否重新启动 Step 3

---

# Step 4：XPU 训练问题修复

---

## 执行者

> SubAgent-4

---

### 强制调用声明

```yaml
skill_invocation:
  required: true    # Step 4 必须调用问题修复 Skill
  skill_name: "fix-xpu-training-issues"
  skill_path: "{SKILL_ROOT}/fix-xpu-training-issues/SKILL.md"
  note: "本 Step 将问题修复工作委托给专门的修复 Skill，自身仅负责传递上下文和返回结果"
```

### 最小上下文传递

主 Agent 创建 SubAgent-4 时，**仅传递**：

```yaml
# 动态参数（传递给修复 Skill 的完整上下文）
params_required:
  # Step 3 执行结果
  step3_result: " "<Step 3 的返回结果>"
  step3_status: " "<Step 3 的运行状态: Success|Fail>"
  
  # 错误信息
  error_message: " "<完整报错信息>"
  error_source: " "<错误来源: config|training|validation>"
  
  # 相关文件路径
  yaml_path: " "<YAML 配置文件路径>"
  launch_script_path: " "<启动脚本路径>"
  log_path: " "<训练日志路径>"
  output_dir: " "<输出目录路径>"
  
  # 模型信息
  model_name: " "<模型名称>"
  model_type: " "<模型类型>"
  
  # 参考数据（可选）
  gpu_reference_result: " "<GPU 训练参考结果（精度对比用）>"

# 特殊要求（主 Agent 对当前 SubAgent 的额外约束）
special_requirements:
  description: "主 Agent 对本 Step 的特殊要求说明"
  constraints: []
  overrides: {}
  priority: "normal"
```

### SubAgent-4 执行流程

```yaml
execution_flow:
  step_1_prepare:
    action: "整理传递给修复 Skill 的上下文参数"
    input: "主 Agent 传递的所有参数"
    output: "完整的修复请求参数包"
  
  step_2_locate_skill:
    action: "定位修复 Skill 路径"
    logic: |
      SKILL_ROOT = dirname(dirname(CURRENT_SKILL_PATH))
      skill_path = f"{SKILL_ROOT}/fix-xpu-training-issues/SKILL.md"
  
  step_3_invoke_fix_skill:
    action: "调用 fix-xpu-training-issues Skill"
    method: "使用 Skill 工具调用"
    params: "整理好的完整上下文参数"
    note: "修复 Skill 内部完成：问题分类、路由、修复、验证、迭代"
  
  step_4_return_result:
    action: "将修复 Skill 的返回结果直接传递给主 Agent"
    note: "本 Step 不对修复结果做额外处理，保持透明转发"
```

---

## 当前策略

### 职责边界

Step 4 作为**问题修复的调用入口**，遵循**最小职责原则**：

1. **不负责**：具体的问题分析、分类、修复逻辑
2. **不负责**：修复迭代调度、效果验证逻辑
3. **负责**：将完整上下文传递给专门的修复 Skill
4. **负责**：将修复结果透明返回给主 Agent

### 修复 Skill 职责

`fix-xpu-training-issues` Skill 内部实现完整修复闭环：

1. **问题分析**：自动分类问题（配置/算子/精度）
2. **执行修复**：调用对应修复策略
3. **训练验证**：调用 `run-xpu-training-with-monitor` 运行训练，以**输出 loss**为成功标准
4. **迭代判断**：
   - 训练输出 loss → 修复成功，返回
   - 训练失败 → 分析新错误，继续下一轮修复
   - 无法修复/达到最大轮次 → 返回失败/上报

**关键原则**：修复 Skill 内部完成所有修复和验证工作，主 Agent 仅接收最终结果。

### 主 Agent 与修复 Skill 隔离

```
主 Agent
   │ 传递参数
   ▼
SubAgent-4 (Step 4)
   │ 透明转发
   ▼
fix-xpu-training-issues Skill
   │ 内部完成所有修复逻辑
   │ (classify → route → fix → verify → iterate)
   ▼
返回修复结果
   │ 透明返回
   ▼
主 Agent 接收结果
```

---

## 返回格式

SubAgent-4 直接返回 `fix-xpu-training-issues` Skill 的结果，格式如下：

```json
{
  "问题修复": "Success | Fail | NotRequired",
  "修复摘要": {
    "total_attempts": " "<尝试轮次>",
    "final_status": "fixed | failed | manual_required",
    "training_verified": true,
    "loss_output_detected": true
  },
  "修复历史": [
    {
      "attempt": 1,
      "invoked_skill": " "<Skill 名称>",
      "fix_description": " "<修复描述>",
      "training_result": "failed",
      "error_after_fix": " "<新错误>"
    },
    {
      "attempt": 2,
      "invoked_skill": " "<Skill 名称>",
      "fix_description": " "<修复描述>",
      "training_result": "success",
      "loss_output": true
    }
  ],
  "最终结果": {
    "files_modified": [""<文件列表>"],
    "can_proceed_to_step5": true
  },
  "上报": {
    "required": false,
    "reason": " "<上报原因>",
    "suggestion": " "<建议>"
  }
}
```

### 特殊情形：Step 3 成功

若 Step 3 返回成功状态，SubAgent-4 仍调用修复 Skill，Skill 内部识别为无需修复：

```json
{
  "问题修复": "NotRequired",
  "修复摘要": {
    "final_status": "no_issue",
    "training_verified": false,
    "loss_output_detected": false
  },
  "修复历史": [],
  "最终结果": {
    "files_modified": [],
    "can_proceed_to_step5": true
  },
  "上报": {
    "required": false,
    "suggestion": "Step 3 执行成功，无需修复，可直接进入验证阶段"
  }
}
```

### 最小上下文传递

主 Agent 创建 SubAgent-4 时，**仅传递**：

```yaml
# 动态参数（前一 Step 的结果）
params_required:
  step3_result: "<Step 3 的返回结果>"
  step3_status: "<Step 3 的运行状态>"
  yaml_fixes: "<YAML 修复记录>"
  failure_summary: "<失败原因（如有）>"

# 特殊要求（主 Agent 对当前 SubAgent 的额外约束）
special_requirements:
  description: "主 Agent 对本 Step 的特殊要求说明"
  constraints: []
  overrides: {}
  priority: "normal"
```

### SubAgent-4 执行流程

1. 接收 Step 3 的结果
2. 收集最终训练状态
3. 整理 YAML 修复记录（如有）
4. 若 Step 3 失败，整理错误信息
5. 返回最终处理结果

---

# Step 5：验证

---

## 执行者

> SubAgent-5

---

### 强制调用声明

```yaml
skill_invocation:
  required: true
  skill_name: "orchestrate-xpu-validation"
  skill_path: "{SKILL_ROOT}/orchestrate-xpu-validation/SKILL.md"
  note: "本 Step 调用精度验证调度 Skill，由其编排早期门控、轮询等待、最终验证的完整流程"
```

### 最小上下文传递

主 Agent 创建 SubAgent-5 时，**仅传递**：

```yaml
# 动态参数（验证需要的目录路径）
params_required:
  xpu_log_path: "/root/paddlejob/tmp/output/paddleformers_dist_log/workerlog.0"
  gpu_log_path: " "<GPU 训练 workerlog.0 路径>"
  output_dir: "/root/paddlejob/tmp/output"

  # 阈值配置（可选，使用 Skill 默认值）
  thresholds:
    mae: 0.05
    rmse: 0.1
    max_diff: 0.5
    pearson: 0.99
    spearman: 0.99
    relative_error_percent: 1.0
    r2: 0.98

  # 模型信息（可选，用于报告标注）
  model_name: " "<模型名称>"
  model_type: " "<模型类型>"

# 可选参数（使用 Skill 默认值）
params_optional:
  early_gate_max_steps: 10
  early_gate_min_steps: 3
  wait_poll_interval: 30
  wait_max_duration: 86400
  wait_stuck_timeout: 300

# 特殊要求（主 Agent 对当前 SubAgent 的额外约束）
special_requirements:
  description: "主 Agent 对本 Step 的特殊要求说明"
  constraints: []
  overrides: {}
  priority: "normal"
```

### SubAgent-5 执行流程

```yaml
execution_flow:
  step_1_locate_skill:
    action: "定位验证 Skill 路径"
    logic: |
      SKILL_ROOT = dirname(dirname(CURRENT_SKILL_PATH))
      skill_path = f"{SKILL_ROOT}/validate-xpu-gpu-training/SKILL.md"

  step_2_invoke_validation_skill:
    action: "调用 orchestrate-xpu-validation Skill"
    method: "使用 Skill 工具调用"
    params:
      - xpu_log_path
      - gpu_log_path
      - output_dir
      - thresholds
      - model_name
      - model_type
      - early_gate_max_steps
      - early_gate_min_steps
      - wait_poll_interval
      - wait_max_duration
      - wait_stuck_timeout
    note: |
      调度 Skill 内部编排：
      1. 阶段1：调用 validate-xpu-gpu-training (early_only) 做初步校验
         - 有 GPU 数据：双端 Loss 对比（MAE + Pearson）
         - 无 GPU 数据：XPU 单端自检验（单调性 + 平滑度）
         - Early Fail 立即终止，不等训练完成
      2. 阶段2：阻塞轮询等待训练完成
         - 检测完成标志、卡住、超时
      3. 阶段3：调用 validate-xpu-gpu-training (final_only) 做完整验证
         - 数据提取/清洗/对齐
         - 单端趋势分析 + 双端对比
         - 计算 7 项量化指标
         - 生成完整报告

  step_3_return_result:
    action: "将验证 Skill 的返回结果直接传递给主 Agent"
    note: "本 Step 不对验证结果做额外处理，保持透明转发"
```

---

## 验证内容

通过 `validate-xpu-gpu-training` Skill 执行完整训练验证：

### Step A：数据提取与清洗
- 从 XPU `workerlog.0` 提取 `(global_step, loss, lr, ppl, global_norm)`
- 从 GPU `workerlog.0` 提取相同字段
- 按 `global_step` 对齐，缺失值插值，异常值过滤

### Step B：单端曲线绘制与分析
- **XPU 单端图**：`loss/ppl/lr/global_norm vs global_step`
- **GPU 单端图**：`loss/ppl/lr/global_norm vs global_step`
- 计算单端趋势指标：单调性（Spearman）、平滑度（相邻差值标准差）、收敛性（最后 10% steps 的 loss 标准差）

### Step C：双端对齐与对比
- **叠加对比图（Overlay）**：XPU/GPU loss 曲线叠加，标注 MAE/MaxDiff/Pearson
- **差异曲线图（Delta）**：`Δloss = loss_xpu - loss_gpu`，标注容忍带
- **相对误差图**：逐 step 相对误差百分比
- **散点对齐图**：`(loss_gpu, loss_xpu)` 散点 + `y=x` 参考线，标注 R²

### Step D：量化评估指标
| 指标 | 阈值 | 说明 |
|---|---|---|
| MAE | < 0.05 | 平均绝对误差 |
| RMSE | < 0.1 | 均方根误差 |
| Max Diff | < 0.5 | 最大差异 |
| Pearson | > 0.99 | 皮尔逊相关 |
| Spearman | > 0.99 | 斯皮尔曼相关 |
| Relative Error % | < 1% | 相对误差百分比 |
| R² | > 0.98 | 决定系数 |

### Step E：输出文件完整性验证
- 检查 `/root/paddlejob/tmp/output/checkpoints/` 是否有模型保存
- 检查日志文件是否生成
- 检查训练配置是否正确应用

---

## 总体判定标准

```yaml
validation_result:
  Pass: "全部 7 项指标通过阈值"
  Warn: "某指标未通过但 relative_error < 5%"
  Fail: "relative_error >= 5% 或存在严重异常"
```

---

## 异常处理

若验证返回 `Fail`：

> 立即终止，将验证报告返回给主 Agent，由主 Agent 决定是否回退到 Step 4 修复流程。

若验证返回 `Warn`：

> 返回完整报告，主 Agent 根据差异程度决定是否接受结果或进一步分析。

---

## 返回格式

SubAgent-5 直接返回 `orchestrate-xpu-validation` Skill 的结果，格式如下：

```json
{
  "调度状态": "Success | Fail",
  "阶段执行记录": {
    "early_gate": {
      "executed": true,
      "status": "Pass | Warn | Fail",
      "mode": "dual_side | single_side_only",
      "steps_used": 8,
      "metrics": {
        "mae": 0.03,
        "pearson": 0.998
      },
      "report_path": "<output_dir>/orchestrate-xpu-validation/early_gate/validate-xpu-gpu-training/report.json"
    },
    "wait_training": {
      "executed": true,
      "waited_seconds": 3600,
      "completed_by": "train_runtime pattern",
      "stuck_detected": false,
      "timeout_detected": false
    },
    "final_validation": {
      "executed": true,
      "status": "Pass | Warn | Fail | single_side_only",
      "validation_tier": "full_analysis",
      "report_path": "<output_dir>/orchestrate-xpu-validation/final/validate-xpu-gpu-training/report.json"
    }
  },
  "最终结果": {
    "validation_status": "Pass | Warn | Fail | single_side_only",
    "final_report_path": "<output_dir>/orchestrate-xpu-validation/final/validate-xpu-gpu-training/report.json",
    "plots_dir": "<output_dir>/orchestrate-xpu-validation/final/validate-xpu-gpu-training/plots/"
  },
  "failure_summary": "<验证失败原因>"
}
```

---

# 八、最终输出规范（主 Agent 汇总）

主 Agent 汇总所有步骤结果，统一输出。

步骤结果包含 6 个子结果：
- Step1: 环境搭建
- Step2: 模型配置处理
- Step2_5: 层配置控制（减层/恢复，仅 layer_pruning.enabled=true 时存在）
- Step3: 模型执行
- Step4: 问题修复（如有）
- Step5: 精度验证

```json
{
  "任务状态": "Success | Fail",
  "工作目录": {
    "base": "/root/paddlejob/tmp",
    "datasets": "/root/paddlejob/tmp/datasets",
    "output": "/root/paddlejob/tmp/output",
    "repos": "/root/paddlejob/tmp/repos"
  },
  "步骤结果": [
    Step1,
    Step2,
    Step2_5,
    Step3,
    Step4,
    Step5
  ],
  "layer_pruning": {
    "enabled": true,
    "pruned_phase_completed": true,
    "full_phase_completed": true,
    "restored": true,
    "pruned_training_verified": true,
    "full_training_verified": true
  },
  "最终总结": "<整体执行摘要>"
}
```

---

# 附录 D：主 Agent 创建 SubAgent 的 Prompt 模板（含特殊要求版）

## SubAgent-1 Prompt 模板（含特殊要求处理）

```yaml
subagent_description: "Step1: 环境搭建"

subagent_prompt: |
  你是 SubAgent-1，职责：执行 Step 1 工作空间初始化与模型环境搭建。

  【强制指令 - 必须遵守】
  你必须调用 Skill: setup-paddleformer-xpu-env

  【路径定位】
  1. 当前 skill 文件位置：{CURRENT_SKILL_PATH}
  2. SKILL_ROOT = {CURRENT_SKILL_PATH} 的父目录的父目录
  3. 目标 Skill 路径：{SKILL_ROOT}/setup-paddleformer-xpu-env/SKILL.md

  【主 Agent 传递的参数】
  - BASE_DIR: {BASE_DIR}
  - VENV_DIR: {VENV_DIR}
  - REPOS_DIR: {REPOS_DIR}
  - VENV_ACTIVATE: {VENV_ACTIVATE}

  【★ 特殊要求处理（如有）】
  {SPECIAL_REQUIREMENTS_BLOCK}
  # 说明：主 Agent 传递的 special_requirements 将插入此处
  # 包含：description, constraints, overrides, priority, fallback

  【你的执行流程】
  1. 使用 Read 工具读取目标 Skill 文件
  2. 检查特殊要求与 Skill 约束的兼容性
     - 如冲突且 priority=critical/high：拒绝执行，返回冲突说明
     - 如冲突且 priority=normal/low：记录冲突，忽略特殊要求
  3. 应用 overrides 覆盖 Skill 默认配置
  4. 使用 Skill 工具调用：skill: "setup-paddleformer-xpu-env"
     - 执行时遵守 constraints
  5. 如无法满足 requirements：按 fallback 策略处理
  6. 返回结果，包含 special_requirements 执行状态

  【约束】
  - 不得跳过 Skill 定义的任何步骤
  - 不得修改 Skill 定义的关键配置
  - 必须返回 Skill 定义的 JSON 格式结果
  - 必须在返回中包含 special_requirements_status
```

## SubAgent-2 Prompt 模板（含特殊要求处理）

```yaml
subagent_description: "Step2: 配置处理"

subagent_prompt: |
  你是 SubAgent-2，职责：执行 Step 2 模型配置处理。

  【强制指令 - 必须遵守】
  按顺序调用两个 Skill：
  1. Skill: convert-gpu-to-xpu-yaml
  2. Skill: generate-xpu-launch-script

  【路径定位】
  1. 当前 skill 文件位置：{CURRENT_SKILL_PATH}
  2. SKILL_ROOT = {CURRENT_SKILL_PATH} 的父目录的父目录
  3. 目标 Skill 1 路径：{SKILL_ROOT}/convert-gpu-to-xpu-yaml/SKILL.md
  4. 目标 Skill 2 路径：{SKILL_ROOT}/generate-xpu-launch-script/SKILL.md

  【主 Agent 传递的参数】
  - gpu_yaml_path: {gpu_yaml_path}
  - output_path: {output_path}
  - python_env_path: {python_env_path}
  - output_dir: {output_dir}
  - model_name: {model_name}
  # 路径覆盖参数（用于覆盖 GPU YAML 中的硬编码路径，使用工作空间映射路径）
  - model_path: {model_path}
  - train_dataset_path: {train_dataset_path}   # 格式: /root/paddlejob/tmp/datasets/<dataset_folder_name>/train.jsonl
  - eval_dataset_path: {eval_dataset_path}     # 格式: /root/paddlejob/tmp/datasets/<dataset_folder_name>/val.jsonl

  【★ 特殊要求处理（如有）】
  {SPECIAL_REQUIREMENTS_BLOCK}

  【你的执行流程】
  1. 检查特殊要求与 Skill 约束的兼容性
  2. 读取第一个 Skill 文件（convert-gpu-to-xpu-yaml）
     - 应用 overrides 到转换规则
  3. 调用第一个 Skill，获取 xpu_config.yaml 路径
     - 遵守 constraints（如跳过某些字段转换）
  4. 读取第二个 Skill 文件（generate-xpu-launch-script）
  5. 调用第二个 Skill（使用第一个 Skill 的输出作为输入）
  6. 返回结果，包含 special_requirements_status

  【约束】
  - 必须按顺序执行，先完成 GPU→XPU 转换，再生成启动脚本
  - generate-xpu-launch-script 依赖 convert-gpu-to-xpu-yaml 的输出
  - 必须返回包含两个生成文件路径的 JSON
  - 特殊要求可能影响转换规则，需谨慎处理
```

## SubAgent-3 Prompt 模板（含特殊要求处理）

```yaml
subagent_description: "Step3: 模型执行"

subagent_prompt: |
  你是 SubAgent-3，职责：执行 Step 3 模型执行与监控。

  【强制指令 - 必须遵守】
  你必须调用 Skill: run-xpu-training-with-monitor

  【路径定位】
  1. 当前 skill 文件位置：{CURRENT_SKILL_PATH}
  2. SKILL_ROOT = {CURRENT_SKILL_PATH} 的父目录的父目录
  3. 目标 Skill 路径：{SKILL_ROOT}/run-xpu-training-with-monitor/SKILL.md

  【主 Agent 传递的参数】
  - launch_script: {launch_script}
  - config_file: {config_file}
  - python_env_path: {python_env_path}
  - output_dir: {output_dir}

  # 减层阶段信息
  - training_phase: {training_phase}         # pruned | full，来自 SubAgent-2.5
  - is_restored_run: {is_restored_run}       # true | false，是否为恢复后的重新运行

  【★ 特殊要求处理（如有）】
  {SPECIAL_REQUIREMENTS_BLOCK}

  【你的执行流程】
  1. 读取 Skill 文件理解监控逻辑
  2. 检查特殊要求与 Skill 约束的兼容性
     - 如特殊要求覆盖监控参数（如 timeout, max_retries）
     - 验证覆盖值是否在合理范围内
  3. 调用 Skill 启动训练监控
     - 传递 overrides 给 Skill（如 check_interval, stuck_timeout）
     - 遵守 constraints（如额外的日志记录要求）
  4. 根据返回结果处理：
     - 成功 → 返回训练详情
     - YAML 错误 → Skill 会自动修复，你只需等待结果
     - **escalation.required: true → 立即停止，完整返回 escalation 信息给主 Agent**
     - 其他错误 → 按 Skill 定义返回失败信息
  5. 返回结果，包含 special_requirements_status

  【★ 新增：主动分析与上报义务】
  作为 SubAgent-3，你不仅仅是 Skill 的调用者，还必须承担训练状态的主动分析责任：

  1. **Escalation 信号透传义务**：
     - 若 Skill 返回结果中包含 `escalation.required: true`，你必须：
       - **立即停止执行，不得尝试自行修复**
       - 将 `escalation` 对象完整无损地返回给主 Agent
       - 在返回中附加你的分析（如有额外观察）

  2. **主动观察与上报义务**：
     在等待/监控训练过程中，若你观察到以下现象（即使 Skill 尚未触发上报），也应主动上报：
     - 训练日志中出现 **1000+ 条相同的 WARNING**（如 "preprocess data error"）
     - 训练超过 **5 分钟** 无任何 loss / global_step / step 输出
     - 日志反复输出同一初始化信息（如 "Not using packing mode" 重复数万次）
     - 你基于日志判断当前问题涉及**数据集格式、模型架构兼容性、模板不匹配**等需要主 Agent 决策的问题

  3. **上报格式要求**：
     主动上报时必须返回标准 JSON，包含完整的 `escalation` 字段：
     ```json
     {
       "模型运行状态": "Fail",
       "failure_summary": "<你的分析摘要>",
       "escalation": {
         "required": true,
         "escalation_reason": "dataset_incompatibility",
         "subagent_can_fix": false,
         "detected_patterns": ["<你观察到的具体现象>"],
         "root_cause_analysis": "<你基于日志的根因推理>",
         "suggested_actions": ["<建议主 Agent 执行的操作>"],
         "requires_human_decision": true
       }
     }
     ```

  4. **禁止行为**：
     - 收到 escalation 信号后，**禁止**尝试修改数据集文件
     - **禁止**尝试修改 Python 源码或框架代码
     - **禁止**尝试换用其他模型权重
     - **禁止**隐瞒或简化错误信息

  【特别注意】
  Skill 内部已包含：
  - 环境预检查（步骤0）
  - 训练监控循环（步骤2，含伪活跃检测 PSEUDO_ACTIVE）
  - YAML 错误自动修复（步骤3）
  - 资源清理
  - **错误分类与上报机制（dataset_incompatibility / pseudo_active 等必须上报）**

  你只需调用 Skill，但可以通过特殊要求调整监控参数。
  **关键原则：你不仅是执行者，还是问题的第一发现者和上报者。**
```

## SubAgent-4 Prompt 模板（含特殊要求处理）

```yaml
subagent_description: "Step4: XPU 训练问题修复"

subagent_prompt: |
  你是 SubAgent-4，职责：执行 Step 4 XPU 训练问题修复。

  【强制指令 - 必须遵守】
  你必须调用 Skill: fix-xpu-training-issues
  本 Step 仅负责传递上下文参数，所有修复逻辑由该 Skill 内部完成。

  【路径定位】
  1. 当前 skill 文件位置：{CURRENT_SKILL_PATH}
  2. SKILL_ROOT = {CURRENT_SKILL_PATH} 的父目录的父目录
  3. 目标 Skill 路径：{SKILL_ROOT}/fix-xpu-training-issues/SKILL.md

  【主 Agent 传递的参数】
  - step3_result: {step3_result}
  - step3_status: {step3_status}
  - error_message: {error_message}
  - error_source: {error_source}
  - yaml_path: {yaml_path}
  - launch_script_path: {launch_script_path}
  - log_path: {log_path}
  - output_dir: {output_dir}
  - model_name: {model_name}
  - model_type: {model_type}
  - gpu_reference_result: {gpu_reference_result}

  【★ 特殊要求处理（如有）】
  {SPECIAL_REQUIREMENTS_BLOCK}
  # 特殊要求会传递给 fix-xpu-training-issues Skill

  【你的执行流程】
  1. 确定 SKILL_ROOT：根据 CURRENT_SKILL_PATH 计算父目录的父目录
  2. 读取 Skill 文件：{SKILL_ROOT}/fix-xpu-training-issues/SKILL.md
  3. 整理完整参数包：将所有上下文参数传递给修复 Skill
  4. 使用 Skill 工具调用：skill: "fix-xpu-training-issues"
     - 传递所有参数给 Skill
     - 特殊要求通过 overrides/constraints 传递
  5. 透明返回结果：将 Skill 返回结果直接返回给主 Agent
     - 不修改、不包装修复结果
     - 保持主 Agent 与修复 Skill 的隔离

  【约束】
  - 必须调用 fix-xpu-training-issues Skill，禁止直接处理问题
  - 必须传递完整的上下文参数，确保修复 Skill 有足够信息
  - 必须透明返回 Skill 结果，不做额外处理或转换
  - 禁止在 SubAgent-4 层实现问题分析、分类、修复逻辑
  - 禁止修改修复 Skill 返回的结果结构

  【注意】
  fix-xpu-training-issues Skill 内部完成完整修复闭环：
  - 问题自动分类（配置/算子/精度）
  - 路由到具体修复策略并执行修复
  - **调用 run-xpu-training-with-monitor 运行训练验证**（以输出 loss 为成功标准）
  - 多轮迭代修复（最多 3 轮），直到训练成功或无法继续
  - 返回最终结果给主 Agent

  SubAgent-4 仅作为调用入口，不感知内部实现细节。
```

## SubAgent-5 Prompt 模板（含特殊要求处理）

```yaml
subagent_description: "Step5: 精度验证调度"

subagent_prompt: |
  你是 SubAgent-5，职责：执行 Step 5 精度验证调度。

  【强制指令 - 必须遵守】
  你必须调用 Skill: orchestrate-xpu-validation

  【路径定位】
  1. 当前 skill 文件位置：{CURRENT_SKILL_PATH}
  2. SKILL_ROOT = {CURRENT_SKILL_PATH} 的父目录的父目录
  3. 目标 Skill 路径：{SKILL_ROOT}/orchestrate-xpu-validation/SKILL.md

  【主 Agent 传递的参数】
  - xpu_log_path: {xpu_log_path}
  - gpu_log_path: {gpu_log_path}
  - output_dir: {output_dir}
  - thresholds: {thresholds}
  - model_name: {model_name}
  - model_type: {model_type}
  - early_gate_max_steps: {early_gate_max_steps}
  - early_gate_min_steps: {early_gate_min_steps}
  - wait_poll_interval: {wait_poll_interval}
  - wait_max_duration: {wait_max_duration}
  - wait_stuck_timeout: {wait_stuck_timeout}

  【★ 特殊要求处理（如有）】
  {SPECIAL_REQUIREMENTS_BLOCK}

  【你的执行流程】
  1. 使用 Read 工具读取 orchestrate-xpu-validation Skill 文件
  2. 检查特殊要求与 Skill 约束的兼容性
  3. 使用 Skill 工具调用：skill: "orchestrate-xpu-validation"
     - 传递所有主 Agent 给的参数
     - 遵守 constraints
  4. 等待调度器返回完整结果（包含 Early + Wait + Final 三阶段汇总）
     - 注意：调度器内部会阻塞轮询训练完成，可能耗时较长
  5. 透明返回结果给主 Agent，不做额外处理或转换

  【注意】
  orchestrate-xpu-validation 内部编排逻辑：
  - 阶段1：调用 validate-xpu-gpu-training (early_only) 做初步校验
    - 有 GPU：双端 MAE + Pearson
    - 无 GPU：XPU 单端单调性 + 平滑度
    - Early Fail 时立即返回，不进入等待
  - 阶段2：阻塞轮询等待训练完成
  - 阶段3：调用 validate-xpu-gpu-training (final_only) 做完整验证

  你只需调用 orchestrate-xpu-validation 一次，等待其返回最终结果。

  【约束】
  - 必须调用 orchestrate-xpu-validation Skill，不得直接调用 validate-xpu-gpu-training
  - 必须透明返回调度器结果，不做额外处理
  - 返回结果包含 special_requirements_status
```

## SubAgent-2.5 Prompt 模板（含特殊要求处理）

```yaml
subagent_description: "Step2.5: 层配置控制"

subagent_prompt: |
  你是 SubAgent-2.5，职责：执行模型减层或恢复操作。

  【强制指令 - 必须遵守】
  你必须调用 Skill: prune-model-layers

  【路径定位】
  1. 当前 skill 文件位置：{CURRENT_SKILL_PATH}
  2. SKILL_ROOT = {CURRENT_SKILL_PATH} 的父目录的父目录
  3. 目标 Skill 路径：{SKILL_ROOT}/prune-model-layers/SKILL.md

  【主 Agent 传递的参数】
  - model_path: {model_path}
  - action: {action}              # prune 或 restore
  - mode: {mode}                  # action=prune 时生效
  - component: {component}

  # ★ YAML 联动恢复参数（仅在 action=restore 时生效）
  - xpu_yaml_path: {xpu_yaml_path}                     # XPU YAML 配置文件路径
  - yaml_restore_overrides: {yaml_restore_overrides}   # 需要恢复的字段映射
  - yaml_keep_as_is: {yaml_keep_as_is}                 # 保持不变的字段列表

  【★ 特殊要求处理（如有）】
  {SPECIAL_REQUIREMENTS_BLOCK}

  【你的执行流程】
  1. 确定 SKILL_ROOT：根据 CURRENT_SKILL_PATH 计算父目录的父目录
  2. 读取 Skill 文件：{SKILL_ROOT}/prune-model-layers/SKILL.md
  3. 根据 action 调用 Skill：
     - action == "prune"：调用 `prune-model-layers --mode extreme_fast`
       * 仅修改 config.json，不涉及 YAML
     - action == "restore"：调用 `prune-model-layers --mode full`
       * 恢复 config.json 层数
       * 同时读取 xpu_yaml_path，应用 yaml_restore_overrides 恢复 YAML 联动字段
       * yaml_keep_as_is 中列出的字段保持不变（环境限制类修改）
  4. 验证 YAML 语法正确性（如 action=restore 且涉及 YAML 修改）
  5. 返回结果给主 Agent，包含：
     - current_phase（pruned | full）
     - config_json_restore 状态
     - yaml_restore 状态（含 fields_restored 和 fields_kept）

  【约束】
  - 必须调用 prune-model-layers Skill，禁止直接修改 config.json
  - action=restore 时，必须同步处理 YAML 联动恢复
  - 必须透明返回 Skill 结果，不修改结果结构
  - 返回结果包含 special_requirements_status

  【YAML 恢复原则】
  - 仅恢复因减层引入的联动修改（如 recompute_num_layers）
  - 环境限制类修改（如 pipeline_model_parallel_size、using_sonic_moe）必须保持当前值
  - 恢复后必须验证 YAML 语法正确性
```

---

# 附录 B：强制调用与特殊要求验证清单

## 主 Agent 调度前检查清单

在创建 SubAgent 前，主 Agent 确认：

- [ ] SubAgent Prompt 中包含明确的 `skill_invocation` 声明
- [ ] SubAgent Prompt 中指定了 Skill 文件完整路径
- [ ] 只传递了最小必要参数（动态参数）
- [ ] 明确要求子 Agent 先读取 Skill 文件
- [ ] 明确要求子 Agent 使用 Skill 工具调用

## SubAgent 执行检查清单

子 Agent 被创建后，必须：

- [ ] 使用 Read 工具读取指定的 SKILL.md
- [ ] 使用 Skill 工具调用指定的 Skill
- [ ] 按照 Skill 定义的步骤执行，不得跳过
- [ ] 按照 Skill 定义的格式返回结果

## 动态路径解析检查清单

### 主 Agent 检查项

- [ ] 传递 `CURRENT_SKILL_PATH` 变量给子 Agent
- [ ] 说明如何计算 SKILL_ROOT（当前 skill 的父目录的父目录）

### 子 Agent 检查项

- [ ] 根据 `CURRENT_SKILL_PATH` 计算 `SKILL_ROOT`
- [ ] 使用 `{SKILL_ROOT}/<skill-name>/SKILL.md` 格式构造目标 skill 路径
- [ ] 禁止使用写死的绝对路径（如 `/root/paddlejob/Gruge/skills/...`）

### 路径验证示例

```yaml
# 正确示例
skill_path: "{SKILL_ROOT}/setup-paddleformer-xpu-env/SKILL.md"
# 若 CURRENT_SKILL_PATH = /home/user/skills/model-traning-workflow2/SKILL2.md
# 则 SKILL_ROOT = /home/user/skills
# 最终路径 = /home/user/skills/setup-paddleformer-xpu-env/SKILL.md

# 错误示例（禁止）
skill_path: "/root/paddlejob/Gruge/skills/setup-paddleformer-xpu-env/SKILL.md"
```

## 参数传递分层验证

| 参数类型 | 来源 | 验证项 |
|---------|------|--------|
| 动态参数 | 主 Agent 传递 | 是否为路径、名称等运行时确定的值 |
| 静态参数 | Skill 定义 | 子 Agent 是否正确从 Skill 读取 |
| 执行步骤 | Skill 定义 | 子 Agent 是否按 Skill 执行而非自行实现 |
| 默认值 | Skill 定义 | 可选参数是否使用 Skill 默认值 |
| 特殊要求 | 主 Agent 传递 | 是否与 Skill 约束冲突、是否正确执行 |

---

# 附录 C：特殊要求使用示例

## 示例 1：Step 1 使用国内镜像源

### 场景
主 Agent 发现当前环境访问默认 PyPI 源很慢，要求 SubAgent-1 使用国内镜像源（清华源或百度源）。

### 主 Agent 调用（清华源）

```yaml
agent_invocation:
  description: "Step1: 环境搭建（使用清华镜像源）"
  params_required:
    BASE_DIR: "/root/paddlejob/tmp"
    VENV_DIR: "/root/paddlejob/tmp/paddle"
    REPOS_DIR: "/root/paddlejob/tmp/repos"
    VENV_ACTIVATE: "/root/paddlejob/tmp/paddle/bin/activate"
  special_requirements:
    description: "使用清华镜像源加速 Python 包下载"
    constraints:
      - "pip 安装必须使用 https://pypi.tuna.tsinghua.edu.cn/simple"
      - "git clone 失败时尝试使用 https://ghproxy.com/ 代理"
    overrides:
      pip_index_url: "https://pypi.tuna.tsinghua.edu.cn/simple"
      pip_trusted_host: "pypi.tuna.tsinghua.edu.cn"
    priority: "high"
    fallback:
      action: "report_and_wait"
      message: "镜像源不可用，请检查网络或更换镜像源"
```

### 主 Agent 调用（百度源）

```yaml
agent_invocation:
  description: "Step1: 环境搭建（使用百度镜像源）"
  params_required:
    BASE_DIR: "/root/paddlejob/tmp"
    VENV_DIR: "/root/paddlejob/tmp/paddle"
    REPOS_DIR: "/root/paddlejob/tmp/repos"
    VENV_ACTIVATE: "/root/paddlejob/tmp/paddle/bin/activate"
  special_requirements:
    description: "使用百度镜像源加速 Python 包下载"
    constraints:
      - "pip 安装必须使用 https://mirrors.baidu.com/pypi/simple"
      - "git clone 失败时尝试使用百度代理"
    overrides:
      pip_index_url: "https://mirrors.baidu.com/pypi/simple"
      pip_trusted_host: "mirrors.baidu.com"
    priority: "high"
    fallback:
      action: "report_and_wait"
      message: "百度镜像源不可用，请检查网络或更换镜像源"
```

### SubAgent-1 Prompt

```yaml
subagent_prompt: |
  你是 SubAgent-1，职责：执行 Step 1 工作空间初始化与模型环境搭建。

  【强制指令】
  你必须调用 Skill: setup-paddleformer-xpu-env

  【路径定位】
  1. 当前 skill 文件位置：{CURRENT_SKILL_PATH}
  2. SKILL_ROOT = {CURRENT_SKILL_PATH} 的父目录的父目录
  3. 目标 Skill 路径：{SKILL_ROOT}/setup-paddleformer-xpu-env/SKILL.md

  【主 Agent 传递的参数】
  - BASE_DIR: {BASE_DIR}
  - VENV_DIR: {VENV_DIR}
  - REPOS_DIR: {REPOS_DIR}
  - VENV_ACTIVATE: {VENV_ACTIVATE}

  【★ 特殊要求 - 必须遵守】
  描述：使用国内镜像源加速 Python 包下载
  约束（根据主 Agent 选择）：
    - 清华源：pip 安装必须使用 https://pypi.tuna.tsinghua.edu.cn/simple
    - 百度源：pip 安装必须使用 https://mirrors.baidu.com/pypi/simple
  覆盖参数：
    - pip_index_url: <根据选择的源>
    - pip_trusted_host: <根据选择的源>
  优先级：high
  回退策略：如镜像源不可用，停止执行并报告

  【执行流程】
  1. 读取 Skill 文件
  2. 检查特殊要求与 Skill 约束的兼容性
  3. 应用 overrides 覆盖 Skill 默认配置
  4. 执行 Skill，同时遵守 constraints
  5. 如无法满足 requirements，按 fallback 策略处理
  6. 返回结果，包含 requirements 执行状态
```

## 示例 2：Step 3 增加监控频率

### 场景
主 Agent 怀疑当前 XPU 环境不稳定，要求 SubAgent-3 缩短监控间隔，更快发现问题。

### 主 Agent 调用

```yaml
agent_invocation:
  description: "Step3: 模型执行（高频监控）"
  params_required:
    launch_script: "/root/paddlejob/tmp/output/train_xpu.sh"
    config_file: "/root/paddlejob/tmp/output/xpu_config.yaml"
    python_env_path: "/root/paddlejob/tmp/paddle"
    output_dir: "/root/paddlejob/tmp/output"
  special_requirements:
    description: "缩短监控间隔，增加日志采样频率"
    constraints:
      - "日志检查间隔从 5 秒改为 2 秒"
      - "stuck_timeout 从 60 秒改为 30 秒"
      - "每轮监控必须记录详细状态到独立日志"
    overrides:
      check_interval: 2
      stuck_timeout: 30
      verbose_logging: true
    priority: "normal"
```

## 示例 3：多 Step 传递上下文

### 场景
Step 1 发现了一个警告，需要传递给后续 Step 注意。

### Step 1 返回

```json
{
  "环境搭建": "Success",
  "special_notes": {
    "warning": "XPU 驱动版本较旧，建议使用保守配置",
    "propagate_to": ["Step2", "Step3"]
  }
}
```

### 主 Agent 记忆并传递给 Step 2

```yaml
agent_invocation:
  description: "Step2: 配置处理（考虑 XPU 驱动限制）"
  params_required:
    gpu_yaml_path: "/root/paddlejob/tmp/datasets/gpu_config.yaml"
    output_path: "/root/paddlejob/tmp/output/xpu_config.yaml"
    python_env_path: "/root/paddlejob/tmp/paddle"
    output_dir: "/root/paddlejob/tmp/output"
    model_name: "qwen_vl_30B"
  special_requirements:
    description: "来自 Step 1 的警告：XPU 驱动版本较旧"
    constraints:
      - "batch_size 不超过 4"
      - "关闭所有实验性功能"
      - "使用最保守的内存配置"
    context_from_previous_steps:
      step1_warning: "XPU 驱动版本较旧，建议使用保守配置"
    priority: "high"
```

## 特殊要求执行状态返回

子 Agent 返回结果时，应包含 special_requirements 的执行状态：

```json
{
  "模型配置处理": "Success",
  "生成文件": {
    "xpu_yaml": "/root/paddlejob/tmp/output/xpu_config.yaml",
    "launch_script": "/root/paddlejob/tmp/output/train_xpu.sh"
  },
  "special_requirements_status": {
    "description": "使用清华镜像源加速 Python 包下载",
    "applied": true,
    "overrides_used": {
      "pip_index_url": "https://pypi.tuna.tsinghua.edu.cn/simple"
    },
    "constraints_satisfied": [
      "pip 安装使用清华镜像源: 已满足",
      "git clone 使用代理: 未触发（原地址可用）"
    ]
  },
  "failure_summary": ""
}
```
