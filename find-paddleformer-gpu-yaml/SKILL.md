---
name: find-paddleformer-gpu-yaml
description: |
  根据环境定位 PaddleFormers 仓库，并智能检索最匹配目标模型的 GPU YAML 配置文件。
  支持指定 Python 环境、仓库路径或 YAML 文件夹路径，基于语义相似性进行候选文件评分和最优模板选择。
keywords: paddleformer, gpu, yaml, config, find, search, 查找, 配置
---

# 查找 PaddleFormers GPU YAML 配置

本 Skill 指导 Agent 完成两个核心任务：
1. **定位 PaddleFormers 代码仓库**（或直接使用指定路径）
2. **检索并选择最匹配的 GPU YAML 配置文件**

## 输入参数

| 参数 | 必需 | 说明 |
|------|------|------|
| `model_name` | 是 | 目标模型名称，如 "Qwen3-VL-30B-A3B-Instruct" |
| `yaml_dir` | 否 | 直接指定 YAML 配置文件夹路径，优先级最高 |
| `repo_path` | 否 | 直接指定 PaddleFormers 仓库根目录 |
| `python_path` | 否 | 指定 Python 解释器路径，用于从特定环境定位 paddleformers |

**参数优先级**: `yaml_dir` > `repo_path` > `python_path` > 自动定位

## 执行流程概览

```
输入: model_name + 可选参数(yaml_dir/repo_path/python_path)
  │
  ▼
┌─────────────────────────────────────────┐
│  步骤1: 确定 YAML 搜索范围               │
│  - 如果提供 yaml_dir: 直接使用该文件夹   │
│  - 如果提供 repo_path: 在该仓库内扫描    │
│  - 如果提供 python_path: 从该环境定位    │
│  - 否则: 自动分层检索定位                │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────┐
│  步骤2: 检索候选 YAML    │
│  - 扫描指定范围内的配置  │
│  - 提取文件语义特征      │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  步骤3: 选择最优模板     │
│  - 模型特征匹配分析      │
│  - 综合评分排序          │
│  - 输出推荐结果          │
└─────────────────────────┘
           │
           ▼
输出: 最佳匹配的 GPU YAML 路径 + 选择理由
```

## 步骤1: 确定 YAML 搜索范围

根据用户提供的参数，按以下优先级确定搜索范围：

### 优先级1: 直接指定 YAML 文件夹（`yaml_dir`）

**如果用户提供了 `yaml_dir`**：
- **直接使用该文件夹作为搜索范围**
- 跳过所有仓库定位步骤
- 验证该文件夹存在且包含 YAML 文件

**验证命令**：
```bash
ls {yaml_dir}/*.yaml {yaml_dir}/*.yml 2>/dev/null | head -5
```

**成功标志**：返回至少一个 YAML 文件路径

### 优先级2: 直接指定仓库路径（`repo_path`）

**如果用户提供了 `repo_path`**：
- 使用该路径作为 PaddleFormers 仓库根目录
- 执行仓库验证
- 在仓库内扫描 YAML 文件

**仓库验证**：
```bash
ls {repo_path}/examples/configs/ 2>/dev/null || ls {repo_path}/configs/ 2>/dev/null || ls {repo_path}/tests/config/ 2>/dev/null
```

**必需存在**：`examples/` 或 `configs/` 或 `tests/config/` 目录

### 优先级3: 指定 Python 环境（`python_path`）

**如果用户提供了 `python_path`**：
- 使用该 Python 解释器导入 paddleformers
- 从该环境获取仓库路径

**执行**：
```bash
{python_path} -c "import paddleformers; print(paddleformers.__path__[0])"
```

**成功标志**：返回有效路径，如 `/root/paddlejob/PaddleFormers/paddleformers`

**处理**：
- 提取父目录作为仓库根目录
- 验证该目录下存在配置目录

### 优先级4: 自动分层检索（默认）

如以上参数均未提供，执行自动分层检索：

#### Layer 1: Python 运行环境（默认 python）

**执行**：
```bash
python -c "import paddleformers; print(paddleformers.__path__[0])"
```

**处理**：
- 提取父目录作为仓库根目录
- 验证该目录下存在配置目录

#### Layer 2: 常见开发目录

如 Layer 1 失败，检查以下常见位置：

| 检查路径 | 验证标志 |
|---------|---------|
| `./PaddleFormers` | 存在 `examples/` 或 `configs/` 目录 |
| `../PaddleFormers` | 存在 `examples/` 或 `configs/` 目录 |
| `~/PaddleFormers` | 存在 `examples/` 或 `configs/` 目录 |
| `/workspace/PaddleFormers` | 存在 `examples/` 或 `configs/` 目录 |
| 环境变量 `$PADDLEFORMERS_ROOT` | 指向有效目录 |

#### Layer 3: 扩展搜索（最后手段）

如以上均失败，执行受限全局搜索：

```bash
# 限制深度为3层，避免耗时过长
find /root /workspace /home -maxdepth 3 -type d -name "PaddleFormers" 2>/dev/null | head -5
```

---

## 步骤2: 检索候选 GPU YAML 文件

### 2.1 扫描配置目录

根据步骤1确定的搜索范围，执行相应的扫描：

**场景A: 直接指定了 `yaml_dir`**

在该文件夹内查找所有 YAML 文件：
```bash
find {yaml_dir} -type f \( -name "*.yaml" -o -name "*.yml" \)
```

**注意**：不限制 `grep -i gpu`，由 Agent 根据内容判断是否为 GPU 配置

**场景B: 指定了仓库路径（`repo_path` 或从 `python_path` 定位）**

在仓库内递归查找 GPU 相关 YAML 文件：

**搜索范围**：
```
{REPO_ROOT}/
├── examples/configs/**/*gpu*.yaml
├── examples/config/**/*gpu*.yaml
├── configs/**/*.yaml
├── tests/config/**/*.yaml
└── **/sft/**/*.yaml
└── **/pretrain/**/*.yaml
```

**执行**：
```bash
find {REPO_ROOT} -type f \( -name "*.yaml" -o -name "*.yml" \) | grep -i gpu
```

### 2.2 解析模型特征

从输入的 `model_name` 提取关键特征：

| 特征维度 | 提取方法 | 示例 |
|---------|---------|------|
| `family` | 前缀匹配 | `qwen3`, `llama`, `deepseek` |
| `variant` | 结构标识 | `vl` (视觉), `text`, `audio` |
| `size` | 参数规模 | `7B`, `30B`, `A3B` |
| `structure` | 架构特征 | `moe`, `dense` |
| `task_type` | 训练阶段 | `sft`, `pretrain`, `instruct` |

**示例解析**：
```
输入: "Qwen3-VL-30B-A3B-Instruct"
  │
  ├── family: "qwen3"
  ├── variant: "vl"
  ├── size: "30B"
  ├── structure: ["moe"]      # 从 A3B 推断为 MoE
  └── task_type: "instruct"
```

### 2.3 提取 YAML 文件语义

对每个候选 YAML 文件，提取以下信息：

**从文件路径提取**：
- `path_hints.model_family_from_path`: 路径中的模型标识（如 `qwen3vl`）
- `path_hints.task_from_path`: 路径中的任务标识（如 `sft`, `pretrain`）
- `path_hints.device_from_path`: 设备标识（如 `gpu`, `xpu`）

**从文件内容提取**（读取前50行）：
- `model_name_or_path`: 配置中的模型名称
- `stage`: 训练阶段
- 关键参数: `per_device_train_batch_size`, `gradient_accumulation_steps` 等

---

## 步骤3: 选择最优 GPU YAML 模板

### 3.1 评分维度

Agent 对每个候选文件进行综合评分：

| 维度 | 权重 | 评分标准 |
|------|------|---------|
| **模型系列匹配** | 40% | 文件名或路径包含模型系列（如 `qwen3vl`） |
| **任务类型匹配** | 25% | `stage` 字段与目标 task_type 匹配 |
| **结构特征匹配** | 20% | 支持的结构类型（如 MoE、VL）匹配 |
| **配置完整性** | 10% | 必需字段是否齐全 |
| **设备适配性** | 5% | 是否为 GPU 配置（优先） |

### 3.2 决策流程

```
对每个候选文件:
  1. 计算各维度匹配分数
  2. 加权计算总分
  3. 记录评分理由

按总分降序排序，选择前3名:
  - 第1名总分 >= 80: 直接推荐为最优模板
  - 第1名总分 60-80: 推荐为最优，但提示需人工复核
  - 第1名总分 < 60: 列出Top-3供用户选择，或提示未找到匹配配置
```

### 3.3 输出格式

**成功找到匹配配置**：
```markdown
✅ 已找到最佳匹配的 GPU YAML 配置

📋 匹配结果:
   文件路径: {relative_path}
   绝对路径: {absolute_path}

🔍 模型特征分析:
   - 系列: {family}
   - 结构: {variant}
   - 规模: {size}
   - 任务: {task_type}

📊 候选评分:
   1. {file1}: {score}分
      - 系列匹配: ✓
      - 任务匹配: ✓
      - 理由: {reasoning}
   
   2. {file2}: {score}分
      ...

🎯 推荐配置: {file1}
   理由: {综合选择理由}
```

**未找到匹配配置**：
```markdown
⚠️ 未找到高度匹配的 GPU YAML 配置

📋 搜索结果:
   扫描文件数: {N}
   最佳匹配: {file} (得分: {score})

💡 建议:
   1. 确认模型名称拼写正确
   2. 检查 PaddleFormers 仓库路径
   3. 手动指定 GPU 配置路径
   4. 使用参考配置作为替代
```

---

## 执行示例

### 示例1: 直接指定 YAML 文件夹（推荐，最快速）

**用户输入**: 
- `model_name = "Qwen3-VL-30B-A3B-Instruct"`
- `yaml_dir = "/data/configs/gpu"`

**Agent 执行**:

1. **直接使用指定文件夹**:
   ```bash
   ls /data/configs/gpu/*.yaml
   # 找到: qwen3vl_sft.yaml, qwen3vl_pretrain.yaml, llama3_8b_sft.yaml...
   ```

2. **解析并评分**（跳过仓库定位）:
   ```
   Qwen3-VL-30B-A3B-Instruct 特征: {family: qwen3, variant: vl, size: 30B, task: instruct}
   
   候选评分:
   - qwen3vl_sft.yaml: 85分 (系列匹配+任务相近)
   - qwen3vl_pretrain.yaml: 60分 (系列匹配+任务不匹配)
   - llama3_8b_sft.yaml: 20分 (系列不匹配)
   ```

3. **输出结果**: 推荐 `qwen3vl_sft.yaml`

### 示例2: 指定 Python 环境定位

**用户输入**: 
- `model_name = "Qwen3-VL-30B-A3B-Instruct"`
- `python_path = "/root/paddlejob/Gruge/Gruge_env/paddle/bin/python"`

**Agent 执行**:

1. **使用指定 Python 定位仓库**:
   ```bash
   /root/paddlejob/Gruge/Gruge_env/paddle/bin/python -c "import paddleformers; print(paddleformers.__path__[0])"
   # 输出: /root/paddlejob/PaddleFormers/paddleformers
   ```

2. **检索候选**:
   ```bash
   find /root/paddlejob/PaddleFormers -name "*.yaml" | grep -i gpu
   ```

3. **评分并输出结果**

### 示例3: 直接指定仓库路径

**用户输入**: 
- `model_name = "Qwen3-VL-30B-A3B-Instruct"`
- `repo_path = "/workspace/PaddleFormers"`

**Agent 执行**:

1. **验证仓库路径**:
   ```bash
   ls /workspace/PaddleFormers/examples/configs/
   # 确认存在配置目录
   ```

2. **在指定仓库内扫描**:
   ```bash
   find /workspace/PaddleFormers -name "*.yaml" | grep -i gpu
   ```

3. **评分并输出结果**

### 示例4: 自动定位（默认方式）

**用户输入**: `model_name = "Qwen3-VL-30B-A3B-Instruct"`（无其他参数）

**Agent 执行**:

1. **自动分层检索**:
   ```bash
   python -c "import paddleformers; print(paddleformers.__path__[0])"
   # 或尝试其他 Layer...
   ```

2. **在定位到的仓库内扫描 YAML**

3. **评分并输出结果**

### 示例5: 定位失败处理

**用户输入**: `model_name = "Qwen3-VL-30B-A3B-Instruct"`（自动定位失败）

**Agent 执行**:
```bash
python -c "import paddleformers..."  # 失败（未安装）
ls ./PaddleFormers  # 不存在
find /root -maxdepth 3 -name "PaddleFormers"  # 未找到
```

**输出**:
```markdown
❌ 无法定位 PaddleFormers 仓库

尝试过的方法:
   1. Python 环境导入 - 失败
   2. 常见目录检查 - 失败
   3. 扩展搜索 - 失败

💡 请通过以下方式之一指定:
   1. 提供 yaml_dir: 直接指定 YAML 配置文件夹
   2. 提供 repo_path: 直接指定 PaddleFormers 仓库路径
   3. 提供 python_path: 指定包含 paddleformers 的 Python 环境
```

---

## 注意事项

1. **参数优先级**: `yaml_dir` > `repo_path` > `python_path` > 自动定位。提供高优先级参数时，跳过低优先级的检索步骤

2. **推荐使用 `yaml_dir`**: 如果已知配置文件夹路径，直接指定 `yaml_dir` 是最快速、最可靠的方式，跳过所有定位步骤

3. **`python_path` 使用场景**: 当系统默认 python 没有安装 paddleformers，而特定虚拟环境有时，使用该参数指定环境路径

4. **分层检索顺序**: 自动定位时，必须按 Layer 1 → Layer 2 → Layer 3 顺序执行

5. **评分权重可调**: 如果特定场景下某维度更重要（如任务类型优先），Agent 可调整权重

6. **配置完整性检查**: 推荐的 YAML 必须包含以下必需字段:
   - `model_name_or_path`
   - `stage`
   - `per_device_train_batch_size`
   - 至少一个训练数据路径字段

7. **避免过度依赖文件名**: 应结合文件路径、内容和配置字段综合判断，而非仅匹配文件名

8. **超时控制**: 扩展搜索（Layer 3）应限制时间和深度，避免长时间阻塞
