---
name: generate-xpu-launch-script
description: |
  根据 XPU YAML 训练配置文件生成启动脚本（bash）。
  填充模板变量，生成可直接运行的训练启动脚本。
keywords: xpu, launch, script, bash, train, startup, 启动脚本, 训练
---

# XPU 训练启动脚本生成器

本 Skill 指导 Agent 根据 XPU YAML 配置文件生成训练启动脚本。

## 输入参数

| 参数 | 必需 | 说明 | 示例 |
|------|------|------|------|
| `config_file` | 是 | XPU YAML 配置文件路径 | `/data/configs/train_xpu.yaml` |
| `python_env_path` | 是 | Python 虚拟环境路径 | `/root/paddlejob/env/paddle` |
| `output_dir` | **是** | 训练输出目录，**由主 agent 强制指定** | `/root/paddlejob/tmp/output` |
| `model_name` | 否 | 模型名称，默认从 YAML 读取 | `Qwen3-VL-30B` |
| `num_xpus` | 否 | XPU 设备数量，默认 8 | `8` |
| `xpu_devices` | 否 | XPU 设备列表，默认 `0,1,2,3,4,5,6,7` | `0,1,2,3,4,5,6,7` |
| `api_yaml_path` | 否 | API 追踪配置路径，默认自动生成 | `${OUTPUT_DIR}/api.yaml` |

## 执行流程概览

```
输入: config_file, python_env_path, [其他可选参数]
  │
  ▼
┌─────────────────────────┐
│  步骤1: 读取 YAML 配置    │
│  - 提取关键信息          │
│  - 确定默认参数          │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  步骤2: 准备模板变量      │
│  - 收集所有占位符值       │
│  - 计算动态路径          │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  步骤3: 填充模板          │
│  - 读取模板文件           │
│  - 替换模板占位符         │
│  - ⚠️ 强制使用 paddleformers-cli train │
│  - 🚫 禁止使用其他启动方式 │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  步骤4: 输出脚本          │
│  - 写入文件              │
│  - 添加执行权限          │
│  - ✅ 验证启动命令        │
│  - 生成使用说明          │
└─────────────────────────┘
           │
           ▼
输出: 可执行的 bash 脚本路径
```

> **核心约束**：生成的脚本必须使用 `paddleformers-cli train ${CONFIG_FILE}` 启动训练，禁止使用 `paddle.distributed.launch` 或 `run_finetune.py`。

---

## 步骤1: 读取 YAML 配置

**执行**：
```bash
cat {config_file}
```

**提取信息**：
- `model_name_or_path` → 用于 `MODEL_NAME`
- ~~`output_dir` → 不再从 YAML 读取，强制使用传入参数~~
- `per_device_train_batch_size`, `gradient_accumulation_steps` 等训练参数

**默认值处理**：
| 参数 | 来源 | 默认值 |
|------|------|--------|
| `MODEL_NAME` | `model_name_or_path` 的最后一段 | `unknown_model` |
| `OUTPUT_DIR` | **主 agent 传入的 `output_dir` 参数** | **无（必需）** |

> **重要**：`output_dir` 必须由主 agent 显式传入，禁止从 YAML 配置文件读取，确保所有训练输出统一在指定目录下。

---

## 步骤2: 准备模板变量

### 必需变量（用户必须提供）

| 模板变量 | 值 | 说明 |
|---------|-----|------|
| `{{PYTHON_ENV_PATH}}` | `{python_env_path}` | Python 环境路径 |
| `{{CONFIG_FILE}}` | `{config_file}` | YAML 配置文件路径（绝对路径） |
| `{{OUTPUT_DIR}}` | `{output_dir}` | **训练输出目录（主 agent 强制指定）** |

### 自动计算变量

| 模板变量 | 计算方式 | 说明 |
|---------|---------|------|
| `{{MODEL_NAME}}` | `{model_name}` 或从 YAML 提取 | 模型名称 |
| `{{NUM_XPUS}}` | `{num_xpus}` 或 `8` | XPU 设备数量 |
| `{{XPU_DEVICES}}` | `{xpu_devices}` 或 `0,1,2,3,4,5,6,7` | XPU 设备列表 |
| `{{GENERATED_TIME}}` | `$(date '+%Y-%m-%d %H:%M:%S')` | 生成时间 |
| `{{API_YAML_PATH}}` | `{api_yaml_path}` 或 `${OUTPUT_DIR}/api.yaml` | API 追踪配置 |
| `{{API_CONFIG_PATH}}` | `${OUTPUT_DIR}/api_config.txt` | API 配置文件 |
| `{{FLAGS_TRACE_API}}` | `${API_YAML_PATH},${API_CONFIG_PATH}` | API 追踪标志 |
| `{{LD_LIBRARY_PATH}}` | `${PYTHON_ENV_PATH}/lib/python${VERSION}/site-packages/paddle/libs/` | 动态库路径 |

### Python 版本检测

**执行**：
```bash
{python_env_path}/bin/python --version
```

**提取版本号**（如 `3.10`）用于拼接 `LD_LIBRARY_PATH`

---

## 步骤3: 填充模板

### 模板文件位置

```
templates/xpu_train.sh.template
```

### ⚠️ 强制约束（必须遵守）

**启动命令强制约束**：
- **必须使用**：`paddleformers-cli train ${CONFIG_FILE}`
- **禁止使用**：`paddle.distributed.launch`、`python -m paddle.distributed.launch`、`run_finetune.py` 等任何其他启动方式

**原因**：`paddleformers-cli` 是 PaddleFormers 提供的标准 CLI 工具，`run_finetune.py` 不存在于 PaddleFormers 仓库。

### 模板占位符列表

| 占位符 | 来源 | 示例值 |
|--------|------|--------|
| `{{GENERATED_TIME}}` | 自动生成 | `2025-01-15 10:30:00` |
| `{{MODEL_NAME}}` | 用户提供或 YAML | `Qwen3-VL-30B` |
| `{{CONFIG_FILE}}` | 用户提供 | `/data/configs/train_xpu.yaml` |
| `{{OUTPUT_DIR}}` | 用户提供或 YAML | `./checkpoints/train_001` |
| `{{NUM_XPUS}}` | 用户提供或默认 | `8` |
| `{{XPU_DEVICES}}` | 用户提供或默认 | `0,1,2,3,4,5,6,7` |
| `{{PYTHON_ENV_PATH}}` | 用户提供 | `/root/paddlejob/env/paddle` |
| `{{API_YAML_PATH}}` | 用户提供或生成 | `./checkpoints/train_001/api.yaml` |
| `{{API_CONFIG_PATH}}` | 自动计算 | `./checkpoints/train_001/api_config.txt` |
| `{{FLAGS_TRACE_API}}` | 自动拼接 | `./checkpoints/train_001/api.yaml,./checkpoints/train_001/api_config.txt` |
| `{{LD_LIBRARY_PATH}}` | 动态拼接 | `${PYTHON_ENV_PATH}/lib/python3.10/site-packages/paddle/libs/` |

### 填充步骤

1. **读取模板文件内容**
2. **逐个替换占位符**：
   - 将 `{{MODEL_NAME}}` 替换为实际模型名称
   - 将 `{{PYTHON_ENV_PATH}}` 替换为实际环境路径
   - ...以此类推
3. **确保 `LD_LIBRARY_PATH` 使用变量形式**：
   ```bash
   export LD_LIBRARY_PATH=${PYTHON_ENV_PATH}/lib/python3.10/site-packages/paddle/libs/:${LD_LIBRARY_PATH}
   ```
4. **验证启动命令**：确保生成的脚本包含 `paddleformers-cli train ${CONFIG_FILE}`，不包含 `paddle.distributed.launch` 或 `run_finetune.py`

---

## 步骤4: 输出脚本

### 确定输出路径

- 默认：`{config_file所在目录}/train_xpu.sh`
- 或用户指定路径

### 写入文件

```bash
cat > {output_script_path} << 'EOF'
{填充后的脚本内容}
EOF
```

### 添加执行权限

```bash
chmod +x {output_script_path}
```

### 验证生成的脚本（强制步骤）

**必须执行以下验证**：

1. **验证启动命令**：
   ```bash
   grep -q "paddleformers-cli train" {output_script_path} && echo "✅ 启动命令正确" || echo "❌ 启动命令错误"
   ```

2. **验证没有使用错误命令**：
   ```bash
   if grep -qE "(paddle\.distributed\.launch|run_finetune\.py)" {output_script_path}; then
       echo "❌ 错误：脚本包含禁止的启动方式"
       exit 1
   fi
   ```

3. **验证文件存在且可执行**：
   ```bash
   [ -f {output_script_path} ] && [ -x {output_script_path} ] && echo "✅ 脚本可执行"
   ```

**验证失败处理**：
- 如果验证失败，必须删除生成的脚本并重新生成
- 禁止返回验证失败的脚本给主 Agent

### 生成 API 配置文件

如果 `api_yaml_path` 未提供，创建默认文件：
```bash
mkdir -p {OUTPUT_DIR}
cat > {OUTPUT_DIR}/api.yaml << 'EOF'
apis: []
EOF

cat > {OUTPUT_DIR}/api_config.txt << 'EOF'
EOF
```

---

## 执行示例

### 示例1: 基础生成

**用户输入**:
- `config_file = "/data/configs/qwen3vl_xpu.yaml"`
- `python_env_path = "/root/paddlejob/env/paddle"`

**Agent 执行**:

1. **读取 YAML**:
   ```yaml
   model_name_or_path: Qwen/Qwen3-VL-30B-A3B
   output_dir: ./checkpoints/train_qwen3vl_xpu
   ...
   ```

2. **准备变量**:
   ```json
   {
     "MODEL_NAME": "Qwen3-VL-30B-A3B",
     "CONFIG_FILE": "/data/configs/qwen3vl_xpu.yaml",
     "OUTPUT_DIR": "./checkpoints/train_qwen3vl_xpu",
     "NUM_XPUS": "8",
     "XPU_DEVICES": "0,1,2,3,4,5,6,7",
     "PYTHON_ENV_PATH": "/root/paddlejob/env/paddle",
     "GENERATED_TIME": "2025-01-15 10:30:00",
     "API_YAML_PATH": "./checkpoints/train_qwen3vl_xpu/api.yaml",
     "FLAGS_TRACE_API": "./checkpoints/train_qwen3vl_xpu/api.yaml,./checkpoints/train_qwen3vl_xpu/api_config.txt",
     "LD_LIBRARY_PATH": "${PYTHON_ENV_PATH}/lib/python3.10/site-packages/paddle/libs/"
   }
   ```

3. **填充模板** → 生成脚本

4. **输出**:
   ```markdown
   ✅ 启动脚本已生成

   📄 脚本路径: /data/configs/train_xpu.sh
   📋 配置文件: /data/configs/qwen3vl_xpu.yaml
   🐍 Python环境: /root/paddlejob/env/paddle
   📁 输出目录: ./checkpoints/train_qwen3vl_xpu

   🔧 使用方式:
      cd /data/configs
      ./train_xpu.sh

   📊 监控日志:
      tail -f ./checkpoints/train_qwen3vl_xpu/paddleformers_dist_log/workerlog.0
   ```

---

## 注意事项

1. **启动命令强制约束**（最重要）：
   - **必须使用**：`paddleformers-cli train ${CONFIG_FILE}`
   - **禁止使用**：`python -m paddle.distributed.launch`、`run_finetune.py` 等
   - **原因**：`run_finetune.py` 不存在于 PaddleFormers 仓库，只有 `paddleformers-cli` 是标准启动方式

2. **PYTHON_ENV_PATH 必需**: 这是唯一必须手动指定的参数，用于激活 Paddle 环境

3. **路径使用绝对路径**: 建议使用绝对路径避免工作目录切换问题

4. **LD_LIBRARY_PATH 动态生成**: 禁止写死固定路径，必须基于 `PYTHON_ENV_PATH` 动态拼接

5. **API 配置文件**: 如未提供 `api_yaml_path`，会自动在 `OUTPUT_DIR` 下创建默认文件

6. **执行权限**: 生成脚本后自动添加 `+x` 执行权限

7. **验证环境**: 脚本启动时会验证 `PYTHON_ENV_PATH/bin/activate` 是否存在

8. **必须验证生成结果**: 生成后必须检查脚本包含正确的启动命令，否则必须重新生成
