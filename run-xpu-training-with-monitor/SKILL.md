---
name: run-xpu-training-with-monitor
description: |
  执行 XPU 训练启动脚本并实时监控训练状态。
  以检测到 loss 输出为训练成功标志，自动识别并修复 YAML 配置错误，其他错误直接抛出。
keywords: xpu, train, monitor, loss, error, repair, retry, 训练, 监控, 错误修复
---

> **执行约束**：执行者必须严格遵循本 skill 定义的调用顺序，不得擅自添加前置检查或跳过逻辑。子 skill 内部自行处理安装/更新判断。
>
> **⚠️ 强制限制：本 skill 仅允许修改 YAML 配置文件和 Shell 启动脚本（.sh），禁止修改任何其他类型的文件（如 Python 源码、模型权重、数据文件等）。遇到其他类型文件的错误时，必须向上汇报，不得擅自修复。**

# XPU 训练执行与监控器

本 Skill 指导 Agent 执行 XPU 训练启动脚本，实时监控训练状态，自动处理配置错误。

## 输入参数

| 参数 | 必需 | 说明 | 示例 |
|------|------|------|------|
| `launch_script` | 是 | 训练启动脚本路径 | `./train_xpu.sh` |
| `config_file` | 是 | XPU YAML 配置文件路径（用于修复） | `./train_xpu.yaml` |
| `output_dir` | **是** | 训练输出目录，用于定位日志 | `/root/paddlejob/tmp/output` |
| `log_file` | 否 | 训练日志文件路径，默认基于 `output_dir` | `/root/paddlejob/tmp/output/paddleformers_dist_log/workerlog.0` |
| `timeout` | 否 | 监控超时时间（秒），默认 300 | `300` |
| `max_retries` | 否 | 最大重试次数，默认 3 | `3` |
| `stuck_timeout` | 否 | 进程阻塞检测时间（秒），默认 60 | `60` |
| `pseudo_active_timeout` | 否 | 伪活跃检测时间（秒），默认 120。当日志持续增长但无 loss/global_step 等有效训练进度标志超过此时长，判定为伪活跃 | `120` |
| `python_env_path` | 否 | Python 虚拟环境路径，用于环境预检查 | `/root/paddlejob/env/paddle` |
| `fix_history` | 否 | 修复历史记录，用于检测同一问题无推进重复出现。格式：`[{"signature": "yaml_error:missing_device", "timestamp": "...", "outcome": "same_error"}]` | `[]` |

## 执行流程概览

```
输入: launch_script, config_file
  │
  ▼
┌─────────────────────────┐
│  步骤0: 环境预检查       │
│  - 验证 Python 环境      │
│  - 检查 XPU 设备         │
│  - 检查端口占用          │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  步骤1: 启动训练脚本     │
│  - 执行 bash 脚本       │
│  - 获取训练进程 PID     │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  步骤2: 监控训练状态     │
│  - 循环读取日志         │
│  - 检测 loss/错误/超时  │
└──────────┬──────────────┘
           │
           ▼
     ┌─────┴─────┐
     │           │
    LOSS       ERROR
     │           │
     ▼           ▼
┌────────┐  ┌────────────────┐
│ 成功   │  │ 解析错误类型    │
│ 退出   │  └───────┬────────┘
└────────┘          │
                    ▼
            ┌───────┴───────┐
            │               │
       YAML错误          其他错误
            │               │
            ▼               ▼
     ┌────────────┐    ┌────────┐
     │ 修复YAML   │    │ 抛出   │
     │ 清理资源   │    │ 错误   │
     │ 重新启动   │    │ 退出   │
     │ (重试计数) │    │        │
     └────────────┘    └────────┘
```

---

## 步骤0: 环境预检查

在首次启动训练前，执行环境检查以避免浪费重试次数在环境问题上。

### 0.1 Python 环境检查

**检查虚拟环境是否存在**：
```bash
if [ ! -d "{python_env_path}" ]; then
    echo "❌ Python 环境不存在: {python_env_path}"
    exit 1
fi

if [ ! -f "{python_env_path}/bin/activate" ]; then
    echo "❌ 环境激活脚本不存在: {python_env_path}/bin/activate"
    exit 1
fi
```

**验证 Paddle 安装**：
```bash
source {python_env_path}/bin/activate
python -c "import paddle; print(f'Paddle version: {paddle.__version__}')" || {
    echo "❌ Paddle 未正确安装"
    exit 1
}
```

### 0.2 XPU 设备检查

**检查 XPU 设备可用性**：
```bash
python -c "import paddle; paddle.device.get_available_device()" 2>/dev/null || {
    echo "⚠️  无法检测 XPU 设备，继续尝试..."
}
```

### 0.3 配置文件检查

**检查 YAML 配置文件存在性**：
```bash
if [ ! -f "{config_file}" ]; then
    echo "❌ 配置文件不存在: {config_file}"
    exit 1
fi
```

**基础 YAML 语法检查**：
```bash
python -c "import yaml; yaml.safe_load(open('{config_file}'))" 2>/dev/null || {
    echo "❌ YAML 文件格式错误，请检查配置文件"
    exit 1
}
```

**检查必需字段**：
```bash
# 检查关键字段是否存在
python -c "
import yaml
with open('{config_file}') as f:
    config = yaml.safe_load(f)
    required = ['model_name_or_path', 'output_dir']
    missing = [f for f in required if f not in config]
    if missing:
        print(f'❌ 缺少必需字段: {missing}')
        exit(1)
" || exit 1
```

### 0.4 端口检查

**检查常用端口是否被占用**：
```bash
# 检查 6000-6010 端口（BKCL 常用端口范围）
for port in 6000 6001 6002 6003 6004 6005 6006 6007 6008 6009 6010; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  端口 $port 被占用"
    fi
done
```

### 0.5 磁盘空间检查

**检查输出目录磁盘空间**：
```bash
# 获取输出目录所在分区
output_dir=$(python -c "import yaml; print(yaml.safe_load(open('{config_file}'))['output_dir'])")
available_space=$(df -BG "$output_dir" 2>/dev/null | awk 'NR==2 {print $4}' | tr -d 'G')
if [ "$available_space" -lt 10 ]; then
    echo "⚠️  磁盘空间不足: ${available_space}GB (建议至少 10GB)"
fi
```

### 0.6 预检查输出

**检查通过输出**：
```
✅ 环境预检查通过
   Python 环境: /root/paddlejob/env/paddle
   Paddle 版本: 3.0.0
   XPU 设备: 已检测到 8 个设备
   配置文件: 格式正确，必需字段齐全
   端口状态: 无冲突
   磁盘空间: 充足 (50GB 可用)
```

**检查失败输出**：
```
❌ 环境预检查失败
   失败项:
   - Python 环境不存在: /root/paddlejob/env/paddle
   建议: 请检查 python_env_path 是否正确
```

---

## 步骤1: 启动训练脚本

**执行**：
```bash
cd {launch_script所在目录}
bash {launch_script} &
```

**获取进程信息**：
```bash
echo $!  # 获取后台进程 PID
```

**验证启动**：
- 检查 PID 是否存在
- 等待 5 秒确认进程未立即退出

---

## 步骤2: 监控训练状态

### 2.1 确定日志文件路径

**如果用户提供了 `log_file`**：
- 直接使用该路径

**如果未提供，使用 `output_dir` 构建默认路径**：
```bash
# 使用传入的 output_dir 构建日志路径
log_file="{output_dir}/paddleformers_dist_log/workerlog.0"
echo "📄 日志文件路径: $log_file"
```

**备选方案（从 config_file 读取）**：
```bash
# 如果 output_dir 未传入，尝试从 YAML 读取
if [ -z "{output_dir}" ]; then
    output_dir=$(python -c "import yaml; print(yaml.safe_load(open('{config_file}'))['output_dir'])" 2>/dev/null)
    if [ -z "$output_dir" ]; then
        output_dir="/root/paddlejob/tmp/output"
        echo "⚠️  未能从配置读取 output_dir，使用默认值: $output_dir"
    fi
    log_file="${output_dir}/paddleformers_dist_log/workerlog.0"
fi
```

### 2.2 实时监控循环

**执行逻辑**：
```bash
# 循环监控，直到满足以下条件之一：
# 1. 检测到 loss 输出 → 成功
# 2. 检测到错误 → 进入错误处理
# 3. 进程阻塞 → 进入错误处理
# 4. 超时 → 失败

timeout={timeout}
stuck_timeout={stuck_timeout}
pseudo_active_timeout={pseudo_active_timeout}
start_time=$(date +%s)
last_log_size=0
no_change_count=0
stuck_check_start=0
last_progress_time=""
warning_repeat_count=0

while true; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))

    # ========== 1. 检查进程是否存在 ==========
    if ! ps -p ${TRAIN_PID} > /dev/null 2>&1; then
        echo "❌ 训练进程已退出 (PID: ${TRAIN_PID})"
        exit 1
    fi

    # ========== 2. 检查进程状态（僵尸进程检测）==========
    proc_state=$(cat /proc/${TRAIN_PID}/stat 2>/dev/null | awk '{print $3}')
    if [ "$proc_state" = "Z" ]; then
        echo "❌ 进程已成为僵尸进程 (Zombie)，训练卡死"
        exit 1
    fi

    # ========== 3. 检查日志文件存在性 ==========
    if [ ! -f "${log_file}" ]; then
        echo "⏳ 等待日志文件创建... (${elapsed}s)"
        # 如果进程存在但 30 秒后日志仍未创建，可能是启动失败
        if [ ${elapsed} -gt 30 ] && [ ${no_change_count} -eq 0 ]; then
            echo "⚠️  进程存在但日志文件长时间未创建，可能启动异常"
            no_change_count=$((no_change_count + 1))
        fi
        sleep 5
        continue
    fi

    # ========== 4. 检查日志变化（文件大小对比）==========
    current_log_size=$(stat -c %s "${log_file}" 2>/dev/null || echo 0)

    if [ ${current_log_size} -eq ${last_log_size} ]; then
        # 日志无变化
        no_change_count=$((no_change_count + 1))

        # 如果超过 stuck_timeout 时间无变化，判定为阻塞
        if [ ${no_change_count} -gt $((stuck_timeout / 5)) ]; then
            echo "❌ 检测到进程阻塞：日志 ${stuck_timeout} 秒无变化"
            echo "   进程 PID: ${TRAIN_PID}, 状态: ${proc_state}"
            echo "   最后日志大小: ${current_log_size} 字节"
            exit 1
        fi

        # 超过 3 个周期无变化，提示用户
        if [ ${no_change_count} -eq 3 ]; then
            echo "⏳ 日志 15 秒无更新，可能正在初始化或已阻塞..."
        fi
    else
        # 日志有变化，重置计数器
        if [ ${no_change_count} -ge 3 ]; then
            echo "✅ 日志恢复更新 (+$((current_log_size - last_log_size)) 字节)"
        fi
        no_change_count=0
        last_log_size=${current_log_size}
    fi

    # ========== 5. 读取日志内容并检测状态 ==========
    log_content=$(tail -n 100 "${log_file}" 2>/dev/null)

    # 检查是否检测到 loss
    if echo "$log_content" | grep -E "loss:\s*[0-9]+\.?[0-9]*|train_loss:\s*[0-9]+\.?[0-9]*" > /dev/null; then
        echo "✅ 检测到 loss 输出，训练正常启动"
        exit 0
    fi

    # 检查是否检测到错误（ERROR 级别）
    if echo "$log_content" | grep -Ei "error|exception|fatal|failed" > /dev/null; then
        echo "❌ 检测到错误信号"
        # 进入错误处理流程
        break
    fi

    # 检查是否初始化中
    if echo "$log_content" | grep -Ei "loading|initializing|preparing" > /dev/null; then
        echo "⏳ [${elapsed}s] 初始化中..."
    fi

    # ========== 5a. 伪活跃检测（PSEUDO_ACTIVE）==========
    # 若日志在增长，检查是否有有效训练进度标志
    if [ ${current_log_size} -ne ${last_log_size} ]; then
        # 检查进度标志
        if echo "$log_content" | grep -Ei "global_step|step.*[0-9]+.*loss|train_loss" > /dev/null; then
            # 有有效进度，记录时间
            last_progress_time=${current_time}
            warning_repeat_count=0
        else
            # 日志在增长但无有效进度，检查 WARNING 重复模式
            # 统计最近 100 行中 "preprocess data error" 出现次数
            preprocess_errors=$(echo "$log_content" | grep -c "preprocess data error" 2>/dev/null || echo 0)
            if [ ${preprocess_errors} -gt 10 ]; then
                warning_repeat_count=$((warning_repeat_count + preprocess_errors))
            fi

            # 检查是否超过伪活跃超时时间
            if [ -n "${last_progress_time}" ]; then
                time_since_progress=$((current_time - last_progress_time))
                if [ ${time_since_progress} -gt ${pseudo_active_timeout} ]; then
                    echo "❌ 检测到伪活跃状态：日志持续增长但 ${pseudo_active_timeout} 秒无有效训练进度"
                    echo "   可能原因：数据格式不兼容、模板不匹配、预处理死循环"
                    # 进入错误处理流程（类型：pseudo_active / dataset_incompatibility）
                    break
                fi
            fi

            # 检查是否出现大量重复 WARNING（如 1000+ 条相同预处理错误）
            if [ "${warning_repeat_count}" -gt 1000 ]; then
                echo "❌ 检测到大量重复 WARNING：数据预处理失败 ${warning_repeat_count}+ 次"
                echo "   可能原因：数据集格式与模型/模板不兼容"
                # 进入错误处理流程（类型：dataset_incompatibility）
                break
            fi
        fi
    fi

    # ========== 6. 检查超时 ==========
    if [ ${elapsed} -gt ${timeout} ]; then
        echo "❌ 监控超时 (${timeout}秒)，训练启动失败"
        echo "   可能原因: 初始化时间过长、进程阻塞、配置问题"
        exit 1
    fi

    sleep 5
done
```

### 2.3 状态判定规则

**状态1: RUNNING（训练成功）- 优先级最高**
- **判定条件**：检测到 loss 输出
- **检测模式**：
  - `loss:\s*\d+\.?\d*`
  - `train_loss:\s*\d+\.?\d*`
  - `step:\s*\d+.*loss`
- **结论**：训练已成功启动，退出监控

**状态2: ERROR（训练错误）**
- **判定条件**：检测到错误信号
- **错误类型识别**：

| 错误模式 | 错误类型 | 是否可修复 |
|---------|---------|-----------|
| `KeyError.*yaml\|YAML.*error\|config.*invalid\|missing.*parameter` | `yaml_error` | ✅ 是（自动修复） |
| `template.*not found\|template.*not supported\|unknown template` | `template_error` | ✅ 是（自动修复） |
| `out_of_memory\|OOM\|allocate memory failed` | `out_of_memory` | ❌ 否（需上报主 Agent 决策） |
| `RuntimeError\|Segmentation fault\|AssertionError` | `runtime_error` | ❌ 否（直接失败） |
| `BKCL.*timeout\|NCCL.*timeout\|communication error` | `communication_error` | ❌ 否（直接失败） |
| `Operator.*not supported\|kernel.*not found` | `operator_error` | ❌ 否（需上报主 Agent） |
| `preprocess data error.*Ignore example\|empty messages\|dataset.*format.*invalid\|tokenizer.*error.*data` | `dataset_incompatibility` | ❌ 否（**必须上报主 Agent**） |
| `model architecture.*not supported\|unsupported model type` | `model_unsupported` | ❌ 否（**必须上报主 Agent**） |
| **进程成为僵尸进程** | `process_zombie` | ❌ 否（直接失败） |
| **日志长时间无变化** | `process_stalled` | ❌ 否（直接失败） |
| **进程意外退出** | `process_died` | ❌ 否（直接失败） |
| **日志增长但无有效训练进度** | `pseudo_active` | ❌ 否（**必须上报主 Agent**） |

**状态3: INITIALIZING（初始化中）**
- **判定条件**：检测到初始化日志但无 loss
- **检测模式**：`Loading checkpoint\|Initializing model\|Preparing data`
- **结论**：继续监控

**状态4: STALLED（进程阻塞）**
- **判定条件**：
  - 进程 PID 存在但日志文件超过 `stuck_timeout`（默认 60 秒）无变化
  - 或进程成为僵尸进程（Zombie）
- **检测机制**：
  - 每 5 秒检查一次日志文件大小
  - 对比 `last_log_size` 和 `current_log_size`
  - 检查 `/proc/{PID}/stat` 中的进程状态
- **结论**：进程可能已卡死，训练启动失败
- **处理**：输出诊断信息后退出

**状态5: PSEUDO_ACTIVE（伪活跃/假死）**
- **判定条件**：
  - 进程 PID 存在且日志文件持续增长（排除 STALLED）
  - 但超过 `pseudo_active_timeout`（默认 120 秒）未检测到任何有效训练进度标志
  - 有效训练进度标志包括：`loss` 输出、`global_step`、`train_loss`、`step` 等
  - 或检测到 1000+ 条相同 WARNING（如数据预处理失败）
- **典型场景**：
  - 数据格式不兼容导致预处理循环（如 15,000+ 条 "preprocess data error"）
  - 模板与数据类型不匹配导致迭代器反复重试
  - 日志持续增长但均为重复初始化信息（如 "Not using packing mode"）
- **结论**：训练陷入死循环，无法自行恢复
- **处理**：**必须上报主 Agent**，SubAgent 禁止自行修复

**状态6: TIMEOUT（超时）**
- **判定条件**：超过 `timeout` 时间未检测到 loss
- **结论**：训练启动失败

**状态判定优先级**：
```
RUNNING (loss 检测) > ERROR (错误检测) > STALLED (阻塞检测) > PSEUDO_ACTIVE (伪活跃检测) > TIMEOUT (超时检测)
```

---

## 步骤3: 错误处理

### 3.1 YAML 配置错误处理

**识别 YAML 错误**：
```bash
# 错误模式匹配
grep -i "keyerror\|yaml.*error\|config.*invalid\|missing.*parameter\|expected.*found" {log_file}
```

**常见 YAML 错误类型**：

| 错误信息 | 修复方法 |
|---------|---------|
| `KeyError: 'device'` | 添加 `device: xpu` |
| `KeyError: 'stage'` | 添加 `stage: sft`（或 pretrain）|
| `missing required parameter: xxx` | 根据参考值添加缺失参数 |
| `invalid value for xxx` | 修正为合法值 |

**修复流程**：
1. **备份原配置**：
   ```bash
   cp {config_file} {config_file}.backup.$(date +%Y%m%d_%H%M%S)
   ```

2. **同一问题推进检测（新增）**：
   - **提取错误签名**：`error_signature = "{error_type}:{involved_field}"`
     - 例如：`yaml_error:device`（缺少 device 字段）、`yaml_error:recompute_num_layers`（recompute_num_layers 值不合法）
     - 若无法提取具体字段，使用错误信息前 50 个字符作为签名
   - **检查 fix_history**：
     - 统计 fix_history 中相同 error_signature 的连续出现次数
     - **若相同 signature 已连续出现 ≥3 次**：
       - **判定是否有推进**：
         - ✅ 有推进：修复后进程运行时间比上次长 / 错误信息发生变化 / 从 YAML 错误变为其他错误类型
         - ❌ 无推进：每次修复后仍然报同样的错误，或进程仍在同一阶段崩溃
       - **若无推进**：
         - 立即停止重试，返回 escalation 上报主 Agent
         - 在 escalation 中说明：`同一 YAML 错误 ({error_signature}) 连续 3 次修复无推进，怀疑存在根本性配置冲突`
       - **若有推进**：继续修复流程，但记录本次修复
     - **若未达到 3 次**：继续修复流程
   - **记录当前修复**：将本次 `{signature, timestamp, error_message, outcome}` 追加到 fix_history

3. **分析错误并修复**：
   - 读取错误信息，确定缺失/错误的字段
   - 使用 `sed` 或编辑工具修改 YAML 文件
   - 示例：添加缺失字段
     ```bash
     echo "device: xpu" >> {config_file}
     ```

3. **重试计数**：
   - 重试次数 +1
   - 如果超过 `max_retries`，停止重试并报错

4. **资源清理**（关键步骤，避免重复失败）：
   ```bash
   echo "🧹 清理资源..."
   
   # 4.1 停止残留的训练进程
   pkill -9 -f "paddleformers-cli train" 2>/dev/null || true
   sleep 2
   
   # 4.2 清理共享内存
   for id in $(ipcs -m 2>/dev/null | awk '/0x/ {print $2}'); do
       ipcrm -m $id 2>/dev/null || true
   done
   
   # 4.3 清理旧的分布式日志（保留本次的用于调试）
   output_dir=$(python -c "import yaml; print(yaml.safe_load(open('{config_file}'))['output_dir'])")
   if [ -d "${output_dir}/paddleformers_dist_log" ]; then
       mv ${output_dir}/paddleformers_dist_log ${output_dir}/paddleformers_dist_log.backup.$(date +%Y%m%d_%H%M%S)
   fi
   
   # 4.4 清理环境变量
   unset PADDLE_ELASTIC_JOB_ID 2>/dev/null || true
   unset PADDLE_TRAINER_ENDPOINTS 2>/dev/null || true
   unset DISTRIBUTED_TRAINER_ENDPOINTS 2>/dev/null || true
   
   # 4.5 等待端口释放
   sleep 5
   
   echo "✅ 资源清理完成"
   ```

5. **重新启动**：
   - 返回步骤1，重新执行启动脚本

### 3.2 错误分类与处理

本 Skill 将训练过程中检测到的错误分为三类，每类有不同的处理策略：

#### 类型 A：SubAgent 可自动修复

- `yaml_error` - YAML 配置缺失/错误（如缺少 `device: xpu`）
- `template_error` - 模板配置错误（如模板不存在，可尝试更换模板）
- `launch_script_error` - 启动脚本参数错误

**处理方式**：自动修复 → 清理资源 → 重新启动（受 max_retries 限制）

#### 类型 B：必须上报主 Agent（escalate_to_master）

以下错误涉及需要主 Agent / 用户决策的问题，SubAgent **禁止自行修复**：

- `dataset_incompatibility` - **数据集格式与模型/模板不兼容**
  - 典型表现：大量 "preprocess data error"、"empty messages" WARNING
  - 原因：视觉数据用文本模型训练、模板与数据类型不匹配等
  - 需决策：换数据集、换模型、或改模板

- `model_unsupported` - **模型架构不被当前框架支持**
  - 典型表现："model architecture not supported"
  - 需决策：更换模型或升级框架版本

- `operator_error` - **算子不支持**
  - 典型表现："Operator not supported"、"kernel not found"
  - 需决策：可能需要框架升级或调整算子配置

- `out_of_memory` - **显存不足**
  - 典型表现：OOM、allocate memory failed
  - 需决策：调整 batch_size、启用 gradient checkpointing 等

- `pseudo_active` - **伪活跃/假死**
  - 典型表现：日志持续增长但无有效训练进度超过 120 秒
  - 原因：数据预处理死循环、迭代器卡住
  - 需决策：分析日志判断根因，可能需要更换数据集或调整配置

**处理方式**：
1. 立即停止监控并终止训练进程
2. 输出完整诊断信息
3. **返回包含 `escalation` 字段的 JSON，标记 `required: true`**
4. **不得尝试自行修复，不得重试**

#### 类型 C：直接失败（fatal_exit）

以下错误为运行时致命错误，通常无法恢复：

- `runtime_error` - 运行时错误（段错误、断言失败等）
- `communication_error` - 通信错误（BKCL/NCCL timeout）
- `process_zombie` - 进程成为僵尸进程
- `process_stalled` - 进程阻塞（日志无变化）
- `process_died` - 进程意外退出

**处理方式**：
- 立即停止监控
- 输出完整错误日志
- 返回标准失败 JSON（不含 escalation）

---

### 3.3 阻塞/卡死常见原因参考

1. **数据加载阻塞**：数据集路径错误、数据预处理卡住、DataLoader worker 死锁
2. **通信阻塞**：BKCL 初始化失败、端口冲突、网络配置问题
3. **资源死锁**：共享内存未清理、XPU 设备被占用
4. **初始化死锁**：模型加载卡住、检查点读取失败
5. **数据格式不兼容**：数据集与模型/模板不匹配导致预处理循环

---

## 输出示例

### 成功场景

```
🚀 启动 XPU 训练
   脚本: ./train_xpu.sh
   配置: ./train_xpu.yaml

📊 监控训练状态（超时: 300秒）
   日志: ./checkpoints/train_001/paddleformers_dist_log/workerlog.0

⏱️  [15s] 状态: INITIALIZING - 正在加载模型...
⏱️  [32s] 状态: INITIALIZING - 正在准备数据...
⏱️  [58s] 状态: RUNNING ✅
   检测到 Loss: 2.456, 2.234, 2.012

✅ 训练启动成功！
   启动耗时: 58秒
   进程 PID: 12345
   监控命令: tail -f ./checkpoints/train_001/paddleformers_dist_log/workerlog.0
```

### YAML 错误修复场景

```
✅ 环境预检查通过
   Python 环境: /root/paddlejob/env/paddle
   Paddle 版本: 3.0.0
   XPU 设备: 已检测到 8 个设备
   配置文件: 格式正确，必需字段齐全

🚀 启动 XPU 训练
   脚本: ./train_xpu.sh
   配置: ./train_xpu.yaml

📊 监控训练状态（超时: 300秒）

⏱️  [12s] 状态: INITIALIZING
⏱️  [25s] 状态: ERROR ❌
   错误类型: yaml_error
   错误信息: KeyError: 'device'

⚠️  检测到 YAML 配置错误，尝试修复...
   备份: ./train_xpu.yaml.backup.20250115_143022
   修复: 添加 device: xpu

🧹 清理资源...
   停止残留进程: 完成
   清理共享内存: 完成 (清理 5 个残留段)
   备份旧日志: 完成
   清理环境变量: 完成
   等待端口释放: 完成
   ✅ 资源清理完成

🔄 重新启动训练（第1/3次重试）...

📊 监控训练状态...

⏱️  [45s] 状态: RUNNING ✅
   检测到 Loss: 2.456

✅ 训练启动成功！
   总耗时: 70秒（含修复重试）
   修复次数: 1次
```

### 进程阻塞/卡死场景

```
🚀 启动 XPU 训练
   脚本: ./train_xpu.sh
   配置: ./train_xpu.yaml

📊 监控训练状态（超时: 300秒，阻塞检测: 60秒）

⏱️  [15s] 状态: INITIALIZING - 正在加载模型...
⏱️  [30s] 状态: INITIALIZING
⏳ 日志 15 秒无更新，可能正在初始化或已阻塞...
⏱️  [45s] 状态: INITIALIZING
⏳ 日志 30 秒无更新...
⏱️  [60s] 状态: INITIALIZING
⏳ 日志 45 秒无更新...
⏱️  [75s] 状态: STALLED ❌

❌ 检测到进程阻塞：日志 60 秒无变化
   进程 PID: 12345, 状态: S (Sleeping)
   最后日志大小: 2048 字节
   可能原因:
   1. 数据加载卡住（检查数据集路径和格式）
   2. BKCL 通信初始化失败（检查端口和网络）
   3. 模型加载死锁（检查模型文件完整性）
   4. XPU 设备被占用（检查其他训练进程）

诊断命令:
   查看进程: ps -ef | grep 12345
   查看日志: tail -50 ./checkpoints/train_001/paddleformers_dist_log/workerlog.0
   数据检查: ls -lh {dataset_path}
```

### 僵尸进程场景

```
🚀 启动 XPU 训练
   脚本: ./train_xpu.sh
   配置: ./train_xpu.yaml

📊 监控训练状态（超时: 300秒）

⏱️  [10s] 状态: INITIALIZING
⏱️  [20s] 状态: INITIALIZING

❌ 进程已成为僵尸进程 (Zombie)，训练卡死
   进程 PID: 12345
   可能原因: 父进程异常退出、资源死锁、BKCL 初始化失败
   建议: 检查共享内存清理、重启训练环境
```

### 伪活跃/数据不兼容场景（必须上报主 Agent）

```
🚀 启动 XPU 训练
   脚本: ./train_xpu.sh
   配置: ./train_xpu.yaml

📊 监控训练状态（超时: 300秒，伪活跃检测: 120秒）

⏱️  [15s] 状态: INITIALIZING - 正在加载模型...
⏱️  [30s] 状态: INITIALIZING - 正在准备数据...
⏱️  [60s] 状态: INITIALIZING
⏳ 日志持续增长但无 loss 输出，可能陷入死循环...
⏱️  [90s] 状态: INITIALIZING
⏳ 检测到 500+ 条重复 WARNING: "preprocess data error"
⏱️  [120s] 状态: PSEUDO_ACTIVE ❌

❌ 检测到伪活跃状态：日志持续增长但 120 秒无有效训练进度
   错误类型: dataset_incompatibility
   检测到模式:
   - preprocess data error: Ignore example with empty messages (15000+ 次)
   - Not using packing mode for data iteration (400000+ 次重复)
   - 无 global_step / loss 输出

⚠️  此问题涉及数据集格式与模型/模板兼容性，SubAgent 无法自行修复
   建议上报主 Agent 进行决策:
   1. 更换为兼容的纯文本数据集
   2. 或改用支持当前数据格式的模型
   3. 或调整 template 配置以匹配数据类型

📤 返回 escalation 信号（上报主 Agent）
```

---

## 返回格式规范

### 标准失败返回（含 escalation 字段）

```json
{
  "模型运行状态": "Fail",
  "训练详情": {
    "launch_script": "./train_xpu.sh",
    "config_file": "./train_xpu.yaml",
    "output_dir": "./checkpoints",
    "retry_count": 0,
    "yaml_fixed": false,
    "elapsed_seconds": 120
  },
  "failure_summary": "训练陷入伪活跃状态：日志持续增长但 120 秒内未检测到 loss/global_step。检测到 15000+ 条数据预处理 WARNING，判断为数据集格式与模型不兼容。",

  "fix_history": [
    {
      "signature": "yaml_error:device",
      "timestamp": "2026-05-12T10:00:00",
      "error_message": "KeyError: 'device'",
      "outcome": "fixed"
    },
    {
      "signature": "yaml_error:recompute_num_layers",
      "timestamp": "2026-05-12T10:05:00",
      "error_message": "recompute_num_layers(11) > chunk_size(3)",
      "outcome": "same_error_after_fix"
    }
  ],

  "escalation": {
    "required": true,
    "escalation_reason": "dataset_incompatibility",
    "subagent_can_fix": false,
    "detected_patterns": [
      "preprocess data error: Ignore example with empty messages (15000+ occurrences)",
      "Not using packing mode for data iteration (repeated 400000+ times)",
      "No loss / global_step output after 120 seconds"
    ],
    "root_cause_analysis": "数据集格式（视觉 grounding 数据，包含 <image>、<bbox> 等特殊 token 和 images/objects 字段）与纯文本模型及模板（qwen3）不兼容，导致数据预处理反复失败并陷入循环。",
    "suggested_actions": [
      "更换为纯文本数据集（如 gsm8k、alpaca 等标准 SFT 数据）",
      "或改用支持视觉的模型（如 Qwen2-VL）并配置对应视觉 processor",
      "或修改 template 为 qwen3_vl 并确认模型支持多模态输入"
    ],
    "requires_human_decision": true,
    "affected_files": {
      "dataset": "./datasets/train.jsonl",
      "config": "./train_xpu.yaml"
    }
  }
}
```

### escalation 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `required` | bool | 是否必须上报主 Agent |
| `escalation_reason` | string | 上报原因分类：`dataset_incompatibility` / `model_unsupported` / `operator_error` / `out_of_memory` / `pseudo_active` |
| `subagent_can_fix` | bool | SubAgent 是否能自行修复 |
| `detected_patterns` | array | 检测到的具体异常模式列表 |
| `root_cause_analysis` | string | 根因分析（基于日志的推理） |
| `suggested_actions` | array | 建议主 Agent 执行的操作 |
| `requires_human_decision` | bool | 是否需要人工/主 Agent 决策 |
| `affected_files` | object | 受影响的文件路径 |

---

## 注意事项

1. **必须以 loss 为成功标志**：不能仅以进程存在为成功，必须确认训练已正常开始
2. **YAML 错误自动修复**：仅处理配置缺失/错误，不处理 OOM、运行时错误等
3. **最大重试限制**：防止无限循环，默认最多重试 3 次（仅对可自动修复错误）
4. **备份原配置**：每次修复前自动备份原 YAML 文件和分布式日志
5. **超时设置**：根据模型大小合理设置超时时间（大模型初始化可能需要更长时间）
6. **环境预检查**：首次启动前会检查环境，避免浪费重试次数在环境问题上
7. **资源清理**：每次重试前会清理进程、共享内存、端口等资源，避免重复失败
8. **进程阻塞检测**：通过 `stuck_timeout` 参数设置阻塞检测时间，默认 60 秒日志无变化即判定为阻塞
9. **僵尸进程检测**：自动检测并处理僵尸进程（Zombie），避免无限等待
10. **伪活跃检测（PSEUDO_ACTIVE）**：通过 `pseudo_active_timeout` 检测日志持续增长但无有效训练进度（如数据预处理死循环），默认 120 秒
11. **上报主 Agent**：检测到 `dataset_incompatibility`、`model_unsupported`、`pseudo_active` 等不可由 SubAgent 修复的问题时，必须返回 `escalation: {required: true}`，禁止自行修复
