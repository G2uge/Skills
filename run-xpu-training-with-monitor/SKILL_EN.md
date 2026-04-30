---
name: run-xpu-training-with-monitor
description: |
  Execute XPU training launch script and monitor training status in real-time.
  Use detection of loss output as success indicator, automatically identify and repair YAML configuration errors, throw other errors directly.
keywords: xpu, train, monitor, loss, error, repair, retry, training
---

> **Execution Constraints**: The executor must strictly follow the calling order defined in this skill, and must not arbitrarily add pre-checks or skip logic. Sub-skills handle installation/update judgment internally.
>
> **⚠️ Mandatory Limitation: This skill is only allowed to modify YAML configuration files and Shell startup scripts (.sh). Modifying any other types of files (such as Python source code, model weights, data files, etc.) is strictly prohibited. When encountering errors related to other file types, you must report upward and must not attempt to fix them on your own.**

# XPU Training Execution and Monitor

This Skill guides the Agent to execute XPU training launch script, monitor training status in real-time, and automatically handle configuration errors.

## Input Parameters

| Parameter | Required | Description | Example |
|-----------|----------|-------------|---------|
| `launch_script` | Yes | Training launch script path | `./train_xpu.sh` |
| `config_file` | Yes | XPU YAML configuration file path (for repair) | `./train_xpu.yaml` |
| `output_dir` | **Yes** | Training output directory, used to locate logs | `/root/paddlejob/tmp/output` |
| `log_file` | No | Training log file path, default based on `output_dir` | `/root/paddlejob/tmp/output/paddleformers_dist_log/workerlog.0` |
| `timeout` | No | Monitoring timeout (seconds), default 300 | `300` |
| `max_retries` | No | Maximum retry attempts, default 3 | `3` |
| `stuck_timeout` | No | Process stall detection time (seconds), default 60 | `60` |
| `python_env_path` | No | Python virtual environment path for environment pre-check | `/root/paddlejob/env/paddle` |

## Execution Flow Overview

```
Input: launch_script, config_file
  │
  ▼
┌─────────────────────────────┐
│  Step 0: Environment Check  │
│  - Verify Python env        │
│  - Check XPU devices        │
│  - Check port usage         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Step 1: Launch Training    │
│  - Execute bash script      │
│  - Get training process PID │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Step 2: Monitor Status     │
│  - Loop read logs           │
│  - Detect loss/error/timeout│
└──────────────┬──────────────┘
               │
               ▼
     ┌─────────┴─────────┐
     │                   │
    LOSS                ERROR
     │                   │
     ▼                   ▼
┌──────────┐  ┌──────────────────────┐
│ Success  │  │ Parse Error Type     │
│ Exit     │  └──────────┬───────────┘
└──────────┘             │
                         ▼
              ┌──────────┴──────────┐
              │                     │
         YAML Error             Other Error
              │                     │
              ▼                     ▼
     ┌────────────────┐  ┌────────────────┐
     │ Repair YAML    │  │ Throw Error    │
     │ Cleanup        │  │ Exit           │
     │ Restart        │  │                │
     │ (Retry Count)  │  │                │
     └────────────────┘  └────────────────┘
```

---

## Step 0: Environment Pre-check

Before first training launch, perform environment checks to avoid wasting retry attempts on environment issues.

### 0.1 Python Environment Check

**Check if virtual environment exists**:
```bash
if [ ! -d "{python_env_path}" ]; then
    echo "❌ Python environment not found: {python_env_path}"
    exit 1
fi

if [ ! -f "{python_env_path}/bin/activate" ]; then
    echo "❌ Environment activation script not found: {python_env_path}/bin/activate"
    exit 1
fi
```

**Verify Paddle Installation**:
```bash
source {python_env_path}/bin/activate
python -c "import paddle; print(f'Paddle version: {paddle.__version__}')" || {
    echo "❌ Paddle not properly installed"
    exit 1
}
```

### 0.2 XPU Device Check

**Check XPU device availability**:
```bash
python -c "import paddle; paddle.device.get_available_device()" 2>/dev/null || {
    echo "⚠️  Unable to detect XPU devices, continuing attempt..."
}
```

### 0.3 Configuration File Check

**Check YAML configuration file existence**:
```bash
if [ ! -f "{config_file}" ]; then
    echo "❌ Configuration file not found: {config_file}"
    exit 1
fi
```

**Basic YAML syntax check**:
```bash
python -c "import yaml; yaml.safe_load(open('{config_file}'))" 2>/dev/null || {
    echo "❌ YAML file format error, please check configuration file"
    exit 1
}
```

**Check required fields**:
```bash
# Check if critical fields exist
python -c "
import yaml
with open('{config_file}') as f:
    config = yaml.safe_load(f)
    required = ['model_name_or_path', 'output_dir']
    missing = [f for f in required if f not in config]
    if missing:
        print(f'❌ Missing required fields: {missing}')
        exit(1)
" || exit 1
```

### 0.4 Port Check

**Check if common ports are in use**:
```bash
# Check ports 6000-6010 (BKCL commonly used port range)
for port in 6000 6001 6002 6003 6004 6005 6006 6007 6008 6009 6010; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  Port $port is in use"
    fi
done
```

### 0.5 Disk Space Check

**Check output directory disk space**:
```bash
# Get partition of output directory
output_dir=$(python -c "import yaml; print(yaml.safe_load(open('{config_file}'))['output_dir'])")
available_space=$(df -BG "$output_dir" 2>/dev/null | awk 'NR==2 {print $4}' | tr -d 'G')
if [ "$available_space" -lt 10 ]; then
    echo "⚠️  Insufficient disk space: ${available_space}GB (recommend at least 10GB)"
fi
```

### 0.6 Pre-check Output

**Check passed output**:
```
✅ Environment Pre-check Passed
   Python Environment: /root/paddlejob/env/paddle
   Paddle Version: 3.0.0
   XPU Devices: 8 devices detected
   Config File: Format correct, required fields present
   Port Status: No conflicts
   Disk Space: Sufficient (50GB available)
```

**Check failed output**:
```
❌ Environment Pre-check Failed
   Failed Items:
   - Python environment not found: /root/paddlejob/env/paddle
   Suggestion: Please check if python_env_path is correct
```

---

## Step 1: Launch Training Script

**Execute**:
```bash
cd {launch_script directory}
bash {launch_script} &
```

**Get Process Info**:
```bash
echo $!  # Get background process PID
```

**Verify Launch**:
- Check if PID exists
- Wait 5 seconds to confirm process didn't exit immediately

---

## Step 2: Monitor Training Status

### 2.1 Determine Log File Path

**If user provided `log_file`**:
- Use the path directly

**If not provided, use `output_dir` to build default path**:
```bash
# Use passed output_dir to construct log path
log_file="{output_dir}/paddleformers_dist_log/workerlog.0"
echo "📄 Log file path: $log_file"
```

**Fallback (read from config_file)**:
```bash
# If output_dir not passed, try to read from YAML
if [ -z "{output_dir}" ]; then
    output_dir=$(python -c "import yaml; print(yaml.safe_load(open('{config_file}'))['output_dir'])" 2>/dev/null)
    if [ -z "$output_dir" ]; then
        output_dir="/root/paddlejob/tmp/output"
        echo "⚠️  Could not read output_dir from config, using default: $output_dir"
    fi
    log_file="${output_dir}/paddleformers_dist_log/workerlog.0"
fi
```

### 2.2 Real-time Monitoring Loop

**Execution Logic**:
```bash
# Loop monitoring until one of following conditions is met:
# 1. Loss output detected → Success
# 2. Error detected → Enter error handling
# 3. Process stalled → Enter error handling
# 4. Timeout → Failure

timeout={timeout}
stuck_timeout={stuck_timeout}
start_time=$(date +%s)
last_log_size=0
no_change_count=0
stuck_check_start=0

while true; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))

    # ========== 1. Check if process exists ==========
    if ! ps -p ${TRAIN_PID} > /dev/null 2>&1; then
        echo "❌ Training process exited (PID: ${TRAIN_PID})"
        exit 1
    fi

    # ========== 2. Check process state (Zombie detection) ==========
    proc_state=$(cat /proc/${TRAIN_PID}/stat 2>/dev/null | awk '{print $3}')
    if [ "$proc_state" = "Z" ]; then
        echo "❌ Process has become Zombie, training deadlocked"
        exit 1
    fi

    # ========== 3. Check log file existence ==========
    if [ ! -f "${log_file}" ]; then
        echo "⏳ Waiting for log file creation... (${elapsed}s)"
        # If process exists but log not created after 30 seconds, possible startup failure
        if [ ${elapsed} -gt 30 ] && [ ${no_change_count} -eq 0 ]; then
            echo "⚠️  Process exists but log file not created for a long time, possible startup anomaly"
            no_change_count=$((no_change_count + 1))
        fi
        sleep 5
        continue
    fi

    # ========== 4. Check log changes (file size comparison) ==========
    current_log_size=$(stat -c %s "${log_file}" 2>/dev/null || echo 0)

    if [ ${current_log_size} -eq ${last_log_size} ]; then
        # No log changes
        no_change_count=$((no_change_count + 1))

        # If no changes for stuck_timeout, determine stalled
        if [ ${no_change_count} -gt $((stuck_timeout / 5)) ]; then
            echo "❌ Detected process stall: No log changes for ${stuck_timeout} seconds"
            echo "   Process PID: ${TRAIN_PID}, State: ${proc_state}"
            echo "   Last log size: ${current_log_size} bytes"
            exit 1
        fi

        # More than 3 cycles without changes, notify user
        if [ ${no_change_count} -eq 3 ]; then
            echo "⏳ No log updates for 15 seconds, may be initializing or stalled..."
        fi
    else
        # Log has changes, reset counter
        if [ ${no_change_count} -ge 3 ]; then
            echo "✅ Log resumed (+$((current_log_size - last_log_size)) bytes)"
        fi
        no_change_count=0
        last_log_size=${current_log_size}
    fi

    # ========== 5. Read log content and detect status ==========
    log_content=$(tail -n 50 "${log_file}" 2>/dev/null)

    # Check if loss detected
    if echo "$log_content" | grep -E "loss:\s*[0-9]+\.?[0-9]*|train_loss:\s*[0-9]+\.?[0-9]*" > /dev/null; then
        echo "✅ Loss output detected, training started normally"
        exit 0
    fi

    # Check if error detected
    if echo "$log_content" | grep -Ei "error|exception|fatal|failed" > /dev/null; then
        echo "❌ Error signal detected"
        # Enter error handling flow
        break
    fi

    # Check if initializing
    if echo "$log_content" | grep -Ei "loading|initializing|preparing" > /dev/null; then
        echo "⏳ [${elapsed}s] Initializing..."
    fi

    # ========== 6. Check timeout ==========
    if [ ${elapsed} -gt ${timeout} ]; then
        echo "❌ Monitoring timeout (${timeout}s), training launch failed"
        echo "   Possible causes: Long initialization time, process stalled, configuration issues"
        exit 1
    fi

    sleep 5
done
```

### 2.3 Status Determination Rules

**Status 1: RUNNING (Training Success) - Highest Priority**
- **Condition**: Loss output detected
- **Detection Patterns**:
  - `loss:\s*\d+\.?\d*`
  - `train_loss:\s*\d+\.?\d*`
  - `step:\s*\d+.*loss`
- **Conclusion**: Training successfully started, exit monitoring

**Status 2: ERROR (Training Error)**
- **Condition**: Error signal detected
- **Error Type Identification**:

| Error Pattern | Error Type | Repairable |
|--------------|------------|------------|
| `KeyError.*yaml\|YAML.*error\|config.*invalid\|missing.*parameter` | `yaml_error` | ✅ Yes |
| `out_of_memory\|OOM\|allocate memory failed` | `out_of_memory` | ❌ No (not handled by this Skill) |
| `RuntimeError\|Segmentation fault\|AssertionError` | `runtime_error` | ❌ No |
| `BKCL.*timeout\|NCCL.*timeout\|communication error` | `communication_error` | ❌ No |
| `Operator.*not supported\|kernel.*not found` | `operator_error` | ❌ No |
| **Process becomes Zombie** | `process_zombie` | ❌ No |
| **Log no changes for a long time** | `process_stalled` | ❌ No |
| **Process exited unexpectedly** | `process_died` | ❌ No |

**Status 3: INITIALIZING (Initializing)**
- **Condition**: Initialization log detected but no loss
- **Detection Patterns**: `Loading checkpoint\|Initializing model\|Preparing data`
- **Conclusion**: Continue monitoring

**Status 4: STALLED (Process Stalled)**
- **Condition**:
  - Process PID exists but log file has no changes for `stuck_timeout` (default 60 seconds)
  - Or process has become Zombie
- **Detection Mechanism**:
  - Check log file size every 5 seconds
  - Compare `last_log_size` and `current_log_size`
  - Check process state in `/proc/{PID}/stat`
- **Conclusion**: Process may be deadlocked, training launch failed
- **Action**: Exit after outputting diagnostic information

**Status 5: TIMEOUT (Timeout)**
- **Condition**: No loss detected within `timeout` period
- **Conclusion**: Training launch failed

**Status Priority**:
```
RUNNING (loss detection) > ERROR (error detection) > STALLED (stall detection) > TIMEOUT (timeout detection)
```

---

## Step 3: Error Handling

### 3.1 YAML Configuration Error Handling

**Identify YAML Errors**:
```bash
# Error pattern matching
grep -i "keyerror\|yaml.*error\|config.*invalid\|missing.*parameter\|expected.*found" {log_file}
```

**Common YAML Error Types**:

| Error Message | Repair Method |
|--------------|---------------|
| `KeyError: 'device'` | Add `device: xpu` |
| `KeyError: 'stage'` | Add `stage: sft` (or pretrain) |
| `missing required parameter: xxx` | Add missing parameter based on reference |
| `invalid value for xxx` | Correct to valid value |

**Repair Flow**:
1. **Backup Original Config**:
   ```bash
   cp {config_file} {config_file}.backup.$(date +%Y%m%d_%H%M%S)
   ```

2. **Analyze and Repair**:
   - Read error message, determine missing/incorrect field
   - Use `sed` or editing tool to modify YAML file
   - Example: Add missing field
     ```bash
     echo "device: xpu" >> {config_file}
     ```

3. **Retry Count**:
   - Increment retry count
   - If exceeds `max_retries`, stop retry and report error

4. **Resource Cleanup** (Critical step to avoid repeated failures):
   ```bash
   echo "🧹 Cleaning up resources..."
   
   # 4.1 Stop residual training processes
   pkill -9 -f "paddleformers-cli train" 2>/dev/null || true
   sleep 2
   
   # 4.2 Clean shared memory
   for id in $(ipcs -m 2>/dev/null | awk '/0x/ {print $2}'); do
       ipcrm -m $id 2>/dev/null || true
   done
   
   # 4.3 Clean old distributed logs (keep current for debugging)
   output_dir=$(python -c "import yaml; print(yaml.safe_load(open('{config_file}'))['output_dir'])")
   if [ -d "${output_dir}/paddleformers_dist_log" ]; then
       mv ${output_dir}/paddleformers_dist_log ${output_dir}/paddleformers_dist_log.backup.$(date +%Y%m%d_%H%M%S)
   fi
   
   # 4.4 Clean environment variables
   unset PADDLE_ELASTIC_JOB_ID 2>/dev/null || true
   unset PADDLE_TRAINER_ENDPOINTS 2>/dev/null || true
   unset DISTRIBUTED_TRAINER_ENDPOINTS 2>/dev/null || true
   
   # 4.5 Wait for port release
   sleep 5
   
   echo "✅ Resource cleanup completed"
   ```

5. **Restart**:
   - Return to Step 1, re-execute launch script

### 3.2 Other Error Handling

**Non-repairable Errors**:
- `runtime_error` - Runtime error
- `out_of_memory` - Out of memory (requires batch_size adjustment, not handled by this Skill)
- `communication_error` - Communication error
- `operator_error` - Operator not supported
- `process_zombie` - Process becomes Zombie (usually resource deadlock or parent process issue)
- `process_stalled` - Process stalled (possibly initialization deadlock, data loading stuck, communication waiting)
- `process_died` - Process exited unexpectedly (launch failure, missing dependency, segmentation fault)

**Common Causes of Stalls/Deadlocks**:
1. **Data loading blocked**: Dataset path error, data preprocessing stuck, DataLoader worker deadlock
2. **Communication blocked**: BKCL initialization failure, port conflict, network configuration issue
3. **Resource deadlock**: Shared memory not cleaned up, XPU device occupied
4. **Initialization deadlock**: Model loading stuck, checkpoint reading failure

**Handling Method**:
- Stop monitoring immediately
- Output complete error log
- Return error information and repair suggestions

---

## Output Examples

### Success Scenario

```
🚀 Launching XPU Training
   Script: ./train_xpu.sh
   Config: ./train_xpu.yaml

📊 Monitoring Training Status (Timeout: 300s)
   Log: ./checkpoints/train_001/paddleformers_dist_log/workerlog.0

⏱️  [15s] Status: INITIALIZING - Loading model...
⏱️  [32s] Status: INITIALIZING - Preparing data...
⏱️  [58s] Status: RUNNING ✅
   Loss Detected: 2.456, 2.234, 2.012

✅ Training Launched Successfully!
   Launch Time: 58 seconds
   Process PID: 12345
   Monitor Command: tail -f ./checkpoints/train_001/paddleformers_dist_log/workerlog.0
```

### YAML Error Repair Scenario

```
✅ Environment Pre-check Passed
   Python Environment: /root/paddlejob/env/paddle
   Paddle Version: 3.0.0
   XPU Devices: 8 devices detected
   Config File: Format correct, required fields present

🚀 Launching XPU Training
   Script: ./train_xpu.sh
   Config: ./train_xpu.yaml

📊 Monitoring Training Status (Timeout: 300s)

⏱️  [12s] Status: INITIALIZING
⏱️  [25s] Status: ERROR ❌
   Error Type: yaml_error
   Error Message: KeyError: 'device'

⚠️  YAML Configuration Error Detected, Attempting Repair...
   Backup: ./train_xpu.yaml.backup.20250115_143022
   Repair: Add device: xpu

🧹 Cleaning up resources...
   Stop residual processes: Done
   Clean shared memory: Done (cleaned 5 residual segments)
   Backup old logs: Done
   Clean environment variables: Done
   Wait for port release: Done
   ✅ Resource cleanup completed

🔄 Restarting Training (Retry 1/3)...

📊 Monitoring Training Status...

⏱️  [45s] Status: RUNNING ✅
   Loss Detected: 2.456

✅ Training Launched Successfully!
   Total Time: 70 seconds (including repair retry)
   Repair Count: 1
```

### Process Stall/Deadlock Scenario

```
🚀 Launching XPU Training
   Script: ./train_xpu.sh
   Config: ./train_xpu.yaml

📊 Monitoring Training Status (Timeout: 300s, Stall Detection: 60s)

⏱️  [15s] Status: INITIALIZING - Loading model...
⏱️  [30s] Status: INITIALIZING
⏳ No log updates for 15 seconds, may be initializing or stalled...
⏱️  [45s] Status: INITIALIZING
⏳ No log updates for 30 seconds...
⏱️  [60s] Status: INITIALIZING
⏳ No log updates for 45 seconds...
⏱️  [75s] Status: STALLED ❌

❌ Detected process stall: No log changes for 60 seconds
   Process PID: 12345, State: S (Sleeping)
   Last log size: 2048 bytes
   Possible causes:
   1. Data loading blocked (check dataset path and format)
   2. BKCL communication initialization failed (check ports and network)
   3. Model loading deadlock (check model file integrity)
   4. XPU device occupied (check other training processes)

Diagnostic commands:
   View process: ps -ef | grep 12345
   View log: tail -50 ./checkpoints/train_001/paddleformers_dist_log/workerlog.0
   Data check: ls -lh {dataset_path}
```

### Zombie Process Scenario

```
🚀 Launching XPU Training
   Script: ./train_xpu.sh
   Config: ./train_xpu.yaml

📊 Monitoring Training Status (Timeout: 300s)

⏱️  [10s] Status: INITIALIZING
⏱️  [20s] Status: INITIALIZING

❌ Process has become Zombie (Zombie), training deadlocked
   Process PID: 12345
   Possible causes: Parent process exited abnormally, resource deadlock, BKCL initialization failure
   Suggestion: Check shared memory cleanup, restart training environment
```

### Other Error Scenario

```
🚀 Launching XPU Training
   Script: ./train_xpu.sh
   Config: ./train_xpu.yaml

📊 Monitoring Training Status (Timeout: 300s)

⏱️  [10s] Status: INITIALIZING
⏱️  [20s] Status: ERROR ❌
   Error Type: runtime_error
   Error Message: RuntimeError: CUDA error: device-side assert triggered

❌ Unrepairable Error Encountered, Stopping Training
   Error Type: runtime_error
   Suggestions:
   1. Check if environment configuration is correct
   2. Verify Paddle/PaddleFormers version compatibility
   3. Check if input data is valid
   4. View full log: tail -100 ./checkpoints/train_001/train.log
```

---

## Notes

1. **Must use loss as success indicator**: Cannot use only process existence as success, must confirm training has started normally
2. **YAML Error Auto-repair**: Only handle configuration missing/errors, do not handle OOM, runtime errors, etc.
3. **Maximum Retry Limit**: Prevent infinite loops, default maximum 3 retries
4. **Backup Original Config**: Automatically backup original YAML file and distributed logs before each repair
5. **Timeout Setting**: Set reasonable timeout based on model size (large models may need longer initialization time)
6. **Environment Pre-check**: Environment is checked before first launch to avoid wasting retry attempts on environment issues
7. **Resource Cleanup**: Before each retry, processes, shared memory, ports and other resources are cleaned up to avoid repeated failures
8. **Process Stall Detection**: Set via `stuck_timeout` parameter for stall detection time, default 60 seconds of no log changes determines stall
9. **Zombie Process Detection**: Automatically detects and handles Zombie processes to avoid infinite waiting
