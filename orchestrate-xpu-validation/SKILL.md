---
name: orchestrate-xpu-validation
description: Orchestrate XPU training validation with two-phase strategy. Phase 1 performs a fast Early Gate (first N steps) to detect training anomalies immediately. Phase 2 waits for training completion and triggers full validation. Delegates actual precision computation to validate-xpu-gpu-training Skill via Skill tool invocation.
---

> **定位**：Step 5 验证阶段的调度 Skill，负责编排早期快速门控、轮询等待训练完成、触发最终完整验证。
> **核心方法**：
> 1. **Early Gate（阶段1）**：调用 validate-xpu-gpu-training (early_only)，取前 N 步做极简校验，，< 1 秒完成
> 2. **Wait Loop（阶段2）**：Skill 内部阻塞轮询训练日志，检测完成标志
> 3. **Final Validation（阶段3）**：训练完成后调用 validate-xpu-gpu-training (final_only)，执行完整精度对齐
>
> **调度器不做任何精度计算**，所有计算委托给 validate-xpu-gpu-training Skill。

---

## 输入参数

```yaml
inputs:
  # 日志路径
  xpu_log_path: " " "<XPU workerlog.0 路径>"
  gpu_log_path: " " "<GPU workerlog.0 路径，可选>"

  # 输出目录
  output_dir: " " "<验证报告输出目录>"

  # 模型信息
  model_name: " " "<模型名称>"
  model_type: " " "<模型类型>"

  # Early Gate 控制参数
  early_gate_max_steps: 10          # 最多取前几步做初步校验
  early_gate_min_steps: 3           # 最少需要几步步才触发校验
  early_gate_thresholds:
    mae: 0.2
    pearson: 0.95

  # 等待轮询参数
  wait_poll_interval: 30            # 轮询间隔（秒）
  wait_max_duration: 86400          # 最大等待时长（秒），默认 24h
  wait_stuck_timeout: 300           # 连续无新 step 视为卡住（秒）

  # Final 验证阈值（透传给 validate-xpu-gpu-training）
  thresholds:
    mae: 0.05
    rmse: 0.1
    max_diff: 0.5
    pearson: 0.99
    spearman: 0.99
    relative_error_percent: 1.0
    r2: 0.98
    convergence_std: 0.01
```

---

## 执行流程

```yaml
execution_flow:
  step_0_prepare:
    description: "创建输出目录结构"
    note: |
      调度器创建顶层目录。被调用的 validate-xpu-gpu-training Skill 会在传入的 output_dir
      下自动再创建 validate-xpu-gpu-training/{data,plots}/ 子目录，并将 report.json
      输出到 {传入的output_dir}/validate-xpu-gpu-training/report.json。
      因此实际文件路径存在一层嵌套。
    actions:
      - "mkdir -p {output_dir}/orchestrate-xpu-validation/{early_gate,final}"
    output_dirs:
      - "{output_dir}/orchestrate-xpu-validation/early_gate/"
        # 内部再嵌套: validate-xpu-gpu-training/report.json
      - "{output_dir}/orchestrate-xpu-validation/final/"
        # 内部再嵌套: validate-xpu-gpu-training/report.json

  step_1_log_state_check:
    description: "检查当前日志状态"
    actions:
      - "读取 xpu_log_path，获取当前已输出的 global_step 数（xpu_current_steps）"
      - "检查 gpu_log_path 文件是否存在且包含有效数据"
    output:
      - xpu_current_steps: "当前 XPU 日志步数"
      - gpu_available: "true | false"

  step_2_early_gate:
    description: "触发初步精度校验（Early Gate）"
    note: |
      调用 validate-xpu-gpu-training Skill，传 early_only 模式。
      由 validate skill 内部自动处理有 GPU/无 GPU 两种情况。
    skill_invocation:
      required: true
      skill_name: "validate-xpu-gpu-training"
      skill_path: "{SKILL_ROOT}/validate-xpu-gpu-training/SKILL.md"
    inputs_to_pass:
      validation_mode: "early_only"
      xpu_log_path: "{xpu_log_path}"
      gpu_log_path: "{gpu_log_path}"
      output_dir: "{output_dir}/orchestrate-xpu-validation/early_gate"
      early_gate_max_steps: "{early_gate_max_steps}"
      early_gate_min_steps: "{early_gate_min_steps}"
      early_gate_thresholds: "{early_gate_thresholds}"
      model_name: "{model_name}"
      model_type: "{model_type}"
    expected_return:
      - early_status: "Pass | Warn | Fail"
      - early_metrics: "根据 mode 不同，为 {mae, pearson} 或 {monotonicity, smoothness}"
      - mode_used: "dual_side | single_side_only"
      - steps_used: "实际使用的步数"
      - report_path: "{output_dir}/orchestrate-xpu-validation/early_gate/validate-xpu-gpu-training/report.json"
        note: "validate skill 内部会在传入的 output_dir 下再创建 validate-xpu-gpu-training/ 子目录"

  step_3_early_decision:
    description: "根据 Early Gate 结果决策"
    branching:
      - condition: "early_status == Fail"
        action: "立即终止流程，返回失败结果"
        return_immediately:
          调度状态: "Fail"
          失败阶段: "early_gate"
          失败原因: "初步精度校验未通过，训练初期存在严重异常"
          early_gate_result: " "<step_2 的返回结果>"

      - condition: "early_status == Pass 或 Warn"
        action: "继续执行，进入训练等待阶段"
        next: "step_4_wait_training"

  step_4_wait_training:
    description: "轮询等待训练完成"
    note: |
      阻塞轮询，直到检测到训练完成、卡住或超时。
      此步骤由调度器 Skill 内部执行，对主 Agent 透明。
    polling_logic:
      interval: "{wait_poll_interval} 秒"
      max_wait: "{wait_max_duration} 秒"

      each_poll_actions:
        - "读取 xpu_log_path 最后一条 global_step（current_last_step）"
        - "对比上一轮记录的 last_step"

      completion_detection（满足任一即跳出）:
        log_text_patterns:
          - "train_runtime:"
          - "train_loss:"
          - "Saving model checkpoint"
          - "Training completed"
        file_signals:
          - "workerlog.0 最后修改时间距现在 > 120 秒且文件大小不再增长"
        step_signals:
          - "日志中存在 max_steps 且 current_last_step >= max_steps"

      stuck_detection:
        condition: "连续 {wait_stuck_timeout} 秒 global_step 无增长"
        action: |
          生成 {output_dir}/orchestrate-xpu-validation/alarm_stuck.json
          跳出等待，继续执行 Final（基于已有数据）

      timeout_detection:
        condition: "总等待时间超过 {wait_max_duration}"
        action: |
          生成 {output_dir}/orchestrate-xpu-validation/alarm_timeout.json
          跳出等待，继续执行 Final（基于已有数据）

    branching:
      - condition: "检测到训练完成"
        next: "step_5_final_validation"
        note: "正常完成，基于完整数据执行 Final"
      - condition: "检测到卡住"
        next: "step_5_final_validation"
        note: "训练可能卡住，基于当前已有数据执行 Final，报告中标注意外终止"
      - condition: "检测到超时"
        next: "step_5_final_validation"
        note: "等待超时，基于当前已有数据执行 Final"

  step_5_final_validation:
    description: "训练完成后触发最终精度校验"
    note: |
      调用 validate-xpu-gpu-training Skill，传 final_only 模式。
      使用完整日志执行全部验证流程。
    skill_invocation:
      required: true
      skill_name: "validate-xpu-gpu-training"
      skill_path: "{SKILL_ROOT}/validate-xpu-gpu-training/SKILL.md"
    inputs_to_pass:
      validation_mode: "final_only"
      xpu_log_path: "{xpu_log_path}"
      gpu_log_path: "{gpu_log_path}"
      output_dir: "{output_dir}/orchestrate-xpu-validation/final"
      thresholds: "{thresholds}"
      model_name: "{model_name}"
      model_type: "{model_type}"
    expected_return:
      - validation_status: "Pass | Warn | Fail | single_side_only"
      - validation_tier: "loss_quick_pass | full_analysis"
      - report_path: "{output_dir}/orchestrate-xpu-validation/final/validate-xpu-gpu-training/report.json"
        note: "validate skill 内部会在传入的 output_dir 下再创建 validate-xpu-gpu-training/ 子目录"

  step_6_compose_result:
    description: "汇总 Early Gate + Wait + Final 三阶段结果"
    actions:
      - "合并各阶段输出"
      - "输出统一 JSON 报告"
```

---

## 返回格式

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
      "report_path": "/output/orchestrate-xpu-validation/early_gate/validate-xpu-gpu-training/report.json"
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
      "report_path": "/output/orchestrate-xpu-validation/final/validate-xpu-gpu-training/report.json"
    }
  },
  "最终结果": {
    "validation_status": "Pass | Warn | Fail | single_side_only",
    "final_report_path": "/output/orchestrate-xpu-validation/final/validate-xpu-gpu-training/report.json",
    "plots_dir": "/output/orchestrate-xpu-validation/final/validate-xpu-gpu-training/plots/"
  },
  "failure_summary": "..."
}
```

---

## 与 validate-xpu-gpu-training 的联动

```yaml
context_integration:
  path_architecture_note: |
    validate-xpu-gpu-training 会在传入的 output_dir 下自动再创建
    validate-xpu-gpu-training/{data,plots}/ 子目录。
    因此：
      - Early Gate 实际 report 路径 = {output_dir}/early_gate/validate-xpu-gpu-training/report.json
      - Final 实际 report 路径 = {output_dir}/final/validate-xpu-gpu-training/report.json

  early_gate_call:
    callee: "validate-xpu-gpu-training"
    mode: "early_only"
    purpose: "基于前 N 步做极简精度校验"
    inputs_required:
      - validation_mode: "early_only"
      - xpu_log_path
      - gpu_log_path
      - output_dir: "{output_dir}/orchestrate-xpu-validation/early_gate"
      - early_gate_max_steps
      - early_gate_min_steps
      - early_gate_thresholds
      - model_name
      - model_type

  final_validation_call:
    callee: "validate-xpu-gpu-training"
    mode: "final_only"
    purpose: "训练完成后执行完整精度对齐"
    inputs_required:
      - validation_mode: "final_only"
      - xpu_log_path
      - gpu_log_path
      - output_dir: "{output_dir}/orchestrate-xpu-validation/final"
      - thresholds
      - model_name
      - model_type
```

---

## 异常处理

### Early Gate Fail

> 立即终止流程，不进入 Wait Loop，不等训练完成。
> 返回 Early Gate 结果和失败原因。

### Wait Loop 卡住

> 记录 alarm，基于当前已有数据继续执行 Final Validation。
> Final 报告中标注训练可能未正常完成。

### Wait Loop 超时

> 记录 alarm，基于当前已有数据继续执行 Final Validation。
> Final 报告中标注等待超时。

### validate-xpu-gpu-training Skill 调用失败

> 无论 Early Gate 还是 Final 调用失败，调度器立即终止并返回失败。
> 包含被调用 Skill 的失败原因。

---

## 被调用约定

```yaml
inputs_from_caller:
  xpu_log_path: " "<路径>"
  gpu_log_path: " "<路径，可选>"
  output_dir: " "<路径>"
  model_name: " "<名称>"
  model_type: " "<类型>"
  thresholds: {}  # 可选

expected_return:
  调度状态: "Success | Fail"
  阶段执行记录:
    early_gate: { executed, status, mode, steps_used, metrics, report_path }
    wait_training: { executed, waited_seconds, completed_by, stuck_detected, timeout_detected }
    final_validation: { executed, status, validation_tier, report_path }
  最终结果:
    validation_status: "Pass | Warn | Fail | single_side_only"
    final_report_path: "{output_dir}/orchestrate-xpu-validation/final/validate-xpu-gpu-training/report.json"
    plots_dir: "{output_dir}/orchestrate-xpu-validation/final/validate-xpu-gpu-training/plots/"
  failure_summary: " <失败原因>"
```
