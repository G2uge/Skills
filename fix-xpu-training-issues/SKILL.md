---
name: fix-xpu-training-issues
description: XPU 训练问题修复框架。接收错误上下文，进行问题分类并调用对应修复 Skill，支持多轮迭代修复。
---

> **设计原则**：
> - 本 Skill 仅作为**问题路由框架**，不硬编码具体分类规则
> - 问题分类和修复策略由外部 Skill 动态决定
> - 保持最大灵活性，便于后续扩展

---

## 输入参数

```yaml
inputs:
  # 基础信息
  step3_result: " "<Step 3 返回结果>"
  step3_status: " "<Success|Fail>"
  error_message: " "<完整报错信息>"
  error_source: " "<错误来源标识>"
  
  # 文件路径
  yaml_path: " "<YAML 配置文件路径>"
  launch_script_path: " "<启动脚本路径>"
  log_path: " "<训练日志路径>"
  output_dir: " "<输出目录>"
  
  # 模型信息
  model_name: " "<模型名称>"
  model_type: " "<模型类型>"
  
  # 可选
  gpu_reference_result: " "<GPU 参考结果>"
  max_fix_attempts: 3
  previous_fixes: []
  
  # 特殊要求
  special_requirements: {}
```

---

## 执行流程框架

```yaml
execution_flow:
  step_0_early_exit_check:
    description: "Escalation 信号判断"
    action: "检查 step3_result 是否包含 escalation 上报信号，判断是否需要立即跳过"
    logic: |
      IF step3_result.escalation.required == true AND step3_result.escalation.subagent_can_fix == false:
        -> 检查 escalation_reason 是否属于"无需尝试修复"类别：
           - dataset_incompatibility: 数据集与模型/模板不匹配，需人工决策 → 立即跳转到 step_6_report，final_status = "manual_required"
           - pseudo_active: 训练长时间无进展，原因不明 → 立即跳转到 step_6_report，final_status = "manual_required"
           - model_unsupported: 模型架构不被支持 → 立即跳转到 step_6_report，final_status = "manual_required"
           - source_modification_required: 需修改框架源码 → 进入 step_1_classify，由修复 Skill 判断是否能通过 YAML/脚本绕过
           - operator_error: 算子不支持 → 进入 step_1_classify，由修复 Skill 判断是否能实现回退或绕过
           - 其他: 默认进入 step_1_classify，让修复 Skill 自行判断
    note: |
      Step 3 的 subagent_can_fix: false 仅表示训练执行 SubAgent 无法修复，不代表 fix-xpu-training-issues
      及其子修复 Skill 无法修复。必须先让修复 Skill 尝试判断，除非问题明确需要人工决策。

  step_1_classify:
    description: "问题分类（轻量级，仅提取特征）"
    action: "分析 error_message，提取问题特征标签"
    output:
      issue_features: [""<特征标签列表>"]  # 如: ["yaml", "device", "missing_key"]
      issue_category: " "<初步类别>"         # 可选，不强制定义
      suggested_skill: " "<建议调用的修复 Skill 名称>"
    note: "分类逻辑保持轻量，具体判断交给修复 Skill"
  
  step_2_route:
    description: "路由到修复 Skill"
    logic: |
      根据 suggested_skill 或 issue_features 确定目标 Skill，**优先检查文件实际存在性**：

      1. 若 suggested_skill 已明确指定，优先使用：
         - 若 check_skill_exists(suggested_skill) == true，则调用
         - 若不存在，继续按 issue_features 匹配

      2. 按 issue_features 匹配（按优先级排序）：
         - operator / cuda_only / fused_kernel / NotImplementedError
           → 检查 fix-operator-issues 是否存在
           → 若存在：调用 fix-operator-issues（路径: {SKILL_ROOT}/fix-operator-issues/SKILL.md）
           → 若不存在：尝试 fix-generic-issues
         - cuda_op_import_error / fused_op_unavailable / ImportError
           → 检查 fix-cuda-op-import-issues 是否存在
           → 若存在：调用 fix-cuda-op-import-issues（路径: {SKILL_ROOT}/fix-cuda-op-import-issues/SKILL.md）
           → 若不存在：尝试 fix-generic-issues
         - yaml / config
           → 检查 fix-config-issues 是否存在
           → 若存在：调用 fix-config-issues
           → 若不存在：尝试 fix-generic-issues
         - accuracy / nan / inf
           → 检查 fix-accuracy-issues 是否存在
           → 若存在：调用 fix-accuracy-issues
           → 若不存在：尝试 fix-generic-issues

      3. 兜底：
         - 检查 fix-generic-issues 是否存在
         - 若存在：调用 fix-generic-issues
         - 若不存在：标记为 manual_required

    target_skill_path: "{SKILL_ROOT}/<fix-skill-name>/SKILL.md"
  
  step_3_invoke_fix:
    description: "调用修复 Skill 执行修复"
    params_passed:
      - error_message
      - error_source
      - yaml_path, launch_script_path, log_path
      - model_name, model_type
      - issue_features  # 将分类特征传递给修复 Skill
    return_from_skill:
      - fix_status: "success | failed | manual_required"
      - fix_description: " "<修复描述>"
      - modified_files: [""<文件列表>"]
      - new_error: " "<新产生的错误（如有）>"
  
  step_4_verify:
    description: "验证修复效果（运行训练检查是否输出 loss）"
    action: "调用 run-xpu-training-with-monitor skill 进行训练验证"
    skill_invocation:
      skill_name: "run-xpu-training-with-monitor"
      skill_path: "{SKILL_ROOT}/run-xpu-training-with-monitor/SKILL.md"
      inputs:
        launch_script: " "<启动脚本路径>"
        config_file: " "<修复后的 YAML 路径>"
        python_env_path: " "<虚拟环境路径>"
        output_dir: " "<输出目录>"
        timeout: 300
        max_retries: 1
    verify_result: "loss_output | failed"
    note: "以是否输出 loss 作为修复成功的唯一标准"
  
  step_5_iterate:
    description: "判断是否继续迭代"
    logic: |
      IF verify_result == "loss_output":
        -> 跳转到 step_6_report，标记成功
      ELSE IF fix_status == "manual_required":
        -> 跳转到 step_6_report，标记 manual_required
      ELSE IF current_attempt >= max_fix_attempts:
        -> 跳转到 step_6_report，标记 failed
      ELSE:
        -> 更新 error_message 为新错误，current_attempt + 1，回到 step_1_classify
  
  step_6_report:
    description: "汇总结果并返回"
    output: "统一格式的修复结果（见下方返回格式）"
```

---

## 修复 Skill 调用约定

本 Skill 会调用外部修复 Skill，约定如下：

### 被调用的修复 Skill 需实现

```yaml
# 被调用 Skill 的输入（由本 Skill 传递）
inputs_from_caller:
  error_message: " "<原始错误>"
  error_source: " "<错误来源>"
  issue_features: [""<分类特征>"]  # 本 Skill 提取的特征
  yaml_path: " "<路径>"
  launch_script_path: " "<路径>"
  log_path: " "<路径>"
  model_name: " "<名称>"
  model_type: " "<类型>"

# 被调用 Skill 的返回
expected_return:
  fix_status: "success | failed | manual_required"
  fix_description: " "<描述>"
  modified_files: [""<列表>"]
  new_error: " "<新错误>"  # 用于下一轮迭代
  can_continue: " "<是否可继续尝试>"
```

### 可选的修复 Skill 列表

| Skill 名称 | 用途 | 状态 | 路由触发条件 |
|-----------|------|------|-------------|
| `fix-config-issues` | 处理配置/YAML 问题 | 待实现 | `issue_features` 包含 `yaml` / `config` |
| `fix-operator-issues` | 处理算子兼容性问题（CUDA-only fused op 在 XPU 上不可用） | **已就绪** | `issue_features` 包含 `operator` / `cuda_only` / `fused_kernel`，或 `error_message` 匹配 `NotImplementedError: .* is not implemented for non-CUDA backends` |
| `fix-accuracy-issues` | 处理精度/收敛问题 | 待实现 | `issue_features` 包含 `accuracy` / `nan` / `inf` |
| `fix-cuda-op-import-issues` | 修复 CUDA fused 算子硬编码导入导致非 CUDA 设备失败 | **已就绪** | `issue_features` 包含 `cuda_op_import_error` / `fused_op_unavailable`，或 `error_message` 匹配 `from .* import fused_*` / `ImportError.*fused` |
| `fix-generic-issues` | 通用问题处理（兜底） | 待实现 | 以上均不匹配时 |

### 修复 Skill 存在性检查

路由前必须先检查 Skill 文件是否实际存在：
```python
def check_skill_exists(skill_name, skill_root):
    """检查 Skill 文件是否实际存在"""
    skill_path = f"{skill_root}/{skill_name}/SKILL.md"
    return os.path.exists(skill_path)
```

- 若目标 Skill 文件存在：正常调用
- 若目标 Skill 文件不存在：尝试下一个候选 Skill，最终兜底到 `fix-generic-issues`
- 若所有候选 Skill 都不存在：标记 `manual_required`

> **注意**：本 Skill 不强依赖特定 Skill 存在；若目标修复 Skill 不存在，则标记为 `manual_required`。
>
> `fix-cuda-op-import-issues` 的路径：`{SKILL_ROOT}/fix-cuda-op-import-issues/SKILL.md`

---

## 返回格式

```json
{
  "问题修复": "Success | Fail | NotRequired",
  "修复摘要": {
    "total_attempts": " "<尝试次数>",
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

---

## 特殊情形处理

```yaml
special_cases:
  step3_success:
    condition: "step3_status == 'Success'"
    action: "直接返回 NotRequired，不执行修复流程"

  escalation_from_step3:
    condition: "step3_result.escalation.required == true AND step3_result.escalation.subagent_can_fix == false"
    action: |
      不立即终止，先判断 escalation_reason 类型：
      1. 若 escalation_reason 属于 dataset_incompatibility / pseudo_active / model_unsupported：
         → 这些类型通常需要人工决策（换数据集/换模型/改模板）
         → 立即返回 manual_required
      2. 若 escalation_reason 属于 operator_error / source_modification_required / config_error 等：
         → 进入正常修复流程（step_1_classify → step_2_route → step_3_invoke_fix）
         → 让修复 Skill 自行判断是否能修复
         → 修复 Skill 判断无法修复时，再返回 manual_required
      3. 其他 escalation_reason：
         → 默认进入正常修复流程
    note: |
      核心原则：Step 3 的 subagent_can_fix: false 仅表示训练执行 SubAgent 无法修复，
      不代表 fix-xpu-training-issues 及其子修复 Skill 无法修复。
      修复 Skill 有独立的问题分析和修复能力，必须给它尝试的机会。

  skill_not_found:
    condition: "目标修复 Skill 不存在"
    action: "返回 manual_required，提示需要人工处理"

  unfixable_detected:
    condition: "修复 Skill 返回 manual_required"
    action: "立即终止迭代，返回上报建议"

  max_attempts:
    condition: "达到 max_fix_attempts 仍未解决"
    action: "返回 failed，汇总所有尝试历史"
```

---

## 与主 Agent 联动

```yaml
context_integration:
  # 接收主 Agent 通过 SubAgent-4 传递的完整上下文
  receive_from: "SubAgent-4 (Step 4)"
  
  # 返回给主 Agent 的信息
  return_to: "主 Agent"
  return_content:
    - 修复是否成功（以训练输出 loss 为判断标准）
    - 是否可以进入 Step 5 验证
    - 详细的修复历史（包含每轮训练的验证结果）
    - 是否需要人工介入
```

---

## 待扩展点

本 Skill 为框架性实现，以下部分后续可逐步完善：

1. **问题分类规则**：根据实际错误模式，逐步丰富 issue_features 提取逻辑
2. **修复 Skill 映射**：根据问题类型，建立更精准的路由规则
3. **验证方法**：针对不同问题类型，扩展验证策略
4. **知识积累**：记录成功/失败案例，形成修复知识库
