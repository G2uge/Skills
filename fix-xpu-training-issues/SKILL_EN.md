---
name: fix-xpu-training-issues
description: XPU training issue repair framework. Receives error context, classifies issues and invokes corresponding repair skills, supporting multi-round iterative repair.
---

> **Design Principles**:
> - This skill serves only as an **issue routing framework** without hardcoding specific classification rules
> - Issue classification and repair strategies are dynamically determined by external skills
> - Maintains maximum flexibility for future extension

---

## Input Parameters

```yaml
inputs:
  # Basic Information
  step3_result: " "<Step 3 return result>"
  step3_status: " "<Success|Fail>"
  error_message: " "<Complete error message>"
  error_source: " "<Error source identifier>"
  
  # File Paths
  yaml_path: " "<YAML configuration file path>"
  launch_script_path: " "<Launch script path>"
  log_path: " "<Training log path>"
  output_dir: " "<Output directory>"
  
  # Model Information
  model_name: " "<Model name>"
  model_type: " "<Model type>"
  
  # Optional
  gpu_reference_result: " "<GPU reference result>"
  max_fix_attempts: 3
  previous_fixes: []
  
  # Special Requirements
  special_requirements: {}
```

---

## Execution Flow Framework

```yaml
execution_flow:
  step_1_classify:
    description: "Issue classification (lightweight, feature extraction only)"
    action: "Analyze error_message and extract issue feature tags"
    output: 
      issue_features: [""<List of feature tags>"]  # e.g.: ["yaml", "device", "missing_key"]
      issue_category: " "<Preliminary category>"         # Optional, not mandatory
      suggested_skill: " "<Suggested repair skill name>"
    note: "Keep classification logic lightweight, leave specific judgment to repair skills"
  
  step_2_route:
    description: "Route to repair skill"
    logic: |
      Determine target skill based on suggested_skill or issue_features:
      - If {SKILL_ROOT}/fix-<category>-issues/SKILL.md exists, invoke it
      - If not, try invoking generic repair skill: fix-generic-issues
      - If still not available, mark as manual_required
    target_skill_path: "{SKILL_ROOT}/}/<fix-skill-name>/SKILL.md"
  
  step_3_invoke_fix:
    description: "Invoke repair skill to execute fix"
    params_passed:
      - error_message
      - error_source
      - yaml_path, launch_script_path, log_path
      - model_name, model_type
      - issue_features  # Pass classification features to repair skill
    return_from_skill:
      - fix_status: "success | failed | manual_required"
      - fix_description: " "<Fix description>"
      - modified_files: [""<File list>"]
      - new_error: " "<New error generated (if any)>"
  
  step_4_verify:
    description: "Verify repair effect"
    methods:
      - "Check syntax/format of modified_files"
      - "If applicable, launch brief training verification"
    verify_result: "passed | failed"
  
  step_5_iterate:
    description: "Determine whether to continue iteration"
    condition: "verify_result == 'failed' AND max attempts not reached"
    on_continue:
      - Update error_message to new_error
      - current_attempt + 1
      - Return to step_1_classify
    on_terminate:
      - Max attempts reached -> mark failed
      - Skill returns manual_required -> terminate immediately
  
  step_6_report:
    description: "Summarize results and return"
    output: "Unified format repair result (see return format below)"
```

---

## Repair Skill Invocation Convention

This skill will invoke external repair skills with the following conventions:

### Required Implementation for Invoked Repair Skills

```yaml
# Input to called skill (passed by this skill)
inputs_from_caller:
  error_message: " "<Original error>"
  error_source: " "<Error source>"
  issue_features: [""<Classification features>"]  # Features extracted by this skill
  yaml_path: " "<Path>"
  launch_script_path: " "<Path>"
  log_path: " "<Path>"
  model_name: " "<Name>"
  model_type: " "<Type>"

# Return from called skill
expected_return:
  fix_status: "success | failed | manual_required"
  fix_description: " "<Description>"
  modified_files: [""<List>"]
  new_error: " "<New error>"  # Used for next round iteration
  can_continue: " "<Whether can continue trying>"
```

### Optional Repair Skill List (Examples)

| Skill Name | Purpose | Status |
|-----------|---------|--------|
| `fix-config-issues` | Handle configuration/YAML issues | To be implemented |
| `fix-operator-issues` | Handle operator compatibility issues | To be implemented |
| `fix-accuracy-issues` | Handle precision/convergence issues | To be implemented |
| `fix-generic-issues` | Generic issue handling (fallback) | To be implemented |

> **Note**: The above skill list is for example only, can be dynamically extended in practice. This skill does not strictly depend on specific skills.

---

## Return Format

```json
{
  "issue_repair": "Success | Fail | NotRequired",
  "repair_summary": {
    "total_attempts": " "<Number of attempts>",
    "max_attempts": " "<Maximum attempts>",
    "final_status": "fixed | failed | manual_required",
    "issue_features": [""<Feature tags>"],
    "issue_resolved": true | false
  },
  "repair_history": [
    {
      "attempt": 1,
      "invoked_skill": " "<Invoked skill name>",
      "fix_description": " "<Fix description>",
      "verify_result": "passed | failed"
    }
  ],
  "final_result": {
    "files_modified": [""<File list>"],
    "can_retry_training": true | false
  },
  "escalation": {
    "required": true | false,
    "reason": " "<Escalation reason>",
    "suggestion": " "<Suggestion>"
  }
}
```

---

## Special Case Handling

```yaml
special_cases:
  step3_success:
    condition: "step3_status == 'Success'"
    action: "Directly return NotRequired, skip repair flow"
  
  skill_not_found:
    condition: "Target repair skill does not exist"
    action: "Return manual_required, prompt for manual handling"
  
  unfixable_detected:
    condition: "Repair skill returns manual_required"
    action: "Immediately terminate iteration, return escalation suggestion"
  
  max_attempts:
    condition: "Reached max_fix_attempts without resolution"
    action: "Return failed, summarize all attempt history"
```

---

## Integration with Master Agent

```yaml
context_integration:
  # Receive complete context passed by master agent through SubAgent-4
  receive_from: "SubAgent-4 (Step 4)"
  
  # Information returned to master agent
  return_to: "Master Agent"
  return_content:
    - Whether repair was successful
    - Whether training can be retried
    - Detailed repair history (for knowledge accumulation)
    - Whether manual intervention is required
```

---

## Extension Points

This skill is a framework implementation. The following parts can be gradually improved:

1. **Issue Classification Rules**: Gradually enrich issue_features extraction logic based on actual error patterns
2. **Repair Skill Mapping**: Establish more precise routing rules based on issue types
3. **Verification Methods**: Extend verification strategies for different issue types
4. **Knowledge Accumulation**: Record success/failure cases to form a repair knowledge base
