---
name: fix-cuda-op-import-issues
description: >
  Fix CUDA-only fused operator unconditional import errors that break
  training on non-CUDA devices (XPU / CPU).
  This Skill transforms hard-coded imports into guarded conditional imports
  and adds None-guards at call sites.
---

> **设计原则**：
> - 本 Skill 是 `fix-xpu-training-issues` 框架的**专用修复 Skill**
> - 严格遵循 `fix-xpu-training-issues` 定义的输入输出约定
> - 所有文件路径使用相对路径 `{SKILL_ROOT}`，禁止写死绝对路径
> - 不直接执行训练，仅负责源码修复；验证由调用方（`fix-xpu-training-issues` Step 4）完成

---

## 1. 适用范围

当 `error_message` 中出现以下任一特征时，由 `fix-xpu-training-issues` Step 2 路由到本 Skill：

- `from paddlefleet.ops import fused_*` 或 `from paddlenlp.ops import fused_*` 报错
- `ImportError` / `ModuleNotFoundError` 与 fused 算子相关
- `issue_features` 包含 `cuda_op_import_error`
- `issue_features` 包含 `fused_op_unavailable`

---

## 2. 输入参数

```yaml
inputs:
  # 由 fix-xpu-training-issues 传递的必填参数
  error_message: "<完整报错信息>"
  error_source: "<错误来源标识>"
  issue_features: ["<特征标签列表>"]   # 如: ["cuda_op_import_error", "fused_op_unavailable"]
  yaml_path: "<YAML 配置文件路径>"
  launch_script_path: "<启动脚本路径>"
  log_path: "<训练日志路径>"
  model_name: "<模型名称>"
  model_type: "<模型类型>"

  # 可选（调用方可能传递）
  special_requirements: {}
```

---

## 3. 执行流程

所有 Step 均在本 Skill 内部按顺序执行，无需创建额外 SubAgent。

### Step 1：定位问题导入语句

```yaml
step_1_locate:
  description: "在报错堆栈和项目源码中定位无条件导入 CUDA fused 算子的语句"
  patterns:
    - "from paddlefleet\\.ops import (fused_\\w+)"
    - "from paddlenlp\\.ops import (fused_\\w+)"
    - "from \\S+ import (fused_\\w+)"
  search_scope:
    - "{error_message} 的堆栈文件"
    - "${REPOS_DIR}/PaddleFleet"
    - "${REPOS_DIR}/PaddleFormers"
  output:
    - target_file: "<绝对路径>"
    - import_line: "<原始导入语句>"
    - op_name: "<算子变量名，如 fused_apply_rotary_pos_emb_vision>"
    - module_name: "<来源模块，如 paddlefleet.ops>"
```

> **若无法定位文件**：返回 `fix_status: manual_required`，`new_error` 说明无法定位。

---

### Step 2：无条件导入 → 条件导入

将 Step 1 定位到的导入语句替换为以下模板（保持原始缩进和上下文）：

```python
if paddle.is_compiled_with_cuda():
    try:
        from {module_name} import {op_name}
    except ImportError:
        logging.getLogger(__name__).warning(
            "Failed to import optional CUDA op "
            "'{op_name}'; falling back to the "
            "non-fused path."
        )
        {op_name} = None
else:
    {op_name} = None
```

**规则**：
- 必须保留原始的算子名称 `{op_name}`
- Warning 信息中必须包含算子全名和 `"falling back"` 提示
- 非 CUDA 环境下必须显式设为 `None`
- 同一文件中若存在多个此类导入，每个独立处理

---

### Step 3：使用处增加 `is not None` 空值保护

在 `{target_file}` 中搜索 `{op_name}` 的所有使用位置，在调用/条件判断前增加空值保护。

**变换模板**：

```python
# 原始代码示例：
if not rotary_interleaved and mscale == 1.0 and ...:
    t = fused_apply_rotary_pos_emb_vision(t, freqs_half)

# 修复后：
if (
    fused_apply_rotary_pos_emb_vision is not None
    and not rotary_interleaved
    and mscale == 1.0
    and ...
):
    t = fused_apply_rotary_pos_emb_vision(t, freqs_half)
```

**规则**：
- 仅在 fused 路径的条件判断中增加保护，不改变正常逻辑分支
- 若文件中无使用该算子（仅导入未使用），可跳过此步骤
- 不得修改非 fused 路径的代码

---

### Step 4：语法校验

```bash
python -m py_compile {target_file}
```

- 成功：继续 Step 5
- 失败：回滚修改，返回 `fix_status: failed`，`new_error` 包含语法错误详情

---

### Step 5：汇总返回

按 `fix-xpu-training-issues` 约定格式返回：

```yaml
return:
  fix_status: "success | failed | manual_required"
  fix_description: "<修复做了什么>"
  modified_files: ["<文件绝对路径>"]
  new_error: "<若修复后引入新错误>"
  can_continue: true  # 除非返回 manual_required，否则保持 true
```

---

## 4. 返回格式（严格遵循调用方约定）

```json
{
  "fix_status": "success | failed | manual_required",
  "fix_description": "<本次修复的详细描述：修改了哪些文件、做了什么变更>",
  "modified_files": [
    "<修改后的文件绝对路径>"
  ],
  "new_error": "<修复后如果产生新错误，填写新错误信息；否则留空或 null>",
  "can_continue": true
}
```

---

## 5. 边界情况处理

```yaml
edge_cases:
  already_guarded:
    condition: "导入已经是条件导入形式（存在 `if paddle.is_compiled_with_cuda()`）"
    action: "检查使用处是否缺少 `is not None` 保护，补充缺失部分"

  multiple_ops_same_file:
    condition: "一个文件中有多个 CUDA fused 算子无条件导入"
    action: "对每个算子独立应用 Step 2 和 Step 3"

  custom_ops_module:
    condition: "导入来源不是 paddlefleet.ops 或 paddlenlp.ops"
    action: "只要是 `fused_` 前缀算子且报错包含 ImportError，同样适用本 Skill"

  import_in_init:
    condition: "导入在 __init__.py 中，被多个模块依赖"
    action: "同样替换为条件导入，并检查是否影响其他子模块导入"

  file_not_found:
    condition: "从报错堆栈无法定位到具体文件"
    action: "在 ${REPOS_DIR}/PaddleFleet 和 ${REPOS_DIR}/PaddleFormers 全局搜索 `from .* import .*fused_`"
    fallback: "若仍无法定位，返回 manual_required"
```

---

## 6. 与 fix-xpu-training-issues 的联动

```yaml
context_integration:
  # 接收自
  receive_from: "fix-xpu-training-issues / Step 2 (route) & Step 3 (invoke_fix)"

  # 路由触发条件
  routing_trigger:
    issue_features_contains_any:
      - "cuda_op_import_error"
      - "fused_op_unavailable"
    error_message_matches:
      - "from paddlefleet\\.ops import fused_"
      - "from paddlenlp\\.ops import fused_"
      - "cannot import name .*fused"
      - "ImportError.*fused"

  # 返回给
  return_to: "fix-xpu-training-issues / Step 3_invoke_fix"

  # 本 Skill 路径
  skill_path: "{SKILL_ROOT}/fix-cuda-op-import-issues/SKILL.md"
```

---

## 7. 示例（PR #800 映射）

**原始报错特征**：
```
ImportError: cannot import name 'fused_apply_rotary_pos_emb_vision' from 'paddlefleet.ops'
```

**issue_features**：
```json
["cuda_op_import_error", "fused_op_unavailable", "paddlefleet.ops"]
```

**修复动作**：
1. 定位文件：`src/paddlefleet/models/common/embeddings/rope_utils.py`
2. Step 2：将 `from paddlefleet.ops import fused_apply_rotary_pos_emb_vision` 改为条件导入
3. Step 3：在 `_apply_rotary_pos_emb_bshd` 的 `if apply_rope_fusion:` 分支中增加 `fused_apply_rotary_pos_emb_vision is not None`
4. Step 4：`python -m py_compile rope_utils.py` 通过
5. 返回 `fix_status: success`

---

## 8. 禁止事项

- **禁止**修改与 CUDA fused 算子无关的导入语句
- **禁止**在非报错堆栈指向的文件中主动扩散修复范围（除非同文件内有多个同类问题）
- **禁止**跳过语法校验直接返回成功
- **禁止**直接执行训练脚本（验证由调用方负责）
