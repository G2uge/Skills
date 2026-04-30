---
name: fix-operator-issues
description: Fix CUDA-only fused operator compatibility issues on XPU backends by providing native Paddle API fallbacks. Covers NotImplementedError for non-CUDA backends, missing custom ops, and fused kernel unavailability.
---

> **适用范围**：训练过程中因 CUDA-only 算子（custom op / fused kernel）在 XPU 后端不可用而失败的场景。
> **典型错误**：`NotImplementedError: .* is not implemented for non-CUDA backends`
> **修复策略**：分析 CUDA kernel 数学逻辑 → 用原生 Paddle API 实现等价回退 → 修改路由使非 CUDA 后端走回退路径 → 生成测试验证等价性。

---

## 输入参数

```yaml
inputs:
  error_message: "<完整报错信息>"
  error_source: "<config|training|validation>"
  issue_features: ["<分类特征列表>"]  # 如: ["operator", "cuda_only", "fused_kernel"]

  # 路径信息
  yaml_path: "<YAML 配置文件路径>"
  launch_script_path: "<启动脚本路径>"
  log_path: "<训练日志路径>"
  output_dir: "<输出目录，用于存放生成的测试文件>"

  # 模型信息
  model_name: "<模型名称>"
  model_type: "<模型类型>"

  # 可选
  repos_dir: "<代码仓库根目录，默认 /root/paddlejob/tmp/repos>"
```

---

## 执行流程

```yaml
execution_flow:
  step_1_detect:
    description: "检测算子错误类型"
    patterns:
      - regex: "NotImplementedError: .* is not implemented for non-CUDA backends"
        category: "cuda_only_fused_op"
        action: "proceed_to_fix"
      - regex: "cannot import name '.*' from '.*_extensions'"
        category: "missing_custom_op"
        action: "proceed_to_fix"
      - regex: "RuntimeError: .* op is not compiled with XPU"
        category: "op_not_compiled_for_xpu"
        action: "proceed_to_fix"
      - regex: "OSError: .* .so .* cannot open shared object file"
        category: "missing_shared_library"
        action: "check_if_op_fallback_possible"
    output:
      operator_name: "<从日志提取的算子名，如 fused_swiglu_bwd>"
      category: "<错误类别>"
      source_file: "<从 traceback 提取的源码文件>"
      line_number: "<行号>"

  step_2_locate_source:
    description: "定位算子源码"
    actions:
      - "从 error traceback 中提取算子调用链"
      - "在 repos_dir 下搜索对应的 .py / .cu 源码文件"
      - "识别 CUDA kernel 实现和 Python 封装层"
    search_patterns:
      - "def <operator_name>"
      - "PD_BUILD_OP(<operator_name>)"
      - "__global__ .* <operator_name>"
      - "class .*Function(paddle.autograd.PyLayer)"
    expected_files:
      - "<cuda_kernel>.cu"        # CUDA kernel 实现
      - "<python_wrapper>.py"     # Python 封装和 backward 逻辑

  step_3_analyze_kernel:
    description: "分析 CUDA kernel 数学逻辑"
    methodology:
      - "读取 .cu 文件中的 kernel 实现"
      - "提取前向/反向传播的数学公式"
      - "关注: 输入输出维度、切分方式、激活函数、元素级操作"
    extract_items:
      - forward_formula: "<数学表达式>"
      - backward_formula: "<梯度计算表达式>"
      - input_output_specs: "<张量形状和 dtype 要求>"
      - special_handling: "<如 chunk, concat, clamp, cast 等特殊操作>"

  step_4_implement_fallback:
    description: "实现原生 Paddle API 回退"
    principles:
      - "数学逻辑必须与 CUDA kernel 逐元素等价"
      - "仅使用 paddle.* 和 paddle.nn.functional.* 标准 API"
      - "禁止使用 CUDA-specific 操作（如 stream、grid、block）"
      - "保持输入输出 dtype、shape 完全一致"
    fallback_function_template: |
      def _<operator_name>_fallback(*args, **kwargs):
          """XPU/CPU fallback for <operator_name> using native Paddle APIs.

          Forward: <forward_formula>
          Backward: <backward_formula>
          """
          # 使用 paddle.chunk, paddle.concat, paddle.nn.functional.* 等实现
          ...
          return result
    modification_target:
      - "修改 Python 封装层中的路由逻辑"
      - "将 `if paddle.is_compiled_with_cuda(): ... else: raise` 改为 `else: return _fallback(...)`"

  step_5_generate_tests:
    description: "为回退实现生成测试用例"
    output_dir: "{output_dir}/tests"
    test_categories:
      - "correctness: 与原生 Paddle autograd 结果对比"
      - "shapes: 多种输入维度（2D, 3D, 不同 hidden size）"
      - "dtypes: float32, float16, bfloat16, float64"
      - "edge_cases: zeros, ones, negatives, extremes, batch=1, minimal_size"
      - "stability: 数值稳定性测试（极大/极小值）"
      - "reproducibility: 相同输入产生相同输出"
    test_file_name: "test_<operator_name>_fallback.py"

  step_6_run_tests:
    description: "运行测试验证等价性"
    command: "source <venv>/bin/activate && python {output_dir}/tests/test_<operator_name>_fallback.py -v"
    success_criteria: "所有测试通过"
    on_failure: "检查回退实现逻辑，修复后重试"

  step_7_return:
    description: "返回修复结果"
```

---

## 被调用约定（来自 fix-xpu-training-issues）

本 Skill 接收以下输入并返回约定格式：

```yaml
inputs_from_caller:
  error_message: "<原始错误>"
  error_source: "<错误来源>"
  issue_features: ["<特征列表>"]
  yaml_path: "<路径>"
  launch_script_path: "<路径>"
  log_path: "<路径>"
  output_dir: "<输出目录>"
  model_name: "<名称>"
  model_type: "<类型>"

expected_return:
  fix_status: "success | failed | manual_required"
  fix_description: "<修复描述>"
  modified_files: ["<文件列表>"]
  new_error: "<新错误>"
  can_continue: "<是否可继续尝试>"
```

---

## 典型修复案例

### Case 1: fused_swiglu_bwd

**错误**：
```
NotImplementedError: fused_swiglu_bwd is not implemented for non-CUDA backends.
```

**定位**：
- Python 封装：`PaddleFleet/src/paddlefleet/fusions/fused_bias_swiglu.py:80-87`
- CUDA kernel：`PaddleFleet/src/paddlefleet/_extensions/swiglu_kernel.cu`

**分析**：
- Forward: `swiglu(y) = silu(y1) * y2`, where `y = [y1, y2]` along last dim
- Backward:
  - `dx1 = g * y2 * d_silu(y1)`
  - `dx2 = g * silu(y1)`
  - `dx = concat([dx1, dx2], axis=-1)`

**回退实现**：
```python
def _fused_swiglu_bwd_fallback(g, y):
    y1, y2 = paddle.chunk(y, chunks=2, axis=-1)
    sig_y1 = paddle.nn.functional.sigmoid(y1)
    silu_y1 = y1 * sig_y1
    dx2 = g * silu_y1
    d_silu = sig_y1 * (1.0 + y1 * (1.0 - sig_y1))
    dx1 = g * y2 * d_silu
    return paddle.concat([dx1, dx2], axis=-1)
```

**修改点**：
```python
# 修改前
else:
    raise NotImplementedError("...")

# 修改后
else:
    return _fused_swiglu_bwd_fallback(g, y)
```

---

## 测试模板规范

每个回退实现必须生成以下测试用例：

```python
class Test<Operator>Fallback(unittest.TestCase):
    """Test suite for _<operator_name>_fallback."""

    def test_basic_2d(self): ...
    def test_basic_3d(self): ...
    def test_large_batch(self): ...
    def test_large_hidden(self): ...
    def test_batch_size_one(self): ...
    def test_minimal_size(self): ...
    def test_all_zeros(self): ...
    def test_negative_values(self): ...
    def test_positive_values(self): ...
    def test_gradient_all_ones(self): ...
    def test_gradient_all_zeros(self): ...
    def test_extreme_large_values(self): ...
    def test_extreme_small_values(self): ...
    def test_float16(self): ...
    def test_bfloat16(self): ...
    def test_float64(self): ...
    def test_reproducibility(self): ...
    def test_output_shape(self): ...
```

**测试文件输出位置**：`{output_dir}/tests/test_<operator_name>_fallback.py`

---

## 返回格式

```json
{
  "fix_status": "success | failed | manual_required",
  "fix_description": "<详细修复描述，包含修改了哪些文件、如何实现回退>",
  "modified_files": [
    "<被修改的源码文件路径>"
  ],
  "test_files": [
    "<生成的测试文件路径>"
  ],
  "new_error": "<修复后运行产生的新错误（如有）>",
  "can_continue": "true | false",
  "operator_info": {
    "name": "<算子名>",
    "category": "<错误类别>",
    "source_file": "<源码路径>",
    "fallback_function": "<回退函数名>"
  }
}
```

---

## 限制与边界

**可修复的情况**：
- CUDA-only fused operator 有明确的数学公式，可用 Paddle 标准 API 等价实现
- Python 层对 CUDA 的判断是 `paddle.is_compiled_with_cuda()` 形式

**不可修复的情况（返回 manual_required）**：
- 算子涉及复杂的 CUDA-specific 内存操作、warp shuffle、tensor core 等，无法用标准 API 模拟
- 算子在 C++ 层硬编码了 CUDA 调用，Python 层无法介入
- 缺少源码或源码在二进制 so 中不可读
- 回退实现后产生的新错误无法解决

---

## 与 fix-xpu-training-issues 联动

```yaml
context_integration:
  receive_from: "fix-xpu-training-issues (Step 4 路由)"
  return_to: "fix-xpu-training-issues"

  # 当 issue_features 包含 "operator" | "cuda_only" | "fused_kernel" 时，
  # fix-xpu-training-issues 应将问题路由到本 Skill
  routing_rules:
    - condition: "'operator' in issue_features or 'cuda_only' in issue_features"
      target_skill: "fix-operator-issues"
      skill_path: "{SKILL_ROOT}/fix-operator-issues/SKILL.md"
```
