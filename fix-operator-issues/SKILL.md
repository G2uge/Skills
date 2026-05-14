---
name: fix-operator-issues
description: Fix CUDA-only fused operator compatibility issues on XPU backends by providing native Paddle API fallbacks. Covers NotImplementedError for non-CUDA backends, missing custom ops, and fused kernel unavailability.
---

> **适用范围**：训练过程中因 CUDA-only 算子（custom op / fused kernel）在 XPU 后端不可用而失败的场景。
> **典型错误**：`NotImplementedError: .* is not implemented for non-CUDA backends`
> **修复策略**（按优先级排序）：
> 1. 检查 Paddle/PaddleFleet 是否已原生实现该算子的 XPU/CPU 版本。
>    - 若存在：直接调用原生算子作为 fallback，修改底层路由即可（需显式判断硬件后端）。
>    - 若不存在：进入步骤 2。
> 2. 向上追溯调用链，检查上游是否存在非融合的退化逻辑（如 else 分支使用原生 Paddle API 的实现）。
>    - 若存在：修改上游判断条件，增加 `paddle.is_compiled_with_cuda()` 判断，使非 CUDA 后端自动走退化路径。
>    - 若不存在：进入步骤 3。
> 3. 分析 CUDA kernel 数学逻辑 → 用原生 Paddle API 实现等价回退 → 修改底层路由使非 CUDA 后端走回退路径 → 生成测试验证等价性。

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

  step_2b_check_paddle_native_op:
    description: "检查 Paddle/PaddleFleet 是否已原生实现该算子的 XPU/CPU 版本"
    actions:
      - "提取算子核心语义（如 fused_swiglu_bwd → swiglu_grad）"
      - "在 Paddle 源码中搜索：phi/kernels/xpu/、phi/kernels/cpu/、phi/kernels/fusion/xpu/"
      - "在 PaddleFleet 源码中搜索对应实现"
      - "检查 paddle._C_ops、paddle.nn.functional 是否暴露 Python API"
      - "验证原生算子与 CUDA 自定义算子的输入输出语义等价性（参数顺序、返回格式、dtype 支持）"
    search_patterns:
      - "phi/kernels/xpu/<op_name>*_kernel.cc"
      - "phi/kernels/fusion/xpu/<op_name>*_kernel.cc"
      - "paddle._C_ops.<op_name>"
      - "paddle._C_ops.<op_name>_grad"
      - "paddle.nn.functional.<op_name>"
    decision_criteria:
      - "若找到语义等价的原生算子 → 进入 step_2c_use_native_op"
      - "若无原生实现，或语义不等价 → 进入 step_2d_trace_call_chain"

  step_2c_use_native_op:
    description: "使用 Paddle 原生算子作为 fallback"
    principles:
      - "显式判断目标硬件后端（如 paddle.is_compiled_with_xpu()），禁止用 else 做宽泛 fallback"
      - "保持现有 CUDA 路径不变"
      - "其他后端保留 NotImplementedError"
    modification_pattern: |
      # 修改前
      if paddle.is_compiled_with_cuda():
          from xxx import custom_op
          return custom_op(...)
      else:
          raise NotImplementedError("...")

      # 修改后
      if paddle.is_compiled_with_cuda():
          from xxx import custom_op
          return custom_op(...)
      elif paddle.is_compiled_with_xpu():
          dx, _ = paddle._C_ops.xxx_grad(...)
          return dx
      else:
          raise NotImplementedError("...")
    validation:
      - "确认 XPU 后端能正确调用原生算子"
      - "确认 CUDA 后端行为不变"
      - "确认其他后端仍抛异常（不引入未知风险）"
    on_success:
      - "进入 step_5_generate_tests（验证等价性）"
      - "或直接进入 step_7_return（标记修复完成）"

  step_2d_trace_call_chain:
    description: "向上追溯调用链，检查是否存在上游退化路径"
    actions:
      - "从报错算子出发，通过 grep/ast 分析向上追溯 2-3 层调用者"
      - "重点关注：条件分支（if/elif/else）中是否存在非融合退化逻辑"
      - "典型模式：融合路径调用 fused_xxx_impl / xxxFunction.apply，退化路径使用 paddle.chunk + F.xxx + paddle.multiply 等标准 API"
    search_patterns:
      - "调用当前算子/函数的上一层函数（如 bias_swiglu_impl 调用 swiglu_back）"
      - "再上一层调用者（如 mlp.py 中的 bias_activation_fusion 分支）"
      - "else 分支中是否存在非融合的等价实现"
    decision_criteria:
      - "若上游存在 else 分支的非融合退化逻辑 → 进入 step_2e_patch_upstream"
      - "若上游无退化逻辑，或退化逻辑数学不等价 → 跳过到 step_3_analyze_kernel"

  step_2e_patch_upstream:
    description: "修改上游判断条件，使非 CUDA 后端走退化路径"
    principles:
      - "最小侵入：只修改条件判断，不动退化逻辑本身"
      - "保持现有 CUDA 路径不变"
    modification_pattern:
      - "将 `elif self.config.bias_activation_fusion:` 改为 `elif self.config.bias_activation_fusion and paddle.is_compiled_with_cuda():`"
      - "或类似地，在融合路径入口处增加 `and paddle.is_compiled_with_cuda()` 判断"
    validation:
      - "确认非 CUDA 后端会进入 else 分支"
      - "确认 CUDA 后端行为不变"
    on_success:
      - "跳过底层算子 fallback 实现，直接进入 step_5_generate_tests（验证训练能否跑通）"
      - "或直接进入 step_7_return（标记修复完成）"
    on_failure:
      - "若上游修改后仍有其他错误 → 回退到 step_3_analyze_kernel，尝试底层 fallback"

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

### Case 1: fused_swiglu_bwd（Paddle 原生算子直接调用）

**错误**：
```
NotImplementedError: fused_swiglu_bwd is not implemented for non-CUDA backends.
```

**Step 2b: 检查 Paddle 原生算子**
- 报错点：`PaddleFleet/src/paddlefleet/fusions/fused_bias_swiglu.py:80-87`（swiglu_back）
- 在 Paddle 源码中搜索：`phi/kernels/xpu/swiglu_grad_kernel.cc`
- 验证 Python API：`python3 -c "import paddle; print(paddle._C_ops.swiglu_grad)"` 可用
- 语义验证：`fused_swiglu_bwd(g, y)` 与 `paddle._C_ops.swiglu_grad(y, None, g)` 都是计算 swiglu 对输入的梯度，数学上等价
- **结论**：Paddle 框架已原生实现 XPU 的 `swiglu_grad`，可直接调用。

**Step 2c: 直接调用原生算子**
- 修改 `fused_bias_swiglu.py`：
  ```python
  # 修改前
  if paddle.is_compiled_with_cuda():
      from paddlefleet.ops import fused_swiglu_bwd
      return fused_swiglu_bwd(g, y)
  else:
      raise NotImplementedError(
          "fused_swiglu_bwd is not implemented for non-CUDA backends."
      )

  # 修改后
  if paddle.is_compiled_with_cuda():
      from paddlefleet.ops import fused_swiglu_bwd
      return fused_swiglu_bwd(g, y)
  elif paddle.is_compiled_with_xpu():
      dx, _ = paddle._C_ops.swiglu_grad(y, None, g)
      return dx
  else:
      raise NotImplementedError(
          "fused_swiglu_bwd is not implemented for non-CUDA backends."
      )
  ```
- **效果**：XPU 后端直接调用 Paddle 原生 `swiglu_grad` kernel，CUDA 后端保持融合算子不变，其他后端仍抛异常。

---

### Case 2: bias_activation_fusion（上游退化路径）

**错误**：
```
NotImplementedError: fused_swiglu_bwd is not implemented for non-CUDA backends.
```

**Step 2b: 检查 Paddle 原生算子**
- 搜索 Paddle 源码，未发现可直接替代 `fused_swiglu_bwd` 的原生算子
- **结论**：无原生实现，进入 Step 2d。

**Step 2d: 追溯调用链**
- 报错点：`PaddleFleet/src/paddlefleet/fusions/fused_bias_swiglu.py:80-87`（swiglu_back）
- 上一层：`bias_swiglu_impl()` / `BiasSwiGLUFunction.backward()`
- 再上一层：`PaddleFleet/src/paddlefleet/transformer/mlp.py:196`
  ```python
  elif self.config.bias_activation_fusion:
      # 走融合路径 → bias_swiglu_impl → swiglu_back → fused_swiglu_bwd (CUDA only)
      intermediate_parallel = bias_swiglu_impl(...)
  else:
      # 走退化路径：纯 Python GLU，使用 paddle.chunk + F.silu + multiply
      def glu(x):
          x_glu, x_linear = paddle.chunk(x, 2, axis=-1)
          return self.config.hidden_act(x_glu) * x_linear
      intermediate_parallel = glu(intermediate_parallel)
  ```
- **结论**：上游 `else` 分支存在完整的非融合退化实现。

**Step 2e: 修改上游判断条件**
- 修改 `mlp.py:196`：
  ```python
  # 修改前
  elif self.config.bias_activation_fusion:
  # 修改后
  elif self.config.bias_activation_fusion and paddle.is_compiled_with_cuda():
  ```
- **效果**：非 CUDA 后端自动进入 `else` 分支的退化路径，完全避开 `fused_swiglu_bwd`。

---

### Case 3: xxx_op（无原生实现、无退化路径，需手写 fallback）

**错误**：
```
NotImplementedError: xxx_op is not implemented for non-CUDA backends.
```

**Step 2b: 检查 Paddle 原生算子**
- 搜索 Paddle 源码，未发现可直接替代的算子
- **结论**：无原生实现，进入 Step 2d。

**Step 2d: 追溯调用链**
- 报错点：`some_module.py:100`（xxx_op）
- 向上追溯 2-3 层调用者
- **结论**：所有调用路径均直接调用 xxx_op，无 else 分支的退化逻辑。

**Step 3-4: 分析 kernel 并实现底层 fallback**
（此处沿用原有的 analyze_kernel + implement_fallback 流程）

**回退实现**：
```python
def _xxx_op_fallback(*args, **kwargs):
    """XPU/CPU fallback for xxx_op using native Paddle APIs."""
    # 使用 paddle.* 标准 API 实现等价逻辑
    ...
    return result
```

**修改点**：
```python
# 修改前
else:
    raise NotImplementedError("xxx_op is not implemented for non-CUDA backends.")

# 修改后
else:
    return _xxx_op_fallback(*args, **kwargs)
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

**上游退化路径的边界**：
- 退化路径必须是**数学等价**的（输入输出语义一致），不能只处理部分 case
- 退化路径的性能损失需可接受（多 kernel 调度 vs 单 fused kernel）
- 若退化路径依赖上游配置参数（如 `gpt_model_use_experimental_version`），需确认该参数在当前框架版本中可用
- 若上游修改会导致其他模型架构行为变化，需评估影响范围

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
