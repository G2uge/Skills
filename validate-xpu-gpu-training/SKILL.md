---
name: validate-xpu-gpu-training
description: Validate XPU training consistency against GPU baseline by extracting structured metrics, plotting single-side and dual-side curves, and computing quantitative alignment scores.
---

> **定位**：Step 5 验证阶段的专用 Skill，用于判定 XPU 训练结果是否与 GPU 基准一致。
> **核心方法**：先分别绘制 XPU/GPU 单端曲线做独立趋势分析，再叠加对齐做双端对比。

---

## 输入参数

```yaml
inputs:
  # 日志路径（必需）
  xpu_log_path: "<XPU workerlog.0 路径>"
  gpu_log_path: "<GPU workerlog.0 路径>"

  # 输出目录（必需）
  output_dir: "<验证报告输出目录>"

  # 阈值配置（可选，使用默认值）
  thresholds:
    mae: 0.05                 # Mean Absolute Error 阈值
    rmse: 0.1                 # Root Mean Square Error 阈值
    max_diff: 0.5             # 最大绝对差异阈值
    pearson: 0.99             # 皮尔逊相关系数阈值
    spearman: 0.99            # 斯皮尔曼秩相关阈值
    relative_error_percent: 1.0   # 相对误差百分比阈值
    r2: 0.98                  # R² 决定系数阈值
    convergence_std: 0.01     # 收敛性：最后 10% steps 的 loss 标准差阈值

  # 模型信息（可选，用于报告标注）
  model_name: "<模型名称>"
  model_type: "<模型类型>"
```

---

## 执行流程

```yaml
execution_flow:
  step_0_prepare:
    description: "创建输出目录结构"
    actions:
      - "mkdir -p {output_dir}/validate-xpu-gpu-training/{data,plots}"
    output_dirs:
      - "{output_dir}/validate-xpu-gpu-training/data/"
      - "{output_dir}/validate-xpu-gpu-training/plots/"

  step_1_extract:
    description: "从日志提取结构化训练数据"
    source_files:
      - "{xpu_log_path}"
      - "{gpu_log_path}"
    extraction_fields:
      - global_step:    "global_step:\\s*(\\d+)"
      - loss:           "loss:\\s*([\\deE.+-]+)"
      - learning_rate:  "learning_rate:\\s*([\\deE.+-]+)"
      - ppl:            "ppl:\\s*([\\deE.+-]+)"
      - global_norm:    "global_norm:\\s*([\\deE.+-]+)"
      - interval_runtime: "interval_runtime:\\s*([\\deE.+-]+)"
      - interval_samples_per_second: "interval_samples_per_second:\\s*([\\deE.+-]+)"
      - interval_steps_per_second: "interval_steps_per_second:\\s*([\\deE.+-]+)"
      - train_runtime:  "train_runtime:\\s*([\\deE.+-]+)"
      - train_loss:     "train_loss:\\s*([\\deE.+-]+)"
      - current_memory_allocated: "current_memory_allocated:\\s*([\\deE.+-]+)"
      - max_memory_allocated: "max_memory_allocated:\\s*([\\deE.+-]+)"
    output:
      - "{output_dir}/validate-xpu-gpu-training/data/xpu_training_data.json"
      - "{output_dir}/validate-xpu-gpu-training/data/gpu_training_data.json"

  step_2_clean:
    description: "数据清洗与预处理"
    rules:
      - "按 global_step 排序"
      - "重复 step: 保留该 step 最后一次记录"
      - "缺失 step: 线性插值补全"
      - "loss 为 NaN/inf: 标记为异常点，不参与对比"
      - "截取公共区间: [max(min_step_xpu, min_step_gpu), min(max_step_xpu, max_step_gpu)]"
    output:
      - "{output_dir}/validate-xpu-gpu-training/data/aligned_data.json"

  step_3_single_side_plots:
    description: "绘制 XPU/GPU 单端独立曲线"
    plots_per_side:
      - name: "training_curves"
        layout: "2x2"
        subplots:
          - title: "loss vs global_step"
            x: "global_step"
            y: "loss"
            yscale: "linear"
          - title: "ppl vs global_step"
            x: "global_step"
            y: "ppl"
            yscale: "linear"
          - title: "learning_rate vs global_step"
            x: "global_step"
            y: "learning_rate"
            yscale: "linear"
          - title: "global_norm vs global_step"
            x: "global_step"
            y: "global_norm"
            yscale: "linear"
        output:
          xpu: "{output_dir}/validate-xpu-gpu-training/plots/xpu_training_curves.png"
          gpu: "{output_dir}/validate-xpu-gpu-training/plots/gpu_training_curves.png"

  step_4_single_side_analysis:
    description: "单端趋势分析"
    metrics_per_side:
      - name: "monotonicity"
        method: "Spearman rank correlation between step and loss"
        expected: "significantly negative (loss decreases over steps)"
      - name: "smoothness"
        method: "Standard deviation of adjacent step loss differences"
        expected: "small value (smooth curve)"
      - name: "outliers"
        method: "3-sigma rule on loss values"
        expected: "no points beyond 3 standard deviations"
      - name: "convergence"
        method: "Standard deviation of loss in last 10% steps"
        threshold: "< {thresholds.convergence_std}"
    output:
      - "嵌入 report.json 的 single_side_analysis 字段"

  step_5_overlay_plot:
    description: "绘制叠加对比图"
    plot:
      - title: "Loss: XPU vs GPU (Overlay)"
        x_axis: "global_step"
        y_axis: "loss"
        series:
          - label: "XPU"
            color: "blue"
            linestyle: "solid"
          - label: "GPU"
            color: "orange"
            linestyle: "solid"
        annotations:
          - "MAE = {mae}"
          - "Max Diff = {max_diff}"
          - "Pearson = {pearson}"
    output:
      - "{output_dir}/validate-xpu-gpu-training/plots/xpu_gpu_overlay.png"

  step_6_delta_plot:
    description: "绘制差异曲线图"
    plot:
      - title: "Loss Delta: XPU - GPU"
        x_axis: "global_step"
        y_axis: "delta_loss"
        tolerance_band: "±{thresholds.mae}"
        highlight_outliers: "points beyond tolerance band in red"
    output:
      - "{output_dir}/validate-xpu-gpu-training/plots/xpu_gpu_delta.png"

  step_7_relative_error_plot:
    description: "绘制逐 step 相对误差图"
    plot:
      - title: "Relative Error per Step (%)"
        x_axis: "global_step"
        y_axis: "relative_error_percent"
        threshold_line: "{thresholds.relative_error_percent}%"
    formula: "relative_error = abs(loss_xpu - loss_gpu) / abs(loss_gpu) * 100"
    output:
      - "{output_dir}/validate-xpu-gpu-training/plots/xpu_gpu_relative_error.png"

  step_8_scatter_plot:
    description: "绘制散点对齐图"
    plot:
      - title: "Loss Alignment: XPU vs GPU"
        x_axis: "loss_gpu"
        y_axis: "loss_xpu"
        reference_line: "y = x (perfect alignment)"
        annotation: "R² = {r2_score}"
    output:
      - "{output_dir}/validate-xpu-gpu-training/plots/xpu_gpu_scatter.png"

  step_9_quantitative_metrics:
    description: "计算量化评估指标"
    formulas:
      mae:    "mean(abs(loss_xpu - loss_gpu))"
      rmse:   "sqrt(mean((loss_xpu - loss_gpu)²))"
      max_diff: "max(abs(loss_xpu - loss_gpu))"
      pearson:  "pearsonr(loss_xpu, loss_gpu)"
      spearman: "spearmanr(loss_xpu, loss_gpu)"
      relative_error_percent: "mae / mean(abs(loss_gpu)) * 100"
      r2:       "r2_score(loss_gpu, loss_xpu)"
    pass_criteria:
      - "mae < {thresholds.mae}"
      - "rmse < {thresholds.rmse}"
      - "max_diff < {thresholds.max_diff}"
      - "pearson > {thresholds.pearson}"
      - "spearman > {thresholds.spearman}"
      - "relative_error_percent < {thresholds.relative_error_percent}"
      - "r2 > {thresholds.r2}"
    overall_status:
      - condition: "all criteria pass"
        result: "Pass"
      - condition: "any criterion fails but relative_error < 5%"
        result: "Warn"
      - condition: "relative_error >= 5% or any critical failure"
        result: "Fail"

  step_10_generate_json_report:
    description: "生成结构化 JSON 验证报告"
    output:
      - "{output_dir}/validate-xpu-gpu-training/report.json"
    content:
      - "验证状态 (Pass/Warn/Fail)"
      - "元信息 (model_name, model_type, paths, timestamp)"
      - "数据对齐情况 (steps, interpolation, outliers)"
      - "单端分析 (XPU/GPU: monotonicity, smoothness, convergence, outliers)"
      - "双端对比 (7 项量化指标 + threshold checks)"
      - "逐 step 异常分析 (outlier_steps, max_diff_step)"
      - "总体判定与建议"
      - "输出文件清单"

  step_11_generate_markdown_report:
    description: "生成可读性 Markdown 验证报告"
    output:
      - "{output_dir}/validate-xpu-gpu-training/validation_report.md"
    sections:
      - "一、数据对齐情况": "对齐步数、step 范围、插值次数、异常点过滤"
      - "二、单端趋势分析": "XPU/GPU 各自的单调性、平滑度、收敛稳定性、异常点"
      - "三、双端量化对比指标": "Loss/PPL/GlobalNorm 的 MAE/RMSE/MaxDiff/Pearson/Spearman/R2/RelativeError，附阈值检查结果"
      - "四、总体判定": "Pass/Warn/Fail + 原因"
      - "五、异常 Step 分析": "超过 1% 阈值的 steps 列表、Top 5 异常点、最大差异 step"
      - "六、根因分析": "分析相对误差高的原因（如 loss 趋近 0 时的数值放大效应）"
      - "七、性能对比": "Runtime、Steps/sec、Memory 对比"
      - "八、输出文件清单": "所有 data/plots/reports 的路径"
      - "九、结论": "最终判定 + 是否可安全用于生产环境"
    note: "报告需包含具体数值和判定依据，便于人工审阅"

  step_12_package_outputs:
    description: "打包所有输出文件"
    output_structure:
      data/:
        - xpu_training_data.json
        - gpu_training_data.json
        - aligned_data.json
      plots/:
        - xpu_training_curves.png
        - gpu_training_curves.png
        - xpu_gpu_overlay.png
        - xpu_gpu_delta.png
        - xpu_gpu_relative_error.png
        - xpu_gpu_scatter.png
      reports/:
        - report.json
        - validation_report.md
```

---

## 被调用约定

本 Skill 由主 Agent 或 Step 5 SubAgent 调用，输入输出约定如下：

```yaml
inputs_from_caller:
  xpu_log_path: "<路径>"
  gpu_log_path: "<路径>"
  output_dir: "<路径>"
  model_name: "<名称>"
  model_type: "<类型>"
  thresholds: {}  # 可选

expected_return:
  validation_status: "Pass | Warn | Fail"
  report_path: "<report.json 路径>"
  plots_dir: "<plots 目录路径>"
  summary:
    mae: "<数值>"
    rmse: "<数值>"
    max_diff: "<数值>"
    pearson: "<数值>"
    relative_error_percent: "<数值>"
  can_proceed: "true | false"
  failure_reason: "<如 Fail 的原因>"
```

---

## report.json 完整格式

```json
{
  "验证状态": "Pass | Warn | Fail",
  "元信息": {
    "model_name": "<模型名>",
    "model_type": "<模型类型>",
    "xpu_log_path": "<路径>",
    "gpu_log_path": "<路径>",
    "report_time": "<生成时间>"
  },
  "数据对齐": {
    "xpu_total_steps": 30,
    "gpu_total_steps": 30,
    "aligned_steps": 30,
    "interpolation_count": 0,
    "outlier_filtered": 0,
    "公共区间": [1, 30]
  },
  "单端分析": {
    "XPU": {
      "loss_monotonicity": -0.98,
      "loss_smoothness": 0.12,
      "outliers_detected": 0,
      "convergence_std": 0.001,
      "convergence_pass": true,
      "train_runtime": 20.05,
      "train_loss": 1.7899,
      "max_memory_GB": 13.82,
      "steps_per_second": 1.4966
    },
    "GPU": {
      "loss_monotonicity": -0.97,
      "loss_smoothness": 0.15,
      "outliers_detected": 0,
      "convergence_std": 0.001,
      "convergence_pass": true,
      "train_runtime": 25.56,
      "train_loss": 1.7716,
      "max_memory_GB": 13.78,
      "steps_per_second": 1.1739
    }
  },
  "双端对比": {
    "mae": 0.012,
    "rmse": 0.018,
    "max_diff": 0.038,
    "pearson": 0.9998,
    "spearman": 0.9999,
    "relative_error_percent": 0.85,
    "r2": 0.9996,
    "criteria_check": {
      "mae_pass": true,
      "rmse_pass": true,
      "max_diff_pass": true,
      "pearson_pass": true,
      "spearman_pass": true,
      "relative_error_pass": true,
      "r2_pass": true
    }
  },
  "逐_step异常": {
    "超过1%阈值的_steps": [],
    "最大差异_step": {"step": 15, "xpu_loss": 0.7782, "gpu_loss": 0.7589, "diff": 0.0193}
  },
  "总体判定": "XPU 与 GPU loss 曲线高度一致，适配验证通过",
  "输出文件": {
    "data": {
      "xpu_training_data.json": "<路径>",
      "gpu_training_data.json": "<路径>",
      "aligned_data.json": "<路径>"
    },
    "plots": {
      "xpu_training_curves.png": "<路径>",
      "gpu_training_curves.png": "<路径>",
      "xpu_gpu_overlay.png": "<路径>",
      "xpu_gpu_delta.png": "<路径>",
      "xpu_gpu_relative_error.png": "<路径>",
      "xpu_gpu_scatter.png": "<路径>"
    },
    "reports": {
      "report_json": "<路径>",
      "validation_report_md": "<路径>"
    }
  }
}
```

---

## 与主 Skill 联动

```yaml
context_integration:
  receive_from: "主 Agent (Step 5)"
  return_to: "主 Agent"

  routing_conditions:
    - condition: "Step 3 返回 Success 且 GPU 基准日志存在"
      action: "调用本 Skill 进行 XPU vs GPU 验证"
    - condition: "Step 3 返回 Success 但无 GPU 基准日志"
      action: "仅执行单端分析，标记 validation_status = single_side_only"
    - condition: "验证返回 Fail"
      action: "回退到 Step 4 修复流程"
    - condition: "验证返回 Pass 或 Warn"
      action: "进入最终报告阶段，标记任务完成"
```

---

## 限制与边界

**可执行的情况**：
- XPU 和 GPU 日志均存在且包含 `global_step` 和 `loss` 字段
- 训练步数 > 1（否则无法绘制曲线）
- 日志格式为 PaddleFormers 标准输出格式

**降级处理**：
- 若缺少 GPU 基准日志：仅执行 XPU 单端分析，生成单端曲线，不计算双端指标
- 若日志格式非标准：尝试通用正则匹配，失败则返回 `manual_required`
- 若步数不一致且无法对齐：截取公共区间，剩余部分单独标注

---

## 依赖

```bash
pip install matplotlib numpy pandas scipy
```

---

## 附录：典型问题诊断速查表

| 现象 | 可能原因 | 建议 |
|---|---|---|
| MAE > 0.5 | 算子回退实现数学逻辑有误 | 检查 fallback 实现公式 |
| Pearson < 0.95 | 整体训练趋势不一致 | 检查 learning_rate schedule、warmup 配置 |
| 某 step 相对误差突增 | 该 step 的特定算子精度问题 | 定位对应 step 的操作 |
| XPU loss 发散 / 不下降 | 数值稳定性问题 | 检查 fp16/bf16 混合精度配置 |
| XPU 吞吐显著低于 GPU | XPU 后端优化不足 | 记录性能基线，非精度问题 |
| 最后几步 loss 震荡大 | 学习率未衰减到 0 | 检查 lr_scheduler 配置 |

---

## 附录 B：Markdown 报告模板 (validation_report.md)

本附录提供 `validation_report.md` 的标准结构与内容规范，供 Skill 实现参考。

```markdown
# XPU vs GPU 训练精度对比验证报告

**模型**: {model_name}
**验证时间**: {timestamp}
**XPU 日志**: {xpu_log_path}
**GPU 日志**: {gpu_log_path}
**输出目录**: {output_dir}

---

## 一、数据对齐情况

| 项目 | 数值 |
|---|---|
| XPU 总步数 | {xpu_total_steps} |
| GPU 总步数 | {gpu_total_steps} |
| 对齐步数 | {aligned_steps} |
| Step 范围 | {step_range} |
| 插值次数 | {interpolation_count} |
| 异常点过滤 | {outlier_filtered} |

---

## 二、单端趋势分析

### XPU

| 指标 | 数值 | 说明 |
|---|---|---|
| Loss 单调性 (Spearman) | {xpu_mono} | {判定} |
| 曲线平滑度 (std of Δ) | {xpu_smooth} | — |
| 收敛稳定性 (last 10% std) | {xpu_conv_std} | {判定} |
| 异常点 (3-sigma) | {xpu_outliers} | — |
| 初始 Loss | {xpu_initial_loss} | — |
| 最终 Loss | {xpu_final_loss} | — |
| Loss 下降幅度 | {xpu_loss_range} | — |

### GPU

（同上结构）

---

## 三、双端量化对比指标

### Loss 对比

| 指标 | 数值 | 阈值 | 状态 |
|---|---|---|---|
| MAE | {mae} | < {threshold_mae} | {status} |
| RMSE | {rmse} | < {threshold_rmse} | {status} |
| Max Diff | {max_diff} | < {threshold_max_diff} | {status} |
| Pearson | {pearson} | > {threshold_pearson} | {status} |
| Spearman | {spearman} | > {threshold_spearman} | {status} |
| R² | {r2} | > {threshold_r2} | {status} |
| Relative Error (%) | {rel_err}% | < {threshold_rel_err}% | {status} |

### PPL 对比

| 指标 | 数值 |
|---|---|
| MAE | {ppl_mae} |
| RMSE | {ppl_rmse} |

### Global Norm 对比

| 指标 | 数值 |
|---|---|
| MAE | {norm_mae} |

### 阈值检查汇总

- **通过**: {passed_count} / {total_count}

---

## 四、总体判定

**{status_emoji} {overall_status}**

{overall_reason}

---

## 五、异常 Step 分析

### 超过 {threshold_rel_err}% 相对误差的 step 数量

{outlier_count} / {total_steps}

### 关键异常点（Top 5 按相对误差排序）

| Step | XPU Loss | GPU Loss | 绝对差异 | 相对误差 |
|---|---|---|---|---|
| {step} | {xpu_loss} | {gpu_loss} | {diff} | {rel_err}% |
| ... | ... | ... | ... | ... |

### 最大绝对差异

| 属性 | 数值 |
|---|---|
| Step | {max_diff_step} |
| XPU Loss | {max_diff_xpu} |
| GPU Loss | {max_diff_gpu} |
| 差异 | {max_diff_value} |
| 相对误差 | {max_diff_rel_err}% |

---

## 六、根因分析

{root_cause_analysis}

---

## 七、性能对比

| 指标 | XPU | GPU |
|---|---|---|
| Train Runtime | {xpu_runtime}s | {gpu_runtime}s |
| Train Samples/sec | {xpu_samples_per_sec} | {gpu_samples_per_sec} |
| Train Steps/sec | {xpu_steps_per_sec} | {gpu_steps_per_sec} |
| Max Memory (GB) | {xpu_max_memory} | {gpu_max_memory} |

---

## 八、输出文件清单

### 数据文件

| 文件 | 路径 |
|---|---|
| XPU 结构化数据 | {xpu_data_path} |
| GPU 结构化数据 | {gpu_data_path} |
| 对齐数据 | {aligned_data_path} |
| 结构化报告 (JSON) | {report_json_path} |
| 本报告 (Markdown) | {report_md_path} |

### 图表文件

| 图表 | 路径 |
|---|---|
| XPU 单端曲线 | {xpu_curves_path} |
| GPU 单端曲线 | {gpu_curves_path} |
| 叠加对比图 | {overlay_path} |
| 差异曲线图 | {delta_path} |
| 相对误差图 | {relative_error_path} |
| 散点对齐图 | {scatter_path} |

---

## 九、结论

{final_conclusion}

---

*报告生成工具: validate-xpu-gpu-training Skill*
*分析框架: Step 5 三步验证法（单端 → 双端 → 量化）*
```

### 报告生成要求

1. **所有占位符必须替换为实际数值**，不得留空
2. **状态列必须使用 `[PASS]` / `[FAIL]` / `[WARN]` 标注**
3. **根因分析部分必须具体**，不能仅写 "精度有差异"
4. **结论部分必须明确回答**："XPU 适配是否通过"
5. **相对误差高的根因必须解释**：区分 "数值精度差异" vs "算子实现错误"
6. **生产环境建议**：明确给出 "可安全使用" 或 "需进一步分析"

### 根因分析模板（相对误差高时）

```markdown
### 为什么相对误差较高？

1. 训练后期 loss 已收敛到极小值（step {N}+，loss < {threshold}）
2. 微小绝对差异被严重放大：
   - 例如 step {X}：绝对差异仅 {abs_diff}，但相对误差达 {rel_err}%
3. 数值精度差异的来源：
   - XPU 使用 {fallback_name} 的原生 Paddle API 回退实现
   - GPU 使用 CUDA fused kernel
   - 两者在 FP16/BF16 混合精度计算中的舍入方式、累加顺序存在微小差异
   - 这种差异在 loss 趋近于 0 时，在相对误差计算中被指数级放大

### 为什么这不是精度问题？

| 证据 | 说明 |
|---|---|
| Pearson = {pearson} | 两曲线几乎完全线性相关 |
| R² = {r2} | XPU 可解释 GPU 的方差比例 |
| MAE = {mae} | 平均绝对误差极低 |
| Max Diff = {max_diff} | 最大差异发生在 loss ≈ {loss_at_max_diff} 时 |
| 收敛趋势一致 | 最终 loss 均收敛到 < {final_loss_threshold} |

**结论**：这是正常的数值精度差异，不影响整体训练收敛趋势和最终模型质量。
```
