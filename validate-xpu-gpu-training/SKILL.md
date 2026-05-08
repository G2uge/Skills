---
name: validate-xpu-gpu-training
description: Validate XPU training consistency against GPU baseline via a two-tier strategy. Tier-1 performs a fast Loss alignment gate; if passed, emits a lightweight report immediately. Tier-2 runs full single-side trend analysis, PPL/Global Norm comparison, performance profiling, and root-cause diagnosis only when the Loss gate fails.
---

> **定位**：Step 5 验证阶段的专用 Skill，用于判定 XPU 训练结果是否与 GPU 基准一致。
> **核心方法**：
> 1. **Loss 快速验证门控（Tier-1）**：优先计算 Loss 核心指标（MAE、Pearson、Relative Error 等），若全部通过阈值，立即输出简化报告，跳过全量分析。
> 2. **全量深度分析（Tier-2）**：仅当 Loss 门控未通过（Warn / Fail）时，才执行单端趋势分析、PPL/Global Norm 对比、性能对比、根因分析等。

---

## 输入参数

```yaml
inputs:
  # 日志路径（必需）
  xpu_log_path: " "<XPU workerlog.0 路径>"
  gpu_log_path: " "<GPU workerlog.0 路径>"

  # 输出目录（必需）
  output_dir: " "<验证报告输出目录>"

  # 阈值配置（可选，使用默认值；主要用于 Loss 快速验证门控）
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
  model_name: " " "<模型名称>"
  model_type: " " "<模型类型>"

  # 验证模式控制（新增）
  validation_mode: "auto"
    # auto：执行完整流程（保持原有行为）
    # early_only：仅执行 Early Gate 极简校验，立即返回
    # final_only：跳过 Early Gate，直接执行完整验证流程

  # Early Gate 控制参数（新增，仅 early_only 模式使用）
  early_gate_max_steps: 10          # Early Gate 最多取前几步
  early_gate_min_steps: 3           # 最少需要几步步才做 Early Gate

  # Early Gate 专用放宽阈值（新增）
  early_gate_thresholds:
    mae: 0.2
    pearson: 0.95
```

---

## 执行流程

```yaml
execution_flow:
  step_0_prepare:
    description: "创建输出目录结构"
    note: |
      本 Skill 会在传入的 output_dir 下自动创建 validate-xpu-gpu-training/{data,plots}/ 子目录。
      所有报告和图表均输出到该子目录下。
      调用方如需定位文件，需知晓此嵌套关系。
    actions:
      - "mkdir -p {output_dir}/validate-xpu-gpu-training/{data,plots}"
    output_dirs:
      - "{output_dir}/validate-xpu-gpu-training/data/"
      - "{output_dir}/validate-xpu-gpu-training/plots/"

  step_0b_mode_dispatch:
    description: "根据 validation_mode 决定执行路径"
    note: |
      auto / final_only：进入原有完整验证流程
      early_only：进入极简 Early Gate 流程，仅取前 N 步做快速校验
    branching:
      - condition: "validation_mode == 'early_only'"
        next: "step_1_early_gate"
      - condition: "validation_mode == 'final_only' 或 validation_mode == 'auto'（默认）"
        next: "step_1_extract"

  # ========== Early Only 分支：极简 Early Gate ==========
  step_1_early_gate:
    description: "Early Gate 极简精度门控"
    note: |
      仅取前 N 步数据，计算最核心指标，不画图，不做单端深度分析。
      生成简易 report.json 供上游调度器读取，不含图表、不含单端深度分析。
      目标 < 1 秒完成。支持有 GPU 数据（双端对比）和无 GPU 数据（单端自检验）两种模式。
    actions:
      1_pre_check:
        - "读取 xpu_log_path，解析 global_step 和 loss，统计当前步数（xpu_steps）"
        - "若 xpu_steps < early_gate_min_steps："
        - "  early_status = Fail"
        - "  failure_reason = '可用训练步数不足，训练可能未正常启动'"
        - "  跳过计算，直接返回"
      2_gpu_check:
        - "检查 gpu_log_path 是否存在且可解析出 >= early_gate_min_steps 的有效 loss 数据"
        - "判定 gpu_available = true | false"
    branching_by_gpu:
      - condition: "gpu_available == true"
        mode: "dual_side"
        actions:
          - "截取 XPU 前 min(xpu_steps, early_gate_max_steps) 步的 loss"
          - "截取 GPU 相同 global_step 区间的 loss"
          - "仅计算 2 个指标："
          - "  loss_mae = mean(abs(loss_xpu - loss_gpu))"
          - "  loss_pearson = pearsonr(loss_xpu, loss_gpu)"
        decision_rules:
          - "mae < early_gate_thresholds.mae (0.2) 且 pearson > early_gate_thresholds.pearson (0.95)"
            → early_status: "Pass"
          - "mae >= 0.5 或 pearson < 0.90"
            → early_status: "Fail（严重偏离）"
          - "其他"
            → early_status: "Warn"
      - condition: "gpu_available == false"
        mode: "single_side_only"
        actions:
          - "读取 XPU 前 min(xpu_steps, early_gate_max_steps) 步的 loss"
          - "计算单端指标："
          - "  monotonicity = spearmanr(global_step, loss)"
          - "  smoothness = std(diff(loss))"
          - "  initial_loss = loss[0]"
          - "  latest_loss = loss[-1]"
        decision_rules:
          - "monotonicity < -0.80 且 smoothness < 0.5 且 latest_loss < initial_loss"
            → early_status: "Pass"
            → 判定: "XPU 训练初期趋势正常，loss 正常下降且平滑"
          - "monotonicity > 0（loss 上升）或 latest_loss > initial_loss * 2（明显发散）"
            → early_status: "Fail"
            → 判定: "XPU 训练初期异常，loss 不下降或发散"
          - "其他"
            → early_status: "Warn"
            → 判定: "XPU 训练初期趋势可疑，需继续观察"
    output:
      early_status: "Pass | Warn | Fail"
      mode_used: "dual_side | single_side_only"
      steps_used: "<N>"
      metrics:
        # dual_side
        mae: "<数值>"
        pearson: "<数值>"
        # single_side_only
        monotonicity: "<数值>"
        smoothness: "<数值>"
        initial_loss: "<数值>"
        latest_loss: "<数值>"
    next: "step_1c_generate_early_report"

  step_1c_generate_early_report:
    description: "Early Gate 生成简易报告"
    note: |
      early_only 模式仍需生成 report.json，供上游调度器统一读取。
      报告为简化格式，仅包含核心判定信息和指标，不含图表、单端深度分析。
    output:
      report_json: "{output_dir}/validate-xpu-gpu-training/report.json"
    report_content:
      验证状态: " "<early_status>"
      验证层级: "early_gate_only"
      阶段: "early_gate"
      mode_used: " "<dual_side | single_side_only>"
      steps_used: " "<N>"
      元信息:
        model_name: "{model_name}"
        model_type: "{model_type}"
        xpu_log_path: "{xpu_log_path}"
        gpu_log_path: "{gpu_log_path}"
        report_time: " "<timestamp>"
      验证指标: " "<根据 mode 的指标>"
      判定: " "<判定说明>"
      限制: "无 GPU 数据时无法做双端精度对比，仅能判断训练健康度"
      failure_reason: " "<如 Fail 的原因>"
      can_proceed: "true（Pass/Warn）| false（Fail）"
    next: "step_1d_early_return"

  step_1d_early_return:
    description: "Early Gate 结果直接返回"
    note: "early_only 模式在此结束，不进入后续完整验证流程"
    return_format:
      验证状态: " "<early_status>"
      验证层级: "early_gate_only"
      阶段: "early_gate"
      mode_used: " "<dual_side | single_side_only>"
      steps_used: " "<N>"
      验证指标: " "<根据 mode 的指标>"
      判定: " "<判定说明>"
      限制: "无 GPU 数据时无法做双端精度对比，仅能判断训练健康度"
      failure_reason: " "<如 Fail 的原因>"
      can_proceed: "true（Pass/Warn）| false（Fail）"
      report_path: "{output_dir}/validate-xpu-gpu-training/report.json"

  # ========== Auto / Final Only 分支：原有完整流程 ==========
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

  # ========== Tier-1: Loss 快速验证门控 ==========
  step_3_loss_quick_check:
    description: "Loss 核心指标快速验证门控"
    actions:
      - "基于 aligned_data 计算 loss 的 MAE、RMSE、MaxDiff、Pearson、Spearman、R²、relative_error_percent"
      - "与 thresholds 对比，判定 loss_quick_status: Pass / Warn / Fail"
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
    loss_quick_status:
      - condition: "all criteria pass"
        result: "Pass"
      - condition: "any criterion fails but relative_error < 5%"
        result: "Warn"
      - condition: "relative_error >= 5% or any critical failure"
        result: "Fail"
    branching:
      - condition: "loss_quick_status == Pass"
        next: "step_4a_quick_pass_outputs"
        note: "Loss 对齐通过，跳过全量分析，直接生成简化报告与图表"
      - condition: "loss_quick_status in [Warn, Fail]"
        next: "step_4b_single_side_plots"
        note: "Loss 不对齐，进入全量深度分析流程"

  # ========== Tier-1 Pass: 简化输出分支 ==========
  step_4a_quick_pass_outputs:
    description: "Loss 验证通过：生成简化图表与报告"
    actions:
      - "生成双端对比图表（overlay、delta、relative_error、scatter）"
      - "生成简化 JSON 报告（仅含元信息、数据对齐、loss 指标、总体判定）"
      - "生成简化 Markdown 报告（仅 4 个章节）"
    plots:
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
        output: "{output_dir}/validate-xpu-gpu-training/plots/xpu_gpu_overlay.png"
      - title: "Loss Delta: XPU - GPU"
        x_axis: "global_step"
        y_axis: "delta_loss"
        tolerance_band: "±{thresholds.mae}"
        highlight_outliers: "points beyond tolerance band in red"
        output: "{output_dir}/validate-xpu-gpu-training/plots/xpu_gpu_delta.png"
      - title: "Relative Error per Step (%)"
        x_axis: "global_step"
        y_axis: "relative_error_percent"
        threshold_line: "{thresholds.relative_error_percent}%"
        formula: "relative_error = abs(loss_xpu - loss_gpu) / abs(loss_gpu) * 100"
        output: "{output_dir}/validate-xpu-gpu-training/plots/xpu_gpu_relative_error.png"
      - title: "Loss Alignment: XPU vs GPU"
        x_axis: "loss_gpu"
        y_axis: "loss_xpu"
        reference_line: "y = x (perfect alignment)"
        annotation: "R² = {r2_score}"
        output: "{output_dir}/validate-xpu-gpu-training/plots/xpu_gpu_scatter.png"
    report_json:
      output: "{output_dir}/validate-xpu-gpu-training/report.json"
      content:
        - "验证状态: Pass"
        - "验证层级: loss_quick_pass"
        - "元信息 (model_name, model_type, paths, timestamp)"
        - "数据对齐情况 (steps, interpolation, outliers)"
        - "Loss 双端对比 (7 项量化指标 + threshold checks)"
        - "总体判定: Loss 对齐通过，无需深度分析"
        - "输出文件清单"
    report_md:
      output: "{output_dir}/validate-xpu-gpu-training/validation_report.md"
      sections:
        - "一、数据对齐情况": "对齐步数、step 范围、插值次数、异常点过滤"
        - "二、Loss 双端量化对比指标": "仅 Loss 的 MAE/RMSE/MaxDiff/Pearson/Spearman/R2/RelativeError，附阈值检查结果"
        - "三、总体判定": "Pass + 原因"
        - "四、结论": "Loss 快速验证通过，无需进一步分析其他指标"
    note: "简化报告跳过单端趋势分析、PPL/GlobalNorm 对比、性能对比、根因分析、逐 step 异常 Top5"
    next: "step_12_package_outputs"

  # ========== Tier-2 Fail/Warn: 全量深度分析分支 ==========
  step_4b_single_side_plots:
    description: "绘制 XPU/GPU 单端独立曲线（全量分析分支）"
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

  step_5b_single_side_analysis:
    description: "单端趋势分析（全量分析分支）"
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
        threshold: " "< {thresholds.convergence_std}"
    output:
      - "嵌入 report.json 的 single_side_analysis 字段"

  step_6b_overlay_plot:
    description: "绘制叠加对比图（全量分析分支）"
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

  step_7b_delta_plot:
    description: "绘制差异曲线图（全量分析分支）"
    plot:
      - title: "Loss Delta: XPU - GPU"
        x_axis: "global_step"
        y_axis: "delta_loss"
        tolerance_band: "±{thresholds.mae}"
        highlight_outliers: "points beyond tolerance band in red"
    output:
      - "{output_dir}/validate-xpu-gpu-training/plots/xpu_gpu_delta.png"

  step_8b_relative_error_plot:
    description: "绘制逐 step 相对误差图（全量分析分支）"
    plot:
      - title: "Relative Error per Step (%)"
        x_axis: "global_step"
        y_axis: "relative_error_percent"
        threshold_line: "{thresholds.relative_error_percent}%"
    formula: "relative_error = abs(loss_xpu - loss_gpu) / abs(loss_gpu) * 100"
    output:
      - "{output_dir}/validate-xpu-gpu-training/plots/xpu_gpu_relative_error.png"

  step_9b_scatter_plot:
    description: "绘制散点对齐图（全量分析分支）"
    plot:
      - title: "Loss Alignment: XPU vs GPU"
        x_axis: "loss_gpu"
        y_axis: "loss_xpu"
        reference_line: "y = x (perfect alignment)"
        annotation: "R² = {r2_score}"
    output:
      - "{output_dir}/validate-xpu-gpu-training/plots/xpu_gpu_scatter.png"

  step_10b_full_quantitative_metrics:
    description: "计算全量量化评估指标（含 Loss、PPL、Global Norm）"
    formulas:
      loss_mae:    "mean(abs(loss_xpu - loss_gpu))"
      loss_rmse:   "sqrt(mean((loss_xpu - loss_gpu)²))"
      loss_max_diff: "max(abs(loss_xpu - loss_gpu))"
      loss_pearson:  "pearsonr(loss_xpu, loss_gpu)"
      loss_spearman: "spearmanr(loss_xpu, loss_gpu)"
      loss_relative_error_percent: "loss_mae / mean(abs(loss_gpu)) * 100"
      loss_r2:       "r2_score(loss_gpu, loss_xpu)"
      ppl_mae:       "mean(abs(ppl_xpu - ppl_gpu))"
      ppl_rmse:      "sqrt(mean((ppl_xpu - ppl_gpu)²))"
      norm_mae:      "mean(abs(global_norm_xpu - global_norm_gpu))"
    pass_criteria:
      - "loss_mae < {thresholds.mae}"
      - "loss_rmse < {thresholds.rmse}"
      - "loss_max_diff < {thresholds.max_diff}"
      - "loss_pearson > {thresholds.pearson}"
      - "loss_spearman > {thresholds.spearman}"
      - "loss_relative_error_percent < {thresholds.relative_error_percent}"
      - "loss_r2 > {thresholds.r2}"
    overall_status:
      - condition: "all criteria pass"
        result: "Pass"
      - condition: "any criterion fails but loss_relative_error < 5%"
        result: "Warn"
      - condition: "loss_relative_error >= 5% or any critical failure"
        result: "Fail"

  step_11b_generate_full_reports:
    description: "生成完整结构化 JSON 验证报告与 Markdown 报告（全量分析分支）"
    json_output:
      - "{output_dir}/validate-xpu-gpu-training/report.json"
    json_content:
      - "验证状态 (Pass/Warn/Fail)"
      - "验证层级: full_analysis"
      - "元信息 (model_name, model_type, paths, timestamp)"
      - "数据对齐情况 (steps, interpolation, outliers)"
      - "单端分析 (XPU/GPU: monotonicity, smoothness, convergence, outliers)"
      - "双端对比 (Loss 7 项量化指标 + PPL/Global Norm 指标 + threshold checks)"
      - "性能对比 (runtime, throughput, memory)"
      - "逐 step 异常分析 (outlier_steps, max_diff_step, top5_outliers)"
      - "根因分析"
      - "总体判定与建议"
      - "输出文件清单"
    md_output:
      - "{output_dir}/validate-xpu-gpu-training/validation_report.md"
    md_sections:
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
    note: "根据分支不同，实际输出文件清单有所差异"
    output_structure_quick_pass:
      data/:
        - xpu_training_data.json
        - gpu_training_data.json
        - aligned_data.json
      plots/:
        - xpu_gpu_overlay.png
        - xpu_gpu_delta.png
        - xpu_gpu_relative_error.png
        - xpu_gpu_scatter.png
      reports/:
        - report.json
        - validation_report.md
    output_structure_full_analysis:
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
  xpu_log_path: " "<路径>"
  gpu_log_path: " "<路径>"
  output_dir: " "<路径>"
  model_name: " "<名称>"
  model_type: " "<类型>"
  thresholds: {}  # 可选

expected_return:
  validation_status: "Pass | Warn | Fail | single_side_only"
  validation_tier: "loss_quick_pass | full_analysis | early_gate_only"

  # 当 validation_mode == early_only 时的返回
  early_gate_result:
    early_status: "Pass | Warn | Fail"
    mode_used: "dual_side | single_side_only"
    steps_used: " "<N>"
    metrics:
      # dual_side 模式
      mae: " "<数值>"
      pearson: " "<数值>"
      # single_side_only 模式
      monotonicity: " "<数值>"
      smoothness: " "<数值>"
      initial_loss: " "<数值>"
      latest_loss: " "<数值>"
    can_proceed: "true | false"
    failure_reason: " "<如 Fail 的原因>"
  report_path: "{output_dir}/validate-xpu-gpu-training/report.json"
    note: "本 Skill 会在传入的 output_dir 下自动创建 validate-xpu-gpu-training/ 子目录"
  plots_dir: "{output_dir}/validate-xpu-gpu-training/plots/"
    note: "本 Skill 会在传入的 output_dir 下自动创建 validate-xpu-gpu-training/plots/ 子目录"
  summary:
    mae: " "<数值>"
    rmse: " "<数值>"
    max_diff: " "<数值>"
    pearson: " "<数值>"
    relative_error_percent: " "<数值>"
  can_proceed: "true | false"
  failure_reason: " "<如 Fail 的原因>"
```

---

## report.json 完整格式

```json
{
  "验证状态": "Pass | Warn | Fail",
  "验证层级": "loss_quick_pass | full_analysis",
  "元信息": {
    "model_name": " "<模型名>",
    "model_type": " "<模型类型>",
    "xpu_log_path": " "<路径>",
    "gpu_log_path": " "<路径>",
    "report_time": " "<生成时间>"
  },
  "数据对齐": {
    "xpu_total_steps": 30,
    "gpu_total_steps": 30,
    "aligned_steps": 30,
    "interpolation_count": 0,
    "outlier_filtered": 0,
    "公共区间": [1, 30]
  },
  "Loss快速验证结果": {
    "loss_status": "Pass",
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
    "loss_mae": 0.012,
    "loss_rmse": 0.018,
    "loss_max_diff": 0.038,
    "loss_pearson": 0.9998,
    "loss_spearman": 0.9999,
    "loss_relative_error_percent": 0.85,
    "loss_r2": 0.9996,
    "ppl_mae": 0.021,
    "ppl_rmse": 0.035,
    "norm_mae": 0.005,
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
  "性能对比": {
    "xpu_train_runtime": 20.05,
    "gpu_train_runtime": 25.56,
    "xpu_steps_per_second": 1.4966,
    "gpu_steps_per_second": 1.1739,
    "xpu_max_memory_GB": 13.82,
    "gpu_max_memory_GB": 13.78
  },
  "根因分析": " "<数值精度差异或算子实现错误的分析结论>",
  "总体判定": "XPU 与 GPU loss 曲线高度一致，适配验证通过",
  "输出文件": {
    "data": {
      "xpu_training_data.json": " "<路径>",
      "gpu_training_data.json": " "<路径>",
      "aligned_data.json": " "<路径>"
    },
    "plots": {
      "xpu_training_curves.png": " "<路径>",
      "gpu_training_curves.png": " "<路径>",
      "xpu_gpu_overlay.png": " "<路径>",
      "xpu_gpu_delta.png": " "<路径>",
      "xpu_gpu_relative_error.png": " "<路径>",
      "xpu_gpu_scatter.png": " "<路径>"
    },
    "reports": {
      "report_json": " "<路径>",
      "validation_report_md": " "<路径>"
    }
  }
}
```

> **说明**：
> - 当 `验证层级` = `loss_quick_pass` 时，`单端分析`、`逐_step异常`、`性能对比`、`根因分析` 字段可为 `null` 或省略。
> - 当 `验证层级` = `full_analysis` 时，所有字段均需提供。

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

## 附录 B：Markdown 报告模板

本附录提供 `validation_report.md` 的标准结构与内容规范。根据验证层级不同，实际生成的报告分为 **简化版（Tier-1 Pass）** 和 **完整版（Tier-2 Fail/Warn）**。

### B.1 简化版报告模板（loss_quick_pass）

```markdown
# XPU vs GPU 训练精度对比验证报告（Loss 快速验证通过）

**模型**: {model_name}
**验证时间**: {timestamp}
**XPU 日志**: {xpu_log_path}
**GPU 日志**: {gpu_log_path}
**输出目录**: {output_dir}
**验证层级**: loss_quick_pass

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

## 二、Loss 双端量化对比指标

| 指标 | 数值 | 阈值 | 状态 |
|---|---|---|---|
| MAE | {mae} | < {threshold_mae} | {status} |
| RMSE | {rmse} | < {threshold_rmse} | {status} |
| Max Diff | {max_diff} | < {threshold_max_diff} | {status} |
| Pearson | {pearson} | > {threshold_pearson} | {status} |
| Spearman | {spearman} | > {threshold_spearman} | {status} |
| R² | {r2} | > {threshold_r2} | {status} |
| Relative Error (%) | {rel_err}% | < {threshold_rel_err}% | {status} |

### 阈值检查汇总

- **通过**: {passed_count} / {total_count}

---

## 三、总体判定

**{status_emoji} Loss 快速验证通过**

Loss 核心指标均在阈值范围内，XPU 与 GPU 训练结果高度一致。

---

## 四、结论

Loss 快速验证通过，无需进一步分析 PPL、Global Norm、性能等其他指标。
模型可安全用于生产环境。

---

*报告生成工具: validate-xpu-gpu-training Skill*
*分析框架: Tier-1 Loss 快速验证门控*
```

### B.2 完整版报告模板（full_analysis）

```markdown
# XPU vs GPU 训练精度对比验证报告（全量深度分析）

**模型**: {model_name}
**验证时间**: {timestamp}
**XPU 日志**: {xpu_log_path}
**GPU 日志**: {gpu_log_path}
**输出目录**: {output_dir}
**验证层级**: full_analysis

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
*分析框架: Tier-1 Loss 快速验证门控 + Tier-2 全量深度分析*
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
