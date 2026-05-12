---
name: diagnose-xpu-oom
description: |
  XPU 训练 OOM（Out of Memory）问题的系统化排查与根因定位 Skill。
  通过对比原始 GPU 配置与当前 XPU 环境的并行规模、模型结构、内存优化策略，
  精准定位 OOM 根因并生成可落地的降显存方案。
keywords: oom, memory, xpu, diagnose, parallel, moe, pipeline, tensor
---

> **定位**：当 Step 3（模型执行）或 Step 4（问题修复）遇到 OOM 错误时，
> 优先调用本 Skill 进行根因分析，而非盲目尝试修改配置。
>
> **原则**：先诊断、后开方；量化分析、有据可依。

---

## 输入参数

```yaml
inputs:
  # 必需参数
  gpu_yaml_path: " " # 原始 GPU YAML 配置文件路径
  xpu_yaml_path: " " # 当前 XPU YAML 配置文件路径
  model_path: " " # 模型权重目录路径（需包含 config.json）
  num_xpus: 8 # 当前环境可用 XPU 卡数

  # 可选参数
  oom_log_path: null # OOM 错误日志路径，用于提取具体错误信息
  xpu_memory_per_card_gb: null # 单卡 XPU 显存（GB），如无法自动检测则使用默认值 64
  output_dir: "/root/paddlejob/tmp/output" # 诊断报告输出目录
```

---

## 执行流程

```yaml
execution_flow:
  step_1_collect:
    description: "收集模型和硬件信息"
    action: |
      1. 读取模型 config.json，提取关键结构参数：
         - num_hidden_layers（层数）
         - num_attention_heads / num_key_value_heads
         - hidden_size
         - intermediate_size 或 moe_intermediate_size
         - num_local_experts / num_experts（MoE 模型）
         - vocab_size
      2. 读取原始 GPU YAML，提取并行配置：
         - tensor_model_parallel_size (TP)
         - pipeline_model_parallel_size (PP)
         - expert_model_parallel_size (EP)
         - use_expert_parallel
         - per_device_train_batch_size
         - max_seq_len
         - recompute_granularity / recompute_num_layers
         - sharding 策略
      3. 读取当前 XPU YAML，提取相同字段
      4. 检测当前环境 XPU 卡数和单卡显存

  step_2_scale_compare:
    description: "对比原始 GPU vs 当前 XPU 并行规模"
    action: |
      计算原始 GPU 配置所需总卡数：
        total_gpus_needed = TP_gpu × PP_gpu × EP_gpu
        （当 use_expert_parallel=true 时 EP 计入，否则 EP=1）

      计算当前 XPU 有效并行度：
        total_xpus_effective = TP_xpu × PP_xpu × EP_xpu

      生成规模对比报告：
        - 原始需要卡数 vs 当前可用卡数
        - 规模比例 = num_xpus / total_gpus_needed

      判定规则：
        - 若 scale_ratio < 0.5：极大概率 OOM（当前环境只有原始规模的不到一半）
        - 若 0.5 <= scale_ratio < 1.0：可能 OOM，需要大量降显存优化
        - 若 scale_ratio >= 1.0：规模足够，OOM 根因在其他地方

  step_3_root_cause:
    description: "定位 OOM 根因"
    action: |
      按优先级逐一排查：

      P0 - 并行规模不足（最高优先级）
        检查项：
          - total_gpus_needed > num_xpus
          - EP_gpu 从 >1 降到 1（MoE Expert 未分布式）
          - PP_gpu 从 >1 降到 1（所有层集中到单卡）
        判定：若满足任一 → 根因 = parallel_scale_mismatch

      P1 - MoE Expert 显存爆炸
        检查项：
          - 模型为 MoE（num_experts > 1）
          - use_expert_parallel=false 或 expert_model_parallel_size=1
          - 原始配置 use_expert_parallel=true 且 EP > 1
        判定：若满足 → 根因 = moe_expert_not_sharded
        说明：MoE 的 Expert 权重通常占模型参数量 50%~80%，
              未分布式时每张卡保存全部 Expert，显存是分布式时的 EP 倍

      P2 - Pipeline Parallel 配置不当
        检查项：
          - PP_xpu < PP_gpu
          - num_hidden_layers 不能被 PP_xpu 整除
          - num_empty_layers_add_in_head/tail 被修改导致层数变化
        判定：若满足 → 根因 = pipeline_parallel_insufficient

      P3 - Activation 内存过大
        检查项：
          - max_seq_len >= 8192
          - per_device_train_batch_size >= 2
          - recompute_num_layers 过小（如 < 层数/2）
          - recompute_granularity != "full"
        判定：若满足 → 根因 = activation_memory_overflow

      P4 - 优化器状态内存
        检查项：
          - sharding 被关闭或降级（如 stage2 → stage1 或关闭）
          - tensorwise_offload_optimizer=false
          - bf16=false（使用 fp32 优化器状态）
        判定：若满足 → 根因 = optimizer_state_overflow

      P5 - 其他配置问题
        检查项：
          - fp32_residual_connection=true
          - save_checkpoint_format 占用额外内存
          - continue_training=true 加载完整 checkpoint
        判定：若满足 → 根因 = other_config_issue

  step_4_generate_plan:
    description: "生成降显存方案"
    action: |
      根据根因生成针对性修改方案，按推荐优先级排序：

      若根因包含 parallel_scale_mismatch：
        方案A: 增大 TP（如 TP=1→2→4→8）
        方案B: 恢复 PP（在层数可整除的前提下，如 PP=1→2）
        方案C: 若模型支持，开启 EP（需框架支持 XPU 上的 expert parallel）
        方案D: 综合调整 TP×PP×EP，使有效并行度尽量接近原始配置

      若根因包含 moe_expert_not_sharded：
        方案A: 开启 use_expert_parallel=true 并设置合理的 EP
        方案B: 若框架不支持 XPU EP，尝试减小 num_experts_per_tok 或关闭部分 Expert
        方案C: 使用 CPU offload 存储 Expert 权重（如框架支持）

      若根因包含 activation_memory_overflow：
        方案A: 减小 max_seq_len（如 8192→4096→2048）
        方案B: 减小 per_device_train_batch_size（如 2→1）
        方案C: 增大 recompute_num_layers（如 1→11→23→46）
        方案D: 开启 full recompute（recompute_granularity: full）

      若根因包含 optimizer_state_overflow：
        方案A: 开启 sharding stage1/stage2
        方案B: 开启 tensorwise_offload_optimizer
        方案C: 使用 bf16 + amp_master_grad

      每种方案标注：
        - 预计节省显存比例
        - 对训练速度的影响
        - 对收敛精度的影响
        - 可行性（高/中/低）

  step_5_evaluate:
    description: "评估方案可行性"
    action: |
      对生成的方案进行综合评估：

      1. 计算理论最低显存需求：
         model_params_gb = total_params × 2 / 1e9  # bf16 参数
         optimizer_gb = model_params_gb × 2  # Adam 状态（fp32 copy + momentum）
         activation_gb = batch_size × seq_len × hidden_size × layers × 4 / 1e9  # 粗略估算
         expert_gb = (num_experts × expert_size) × 2 / 1e9  # 若未分布式
         total_min_gb = (model_params_gb + optimizer_gb + activation_gb + expert_gb) / num_xpus

      2. 与单卡显存对比：
         - 若 total_min_gb > xpu_memory_per_card_gb × 0.9：方案可能仍不足
         - 若 total_min_gb < xpu_memory_per_card_gb × 0.7：方案可行

      3. 给出最终建议：
         - 推荐方案（最有希望成功的修改组合）
         - 备选方案
         - 若所有方案均不可行：建议接受减层验证或增加硬件

  step_6_output:
    description: "输出诊断报告"
    action: "按下方返回格式生成 JSON 报告并保存到 output_dir"
```

---

## OOM 排查速查表

| 检查项 | 原始值 vs 当前值 | 正常范围 | 异常标志 |
|--------|-----------------|----------|----------|
| 总并行卡数 | GPU: TP×PP×EP | 当前: num_xpus | 当前 < 原始 50% |
| TP | 原始值 | 1,2,4,8 | 未调整或过大 |
| PP | 原始值 | 能整除层数 | 层数%PP≠0 |
| EP | 原始值 | MoE 时>1 | MoE 时=1 |
| Batch Size | 原始值 | ≥1 | 未减小 |
| Seq Length | 原始值 | 根据显存 | ≥8192 且显存小 |
| Recompute | recompute_num_layers | ≥层数/2 | 过小或关闭 |
| Sharding | stage1/stage2/off | 显存紧张时应有 | 关闭 |

---

## 返回格式

```json
{
  "诊断状态": "Success | Fail",
  "oom根因": {
    "primary_cause": "parallel_scale_mismatch | moe_expert_not_sharded | activation_memory_overflow | optimizer_state_overflow | pipeline_parallel_insufficient | other_config_issue",
    "secondary_causes": ["..."],
    "根因说明": "..."
  },
  "规模对比": {
    "原始gpu总卡数": 32,
    "当前xpu可用卡数": 8,
    "规模比例": 0.25,
    "并行配置对比": {
      "TP": {"原始": 1, "当前": 4},
      "PP": {"原始": 4, "当前": 1},
      "EP": {"原始": 8, "当前": 1},
      "有效并行度": {"原始": 32, "当前": 4}
    }
  },
  "模型结构": {
    "num_hidden_layers": 46,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_experts": 128,
    "moe_intermediate_size": 6400,
    "总参数量估算": "32B"
  },
  "显存分析": {
    "单卡参数显存_gb": 8.0,
    "单卡优化器状态_gb": 16.0,
    "单卡activation估算_gb": 12.0,
    "单卡expert权重_gb": 24.0,
    "理论最低显存_gb": 60.0,
    "单卡可用显存_gb": 64.0,
    "余量比例": 0.06
  },
  "降显存方案": [
    {
      "方案编号": "A",
      "方案描述": "增大TP=8，恢复PP=2（46层+2空层=48，可被2整除），开启sharding stage1",
      "修改项": {
        "tensor_model_parallel_size": 8,
        "pipeline_model_parallel_size": 2,
        "sharding": "stage1",
        "recompute_num_layers": 23
      },
      "预计节省显存": "40%",
      "对速度影响": "TP通信增加，略降速",
      "对精度影响": "无",
      "可行性": "高"
    },
    {
      "方案编号": "B",
      "方案描述": "减小seq_len到2048，batch_size保持1，增大recompute",
      "修改项": {
        "max_seq_len": 2048,
        "recompute_num_layers": 46
      },
      "预计节省显存": "35%",
      "对速度影响": "recompute增加计算",
      "对精度影响": "seq_len减半可能影响长文本能力",
      "可行性": "中"
    }
  ],
  "最终建议": {
    "推荐方案": ["A", "B"],
    "推荐理由": "...",
    "若仍不可行": "建议接受减层验证（如保留24层），或增加XPU卡数到16+"
  },
  "诊断报告路径": "/root/paddlejob/tmp/output/oom_diagnosis_report.json"
}
```

---

## 与主工作流的集成

```yaml
integration:
  触发时机:
    - "Step 3 返回 escalation_reason = out_of_memory"
    - "Step 4 修复 Skill 判断为 OOM 且需要量化分析"

  调用位置:
    - "SubAgent-4 (Step 4) 中，在调用 fix-xpu-training-issues 之前或之后"
    - "或作为 fix-xpu-training-issues 的子 Skill 被调用"

  输入来源:
    gpu_yaml_path: "主 Agent 记忆中的 user_inputs.gpu_yaml_path"
    xpu_yaml_path: "主 Agent 记忆中的 generated_files.xpu_yaml"
    model_path: "主 Agent 记忆中的 user_inputs.model_path"
    num_xpus: "从环境自动检测或启动脚本中获取"
    oom_log_path: "Step 3 返回的 affected_files.log"

  输出去向:
    - "返回给主 Agent，用于决策是否继续尝试修复或接受减层"
    - "若推荐方案可行，主 Agent 可指导 SubAgent-4 按方案修改 YAML 后重试"
```

---

## 典型案例分析（GLM-4.5-Air）

**背景**：
- 模型：GLM-4.5-Air，46 层，128 routed experts
- 原始 GPU 配置：TP=1, PP=4, EP=8，共需 32 卡
- 当前 XPU 环境：8 卡

**诊断结果**：
1. **规模比例 = 8/32 = 0.25**，远低于 0.5 阈值
2. **MoE Expert 未分布式**：EP 从 8 降到 1，Expert 权重显存膨胀 8 倍
3. **PP 取消**：PP 从 4 降到 1，所有 46 层集中到单卡
4. **理论最低显存 ≈ 60GB/卡**，单卡可用约 64GB，余量仅 6%

**结论**：
8 卡 XPU 在现有框架实现下，无法支持 46 层 GLM-4.5-Air 全量 SFT。
即使 TP=8、seq_len=2048、full recompute，仍可能因 Expert 权重和 Activation 内存而 OOM。

**建议**：
- 优先方案：减层到 24~30 层进行冒烟验证
- 备选方案：增加 XPU 卡数到 16~32 张
- 不推荐：在 8 卡上反复尝试各种降显存组合，成功率极低
```