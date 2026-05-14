---
name: prune-model-layers
description: 对任意模型进行可配置减层，缩短 launch 时间，验证前向/反向通路。通过自动探测 config 层数字段和权重 key 模式，零预设适配各类架构。支持基于模型架构、硬件环境和训练配置的智能减层决策。
---

> **定位**：模型调试前置步骤，通过修改 config 中的层数声明，让框架只加载部分层权重，从而加速初始化。
> **原则**：只改配置，不动权重文件；自动探测，零预设架构；安全备份，一键还原；智能决策，灵活执行。

---

## 输入参数

```yaml
inputs:
  model_path: " "              # 模型目录路径，需包含 config.json 和权重索引文件
  mode: "auto"                 # 减层模式：auto | extreme_fast | fast | balanced | full
  component: "auto"            # 作用范围：auto | all | {具体字段路径}

  # 新增：环境上下文（可选，执行侧负责探测或让用户提供）
  hardware_context: null       # 若提供，应包含 device_type / world_size / per_device_memory
  train_config: null           # 训练配置文件路径（yaml/json），用于提取并行策略、batch、seq_len 等

  # YAML 联动恢复参数（仅在 mode=full 时生效）
  xpu_yaml_path: null          # XPU YAML 配置文件路径，恢复时同步修复 YAML 联动字段
  yaml_restore_overrides: {}   # YAML 恢复字段映射，如 {recompute_num_layers: 11}
  yaml_keep_as_is: []          # YAML 中保持不变的字段列表（环境限制类修改）
```

**原则**：所有新增字段均为可选。`mode=auto` 时执行侧应尽力收集信息；收集不到则按保守策略回退。

---

## 执行流程

```yaml
execution_flow:
  step_1_backup:
    goal: "保留原始配置的可恢复副本"
    constraints:
      - 备份文件需能唯一标识（建议带时间戳）
      - 若备份已存在，避免覆盖，可复用或生成新标识
    output: "备份文件路径"

  step_2_probe:
    goal: "识别模型中层数可调控的组件，并理解其架构属性"
    input: "config.json, 权重索引文件"
    requirements:
      - 找出所有声明层数的字段及其当前值
      - 通过权重 key 模式验证哪些组件的层是可裁剪的
      - 识别每个组件的架构类型：标准 Transformer / MoE / Vision / Audio / 其他
      - 估算各组件的单层计算/显存权重（相对开销），用于后续决策
    output: "组件列表（含字段路径、当前层数、架构类型、单层权重、是否可裁剪）"

  step_3_analyze:
    goal: "综合模型、硬件、训练配置，计算当前显存压力并决定目标层数"
    condition: "mode == 'auto'"
    input: "step_2 的组件列表 + hardware_context + train_config"
    requirements:
      - 建立显存预算模型：至少区分 参数静态占用 / 激活值动态占用 / 优化器状态 三类
      - 根据硬件总容量和训练并行策略，评估是否能容纳全量模型
      - 对可裁剪组件，计算能使显存压力降至安全区间的目标层数
      - 推导过程必须可解释（后续报告中要展示关键推理步骤）
    constraints:
      - 任何组件最终层数 >= 4
      - 若存在 pipeline parallelism，目标层数需满足其整除约束
      - MoE 组件的目标层数应比同规模标准 Transformer 更保守
      - 多模态中的非文本组件，默认不参与裁剪
      - 原层数 <= 12 时，避免过度裁剪（建议不低于原层数 // 3）
    output: "各组件目标层数 + 推荐内部 mode 标记 + 显存压力结论"

  step_4_resolve:
    goal: "当 mode 为手动指定时，计算目标层数"
    condition: "mode != 'auto'"
    input: "step_2 的组件列表 + train_config（如有）"
    requirements:
      - 将固定比例（extreme_fast / fast / balanced / full）映射为各组件目标层数
      - 应用与 step_3 相同的约束（下限、PP 整除、MoE 保护、多模态保护）
    output: "各组件目标层数"

  step_5_apply:
    goal: "修改配置，使框架按新的层数声明加载模型"
    input: "step_3 或 step_4 的输出"
    requirements:
      - 仅修改 config.json 中的层数字段，不动权重文件
      - 若 train_config 中存在与层数联动的字段（如 recompute_num_layers），应同步调整以保持逻辑一致
      - 所有修改前必须确认备份已就绪
    output: "修改后的 config.json + 联动配置变更清单"

  step_6_report:
    goal: "输出变更摘要与决策依据"
    output: |
      必须包含：
      - 每个组件的原层数 -> 目标层数
      - 最终选用的 mode（或 auto 模式下内部推导的等价 mode）
      - 若走了 auto 路径，显存压力的关键结论和裁剪理由
      - 恢复指令（如何回到 full）

  step_7_teardown:
    goal: "从减层阶段切换到全量阶段时，清理旧训练现场"
    condition: "mode == 'full'"
    requirements:
      - 终止可能仍在运行的、基于该模型的训练进程
      - 清理该进程占用的系统资源（如共享内存、端口）
      - 保留旧日志供追溯，但避免与即将启动的全量训练冲突
      - 等待并确认环境就绪后再返回
    note: "此步骤不可跳过，否则全量训练启动时可能因资源冲突失败"

  step_8_restore:
    goal: "恢复 YAML 中与层数联动的字段"
    condition: "mode == 'full' 且用户提供了 yaml 恢复参数"
    requirements:
      - 仅恢复因减层而被动修改的字段（如 recompute_num_layers）
      - 保护用户因环境限制而主动修改的字段不被还原
      - 恢复后验证文件语法有效性
```

---

## 返回格式

```json
{
  "status": "Success | Fail",
  "mode": "auto | extreme_fast | fast | balanced | full",
  "backup_path": "config.json.bak.xxx",
  "components_modified": [
    {
      "field_path": "text_config.num_hidden_layers",
      "original": 36,
      "target": 6,
      "architecture_tag": "transformer | moe | vision | audio | other",
      "action": "modified | skipped | not_found"
    }
  ],

  "decision": {
    "path": "auto | manual",
    "pressure_conclusion": "string",
    "target_derivation_trace": ["string"]
  },

  "summary": {
    "total_components_found": 2,
    "components_modified": 1,
    "components_skipped": 1,
    "reason_skipped": "auto mode: non-text component kept unchanged"
  },

  "yaml_adjustments": {
    "status": "Success | Fail | NotRequired",
    "fields_changed": [
      {
        "field": "recompute_num_layers",
        "from": 7,
        "to": 11
      }
    ]
  },

  "restore_command": "prune-model-layers <model_path> --mode full"
}
```

---

## 调用方式

```bash
# 智能决策（默认）
prune-model-layers <model_path>

# 智能决策 + 提供训练配置
prune-model-layers <model_path> --train-config ./configs/train.yaml

# 指定模式
prune-model-layers <model_path> --mode fast
prune-model-layers <model_path> --mode balanced
prune-model-layers <model_path> --mode full          # 一键还原

# 指定组件范围
prune-model-layers <model_path> --mode auto --component all
prune-model-layers <model_path> --component text_config.num_hidden_layers

# 指定硬件上下文（探测失败时使用）
prune-model-layers <model_path> --hardware-context '{"world_size":8,"per_device_memory_mb":65536}'
```

---

## 判定准则

| 规则 | 说明 |
|---|---|
| **绝对下限** | 任何可裁剪组件不少于 4 层。Transformer 至少需 3~4 层堆叠才能验证 Attention、FFN、Residual、Norm 的连通性 |
| **PP 整除** | 若训练配置中存在 pipeline parallelism，目标层数必须能被 PP size 整除；若无法满足且不能调整 PP，则报错 |
| **MoE 保守** | MoE 类型组件的目标层数应比同参数规模标准 Transformer 更保守（建议不低于原层数 // 4） |
| **小模型保护** | 原层数 <= 12 时，避免过度裁剪（建议不低于原层数 // 3） |
| **多模态保护** | auto 模式下，非文本组件默认不裁剪 |
| **auto 回退** | 信息不足无法做显存建模时，回退到 fast 模式并提示 |
| **硬件不可用时回退** | 无法探测硬件时，假设单卡 32GB 继续计算，但最终报告中必须提示用户复核 |
| **已修改检测** | 执行前检查 config 是否与备份一致，不一致则提示先还原或确认覆盖 |

---

## 验证要点

减层后运行训练或推理，确认以下三点即视为成功：

1. 模型初始化不报错
2. 前向传播通过（dummy input 可正常推理）
3. 反向梯度可回传（loss.backward() 不报错）

> **注意**：extreme_fast 模式下 loss 可能不下降，属正常。目标是"跑通"而非"收敛"。

---

## 异常处理

```yaml
special_cases:
  config_not_found:
    condition: "model_path 下不存在 config.json"
    behavior: "失败，提示检查路径"

  index_not_found:
    condition: "权重索引文件不存在"
    behavior: "失败，提示无法分析权重结构"

  no_layer_detected:
    condition: "未探测到任何可裁剪的层结构"
    behavior: "失败，提示该模型可能不支持减层"

  backup_conflict:
    condition: "已存在同名备份"
    behavior: "跳过备份，继续执行"

  restore_no_backup:
    condition: "mode=full 但无可用备份"
    behavior: "失败，提示无法还原"

  pp_constraint_unsatisfiable:
    condition: "PP 整除约束导致目标层数 < 4"
    behavior: "失败，建议用户调整 pipeline_model_parallel_size 或改用手动 mode"

  hardware_insufficient:
    condition: "显存压力极高，即使压到 4 层仍严重超出可用显存"
    behavior: "成功执行减层，但附加 warning 提示用户可能需要增加卡数、降低 seq_length 或启用 offload"

  yaml_not_found:
    condition: "提供了 xpu_yaml_path 但文件不存在"
    behavior: "config.json 仍正常恢复；yaml_adjustments.status = Fail，提示检查路径"

  yaml_restore_field_not_found:
    condition: "yaml_restore_overrides 中的字段在 YAML 中不存在"
    behavior: "跳过该字段，继续恢复其他字段；在失败报告中提示"

  yaml_syntax_error_after_restore:
    condition: "恢复后 YAML 语法验证失败"
    behavior: "失败，建议手动检查 YAML 文件"
```
