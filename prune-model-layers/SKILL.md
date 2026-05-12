---
name: prune-model-layers
description: 对任意模型进行可配置减层，缩短 launch 时间，验证前向/反向通路。通过自动探测 config 层数字段和权重 key 模式，零预设适配各类架构。
---

> **定位**：模型调试前置步骤，通过修改 config 中的层数声明，让框架只加载部分层权重，从而加速初始化。
> **原则**：只改配置，不动权重文件；自动探测，零预设架构；安全备份，一键还原。

---

## 输入参数

```yaml
inputs:
  model_path: " " # 模型目录路径，需包含 config.json 和权重索引文件
  mode: "extreme_fast" # 减层模式：extreme_fast | fast | balanced | full
  component: "auto" # 作用范围：auto | all | {具体字段路径}

  # ★ YAML 联动恢复参数（仅在 mode=full 时生效）
  xpu_yaml_path: null # XPU YAML 配置文件路径，恢复时同步修复 YAML 联动字段
  yaml_restore_overrides: {} # YAML 恢复字段映射，如 {recompute_num_layers: 11}
  yaml_keep_as_is: [] # YAML 中保持不变的字段列表（环境限制类修改），如 [pipeline_model_parallel_size, using_sonic_moe]
```

---

## 执行流程

```yaml
execution_flow:
  step_1_backup:
    description: "备份原始配置"
    action: "复制 config.json 为 config.json.bak.{timestamp}"
    note: |
      若已存在同时间戳备份则跳过。
      --mode full 时从最近备份还原。

  step_2_probe:
    description: "自动探测层结构"
    action: |
      1. 扫描 config.json 中所有层数字段
         - 匹配模式：字段名包含 num_hidden_layers / num_layers / depth / n_layer / encoder_layers / decoder_layers
         - 记录完整路径（如 text_config.num_hidden_layers）和当前值
      2. 扫描权重索引文件（*.index.json）中的 weight_map
         - 通过正则提取含数字索引的层参数前缀
         - 常见模式：*.layers.{N}.* / *.layer.{N}.* / *.h.{N}.* / *.blocks.{N}.*
         - 按前缀分组，统计每组的最小/最大索引、层数、每层 key 数
      3. 交叉验证
         - 将 config 层数字段与权重前缀关联，识别可独立调控的层组件
         - 若权重有层 key 但 config 无对应字段，标记为不可调控

  step_3_map:
    description: "计算目标层数"
    logic: |
      对每个可调控组件，根据 mode 计算：

      extreme_fast（默认）:
        target = max(4, original // 6)
        用途：仅验证网络能否跑通前向+反向，launch 最快

      fast:
        target = original // 3
        用途：冒烟测试 + 基本训练健康度验证，loss 应能下降

      balanced:
        target = original // 2
        用途：验证收敛趋势，适合精度对比前稳定通跑

      full:
        target = original
        用途：从备份还原，恢复原始层数

      保护规则：
        - 任何组件不低于 4 层
        - 若原层数 <= 12，extreme_fast 退化为 fast（即 //3）

      component 过滤：
        - auto（默认）：只减识别为主干的组件（通常含 hidden_layers 的文本模型），其余保持
        - all：所有识别到的层组件均减层
        - {具体路径}：仅对该 config 字段操作

  step_4_apply:
    description: "修改 config.json"
    action: "将目标组件的层数字段修改为目标值，其余字段一律不动"
    note: "不修改 .safetensors / .bin 权重文件，框架按 config 声明自动过滤加载"

  step_5_report:
    description: "输出 config.json 修改摘要"

  step_5_5_stop_pruned_training:
    description: "停止减层阶段的训练进程（仅在 mode=full 时执行）"
    condition: "mode == 'full'"
    action: |
      1. 查找并终止与当前模型相关的训练进程：
         - pkill -9 -f "paddleformers-cli" 2>/dev/null || true
         - pkill -9 -f "launcher.py" 2>/dev/null || true
         - ps -ef | grep -E "paddleformers|launcher" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
         - 等待 5 秒确认进程已终止（轮询检查直到相关进程数为 0 或超时 15 秒）
      2. 清理共享内存残留：
         - for id in $(ipcs -m 2>/dev/null | awk '/0x/ {print $2}'); do ipcrm -m $id 2>/dev/null || true; done
      3. 备份旧的分布式日志：
         - 若 output_dir/paddleformers_dist_log 存在，将其重命名为 paddleformers_dist_log.pruned.{timestamp}
         - 若 output_dir/train.log 存在，将其重命名为 train.log.pruned.{timestamp}
      4. 清理 Paddle 分布式环境变量：
         - unset PADDLE_ELASTIC_JOB_ID PADDLE_TRAINER_ENDPOINTS DISTRIBUTED_TRAINER_ENDPOINTS FLAGS_START_PORT PADDLE_ELASTIC_TIMEOUT
      5. 等待端口释放（sleep 5）
    note: |
      **此步骤为强制步骤，不可跳过。**
      当从减层阶段（pruned）恢复到全量阶段（full）时，减层训练进程可能仍在后台运行。
      若不减层进程未清理，全量训练启动后将产生以下问题：
      - 日志文件冲突（旧进程持续写入 workerlog.0）
      - 进程误判（监控逻辑将旧进程误认为新启动的全量训练）
      - XPU 设备/端口占用导致全量训练启动失败
      执行此步骤后，必须验证相关进程已完全终止，方可继续后续全量训练监控。

  step_6_yaml_restore:
    description: "YAML 联动恢复（仅在 mode=full 且提供了 xpu_yaml_path 时执行）"
    condition: "mode == 'full' AND xpu_yaml_path is not null AND yaml_restore_overrides is not empty"
    action: |
      1. 读取 xpu_yaml_path 指定的 YAML 文件
      2. 遍历 yaml_restore_overrides 中的每个字段：
         - 字段名: yaml 中的字段名（支持嵌套字段用 '.' 分隔，如 performance.recompute_num_layers）
         - 目标值: 恢复后的原始值
      3. 使用 Edit 工具修改 YAML 文件中的对应字段值
      4. 对于 yaml_keep_as_is 中列出的字段：
         - 检查当前值是否与原始 GPU YAML 一致
         - 如一致，保持当前值不变（这些是环境限制类修改，不应恢复）
         - 在报告中标记为 [环境限制-保持]
      5. 验证 YAML 语法正确性（使用 python yaml.safe_load）
      6. 生成 YAML 恢复报告
    note: |
      YAML 恢复仅处理因减层引入的联动修改（如 recompute_num_layers）。
      环境限制类修改（如 pipeline_model_parallel_size、using_sonic_moe）由 yaml_keep_as_is 保护，不应恢复。
```

---

## 返回格式

```json
{
  "status": "Success | Fail",
  "mode": "extreme_fast | fast | balanced | full",
  "backup_path": "config.json.bak.xxx",
  "components_modified": [
    {
      "field_path": "text_config.num_hidden_layers",
      "original": 36,
      "target": 6,
      "weight_prefix": "model.language_model.layers.*",
      "action": "modified | skipped | not_found"
    }
  ],
  "summary": {
    "total_components_found": 2,
    "components_modified": 1,
    "components_skipped": 1,
    "reason_skipped": "auto mode: non-text component kept unchanged"
  },
  "restore_command": "prune-model-layers <model_path> --mode full",

  "yaml_restore": {
    "status": "Success | Fail | NotRequired",
    "yaml_path": "/path/to/xpu_config.yaml",
    "fields_restored": [
      {
        "field": "recompute_num_layers",
        "from": 7,
        "to": 11,
        "reason": "减层联动恢复：全量模型 chunk_size 恢复后需同步提升 recompute 层数"
      }
    ],
    "fields_kept": [
      {
        "field": "pipeline_model_parallel_size",
        "value": 1,
        "reason": "环境限制：8 卡 world_size 不匹配 PP=4"
      }
    ],
    "failure_summary": ""
  }
}
```

---

## 调用方式

```bash
# 默认：极限快速，自动探测，只减主干
prune-model-layers <model_path>

# 指定模式
prune-model-layers <model_path> --mode fast
prune-model-layers <model_path> --mode balanced
prune-model-layers <model_path> --mode full          # 一键还原

# 指定组件范围
prune-model-layers <model_path> --mode extreme_fast --component all
prune-model-layers <model_path> --component text_config.num_hidden_layers
```

---

## 判定准则

| 场景 | 处理逻辑 |
|---|---|
| **绝对下限** | 任何组件保留不少于 4 层。Transformer 至少需 3~4 层堆叠才能验证 Attention、FFN、Residual、Norm 的连通性 |
| **小模型保护** | 原层数 <= 12 时，extreme_fast 退化为 fast（//3），避免过度裁剪 |
| **单组件 vs 多组件** | 默认只减主干（auto），防止视觉/音频等侧误裁剪导致特征维度不匹配 |
| **Config 缺失保护** | 权重有层 key 但 config 无对应字段时，跳过该组件并提示，不硬改 |
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
    action: "返回 Fail，提示检查路径"

  index_not_found:
    condition: "权重索引文件不存在"
    action: "返回 Fail，提示无法分析权重结构"

  no_layer_detected:
    condition: "config 和权重索引中均未探测到层结构"
    action: "返回 Fail，提示该模型可能不支持减层"

  backup_conflict:
    condition: "已存在同名备份"
    action: "跳过备份，继续执行"

  restore_no_backup:
    condition: "--mode full 但备份不存在"
    action: "返回 Fail，提示无法还原"

  yaml_not_found:
    condition: "提供了 xpu_yaml_path 但文件不存在"
    action: "config.json 仍正常恢复；yaml_restore.status = Fail，提示检查路径"

  yaml_restore_field_not_found:
    condition: "yaml_restore_overrides 中的字段在 YAML 中不存在"
    action: "跳过该字段，继续恢复其他字段；在 failure_summary 中提示"

  yaml_syntax_error_after_restore:
    condition: "恢复后 YAML 语法验证失败"
    action: "返回 Fail，建议手动检查 YAML 文件"
```
