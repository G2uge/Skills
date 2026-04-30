---
name: validate-and-fix-training-config
description: Validate and automatically fix Paddle training YAML config and run.sh launch script for XPU training.
---
```

# Skill: Validate & Fix Paddle Training Config

## Purpose

This skill validates and **auto-corrects**:

* training YAML config
* launch script (run.sh)

It ensures:

* config is runnable
* script is executable
* XPU training requirements are satisfied

---

## When to Use

Use this skill when:

* config.yaml 无法运行
* run.sh 启动失败
* 模型刚接入（first-time run）
* 不确定并行参数是否正确
* 需要自动修复配置

---

## Inputs

```yaml
config_yaml: <path>
launch_script: <path>
hardware: xpu
```

---

## Outputs

### 成功（可能包含修复）

```text
[OK] Config validated.
[OK] Launch script validated.
[OK] 单机
[FIXED] Missing fields added.
[FIXED] Invalid values corrected.
[OK] ready to run.
```

### 失败

```text
[FAILED] Stage: Config Validation
Reason: <具体问题>
Suggested Fix:
1. ...
```

---

# 校验与修复规则

---

## 1. Config 文件检查与修复

### Rule 1 — Dataset Path 必须存在且路径正确
检查：

```yaml
train_dataset_path
eval_dataset_path
```
### 错误
#### 1.没有配置路径
example:

```yaml
train_dataset_path: ""
```

#### 2.配置的路径不对
example:
```yaml
train_dataset_path: data/train.jsonl
eval_dataset_path: data/val.jsonl
```
检查路径是否正确。

#### 修复策略
根据用户提供的模型数据集路径和配置文件，例如：
```yaml
train.jsonl
val.jsonl
```
根据用户提供的数据路径，找到对应的配置文件，并在yaml文件中修改成正确的路径。


### 异常情况
1. 找不到对应的数据文件
2. 不同的数据集，可能有相同的文件名，但实际内容不一样。
返回：
```text
→ FAIL（不能自动猜）
```
---

## 2. Model Path 检查

```yaml
model_name_or_path
```

### 校验：
1. 模型文件存在，并且路径正确

### 错误
1. 没有配置模型路径
example：
```yaml
model_name_or_path: ""
```
2. 配置错误的模型路径
example:
```yaml
model_name_or_path: data/Qwen3-VL-30B-A3B-Thinking
```
当前文件夹在当前路径找不到。

### 修复策略
根据用户提供的模型路径，查找Qwen3-VL-30B-A3B-Thinking
```
可以找到，按照查找的路径在对应的yaml文件中修改成正确的路径。

### 异常情况
1. 找不到模型文件
返回：
```text
→ FAIL（不能自动猜）
```
---

## 3. Device 强制修复

### 必须：

```yaml
device: xpu
```

### 错误

```yaml
device: gpu
```

### 自动修复

```yaml
device: xpu
```

---

## 4. 并行参数校验

```yaml
tensor_model_parallel_size
expert_model_parallel_size
pipeline_model_parallel_size
```

---

### 多机设备判断

```text
required_device_count =
  expert_model_parallel_size
× tensor_model_parallel_size
× pipeline_model_parallel_size
```

---

### 校验规则

| 条件       | 行为      |
| -------- | ------- |
| = 8      | ✅ 单机    |
| < 8      | ❌ FAIL |
| > 8 且无多机 | ✅ 多机  |

---
单机 和 多机 判断结果作为输出。

## run.sh 检查与修复

---

### 必须包含项

#### ✅ 必须存在

```bash
ulimit -c unlimited
```

缺失 → 自动添加（顶部）

---

#### 环境变量补全

必须存在：
1. 环境变量
```bash
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
```
2. xpu配置信
```bash
export PADDLE_TRAINERS_NUM=1
export BKCL_TIMEOUT=1000
export BKCL_SOCKET_IFNAME=eth0
export BKCL_ENABLE_XDR=1
export BKCL_FORCE_RDMA_NICS_ORDER=eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4
export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export BKCL_DEEPEP_NORMAL_CLUSTER_NUM=8

export XSHMEM_MODE=1
export XSHMEM_QP_NUM_PER_RANK=24
export BKCL_USE_AR=1
export BKCL_RING_OPT=1
export BKCL_USE_RDMA=1
export BKCL_FORCE_L3_RDMA=0
export BKCL_RDMA_VERBS=1
```

缺失 → 自动补全

---

#### Runtime 激活检查

必须存在：

```bash
source paddle/bin/activate
```
在/root/paddlejob/tmp查找paddle文件，根据实际查找的路径配置该执行路径。如果没有找到直接返回：
```text
[FAILED] Stage: Runtime Activation
Reason: Paddle runtime not activated
```

---

#### 训练命令校验

必须符合：

```bash
paddleformers-cli train xxx.yaml
```
### 异常情况
返回：
```text
→ FAIL（不能自动猜）并提示用户去给出正确的配置
```
---
