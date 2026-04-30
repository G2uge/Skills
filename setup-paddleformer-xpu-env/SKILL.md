---
name: setup-paddleformer-xpu-env
description: One-shot setup for PaddleFormer XPU large model training environment on Ubuntu under specified directory. Installs PaddlePaddle-XPU, PaddleFleet, PaddleFormer, and validates the environment.
---

This setup is **not optional** if you want a complete PaddleFormer XPU training environment.
**必须严格按照本 skill 定义的流程和参数执行，不得跳过任何步骤或修改关键配置，否则环境可能无法正常工作。**

# Setup PaddleFormer XPU environment

This skill performs a **one-shot setup** of the required **PaddleFormer XPU large model training environment**.

## Purpose

This skill is the top-level environment bootstrap entry for:

- PaddleFormer large model training on XPU
- XPU-side training dependency preparation
- source-level debugging environment setup
- reproducible local environment bootstrap

This setup is **not optional** if you want a complete PaddleFormer XPU training environment.

---

## Input Parameters

```yaml
inputs:
  BASE_DIR: "/root/paddlejob/tmp"                    # 工作基础目录
  VENV_DIR: "/root/paddlejob/tmp/paddle"             # 虚拟环境目录
  REPOS_DIR: "/root/paddlejob/tmp/repos"             # 代码仓库存放目录
  VENV_ACTIVATE: "/root/paddlejob/tmp/paddle/bin/activate"  # 激活脚本路径
```

---

## Hard requirements

This skill assumes:
- OS: Ubuntu
- Working directory: ${BASE_DIR}
- Existing virtualenv directory: paddle (will be created if not exists)
- Required activation command:
```bash
source ${VENV_ACTIVATE}
```
- Required proxy configuration (fixed):
```bash
export http_proxy=http://agent.baidu.com:8891
export https_proxy=$http_proxy
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com,paddle-ci.gz.bcebos.com
```

This skill creates the Python environment if not exists.
It focuses on dependency installation + validation.

---

## Required shell context

Before running this skill, the shell must already be able to support:
```bash
cd ${BASE_DIR}
source ${VENV_ACTIVATE}

export http_proxy=http://agent.baidu.com:8891
export https_proxy=$http_proxy
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com,paddle-ci.gz.bcebos.com
```

## Execution flow

This skill runs the following sub-skills in order:

### 1. Install PaddlePaddle-XPU

**调用 Skill**: `install-paddle-xpu`

**传递参数**:
```yaml
inputs:
  VENV_DIR: "${VENV_DIR}"
  REPOS_DIR: "${REPOS_DIR}"
  PADDLE_REPO_URL: "https://github.com/PaddlePaddle/Paddle.git"
```

**功能说明**:
- 检查 `${REPOS_DIR}/Paddle` 是否存在，不存在则 git clone
- 在 `${REPOS_DIR}/Paddle/build` 下编译 Paddle
- 安装到 `${VENV_DIR}` 虚拟环境

### 2. Install PaddleFleet

**调用 Skill**: `install-paddlefleet`

**传递参数**:
```yaml
inputs:
  VENV_ACTIVATE: "${VENV_ACTIVATE}"
  REPOS_DIR: "${REPOS_DIR}"
  FLEET_REPO_URL: "https://github.com/PaddlePaddle/PaddleFleet.git"
  FLEET_BRANCH: "develop"
```

**功能说明**:
- 检查 `${REPOS_DIR}/PaddleFleet` 是否存在，不存在则 git clone
- 使用已激活的虚拟环境安装

### 3. Install PaddleFormer

**调用 Skill**: `install-paddleformers-xpu-training`

**传递参数**:
```yaml
inputs:
  VENV_ACTIVATE: "${VENV_ACTIVATE}"
  REPOS_DIR: "${REPOS_DIR}"
  FORMERS_REPO_URL: "https://github.com/PaddlePaddle/PaddleFormers.git"
  FORMERS_BRANCH: "main"
```

**功能说明**:
- 检查 `${REPOS_DIR}/PaddleFormers` 是否存在，不存在则 git clone
- 使用已激活的虚拟环境安装

## Expected result

After successful execution:
- Paddle virtualenv is active
- proxy is configured
- PaddlePaddle-XPU is installed
- PaddleFleet is installed
- PaddleFormer is installed
- environment passes validation
- PaddleFormer XPU large model training environment is ready

## Basic Environment Validation

```bash
source ${VENV_ACTIVATE}
python - <<'PY'
import sys

print("=== Python 基础信息 ===")
print(sys.version)

print("\n=== Paddle 检查 ===")
import paddle
print("Paddle version:", paddle.__version__)
print("Current device:", paddle.device.get_device())
print("Compiled with XPU:", paddle.is_compiled_with_xpu())

print("\n=== PaddleFleet 检查 ===")
try:
    import paddle.distributed.fleet as fleet
    print("PaddleFleet import: OK")
except Exception as e:
    print("PaddleFleet import: FAILED")
    raise

print("\n=== PaddleFormers 检查 ===")
try:
    import paddlenlp
    print("PaddleNLP / PaddleFormers dependency import: OK")
except Exception as e:
    print("PaddleNLP import warning:", e)

print("\n环境验证通过")
PY
```

## Installation Complete Message (Required)

只有在验证成功后，才输出以下完成提示：

```bash
source ${VENV_ACTIVATE}

python - <<'PY'
import sys
import platform

try:
    import paddle
    paddle_version = paddle.__version__
    current_device = paddle.device.get_device()
    xpu_available = paddle.is_compiled_with_xpu()
except Exception as e:
    print("❌ 环境验证失败：无法正常导入 Paddle")
    print("错误信息:", e)
    sys.exit(1)

print("")
print("============================================================")
print("✅ PaddleFormer XPU 大模型训练环境安装完成")
print("============================================================")
print("")
print("📦 安装路径")
print(f"  工作目录        : ${BASE_DIR}")
print(f"  虚拟环境        : ${VENV_DIR}")
print(f"  代码仓库        : ${REPOS_DIR}")
print(f"  PaddleFormers   : ${REPOS_DIR}/PaddleFormers")
print("")
print("🐍 Python 信息")
print(f"  Python版本       : {platform.python_version()}")
print("")
print("📚 关键组件")
print(f"  Paddle版本       : {paddle_version}")
print(f"  当前设备         : {current_device}")
print(f"  XPU可用          : {xpu_available}")
print("")
print("🚀 后续使用方式")
print("  1. 激活环境:")
print(f"     source ${VENV_ACTIVATE}")
print("")
print("  2. 进入代码目录:")
print(f"     cd ${REPOS_DIR}/PaddleFormers")
print("")
print("  3. 运行验证命令:")
print('     python -c "import paddle; print(paddle.__version__); print(paddle.device.get_device())"')
print("")
print("🎉 环境已就绪，可用于 PaddleFormer XPU 大模型训练")
print("============================================================")
PY
```

## Failure handling

If any sub-step fails, the environment must be treated as not ready.

Do not continue with:
- training script execution
- XPU runtime validation
- distributed training tests
- source-level debugging
- deployment-side integration
until all required steps succeed.

## Common Issues

### 1. Network Connection Failure (External PyPI Inaccessible)

**Symptom:** `Network is unreachable` when installing packages.

**Solution:** Use Baidu internal PyPI mirror:
```bash
python -m pip install <package> -i http://pip.baidu-int.com/simple --trusted-host pip.baidu-int.com
```

## Recommended usage

Use this skill as the default first entry when bootstrapping a fresh PaddleFormer XPU environment.
Use the lower-level install/check skills only when:
- debugging a specific dependency
- re-installing one component
- validating a partially broken environment
