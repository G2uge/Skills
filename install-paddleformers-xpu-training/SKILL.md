---
name: install-paddleformer-xpu-train
description: Set up PaddleFormer large model training environment on XPU under Ubuntu. Requires Python 3.10, virtualenv activation, Paddle-XPU, PaddleFleet, and proxy configuration.
---

# Install PaddleFormer XPU large model training environment

This skill installs **PaddleFormer** as part of the required **XPU large model training environment**.

## Input Parameters

```yaml
inputs:
  VENV_ACTIVATE: "/root/paddlejob/tmp/paddle/bin/activate"  # 虚拟环境激活脚本路径
  REPOS_DIR: "/root/paddlejob/tmp/repos"                    # 代码仓库存放目录
  FORMERS_REPO_URL: "https://github.com/PaddlePaddle/PaddleFormers.git"  # PaddleFormers 仓库地址
  FORMERS_BRANCH: "main"                                    # 分支名
```

## Purpose

This installation is part of the required environment setup for:

- PaddleFormer large model training on XPU
- XPU-side training dependency preparation
- source-level debugging and development
- training environment validation

This is **not optional**.

If this setup is incomplete, the **PaddleFormer XPU training environment is not ready**.

---

## Hard dependency requirements

This skill assumes the current environment already provides the following required dependencies:

- **Python 3.10**
- **Paddle-XPU**
- **PaddleFleet**
- **virtualenv-managed Python environment**

PaddleFormer must be installed into the **same active Python environment** as these dependencies.

---

## Hard environment requirements

This skill assumes and requires all of the following:

- OS: **Ubuntu**
- Python version: **3.10**
- Python environment is managed by **virtualenv**
- Required activation command:

```bash
source ${VENV_ACTIVATE}
```

- Working directory: `${REPOS_DIR}`
- PaddleFormer repo must be managed under `${REPOS_DIR}`
- Proxy must be configured before clone/install

This skill does not create the Python environment.
It installs PaddleFormer into the already prepared XPU training environment.

## Required install location

All repository management must happen under:
```bash
${REPOS_DIR}
```

Expected repo path after clone:
```bash
${REPOS_DIR}/PaddleFormers
```

## Required preconditions

Before installation, the following conditions must already be satisfied.

### 1. Enter repos directory
```bash
cd ${REPOS_DIR}
```

### 2. Activate required Python environment
Run exactly:
```bash
source ${VENV_ACTIVATE}
```

### 3. Configure required proxy
Run exactly:
```bash
export http_proxy=http://agent.baidu.com:8891
export https_proxy=$http_proxy
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com,paddle-ci.gz.bcebos.com
```

If any of these are not true, stop and fix the environment first.

## Step 1: Validate current XPU training environment

Check current working directory:
```bash
pwd
ls
```

Expected:
- current directory is ${REPOS_DIR}
- PaddleFormers/ may or may not exist

Check Python interpreter and version:
```bash
which python
python --version
```

Expected:
- Python should come from the activated paddle virtualenv
- Python version should be 3.10.x
- Python should not be system Python

Check proxy:
```bash
echo $http_proxy
echo $https_proxy
echo $no_proxy
```

Expected:
- http_proxy=http://agent.baidu.com:8891
- https_proxy=http://agent.baidu.com:8891

If validation fails, do not continue installation.

## Step 2: Validate required dependencies

PaddleFormer training on XPU requires the following packages to already exist in the same active environment:
- Paddle-XPU
- PaddleFleet

Check Paddle-related packages:
```bash
python -m pip list | grep -i paddle
```

Check Fleet-related packages:
```bash
python -m pip list | grep -i fleet
```

Optional import verification:
```bash
python -c "import paddle; print('paddle import success')"
python -c "import paddlefleet; print('paddlefleet import success')"
```

If these dependencies are missing, install them first before continuing.

This skill assumes:
- paddlepaddle-xpu is already installed
- PaddleFleet is already installed

## Step 3: Clone or Update PaddleFormer repo

Enter the repos directory:
```bash
cd ${REPOS_DIR}
```

Check if PaddleFormers repo exists:
```bash
if [ ! -d "PaddleFormers" ]; then
    echo "PaddleFormers not found, cloning..."
    git clone ${FORMERS_REPO_URL}
    cd PaddleFormers
    git checkout ${FORMERS_BRANCH}
else
    echo "PaddleFormers exists, updating..."
    cd PaddleFormers
    git checkout ${FORMERS_BRANCH}
    git pull
fi
```

## Step 4: Install PaddleFormer

Inside the PaddleFormer repo root, run:
```bash
python -m pip install -e .
```

Meaning:
- -e . → editable install
- local source changes take effect immediately
- suitable for training-side debugging and source-level development

This installation is done inside the active XPU training environment.

## Step 5: Verify PaddleFormer installation

Check package metadata:
```bash
python -m pip list | grep -i formers
python -m pip show PaddleFormers
```

Try import verification:
```bash
python -c "import paddlenlp; print('PaddleFormer related package import success')"
```

If the import path differs from package naming, inspect the repo package structure.

## Required operational rule

For any future XPU training shell session, always do the following before using PaddleFormer:

```bash
cd ${REPOS_DIR}
source ${VENV_ACTIVATE}

export http_proxy=http://agent.baidu.com:8891
export https_proxy=$http_proxy
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com,paddle-ci.gz.bcebos.com
```

Then enter repo if needed:
```bash
cd PaddleFormers
```

PaddleFormer is considered valid only under this shell context.

## One-shot install flow

Use this exact sequence for a clean PaddleFormer XPU training install:

```bash
cd ${REPOS_DIR}
source ${VENV_ACTIVATE}

export http_proxy=http://agent.baidu.com:8891
export https_proxy=$http_proxy
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com,paddle-ci.gz.bcebos.com

# Clone or update PaddleFormers
cd ${REPOS_DIR}
if [ ! -d "PaddleFormers" ]; then
    git clone ${FORMERS_REPO_URL}
    cd PaddleFormers
    git checkout ${FORMERS_BRANCH}
else
    cd PaddleFormers
    git checkout ${FORMERS_BRANCH}
    git pull
fi

python -m pip install -e .
```

## Failure handling

Installation failure means training environment is incomplete.

If any of the following fails:
- virtualenv activation
- Python version validation
- proxy configuration
- dependency validation (paddle-xpu, paddlefleet)
- repo clone
- editable install

Then the PaddleFormer XPU training environment must be treated as incomplete.

Do not continue with:
- model training
- training script validation
- runtime testing
- dependency verification
- source debugging

Until installation succeeds.

## Common issues

### 1. Wrong Python interpreter

Check:
```bash
which python
python --version
```

Expected:
```bash
Python is from the activated paddle env
Python version is 3.10.x
```

If not, re-enter environment:
```bash
cd ${REPOS_DIR}
source ${VENV_ACTIVATE}

export http_proxy=http://agent.baidu.com:8891
export https_proxy=$http_proxy
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com,paddle-ci.gz.bcebos.com
```

### 2. Clone timeout / network failure

This environment requires proxy.
Re-check:
```bash
echo $http_proxy
echo $https_proxy
echo $no_proxy
```

If missing, re-export:
```bash
export http_proxy=http://agent.baidu.com:8891
export https_proxy=$http_proxy
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com,paddle-ci.gz.bcebos.com
```

Then retry clone/install.

### 3. pip install -e . fails

Possible causes:
- missing upstream dependency
- Paddle-XPU not installed
- PaddleFleet not installed
- wrong Python environment

Check:
```bash
python -m pip list | grep -E "paddle|fleet|formers"
which python
python -c "import sys; print(sys.executable)"
```

### 4. Editable install succeeds but import fails

Usually this means:
- wrong shell environment
- wrong interpreter
- install went into another environment

Check:
```bash
which python
python -c "import sys; print(sys.executable)"
python -m pip list | grep -i formers
```

### 5. Dependency installation failure

If Python package installation fails, the agent must not stop immediately.

#### Recovery steps

1. Retry installation using the Tsinghua PyPI mirror:
```bash
python -m pip install <package> -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 60 --retries 2
```

2. If both the primary and fallback mirrors fail:
- print a clear failure message
- stop execution with non-zero exit status

3. If installation succeeds:
- run a lightweight import check when possible

#### Example

```bash
python -m pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 60 --retries 2
python -c "import sklearn; print(sklearn.__version__)"
```

Required behavior:
- Do not silently ignore installation failures.
- Do not continue subsequent setup steps after dependency installation failure.
- Always use fallback mirror retry before final failure.

#### Expected result

After success:
- Required Python 3.10 XPU training environment is active
- Proxy is configured
- Paddle-XPU is available
- PaddleFleet is available
- PaddleFormer source is cloned locally
- PaddleFormer is installed into the same active environment
- PaddleFormer XPU large model training dependency preparation is complete
