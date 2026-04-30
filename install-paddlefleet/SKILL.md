---
name: install-paddlefleet
description: Install PaddleFleet into the required virtualenv under specified directory for PaddleFormer XPU large model training on Ubuntu.
---

# Install PaddleFleet

This skill installs **PaddleFleet** as a required dependency for **PaddleFormer XPU large model training**.

## Input Parameters

```yaml
inputs:
  VENV_ACTIVATE: "/root/paddlejob/tmp/paddle/bin/activate"  # 虚拟环境激活脚本路径
  REPOS_DIR: "/root/paddlejob/tmp/repos"                    # 代码仓库存放目录
  FLEET_REPO_URL: "https://github.com/PaddlePaddle/PaddleFleet.git"  # PaddleFleet 仓库地址
  FLEET_BRANCH: "develop"                                   # 分支名
```

## Hard requirements

This skill assumes:

- OS: Ubuntu
- Working directory: `${REPOS_DIR}`
- Existing virtualenv activation script: `${VENV_ACTIVATE}`
- Required activation command:

```bash id="prkh0g"
source ${VENV_ACTIVATE}
```

- Required proxy configuration (fixed):
```bash
export http_proxy=http://agent.baidu.com:8891
export https_proxy=$http_proxy
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com,paddle-ci.gz.bcebos.com
```

This installation is required for:

- XPU training
- Paddle runtime on XPU
- PaddleFormer training dependency setup

## Step 1: Enter repos directory and activate virtualenv

Enter the repos directory:
```bash
cd ${REPOS_DIR}
```

Activate the paddle virtualenv:
```bash
source ${VENV_ACTIVATE}
```

Verify:
```bash
which python
python --version
```
Expected:

- Python should point to the activated virtualenv
- You should not be using system Python
If this is not true, stop and fix the environment before continuing.

## Step 2: Configure required proxy

This installation requires proxy configuration.
Run exactly:
```bash
export http_proxy=http://agent.baidu.com:8891
export https_proxy=$http_proxy
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com,paddle-ci.gz.bcebos.com
```
Verify:
```bash
echo $http_proxy
echo $https_proxy
echo $no_proxy
```
Expected:
- http_proxy=http://agent.baidu.com:8891
- https_proxy=http://agent.baidu.com:8891

## Step 3: Install uv inside the activated virtualenv

Install uv into the current environment:
```bash
python -m pip install uv
```
Verify:
```bash
python -m uv --help
```

## Step 4: Clone or Update PaddleFleet

Enter the repos directory:
```bash
cd ${REPOS_DIR}
```

Check if PaddleFleet repo exists:
```bash
if [ ! -d "PaddleFleet" ]; then
    echo "PaddleFleet not found, cloning..."
    git clone ${FLEET_REPO_URL}
    cd PaddleFleet
    git checkout ${FLEET_BRANCH}
else
    echo "PaddleFleet exists, updating..."
    cd PaddleFleet
    git checkout ${FLEET_BRANCH}
    git pull
fi
```

## Step 5: Install PaddleFleet in editable mode

Inside the PaddleFleet repo root, run:
```bash
python -m uv pip install -e . -v --no-build-isolation
```
Equivalent command form:
```bash
uv pip install -e . -v --no-build-isolation
```
### Meaning
- -e . → editable install
- -v → verbose logs
- --no-build-isolation → use current environment dependencies directly

This is recommended for:
- local source debugging
- development iteration
- immediate effect after source code changes

## One-shot install flow

If the environment already exists, the full install flow is:
```bash
cd ${REPOS_DIR}
source ${VENV_ACTIVATE}

export http_proxy=http://agent.baidu.com:8891
export https_proxy=$http_proxy
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com,paddle-ci.gz.bcebos.com

python -m pip install uv

# Clone or update PaddleFleet
cd ${REPOS_DIR}
if [ ! -d "PaddleFleet" ]; then
    git clone ${FLEET_REPO_URL}
    cd PaddleFleet
    git checkout ${FLEET_BRANCH}
else
    cd PaddleFleet
    git checkout ${FLEET_BRANCH}
    git pull
fi

python -m uv pip install -e . -v --no-build-isolation
```

## Verify installation

Check current Python:
```bash
which python
python --version
```
Check package installation:
```bash
python -m pip list | grep -i fleet
python -m pip show PaddleFleet
```
Try import verification:
```bash
python -c "import paddlefleet; print('PaddleFleet import success')"
```
If the import path differs from the package name, inspect the repo package structure.

## Required usage rule

Before using PaddleFleet in a new shell session, always run:
```bash
cd ${REPOS_DIR}
source ${VENV_ACTIVATE}

export http_proxy=http://agent.baidu.com:8891
export https_proxy=$http_proxy
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com,paddle-ci.gz.bcebos.com
```
Then enter repo if needed:
```bash
cd PaddleFleet
```

## Common issues
### 1. Wrong Python interpreter
Check:
```bash
which python
python --version
```
If it is using system Python, you probably forgot:
```bash
source ${VENV_ACTIVATE}
```

### 2. uv not found
Install it again inside the activated environment:
```bash
python -m pip install uv
```
Then use:
```bash
python -m uv --help
```

### 3. Clone / install timeout or network failure
This environment requires proxy. Re-check:
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
Then retry:
```bash
cd ${REPOS_DIR}
git clone ${FLEET_REPO_URL}
```
or:
```bash
python -m uv pip install -e .-i https://pypi.tuna.tsinghua.edu.cn/simple -v --no-build-isolation
```

### 4. Editable install succeeds but import fails
Usually this means:
- wrong environment activated
- install happened in a different Python env
- shell session changed after install

Check:
```bash
which python
python -c "import sys; print(sys.executable)"
python -m pip list | grep -i fleet
```

### Expected result
After success:
- Paddle virtualenv is active
- Proxy is configured
- PaddleFleet source is cloned locally
- PaddleFleet is installed in editable mode
- Local source changes take effect immediately
