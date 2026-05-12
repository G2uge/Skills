---
name: install-paddle-xpu
description: Install PaddlePaddle-XPU into the required Python 3.10 virtualenv under specified directory for PaddleFormer XPU large model training on Ubuntu.
---

> **执行约束**：执行者必须严格遵循本 skill 定义的调用顺序，不得擅自添加前置检查或跳过逻辑。


# Install PaddlePaddle-XPU

This skill installs **PaddlePaddle-XPU** as a required dependency for **PaddleFormer XPU large model training**.

## Input Parameters

```yaml
inputs:
  VENV_DIR: "/root/paddlejob/tmp/paddle"                    # 虚拟环境目录
  REPOS_DIR: "/root/paddlejob/tmp/repos"                    # Paddle 源码仓库存放目录
  PADDLE_REPO_URL: "https://github.com/PaddlePaddle/Paddle.git"  # Paddle 仓库地址
```

## Hard requirements

This skill assumes:

- OS: Ubuntu
- Working directory: `${VENV_DIR}`
- Target virtualenv directory: `${VENV_DIR}`
- Required activation command:

```bash id="4t7gxn"
source ${VENV_DIR}/bin/activate
```

- Required proxy configuration (fixed):
```bash
export http_proxy=http://agent.baidu.com:8891
export https_proxy=$http_proxy
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com,paddle-ci.gz.bcebos.com
```

## Hard Constraints
### This skill MUST NOT be used when ANY of the following are true:
- The machine is **macOS**
- The machine is **Windows**
- The environment is **non-Ubuntu Linux** unless the user explicitly confirms compatibility
## Required OS Verification

Before doing anything else, verify the OS:

```bash
uname -a
cat /etc/os-release
```
Expected result should clearly indicate Ubuntu.

## Prerequisites

Make sure the machine has:

- `python3.10`
- `virtualenv`
- Network access to the Paddle XPU nightly package index

Optional but recommended:
- XPU runtime environment already installed
---

## Workspace Setup

### Step 0: Clone or Update Paddle Source Code

Enter the repos directory:
```bash
cd ${REPOS_DIR}
```

Check if Paddle repo exists:
```bash
if [ ! -d "Paddle" ]; then
    echo "Paddle source not found, cloning..."
    git clone ${PADDLE_REPO_URL}
    cd Paddle
else
    echo "Paddle source exists, updating..."
    cd Paddle
    git pull
fi
```

### Step 1: Create Virtual Environment

Create and activate a dedicated Python environment:
```bash
mkdir -p ${VENV_DIR}
cd ${VENV_DIR}
virtualenv -p python3.10 .
source bin/activate
```
Verify Python path and version:
```bash
which python
python --version
```
Expected output should look similar to:
```bash
${VENV_DIR}/bin/python
Python 3.10.x
```

### Step 2: Install Paddle XPU Dependencies

Upgrade pip first:
```bash
python -m pip install --upgrade pip
```

Then build and install paddlepaddle-xpu from source:
```bash
cd ${REPOS_DIR}/Paddle

# 【关键】确保第三方子模块已初始化
git submodule update --init --recursive

# 【关键】清理可能损坏的 XPU SDK 缓存
if [ -d "build/third_party/xpu/src/extern_xpu" ]; then
  if find "build/third_party/xpu/src/extern_xpu" -name "*.tar.gz" -size -1k | grep -q .; then
    echo "[WARN] Found corrupted XPU SDK cache, cleaning..."
    rm -rf build/third_party/xpu
  fi
fi

mkdir -p build && cd build

export {http,https}_proxy=http://10.63.229.53:8891
cmake .. -DPY_VERSION=3.10 \
-DCMAKE_BUILD_TYPE=Release \
-DWITH_GPU=OFF \
-DWITH_XPU=ON \
-DON_INFER=OFF \
-DWITH_PYTHON=ON \
-DWITH_XPU_XRE5=ON \
-DWITH_MKL=OFF \
-DWITH_XPU_BKCL=ON \
-DWITH_TESTING=OFF \
-DWITH_XCCL_RDMA=ON \
-DWITH_XPU_XHPC=ON \
-DBUILD_WHL_PACKAGE=ON \
-DWITH_DISTRIBUTE=ON \
-DARCH_BIN_CONTAINS_90=1

# 编译并保存日志
BUILD_LOG="build_$(date +%Y%m%d_%H%M%S).log"
make -j$(nproc) TARGET=HASWELL 2>&1 | tee -a "${BUILD_LOG}"

python -m pip install python/dist/paddlepaddle-*.whl -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 120 --retries 3
```

### Step 3: Verify Installation

Run the following checks one by one.
#### 3.1 Check Paddle version
```bash
python -c "import paddle; paddle.version.show()"
```
This verifies:
Paddle is installed correctly
Version and build metadata are visible
#### 3.2 Run Paddle built-in environment check
```bash
python -c "import paddle; paddle.utils.run_check()"
```
This verifies:
- Basic Paddle runtime is working
- Core environment is usable
#### 3.3 Check unified marker import
```bash
python -c "from paddle.jit.marker import unified"
```
This verifies:
- JIT marker-related modules are available
- Some graph / compiler-related paths are intact

## One-shot Setup Script

If you want a quick executable version, save this as:
```bash
install_paddle_xpu.sh
```
Then use:
```bash
bash install_paddle_xpu.sh
```

Script content:
```bash
#!/bin/bash
set -euo pipefail

REPOS_DIR="${REPOS_DIR:-/root/paddlejob/tmp/repos}"
VENV_DIR="${VENV_DIR:-/root/paddlejob/Gruge/Gruge_env/paddle}"
PADDLE_REPO_URL="${PADDLE_REPO_URL:-https://github.com/PaddlePaddle/Paddle.git}"

echo "==> Clone or update Paddle source"
cd ${REPOS_DIR}
if [ ! -d "Paddle" ]; then
    git clone --recursive ${PADDLE_REPO_URL} Paddle
    cd Paddle
else
    cd Paddle
    git pull
fi

echo "==> Initialize git submodules"
git submodule update --init --recursive

echo "==> Clean corrupted XPU cache if any"
if [ -d "build/third_party/xpu/src/extern_xpu" ]; then
  if find "build/third_party/xpu/src/extern_xpu" -name "*.tar.gz" -size -1k | grep -q .; then
    echo "[WARN] Found corrupted XPU SDK cache, cleaning..."
    rm -rf build/third_party/xpu
  fi
fi

echo "==> Enter virtualenv workspace"
mkdir -p ${VENV_DIR}
cd ${VENV_DIR}

echo "==> Create virtualenv"
virtualenv -p python3.10 .
source bin/activate

echo "==> Upgrade pip"
python -m pip install --upgrade pip

echo "==> Build and install paddlepaddle-xpu"
cd ${REPOS_DIR}/Paddle/build
export {http,https}_proxy=http://10.63.229.53:8891
cmake .. -DPY_VERSION=3.10 \
-DCMAKE_BUILD_TYPE=Release \
-DWITH_GPU=OFF \
-DWITH_XPU=ON \
-DON_INFER=OFF \
-DWITH_PYTHON=ON \
-DWITH_XPU_XRE5=ON \
-DWITH_MKL=OFF \
-DWITH_XPU_BKCL=ON \
-DWITH_TESTING=OFF \
-DWITH_XCCL_RDMA=ON \
-DWITH_XPU_XHPC=ON \
-DBUILD_WHL_PACKAGE=ON \
-DWITH_DISTRIBUTE=ON \
-DARCH_BIN_CONTAINS_90=1

BUILD_LOG="build_$(date +%Y%m%d_%H%M%S).log"
make -j$(nproc) TARGET=HASWELL 2>&1 | tee -a "${BUILD_LOG}"

python -m pip install python/dist/paddlepaddle-*.whl \
  -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 120 --retries 3

echo "==> Check paddle version"
python -c "import paddle; paddle.version.show()"

echo "==> Run paddle check"
python -c "import paddle; paddle.utils.run_check()"

echo "==> Check unified marker"
python -c "from paddle.jit.marker import unified"

echo "==> Paddle XPU environment setup done."
```

## Troubleshooting
### 1. virtualenv: command not found
Install virtualenv:
```bash
python3.10 -m pip install virtualenv
```
or:
```bash
pip install virtualenv
```
### 2. python3.10: command not found
Check whether Python 3.10 exists:
```bash
which python3.10
python3.10 --version
```
If not, install Python 3.10 first.
### 3. No matching distribution found for paddlepaddle-xpu
Possible causes:

- Python version mismatch
- Unsupported platform / architecture
- Paddle nightly source unavailable
- The wheel for your environment is missing

### 4. If the Paddle nightly source cannot be reached, export proxy
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
Required behavior
- Do not silently ignore installation failures.
- Do not continue subsequent setup steps after dependency installation failure.
- Always use fallback mirror retry before final failure.

### 6. ThreadPool.h: No such file or directory
**原因**：`git submodule` 未完全初始化，第三方头文件缺失。  
**修复**：
```bash
cd ${REPOS_DIR}/Paddle
git submodule update --init --recursive
```

### 7. extern_xpu tar 解压失败 / gzip: trailing garbage ignored
**原因**：昆仑芯 XPU SDK 下载包损坏（网络波动导致）。  
**修复**：
```bash
rm -rf ${REPOS_DIR}/Paddle/build/third_party/xpu
cd ${REPOS_DIR}/Paddle/build
make -j$(nproc) TARGET=HASWELL
```

### 8. 编译长时间无输出，无法判断进度
**检查方法**：
```bash
# 查看实时对象文件增长
watch -n 5 'find ${REPOS_DIR}/Paddle/build -name "*.o" | wc -l'

# 查看后台进度监控日志
tail -f ${REPOS_DIR}/Paddle/build/build_*.log.progress
```

## Recommended Agent Behavior
When using this skill, the agent should:

- Verify python3.10 and virtualenv availability first.
- Use an isolated virtual environment instead of the system Python.
- Prefer mkdir -p tmp to avoid directory creation failures.
- Treat Paddle install success and FastDeploy op import success as separate checks.

## Success Criteria
A successful setup should pass all of the following:
```bash
python -c "import paddle; paddle.version.show()"
python -c "import paddle; paddle.utils.run_check()"
python -c "from paddle.jit.marker import unified"
```
If all commands succeed, the Paddle XPU environment is ready to use.
