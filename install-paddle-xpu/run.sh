#!/usr/bin/env bash
set -euo pipefail

# ==================== 参数化路径 ====================
REPOS_DIR="${REPOS_DIR:-/root/paddlejob/tmp/repos}"
VENV_DIR="${VENV_DIR:-/root/paddlejob/Gruge/Gruge_env/paddle}"
PADDLE_REPO_URL="${PADDLE_REPO_URL:-https://github.com/PaddlePaddle/Paddle.git}"

# ==================== 代理配置（区分用途）====================
# 外网代理：用于 GitHub、pip 等
export http_proxy=http://agent.baidu.com:8891
export https_proxy=$http_proxy
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com,paddle-ci.gz.bcebos.com,icode.baidu.com,baidu-int.com

# 内网代理：用于 cmake 下载昆仑芯 SDK（XRE/XHPC 等）
CMAKE_PROXY_HOST="http://10.63.229.53:8891"

# ==================== 虚拟环境准备 ====================
mkdir -p "${VENV_DIR}"
cd "${VENV_DIR}"

if [ ! -f "bin/activate" ]; then
  echo "[INFO] Creating virtualenv in ${VENV_DIR}"
  virtualenv -p python3.10 .
fi
source bin/activate
python -m pip install --upgrade pip

# ==================== 源码克隆与 submodule 检查 ====================
mkdir -p "${REPOS_DIR}"
cd "${REPOS_DIR}"

if [ ! -d "Paddle" ]; then
  echo "[INFO] Cloning Paddle source..."
  git clone --recursive "${PADDLE_REPO_URL}" Paddle
  cd Paddle
else
  echo "[INFO] Updating Paddle source..."
  cd Paddle
  git pull
fi

# 【关键修复】检查并初始化所有第三方子模块
echo "[INFO] Checking git submodules..."
git submodule update --init --recursive

# ==================== extern_xpu 缓存清理（防止 tar 损坏）====================
XPU_CACHE_DIR="${REPOS_DIR}/Paddle/build/third_party/xpu/src/extern_xpu"
if [ -d "${XPU_CACHE_DIR}" ]; then
  # 检查是否有损坏的 tar.gz（大小为0或解压失败标记）
  if find "${XPU_CACHE_DIR}" -name "*.tar.gz" -size -1k | grep -q .; then
    echo "[WARN] Found corrupted XPU SDK cache, cleaning..."
    rm -rf "${REPOS_DIR}/Paddle/build/third_party/xpu"
  fi
fi

# ==================== CMake 配置 ====================
BUILD_DIR="${REPOS_DIR}/Paddle/build"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

export http_proxy="${CMAKE_PROXY_HOST}"
export https_proxy="${CMAKE_PROXY_HOST}"

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

# ==================== 编译（带日志与进度监控）====================
BUILD_LOG="${BUILD_DIR}/build_$(date +%Y%m%d_%H%M%S).log"
echo "[INFO] Starting build, log: ${BUILD_LOG}"

# 启动后台进度监控（每5分钟写入日志）
(
  while pgrep -f "make -j" > /dev/null 2>&1 || pgrep -f "setup.py bdist_wheel" > /dev/null 2>&1; do
    OBJ_COUNT=$(find "${BUILD_DIR}" -name '*.o' 2>/dev/null | wc -l)
    SO_COUNT=$(find "${BUILD_DIR}" -name '*.so' 2>/dev/null | wc -l)
    echo "[$(date '+%H:%M:%S')] Progress: ${OBJ_COUNT} object files, ${SO_COUNT} shared libraries"
    sleep 300
  done
  echo "[$(date '+%H:%M:%S')] Build process ended"
) >> "${BUILD_LOG}.progress" 2>&1 &

# 执行编译，同时使用 tee 保存完整日志
make -j$(nproc) TARGET=HASWELL 2>&1 | tee -a "${BUILD_LOG}"

# ==================== 安装 whl ====================
echo "[INFO] Installing wheel..."
python -m pip install python/dist/paddlepaddle-*.whl \
  -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 120 --retries 3

# ==================== 验证 ====================
echo "[INFO] Verifying installation..."
python -c "import paddle; paddle.version.show()"
python -c "import paddle; paddle.utils.run_check()"
python -c "from paddle.jit.marker import unified"

echo "[INFO] PaddlePaddle-XPU installed successfully"
