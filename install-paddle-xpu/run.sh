#!/usr/bin/env bash
set -euo pipefail

cd /root/paddlejob/Gruge/Gruge_env

if [ ! -d "paddle" ]; then
  echo "[INFO] Creating virtualenv directory: /root/paddlejob/Gruge/Gruge_env/paddle"
  mkdir -p paddle
fi

cd paddle

if [ ! -f "bin/activate" ]; then
  echo "[INFO] Creating virtualenv in /root/paddlejob/Gruge/Gruge_env/paddle"
  virtualenv -p python3.10 .
fi

source bin/activate

export http_proxy=http://agent.baidu.com:8891
export https_proxy=$http_proxy
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com,paddle-ci.gz.bcebos.com

cd /root/paddlejob/zhangxiao_dev/Paddle/build/
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

make -j$(nproc) TARGET=HASWELL

python -m pip install python/dist/paddlepaddle-*.whl -i https://pypi.tuna.tsinghua.edu.cn/simple 

echo "[INFO] PaddlePaddle-XPU installed successfully"