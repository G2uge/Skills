#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/root/paddlejob/Gruge/Gruge_env"
REPO_DIR="${BASE_DIR}/PaddleFleet"

cd "${BASE_DIR}"

if [ ! -d "paddle" ]; then
  echo "[ERROR] Missing virtualenv directory: ${BASE_DIR}/paddle"
  exit 1
fi

source paddle/bin/activate

export http_proxy=http://agent.baidu.com:8891
export https_proxy=$http_proxy
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com,paddle-ci.gz.bcebos.com

python -m pip install uv

if [ ! -d "${REPO_DIR}" ]; then
  git clone https://github.com/PaddlePaddle/PaddleFleet.git
else
  cd "${REPO_DIR}"
  git pull
  cd "${BASE_DIR}"
fi

cd "${REPO_DIR}"
python -m uv pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple -v --no-build-isolation

echo "[INFO] PaddleFleet installed successfully"