#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/root/paddlejob/Gruge/Gruge_env"
REPO_DIR="${BASE_DIR}/PaddleFormers"

echo "[INFO] Installing PaddleFormer for XPU large model training"

cd "${BASE_DIR}"

# Validate required virtualenv
if [ ! -d "paddle" ]; then
  echo "[ERROR] Required virtualenv directory 'paddle' not found in ${BASE_DIR}"
  echo "[ERROR] XPU training environment is not prepared."
  exit 1
fi

# Required activation
source /root/paddlejob/Gruge/Gruge_env/paddle/bin/activate

# Required proxy
export http_proxy=http://agent.baidu.com:8891
export https_proxy=$http_proxy
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com,paddle-ci.gz.bcebos.com

echo "[INFO] Validating environment"
echo "[INFO] pwd=$(pwd)"
echo "[INFO] python=$(which python)"
echo "[INFO] python_version=$(python --version 2>&1)"
echo "[INFO] http_proxy=${http_proxy}"
echo "[INFO] https_proxy=${https_proxy}"
echo "[INFO] no_proxy=${no_proxy}"

echo "[INFO] Checking required dependencies"
python -m pip list | grep -i paddle || true
python -m pip list | grep -i fleet || true

if [ ! -d "${REPO_DIR}" ]; then
  echo "[INFO] Cloning PaddleFormers"
  git clone https://github.com/PaddlePaddle/PaddleFormers.git
else
  echo "[INFO] Updating existing PaddleFormers repo"
  cd "${REPO_DIR}"
  git pull
  cd "${BASE_DIR}"
fi

cd "${REPO_DIR}"
python -m pip install -e .

echo "[INFO] PaddleFormer installed successfully"
echo "[INFO] XPU large model training environment dependency is ready"