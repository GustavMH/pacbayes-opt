#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

if [ ! -d .venv ]; then
    python3 -m venv .venv || {
        sudo apt-get update
        sudo apt-get install -y python3.12-venv
        python3 -m venv .venv
    }
fi
source .venv/bin/activate
pip install -q --upgrade pip
# default: GPU/B200 (Blackwell) wheel; override with
#   TORCH_INDEX=https://download.pytorch.org/whl/cpu bash scripts/ucloud_run.sh
# uninstall first so a CPU wheel already in the venv is not left in place
pip uninstall -y torch >/dev/null 2>&1 || true
pip install -q torch --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
pip install -q matplotlib
# UCloud sometimes exports CUDA_VISIBLE_DEVICES as empty, which hides the GPU
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python3 -c "import torch, matplotlib; print('torch', torch.__version__, 'cuda_build', torch.version.cuda, 'cuda_available', torch.cuda.is_available(), 'n_gpu', torch.cuda.device_count(), 'matplotlib', matplotlib.__version__)"

mkdir -p logs
python -u -m pacbayes.train_sgd --hidden 600 600 --seed 11 --out-dir results
python -u -m pacbayes.train_pacbayes \
    --sgd results/sgd_T-600-600.pt \
    --no-lambda-constraint --seed 11 \
    --out-dir results-nolambda 2>&1 | tee logs/T-600-600-nolambda.log

bash plot_lambda_tf.sh

echo "DONE"
ls -1 logs/*nolambda* 2>/dev/null || true
