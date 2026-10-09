#!/usr/bin/env bash
# One-shot UCloud run.
#
#   git clone -b tobias/lambda-plot https://github.com/GustavMH/pacbayes-opt.git
#   cd pacbayes-opt
#   bash scripts/ucloud_run.sh
#
# Sets up the environment, generates the controlled unconstrained lambda run for
# T-600^2, and writes lambda_tf_vs_torch.pdf.
#
# /work is wiped when the job ends, so DOWNLOAD these before it stops:
#   lambda_tf_vs_torch.pdf
#   logs/T-600-600-nolambda.log
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

# --- environment -------------------------------------------------------------
if [ ! -d .venv ]; then
    python3 -m venv .venv || {
        sudo apt-get update
        sudo apt-get install -y python3.12-venv
        python3 -m venv .venv
    }
fi
source .venv/bin/activate
pip install -q --upgrade pip
# default CPU wheel; for the B200 use: TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash scripts/ucloud_run.sh
pip install -q torch --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
pip install -q matplotlib
unset CUDA_VISIBLE_DEVICES || true

python3 -c "import torch, matplotlib; print('torch', torch.__version__, 'matplotlib', matplotlib.__version__)"

# --- T-600^2: SGD, then unconstrained PAC-Bayes (the controlled crossing) -----
mkdir -p logs
python -u -m pacbayes.train_sgd --hidden 600 600 --seed 11 --out-dir results
python -u -m pacbayes.train_pacbayes \
    --sgd results/sgd_T-600-600.pt \
    --no-lambda-constraint --seed 11 \
    --out-dir results-nolambda 2>&1 | tee logs/T-600-600-nolambda.log

# --- figure: TF2 + constrained curves come from git, unconstrained from above --
bash plot_lambda_tf.sh

echo
echo "======================================================================"
echo "DONE. Download these from the file browser before the job ends:"
echo "  lambda_tf_vs_torch.pdf"
ls -1 logs/*nolambda* 2>/dev/null || true
echo "======================================================================"
