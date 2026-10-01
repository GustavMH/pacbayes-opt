#!/usr/bin/env bash
#  bash scripts/run_experiment.sh 600 -> T-600
#  bash scripts/run_experiment.sh 600 600 -> T-600-600 (T-600^2 in the paper)
#  bash scripts/run_experiment.sh --random-labels 600 -> R-600

set -euo pipefail
cd "$(dirname "$0")/.."

out_dir="${OUT_DIR:-results}"
seed="${SEED:-11}"

if [[ "${1:-}" == "--random-labels" ]]; then
    shift
    name="R-$(IFS=-; echo "$*")"
    # Sec. 4.2-4.3: 120 SGD epochs, lr 1e-4 for 500,000 iterations (s = |w|/10 is set automatically)
    python -u -m pacbayes.train_sgd --hidden "$@" --random-labels --epochs 120 --seed "$seed" --out-dir "$out_dir"
    python -u -m pacbayes.train_pacbayes --sgd "$out_dir/sgd_$name.pt" --lr1 1e-4 --iters1 500000 --iters2 0 \
        --seed "$seed" --out-dir "$out_dir"
else
    name="T-$(IFS=-; echo "$*")"
    python -u -m pacbayes.train_sgd --hidden "$@" --seed "$seed" --out-dir "$out_dir"
    python -u -m pacbayes.train_pacbayes --sgd "$out_dir/sgd_$name.pt" --seed "$seed" --out-dir "$out_dir"
fi

python -u -m pacbayes.evaluate --posterior "$out_dir/posterior_$name.pt" --seed "$seed" --out-dir "$out_dir"