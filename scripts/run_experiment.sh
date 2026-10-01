#!/usr/bin/env bash
# Runs all three stages for one row of Table 1.
# scripts/run_experiment.sh 600 -> T-600
# scripts/run_experiment.sh 600 600 -> T-600-600 (T-600^2 in the paper)
# scripts/run_experiment.sh --random-labels 600 -> R-600
# Extra settings for all stages can be passed through the environment, e.g. OUT_DIR=results_seed12 SEED=12.
set -euo pipefail
cd "$(dirname "$0")/.."

random_labels=false
if [[ "${1:-}" == "--random-labels" ]]; then
    random_labels=true
    shift
fi
if [[ $# -eq 0 ]]; then
    echo "usage: $0 [--random-labels] HIDDEN [HIDDEN ...]" >&2
    exit 1
fi

out_dir="${OUT_DIR:-results}"
seed="${SEED:-11}"
prefix=$([[ "$random_labels" == true ]] && echo R || echo T)
name="${prefix}-$(IFS=-; echo "$*")"

if [[ "$random_labels" == true ]]; then
    # Sec. 4.2-4.3: 120 SGD epochs, s = |w|/10, lr 1e-4 for 500,000 iterations
    python -u -m pacbayes.train_sgd --hidden "$@" --random-labels --epochs 120 --seed "$seed" --out-dir "$out_dir"
    python -u -m pacbayes.train_pacbayes --sgd "$out_dir/sgd_$name.pt" --s-scale 0.1 --lr1 1e-4 --iters1 500000 \
        --iters2 0 --seed "$seed" --out-dir "$out_dir"
else
    python -u -m pacbayes.train_sgd --hidden "$@" --seed "$seed" --out-dir "$out_dir"
    python -u -m pacbayes.train_pacbayes --sgd "$out_dir/sgd_$name.pt" --seed "$seed" --out-dir "$out_dir"
fi
python -u -m pacbayes.evaluate --posterior "$out_dir/posterior_$name.pt" --seed "$seed" --out-dir "$out_dir"
