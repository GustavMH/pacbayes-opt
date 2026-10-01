#!/usr/bin/env bash
# All rows of Table 1, one after another. For parallel runs, start scripts/run_experiment.sh in separate jobs instead.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
for hidden in "600" "1200" "300 300" "600 600" "1200 1200" "600 600 600"; do
    name="T-${hidden// /-}"
    bash scripts/run_experiment.sh $hidden 2>&1 | tee "logs/$name.log"
done
bash scripts/run_experiment.sh --random-labels 600 2>&1 | tee logs/R-600.log
