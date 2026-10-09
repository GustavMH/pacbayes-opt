#!/usr/bin/env bash
# Plot lambda(t) for the old TF 2.21 implementation vs the PyTorch implementation.
# Run from the pacbayes-opt repo root on UCloud.
#
#   bash plot_lambda_tf.sh
#
# Produces lambda_tf_vs_torch.pdf in the current directory.
set -euo pipefail

# 1. pull the TF2 log off the tf2 branch into logs/
git fetch origin tf2
git show origin/tf2:logs/pacb_600_readme.log > logs/pacb_600_tf2.log

# 2. plot both log formats on one log-scale axis
python - <<'PYEOF'
import math
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# PyTorch logs: "iter N | ... | lambda X | ..."   (550 iterations = 1 epoch)
TP = re.compile(r"^iter\s+(\d+) \|.*\| lambda ([\d.eE+-]+) \|")


def read_torch(path):
    xs, ys = [], []
    for line in open(path):
        m = TP.match(line)
        if m:
            xs.append(int(m.group(1)) / 550.0)
            ys.append(float(m.group(2)))
    return xs, ys


# TF2 log: "Epoch:0001 ... log_prior_std: -3.0000 ..."   -> lambda = exp(2 * rho)
TF = re.compile(r"Epoch:(\d+).*log_prior_std:\s*([-0-9.]+)")


def read_tf(path):
    xs, ys = [], []
    for line in open(path):
        m = TF.search(line)
        if m:
            xs.append(int(m.group(1)))
            ys.append(math.exp(2.0 * float(m.group(2))))
    return xs, ys


curves = [
    ("logs/pacb_600_tf2.log",       "authors' code (TF2.21, T-600)",   read_tf),
    ("logs/T-600-nolambda.log",     "PyTorch unconstrained (T-600)",   read_torch),
    ("logs/T-600-600.log",          "PyTorch constrained (T-600^2)",   read_torch),
    ("logs/T-600-600-nolambda.log", "PyTorch unconstrained (T-600^2)", read_torch),
]

fig, ax = plt.subplots(figsize=(6.0, 3.6))
for path, label, reader in curves:
    try:
        xs, ys = reader(path)
    except FileNotFoundError:
        print("skip (missing):", path)
        continue
    if not xs:
        print("skip (no rows):", path)
        continue
    ax.plot(xs, ys, lw=1.2, label=label)

ax.axhline(0.1, color="k", ls=":", lw=1, label=r"$c=0.1$")
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel(r"$\lambda$")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig("lambda_tf_vs_torch.pdf")
print("wrote lambda_tf_vs_torch.pdf")
PYEOF
