#!/usr/bin/env python3
"""Plot the optimized prior variance lambda during PAC-Bayes training.

Parses the ``iter ...`` lines that pacbayes.train_pacbayes prints every
``--log-every`` iterations and draws lambda and j = b log(c/lambda) against
iteration.  Also writes the parsed values as CSV.

The paper constrains the optimization to lambda in (0, c) (Eq. 4), and the
union bound only covers the grid lambda = c exp(-j/b) for j in N, i.e. j >= 1
(lambda <= c exp(-1/b)).  The horizontal reference lines mark those boundaries;
a curve that presses against them (or, with --no-lambda-constraint, crosses
them) is the point of the figure.

usage:
    python scripts/plot_lambda.py logs/T-*.log --out lambda.pdf --csv lambda.csv
"""
import argparse
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = 100.0                                  # pk level of precision (Sec. 3.1)
C = 0.1                                    # upper bound on lambda (Sec. 3.1)
LAMBDA_MAX = C * math.exp(-1.0 / B)        # j = 1, smallest admissible lambda

LINE = re.compile(
    r"^iter\s+(?P<it>\d+) \|.*\| lambda (?P<lam>[\d.eE+-]+) \| j (?P<j>[-\d.]+)")


def parse(path):
    """Return [(iteration, lambda, j), ...] for one training log."""
    out = []
    with open(path) as f:
        for line in f:
            m = LINE.match(line)
            if m:
                out.append((int(m.group("it")), float(m.group("lam")), float(m.group("j"))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="training logs written by train_pacbayes")
    ap.add_argument("--out", default="lambda.pdf", help="figure to write")
    ap.add_argument("--csv", default=None, help="optional csv of the parsed values")
    args = ap.parse_args()

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6.0, 4.6))
    rows = []
    for path in args.logs:
        name = os.path.basename(path).replace(".log", "")
        tr = parse(path)
        if not tr:
            print("no lambda trajectory in %s (resumed from a finished checkpoint?)" % path)
            continue
        it = [r[0] for r in tr]
        lam = [r[1] for r in tr]
        j = [r[2] for r in tr]
        ax1.plot(it, lam, lw=1, label=name)
        ax2.plot(it, j, lw=1, label=name)
        rows += [(name, a, b, c) for a, b, c in tr]

    ax1.axhline(C, color="k", ls=":", lw=1, label=r"$c=0.1$")
    ax1.axhline(LAMBDA_MAX, color="r", ls="--", lw=1, label=r"$c\,e^{-1/b}$ ($j=1$)")
    ax1.set_yscale("log")
    ax1.set_ylabel(r"$\lambda$")
    ax1.legend(fontsize=6, ncol=2, loc="lower right")

    ax2.axhline(1.0, color="r", ls="--", lw=1, label=r"$j=1$")
    ax2.axhline(0.0, color="k", ls=":", lw=1)
    ax2.set_ylabel(r"$j = b\log(c/\lambda)$")
    ax2.set_xlabel("iteration")

    fig.tight_layout()
    fig.savefig(args.out)
    print("wrote", args.out)

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("experiment,iteration,lambda,j\n")
            for name, it, lam, j in rows:
                f.write("%s,%d,%.6e,%.3f\n" % (name, it, lam, j))
        print("wrote", args.csv)


if __name__ == "__main__":
    main()
