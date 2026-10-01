"""Sec. 3.3, 4.4, Eq. 6
run as e.g.: python -m pacbayes.evaluate --posterior results/posterior_T-600.pt
"""
import argparse
import json
import math
import os
import time
import torch
from pacbayes.data import load_binary_mnist
from pacbayes.networks import forward, predict
from pacbayes.utils import get_device, set_seed, experiment_name, save_json
from pacbayes.bounds import B, C, DELTA, DELTA_PRIME, kl_inverse, kl_to_prior, bre, lambda_from_j


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--posterior", required=True, help="file written by pacbayes.train_pacbayes")
    parser.add_argument("--n-samples", type=int, default=150000, help="networks drawn from the posterior (Sec. 4.4)")
    parser.add_argument("--samples-per-batch", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=5500)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--data-dir", default="mnist")
    parser.add_argument("--out-dir", default="results")
    return parser.parse_args()


def lambda_candidates(posterior, m):
    """Round lambda down and up on the grid lambda = c exp(-j/b), j >= 1 (Sec. 3.3), with KL and BRE for each."""
    lam = math.exp(2 * posterior["rho_lambda"])
    j_star = B * math.log(C / lam)
    candidates = []
    for j in sorted({max(1, math.floor(j_star)), max(1, math.ceil(j_star))}):
        lam_j = lambda_from_j(j)
        kl = kl_to_prior(posterior["means"], posterior["log_stds"], posterior["prior_means"],
                         torch.tensor(lam_j, dtype=torch.float64)).item()
        candidates.append({"j": j, "lambda": lam_j, "KL": kl, "BRE": bre(kl, j, m)})
    return lam, j_star, candidates


def errors_of_samples(weights, x, y, chunk_size):
    """0-1 error of each of the K sampled networks in weights."""
    wrong = torch.zeros(weights[0].shape[0], dtype=torch.float64, device=x.device)
    with torch.no_grad():
        for start in range(0, x.shape[0], chunk_size):
            output = forward(weights, x[start:start + chunk_size])
            wrong += (predict(output) != y[start:start + chunk_size]).sum(dim=1).double()
    return wrong / x.shape[0]


def main():
    args = parse_args()
    device = get_device()
    torch.backends.cuda.matmul.allow_tf32 = False
    posterior = torch.load(args.posterior)
    name = experiment_name(posterior["hidden"], posterior["random_labels"])
    print("Evaluation | %s | %s | %s" % (name, device, vars(args)))

    train_x, train_y, test_x, test_y = load_binary_mnist(args.data_dir, posterior["random_labels"],
                                                         posterior["label_seed"], device)
    m = train_x.shape[0]
    lam, j_star, candidates = lambda_candidates(posterior, m)
    print("optimized lambda %.4e, j = %.3f, candidates %s" % (lam, j_star, candidates))

    # Progress is saved every minute, so an interrupted run continues where it stopped
    os.makedirs(args.out_dir, exist_ok=True)
    state_path = os.path.join(args.out_dir, "evaluate_%s_n%d_state.json" % (name, args.n_samples))
    state = {"done": 0, "sum_train_error": 0.0, "sum_test_error": 0.0, "seconds": 0.0}
    if os.path.exists(state_path):
        with open(state_path) as f:
            state.update(json.load(f))
        print("Resuming with %d samples done" % state["done"])
    set_seed(args.seed + state["done"])

    means = [w.to(device) for w in posterior["means"]]
    stds = [torch.exp(r.to(device)) for r in posterior["log_stds"]]
    last_time = last_save = time.time()
    while state["done"] < args.n_samples:
        k = min(args.samples_per_batch, args.n_samples - state["done"])
        weights = [w + s * torch.randn((k,) + w.shape, device=device) for w, s in zip(means, stds)]
        state["sum_train_error"] += errors_of_samples(weights, train_x, train_y, args.chunk_size).sum().item()
        state["sum_test_error"] += errors_of_samples(weights, test_x, test_y, args.chunk_size).sum().item()
        state["done"] += k
        if time.time() - last_save > 60 or state["done"] >= args.n_samples:
            state["seconds"] += time.time() - last_time
            last_time = last_save = time.time()
            save_json(state, state_path)
            rate = state["done"] / max(state["seconds"], 1e-9)
            print("%d / %d samples | mean SNN train error %.5f | %.1f samples/s | ETA %.1f min" % (
                state["done"], args.n_samples, state["sum_train_error"] / state["done"], rate,
                (args.n_samples - state["done"]) / rate / 60), flush=True)

    n = state["done"]
    train_error = state["sum_train_error"] / n
    test_error = state["sum_test_error"] / n
    c_samples = math.log(2 / DELTA_PRIME) / n   # Theorem 2.2
    train_error_bound = kl_inverse(train_error, c_samples)
    test_error_bound = kl_inverse(test_error, c_samples)
    for candidate in candidates:                # Eq. (6)
        candidate["bound"] = kl_inverse(train_error_bound, candidate["BRE"])
        candidate["sqrt_half_BRE"] = math.sqrt(candidate["BRE"] / 2)
    best = min(candidates, key=lambda c: c["bound"])

    result = {"experiment": name, "n_samples": n, "snn_train_error_estimate": train_error,
              "snn_test_error_estimate": test_error, "snn_train_error": train_error_bound,
              "snn_test_error": test_error_bound, "pac_bayes_bound": best["bound"], "kl_divergence": best["KL"],
              "sqrt_half_bre": best["sqrt_half_BRE"], "j": best["j"], "lambda": best["lambda"],
              "confidence": 1 - DELTA - DELTA_PRIME, "candidates": candidates, "seconds": state["seconds"],
              "args": vars(args)}
    save_json(result, os.path.join(args.out_dir, "evaluate_%s_n%d.json" % (name, n)))
    os.remove(state_path)   # finished: a later run starts fresh instead of reusing these samples
    print("\n===== %s Results to compare with Table 1 =====" % name)
    for key in ["snn_train_error", "snn_test_error", "pac_bayes_bound", "kl_divergence", "sqrt_half_bre", "j",
                "lambda", "n_samples", "confidence"]:
        print("%-16s %s" % (key, result[key]))


if __name__ == "__main__":
    main()