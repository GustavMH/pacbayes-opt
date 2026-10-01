"""Sec. 3, 3.1, 3.2, 4.3: minimize Eq. (4) over the posterior mean w, 1/2 log s and 1/2 log lambda,
starting at the SGD solution, with prior N(w0, lambda I).

Run e.g.: python -m pacbayes.train_pacbayes --sgd results/sgd_T-600.pt
"""
import argparse
import math
import os
import time
import torch
from pacbayes.bounds import B, C, DELTA
from pacbayes.data import load_binary_mnist
from pacbayes.networks import forward, logistic_loss
from pacbayes.utils import get_device, set_seed, experiment_name, save_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sgd", required=True, help="file written by pacbayes.train_sgd")
    parser.add_argument("--iters1", type=int, default=150000, help="iterations at --lr1 (Sec. 4.3)")
    parser.add_argument("--lr1", type=float, default=1e-3)
    parser.add_argument("--iters2", type=int, default=50000, help="iterations at --lr2 (Sec. 4.3)")
    parser.add_argument("--lr2", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--s-init", choices=["paper", "alg1", "code"], default="paper",
                        help="paper: s = |w| (Sec. 4.3); alg1: log std = |w| (Algorithm 1); code: std = 2|w| (authors' code)")
    parser.add_argument("--s-scale", type=float, default=1.0,
                        help="multiplies s for --s-init paper: 1 for true labels, 0.1 for random labels (Sec. 4.3)")
    parser.add_argument("--rho-lambda-init", type=float, default=-3.0,
                        help="initial 1/2 log lambda; -3 gives lambda = e^-6 (Sec. 4.3, Algorithm 1)")
    parser.add_argument("--no-lambda-constraint", action="store_true", help="let lambda exceed c")
    parser.add_argument("--rms-alpha", type=float, default=0.9, help="RMSprop decay (Sec. 4.3)")
    parser.add_argument("--rms-eps", type=float, default=1e-8, help="not stated in the paper")
    parser.add_argument("--log-every", type=int, default=550)
    parser.add_argument("--checkpoint-every", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--data-dir", default="mnist")
    parser.add_argument("--out-dir", default="results")
    return parser.parse_args()


def initial_log_stds(w_sgd, s_init, s_scale):
    if s_init == "paper":
        return [0.5 * torch.log(s_scale * w.abs() + 1e-16) for w in w_sgd]
    if s_init == "alg1":
        return [w.abs() for w in w_sgd]
    return [torch.log(2 * w.abs() + 1e-16) for w in w_sgd]


def kl_to_prior(means, log_stds, prior_means, rho_lambda):
    """Sec. 3.1 (below Eq. 5), summed over all parameters
    s = exp(2 log_std), lambda = exp(2 rho_lambda)."""
    d = sum(w.numel() for w in means)
    lam = torch.exp(2 * rho_lambda.double())
    s_sum = sum(torch.exp(2 * r.double()).sum() for r in log_stds)
    dist2 = sum(((w.double() - p.double()) ** 2).sum() for w, p in zip(means, prior_means))
    log_s_sum = sum((2 * r.double()).sum() for r in log_stds)
    return 0.5 * (s_sum / lam - d + dist2 / lam + d * torch.log(lam) - log_s_sum)


def log_j(rho_lambda, lambda_constraint):
    """log of j = b log(c / lambda), treated as continuous during optimization in Sec. 3.1."""
    inner = math.log(C) - 2 * rho_lambda.double()
    if not lambda_constraint:
        inner = torch.clamp(inner, min=0.01)   # as in the authors' code (snn/core/network.py, line 242)
    return torch.log(B * inner)


def main():
    args = parse_args()
    device = get_device()
    set_seed(args.seed)
    sgd = torch.load(args.sgd)
    name = experiment_name(sgd["hidden"], sgd["random_labels"])
    print("PAC-Bayes | %s | %s | %s" % (name, device, vars(args)))
    lambda_constraint = not args.no_lambda_constraint

    train_x, train_y, _, _ = load_binary_mnist(args.data_dir, sgd["random_labels"], sgd["label_seed"], device)
    m = train_x.shape[0]
    w_sgd = [w.to(device) for w in sgd["w_sgd"]]
    prior_means = [w.to(device) for w in sgd["w0"]]

    means = [torch.nn.Parameter(w.clone()) for w in w_sgd]
    log_stds = [torch.nn.Parameter(r.clone()) for r in initial_log_stds(w_sgd, args.s_init, args.s_scale)]
    rho_lambda = torch.nn.Parameter(torch.tensor(args.rho_lambda_init, device=device))
    parameters = means + log_stds + [rho_lambda]

    optimizer = torch.optim.RMSprop(parameters, lr=args.lr1, alpha=args.rms_alpha, eps=args.rms_eps)
    log_confidence = math.log(math.pi ** 2 * m / (6 * DELTA))
    rho_lambda_max = 0.5 * (math.log(C) - 1.0 / B)   # lambda = c exp(-1/b), i.e. j = 1

    os.makedirs(args.out_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.out_dir, "pacbayes_%s_checkpoint.pt" % name)
    start_iter = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        with torch.no_grad():
            for p, value in zip(parameters, checkpoint["parameters"]):
                p.copy_(value.to(device))
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iter = checkpoint["iteration"]
        set_seed(args.seed + start_iter)
        print("Resumed from %s at iteration %d" % (checkpoint_path, start_iter))

    total_iters = args.iters1 + args.iters2
    shuffle_generator = torch.Generator().manual_seed(args.seed + start_iter)
    permutation, position = torch.randperm(m, generator=shuffle_generator).to(device), 0
    start_time = time.time()
    for it in range(start_iter, total_iters):
        lr = args.lr1 if it < args.iters1 else args.lr2
        for group in optimizer.param_groups:
            group["lr"] = lr
        if position + args.batch_size > m:
            permutation, position = torch.randperm(m, generator=shuffle_generator).to(device), 0
        batch = permutation[position:position + args.batch_size]
        position += args.batch_size

        weights = [w + torch.exp(r) * torch.randn_like(w) for w, r in zip(means, log_stds)] # Sec. 3.2
        surrogate = logistic_loss(forward(weights, train_x[batch]), train_y[batch])
        kl = kl_to_prior(means, log_stds, prior_means, rho_lambda)
        bre = (kl + 2 * log_j(rho_lambda, lambda_constraint) + log_confidence) / (m - 1) # Eq. (5)
        objective = surrogate + torch.sqrt(bre / 2) # Eq. (4)

        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
        if lambda_constraint:
            with torch.no_grad():
                rho_lambda.clamp_(max=rho_lambda_max) # lambda in (0, c)

        if it % args.log_every == 0 or it == total_iters - 1:
            lam = math.exp(2 * rho_lambda.item())
            print("iter %6d | lr %.0e | surrogate %.4f | KL %.1f | lambda %.3e | j %.1f | sqrt(BRE/2) %.4f | "
                  "objective %.4f | %.0f s" % (it, lr, surrogate.item(), kl.item(), lam, B * math.log(C / lam),
                                               math.sqrt(bre.item() / 2), objective.item(), time.time() - start_time),
                  flush=True)
        if (it + 1) % args.checkpoint_every == 0:
            torch.save({"parameters": [p.detach().cpu() for p in parameters], "optimizer": optimizer.state_dict(),
                        "iteration": it + 1}, checkpoint_path)

    torch.save({"means": [w.detach().cpu() for w in means], "log_stds": [r.detach().cpu() for r in log_stds],
                "rho_lambda": rho_lambda.item(), "prior_means": [p.cpu() for p in prior_means],
                "hidden": sgd["hidden"], "random_labels": sgd["random_labels"], "label_seed": sgd["label_seed"],
                "args": vars(args)}, os.path.join(args.out_dir, "posterior_%s.pt" % name))
    save_json({"experiment": name, "iterations": total_iters, "seconds": time.time() - start_time,
               "rho_lambda": rho_lambda.item(), "args": vars(args)},
              os.path.join(args.out_dir, "pacbayes_%s.json" % name))
    print("Saved", os.path.join(args.out_dir, "posterior_%s.pt" % name))


if __name__ == "__main__":
    main()
