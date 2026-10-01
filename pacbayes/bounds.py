"""Constants and bound computations of the paper (Sec. 2.2, 3.1, 3.3, App. A)."""
import math
import torch

DELTA = 0.025        # confidence of the PAC-Bayes bound (Sec. 3.1)
DELTA_PRIME = 0.01   # confidence of the Monte Carlo estimate (Sec. 4.4)
B = 100.0            # level of precision of the discrete set lambda = c exp(-j/b) (Sec. 3.1)
C = 0.1              # upper bound on lambda (Sec. 3.1)


def kl_bernoulli(q, p):
    """KL(q || p) between two Bernoulli distributions (Sec. 2.1)."""
    t1 = 0.0 if q == 0 else q * math.log(q / p)
    t2 = 0.0 if q == 1 else (1 - q) * math.log((1 - q) / (1 - p))
    return t1 + t2


def kl_inverse(q, c, newton_steps=5):
    """KL^{-1}(q | c) with Newton's method, as in App. A."""
    p = q + math.sqrt(c / 2)
    if p >= 1:
        return 1.0
    for _ in range(newton_steps):
        h = kl_bernoulli(q, p) - c
        h_prime = (1 - q) / (1 - p) - (q / p if q > 0 else 0.0)
        p = p - h / h_prime
    return p


def kl_to_prior(means, log_stds, prior_means, lam):
    """KL(N(w, diag(s)) || N(w0, lam I)) from Sec. 3.1, with s = exp(2 log_std), summed over all parameters."""
    d = sum(w.numel() for w in means)
    s_sum = sum(torch.exp(2 * r.double()).sum() for r in log_stds)
    dist2 = sum(((w.double() - p.double()) ** 2).sum() for w, p in zip(means, prior_means))
    log_s_sum = sum((2 * r.double()).sum() for r in log_stds)
    return 0.5 * (s_sum / lam - d + dist2 / lam + d * torch.log(lam) - log_s_sum)


def bre(kl, j, m, delta=DELTA):
    """Eq. (5) for lambda = c exp(-j/b)."""
    return (kl + 2 * math.log(j) + math.log(math.pi ** 2 * m / (6 * delta))) / (m - 1)


def lambda_from_j(j):
    return C * math.exp(-j / B)