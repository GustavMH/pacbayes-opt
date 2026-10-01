import math
import numpy as np

DELTA = 0.025        # Sec. 3.1
DELTA_PRIME = 0.01   # Sec. 4.4
B = 100.0            # Sec. 3.1
C = 0.1              # Sec. 3.1, upper bound on lambda


def kl_bernoulli(q, p):
    """KL(q || p) between Bernoulli distributions (Sec. 2.1)."""
    t1 = 0.0 if q == 0 else q * math.log(q / p)
    t2 = 0.0 if q == 1 else (1 - q) * math.log((1 - q) / (1 - p))
    return t1 + t2


def kl_inverse(q, c, n_newton=5):
    """KL^{-1}(q | c) = sup{p in [0, 1] : KL(q || p) <= c} by Newton's method (App. A):
    starts at q + sqrt(c/2), return 1 if that is >= 1, otherwise take n_newton Newton steps."""
    p = q + math.sqrt(c / 2)
    if p >= 1:
        return 1.0
    for _ in range(n_newton):
        h = kl_bernoulli(q, p) - c
        h_prime = (1 - q) / (1 - p) - (q / p if q > 0 else 0.0)
        p = p - h / h_prime
    return p


def kl_inverse_bisect(q, c, tol=1e-12):
    """KL^{-1}(q|c) by bisection, used to check kl_inverse."""
    if kl_bernoulli(q, 1.0 - 1e-15) <= c:
        return 1.0
    lo, hi = q, 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if kl_bernoulli(q, mid) <= c:
            lo = mid
        else:
            hi = mid
    return lo


def kl_to_prior(means, log_stds, prior_means, lam):
    """KL(N(w, diag(s)) || N(w0, lam*I)) = 1/2 (||s||_1/lam - d + ||w - w0||^2/lam + d log lam - sum log s)
    (Sec. 3.1), with s = exp(2 * log_std)."""
    d = sum(w.size for w in means)
    s_sum = sum(np.exp(2.0 * r.astype(np.float64)).sum() for r in log_stds)
    dist2 = sum(((w.astype(np.float64) - p.astype(np.float64)) ** 2).sum() for w, p in zip(means, prior_means))
    log_s_sum = sum((2.0 * r.astype(np.float64)).sum() for r in log_stds)
    return 0.5 * (s_sum / lam - d + dist2 / lam + d * math.log(lam) - log_s_sum)


def bre(kl, j, m, delta=DELTA):
    """Eq. (5), where 2 log(b log(c/lambda)) = 2 log(j) for lambda = c exp(-j/b)."""
    return (kl + 2 * math.log(j) + math.log(math.pi ** 2 * m / (6 * delta))) / (m - 1)


def lambda_from_j(j):
    return C * math.exp(-j / B)


def j_from_lambda(lam):
    return B * math.log(C / lam)
