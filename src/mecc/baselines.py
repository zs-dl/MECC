"""Zero-inclusive finite power-law Calibrate variants on private OUE counts."""
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import logsumexp


@lru_cache(maxsize=64)
def public_power_prior(n, d):
    k = np.arange(n + 1, dtype=float)
    logk = np.log1p(k)

    def prior(alpha):
        z = -alpha * logk
        return np.exp(z - logsumexp(z))

    target = n / d
    if d == 2:
        alpha = 0.0
    else:
        upper = 2.0
        while np.dot(prior(upper), k) > target:
            upper *= 2.0
        alpha = brentq(lambda a: np.dot(prior(a), k) - target,
                       0.0, upper, xtol=1e-12)
    return alpha, prior(alpha)


def calibrate_pl0(counts, n, epsilon, simplex_projection, *, include_mle=True):
    """Full finite integer sums; shared Gaussian noise as in Calibrate.

    The shifted prior (k+1)^(-alpha), inclusion of zero, finite upper bound n,
    and optional final projection are explicit adaptations to this task.
    MLE searches one dimension using a fixed public grid and local refinements.
    Public-mean uses the exact mean n/d as a stronger parametric control.
    """
    counts = np.asarray(counts, dtype=np.int64)
    d = counts.size
    q = 1.0 / (np.exp(epsilon) + 1.0)
    gap = 0.5 - q
    sigma2 = n * q * (1.0 - q) / gap**2
    s, inv, multiplicity = np.unique(counts, return_inverse=True, return_counts=True)
    y = (s - n * q) / gap
    k = np.arange(n + 1, dtype=float)
    logk = np.log1p(k)
    log_likelihood = -0.5 * (y[:, None] - k[None, :])**2 / sigma2
    row_shift = np.max(log_likelihood, axis=1)
    likelihood = np.exp(log_likelihood - row_shift[:, None])
    h = multiplicity / d

    def posterior(weights):
        den = likelihood @ weights
        if np.any(den <= 0):
            raise FloatingPointError("Calibrate mixture likelihood underflow")
        return ((likelihood @ (weights * k)) / den)[inv] / n

    def weights_for(alpha):
        z = -float(alpha) * logk
        return np.exp(z - logsumexp(z))

    result = {}
    public_alpha, public_weights = public_power_prior(n, d)
    raw = posterior(public_weights)
    result["calibrate_pl0_public"] = (simplex_projection(raw), {
        "alpha": public_alpha, "prior_mean": float(public_weights @ k),
        "pre_projection": raw, "pre_projection_mass": float(raw.sum()),
        "noise_count_variance": sigma2, "support": [0, n],
        "parameter_rule": "exact_public_mean_n_over_d",
        "adaptation": "zero-inclusive shifted power law; full integer sums; simplex",
    })
    if include_mle:
        evaluations = []

        def objective(alpha):
            weights = weights_for(alpha)
            den = likelihood @ weights
            if np.any(den <= 0):
                return float("inf")
            value = -float(h @ (np.log(den) + row_shift))
            evaluations.append((float(alpha), value))
            return value

        grid = np.r_[np.linspace(0, 4, 17), [5, 6, 8, 12, 20, 32]]
        vals = np.array([objective(a) for a in grid])
        choices = [(float(vals[i]), float(grid[i])) for i in range(len(grid))]
        # Refine every sampled local minimum, including endpoint intervals.
        for i in range(len(grid)):
            if (i == 0 or vals[i] <= vals[i-1]) and (i == len(grid)-1 or vals[i] <= vals[i+1]):
                lo, hi = grid[max(0, i-1)], grid[min(len(grid)-1, i+1)]
                if hi > lo:
                    fit = minimize_scalar(objective, bounds=(lo, hi), method="bounded",
                                          options={"xatol": 1e-7, "maxiter": 100})
                    if not fit.success:
                        raise RuntimeError("Calibrate scalar fit failed: " + str(fit.message))
                    choices.append((float(fit.fun), float(fit.x)))
        best_value, alpha = min(choices)
        zero_value = -float(h @ log_likelihood[:, 0])
        if zero_value < best_value:
            # alpha=infinity is the delta_0 boundary. Save JSON-safe identity.
            alpha, best_value = None, zero_value
            raw = np.zeros(d)
            prior_mean = 0.0
        else:
            w = weights_for(alpha)
            raw, prior_mean = posterior(w), float(w @ k)
        result["calibrate_pl0_mle"] = (simplex_projection(raw), {
            "alpha": alpha, "alpha_boundary": "infinity" if alpha is None else None,
            "prior_mean": prior_mean, "pre_projection": raw,
            "pre_projection_mass": float(raw.sum()), "noise_count_variance": sigma2,
            "objective": best_value, "optimization_evaluations": evaluations,
            "alpha_search_grid": grid, "support": [0, n],
            "parameter_rule": "privatized_report_marginal_likelihood",
            "adaptation": "zero-inclusive shifted power law; full integer sums; simplex",
        })
    return result
