"""Simplex projection and controlled finite-binomial summation."""
import numpy as np
from scipy.stats import binom


def simplex_projection(vector, mass=1.0):
    """Euclidean projection onto nonnegative vectors summing to ``mass``."""
    y = np.asarray(vector, dtype=float)
    if y.ndim != 1 or y.size == 0 or not np.all(np.isfinite(y)):
        raise ValueError('Projection needs a nonempty finite vector')
    if not np.isfinite(mass) or mass < 0:
        raise ValueError('Projection mass must be finite and nonnegative')
    if mass == 0:
        return np.zeros_like(y)
    ordered = np.sort(y)[::-1]
    excess = np.cumsum(ordered) - mass
    active = ordered - excess / np.arange(1, y.size + 1) > 0
    rho = np.flatnonzero(active)[-1]
    threshold = excess[rho] / (rho + 1)
    return np.maximum(y - threshold, 0.0)


def _pmf_segment(m, probability, tail_tolerance):
    if m == 0:
        return 0, np.ones(1), 0.0
    if m <= 128 or tail_tolerance == 0:
        lo, hi = 0, m
    else:
        lo = int(binom.ppf(tail_tolerance / 4, m, probability))
        hi = int(binom.ppf(1 - tail_tolerance / 4, m, probability))
    index = np.arange(lo, hi + 1)
    omitted = float(binom.cdf(lo - 1, m, probability) + binom.sf(hi, m, probability))
    return lo, binom.pmf(index, m, probability), omitted
