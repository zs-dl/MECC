"""Fixed empirical inputs; values are zero-based public-domain indices."""
import gzip
from pathlib import Path
import numpy as np
from .reports import validate_values


def synthetic(n, d, distribution="zipf", seed=7, exponent=1.2):
    if not isinstance(n, (int, np.integer)) or n < 1 or d < 2:
        raise ValueError("n must be positive and d at least two")
    if distribution == "zipf":
        if not np.isfinite(exponent) or exponent <= 0:
            raise ValueError("Zipf exponent must be positive and finite")
        weights = np.arange(1, d+1, dtype=float)**(-exponent)
    elif distribution == "uniform":
        weights = np.ones(d, dtype=float)
    else:
        raise ValueError("distribution must be zipf or uniform")
    cumulative = np.cumsum(weights/weights.sum())
    cumulative[-1] = 1.0
    rng = np.random.Generator(np.random.PCG64(seed))
    return np.searchsorted(cumulative, rng.random(n), side="right").astype(np.int64)


def movielens_first(source):
    """Read extracted MovieLens 1M ratings.dat; earliest (time, movie) per user."""
    first = {}
    with Path(source).open(encoding="ascii") as stream:
        for line in stream:
            if not line.strip():
                continue
            user, movie, _, timestamp = map(int, line.strip().split("::"))
            candidate = (timestamp, movie)
            if user not in first or candidate < first[user]:
                first[user] = candidate
    values = np.array([first[u][1]-1 for u in sorted(first)], dtype=np.int64)
    return validate_values(values, 3952)


def kosarak_clicks(source, n=1024, seed=1069001):
    """Sample expanded clicks without replacement and retain original file order."""
    source = Path(source)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="ascii") as stream:
        clicks = np.fromiter((int(token)-1 for line in stream for token in line.split()), dtype=np.int64)
    validate_values(clicks, 41270)
    if not 1 <= n <= len(clicks):
        raise ValueError("n must lie between one and the number of clicks")
    indices = np.random.Generator(np.random.PCG64(seed)).choice(len(clicks), size=n, replace=False)
    return clicks[np.sort(indices)]
