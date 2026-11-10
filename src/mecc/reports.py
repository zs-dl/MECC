"""Complete optimized unary encoding reports in row-major packed form."""
import math
import numpy as np


def validate_values(values, d):
    values = np.asarray(values)
    if not isinstance(d, (int, np.integer)) or isinstance(d, (bool, np.bool_)) or d < 2:
        raise ValueError("d must be an integer at least two")
    if values.ndim != 1 or values.size == 0 or values.dtype.kind not in "iu":
        raise ValueError("values must be a nonempty one-dimensional integer array")
    if values.min() < 0 or values.max() >= d:
        raise ValueError("values must lie in the public domain 0,...,d-1")
    return values.astype(np.int64, copy=False)


def privatize(values, d, epsilon, *, seed=0):
    """Generate one epsilon-LDP OUE report per value using PCG64.

    Returns uint8 shape (n, ceil(d/8)); bit j is stored least-significant-bit
    first. The random stream follows row-major category order. Each report
    has d logical bits; padding in the last byte is zero.
    """
    values = validate_values(values, d)
    if not np.isscalar(epsilon) or not math.isfinite(float(epsilon)) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    z = math.exp(-float(epsilon))
    q = z/(1+z)
    if not 0 < q < 0.5:
        raise ValueError("epsilon is outside the representable OUE range")
    rng = np.random.Generator(np.random.PCG64(seed))
    packed = np.zeros((len(values), (d+7)//8), dtype=np.uint8)
    chunk = max(1, (1 << 19)//d)
    for lo in range(0, len(values), chunk):
        hi = min(lo+chunk, len(values))
        u = rng.random((hi-lo, d))
        bits = u < q
        rows = np.arange(hi-lo)
        bits[rows, values[lo:hi]] = u[rows, values[lo:hi]] < 0.5
        packed[lo:hi] = np.packbits(bits, axis=1, bitorder="little")
    return packed


def column_counts(packed, d):
    packed = np.asarray(packed)
    if packed.dtype != np.uint8 or packed.ndim != 2 or packed.shape[1] != (d+7)//8:
        raise ValueError("packed reports have an incompatible shape or type")
    counts = np.zeros(d, dtype=np.int64)
    for lo in range(0, len(packed), 256):
        counts += np.unpackbits(packed[lo:lo+256], axis=1, count=d,
                                bitorder="little").sum(axis=0, dtype=np.int64)
    return counts
