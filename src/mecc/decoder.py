"""MECC server decoder for unchanged complete OUE reports.

Coefficient construction sees training counts and public parameters only.
Large mechanism states are returned separately for NPZ storage.
"""
from functools import lru_cache
import math
import time

import numpy as np
from .baselines import public_power_prior
from .numerics import _pmf_segment, simplex_projection


def parameters(n, d, epsilon):
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError('n must be a positive integer')
    if isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer)) or d < 2:
        raise ValueError('d must be an integer at least two')
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('epsilon must be finite and positive')
    q = 1.0 / (np.exp(float(epsilon)) + 1.0)
    delta = 0.5-q
    if not 0 < q < 0.5 or delta <= 0:
        raise ValueError('Channel outside finite-precision identifiable range')
    return {'n': int(n), 'd': int(d), 'epsilon': float(epsilon),
            'p': 0.5, 'q': float(q), 'delta': float(delta),
            'v': float(q*(1-q)/delta**2)}


@lru_cache(maxsize=16)
def public_calibrate_table(n, d, epsilon):
    """Unprojected g_n(c), all integer c, identical to Calibrate-PL0-public.

    Full finite integer sums are evaluated in public fixed-size blocks. The
    cache depends only on n,d,epsilon, never on a report window.
    """
    c = parameters(n, d, epsilon)
    alpha, prior = public_power_prior(int(n), int(d))
    k = np.arange(n+1, dtype=float)
    weighted_k = prior*k
    variance = n*c['v']
    table = np.empty(n+1, dtype=float)
    for start in range(0, n+1, 128):
        stop = min(n+1, start+128)
        y = (np.arange(start, stop, dtype=float)-n*c['q'])/c['delta']
        log_likelihood = -0.5*(y[:, None]-k[None, :])**2/variance
        log_likelihood -= np.max(log_likelihood, axis=1)[:, None]
        likelihood = np.exp(log_likelihood)
        denominator = likelihood@prior
        if np.any(denominator <= 0) or not np.all(np.isfinite(denominator)):
            raise FloatingPointError('Public Calibrate table underflow')
        table[start:stop] = (likelihood@weighted_k)/denominator/n
    if not np.all(np.isfinite(table)) or table.min() < -1e-14 or table.max() > 1+1e-14:
        raise FloatingPointError('Invalid public posterior table')
    table.setflags(write=False)
    return table, {'alpha': float(alpha), 'prior_mean': float(prior@k),
                   'noise_count_variance': variance, 'support': [0, int(n)],
                   'computation': 'full_integer_sums_in_public_blocks_of_128',
                   'max_negative_forward_difference': float(max(0.0, -np.min(np.diff(table))))}


def public_split(n, train_fraction=0.2, split_seed=20261109):
    if not np.isfinite(train_fraction) or not 0 < train_fraction <= 1:
        raise ValueError('train_fraction must lie in (0,1]')
    n_A = min(n, int(math.ceil(n*float(train_fraction))))
    order = np.random.Generator(np.random.PCG64(split_seed)).permutation(n)
    return order[:n_A], order[n_A:]


def expected_forward_differences(counts_A, latent_counts, m, table, epsilon, *, tail_tolerance=1e-14):
    """Expectation using the actual unnormalized two-binomial PMF.

    All inputs belong to training information or are public. The support is
    chosen from m-1,k,p,q and the fixed tolerance, not validation observations.
    The function also supports evaluation-side oracle k when called separately.
    """
    counts_A = np.asarray(counts_A, dtype=np.int64)
    latent_counts = np.asarray(latent_counts, dtype=np.int64)
    if counts_A.ndim != 1 or counts_A.shape != latent_counts.shape or m < 1:
        raise ValueError('Expected one matched training-count / latent-count vector and m>=1')
    if not 0 <= tail_tolerance <= 1e-8:
        raise ValueError('tail_tolerance must lie in [0,1e-8]')
    if np.any(latent_counts < 0) or np.any(latent_counts > m-1):
        raise ValueError('Leave-one-out latent counts must lie in [0,m-1]')
    table = np.asarray(table, dtype=float)
    if table.ndim != 1 or np.any(counts_A < 0) or np.any(counts_A+m >= len(table)):
        raise ValueError('Public table does not cover all required counts')
    q = 1.0/(np.exp(float(epsilon))+1.0)
    difference = np.diff(table)
    max_difference = float(np.max(np.abs(difference)))
    groups, inverse = np.unique(np.column_stack((counts_A, latent_counts)), axis=0, return_inverse=True)
    means = np.empty(len(groups))
    bounds = np.empty(len(groups))
    mass_defects = np.empty(len(groups))
    support = np.empty((len(groups), 2), dtype=np.int64)
    for index, (observed_A, k) in enumerate(groups):
        offset_p, pmf_p, lost_p = _pmf_segment(int(k), 0.5, tail_tolerance)
        offset_q, pmf_q, lost_q = _pmf_segment(int(m-1-k), q, tail_tolerance)
        pmf = np.convolve(pmf_p, pmf_q)
        lo = offset_p+offset_q
        hi = lo+len(pmf)-1
        means[index] = float(pmf@difference[int(observed_A)+lo:int(observed_A)+hi+1])
        bounds[index] = lost_p+lost_q
        mass_defects[index] = abs(float(pmf.sum())-1)
        support[index] = lo, hi
    max_lost = float(bounds.max(initial=0.0))
    max_defect = float(mass_defects.max(initial=0.0))
    if max_lost > tail_tolerance + 1e-15 or max_defect > max_lost + 5e-10:
        raise FloatingPointError('Two-binomial PMF numerical mass check failed')
    metadata = {'model': 'Bin(k,1/2)+Bin(m-1-k,q)',
                'computation': 'nonnegative_direct_convolution_without_renormalization',
                'support_information': 'training_counts_and_public_parameters_only',
                'tail_tolerance': float(tail_tolerance),
                'max_omitted_probability_bound': float(bounds.max(initial=0.0)),
                'max_mass_defect': float(mass_defects.max(initial=0.0)),
                'max_abs_forward_difference': max_difference,
                'max_mu_truncation_bound': float(bounds.max(initial=0.0))*max_difference,
                'distinct_training_groups': len(groups)}
    return means[inverse], metadata, {'mu_groups': groups, 'mu_group_inverse': inverse,
                                     'mu_group_values': means, 'mu_group_support': support,
                                     'mu_group_omitted_probability_bound': bounds,
                                     'mu_group_mass_defect': mass_defects}


def build_coefficients(counts_A, r, n_A, n, d, epsilon, *, max_controls=32,
                       tie_seed=20261110, tail_tolerance=1e-14):
    """Construct H and beta from A only; no B counts, rows or truth argument."""
    c = parameters(n, d, epsilon)
    counts_A, r = np.asarray(counts_A), np.asarray(r, dtype=float)
    if (counts_A.shape != (d,) or counts_A.dtype.kind not in 'iu'
            or np.any(counts_A < 0) or np.any(counts_A > n_A)):
        raise ValueError('Invalid training counts')
    if (r.shape != (d,) or not np.all(np.isfinite(r)) or np.min(r) < -1e-12
            or abs(r.sum()-1) > 1e-8):
        raise ValueError('Training estimate r must be a full-domain simplex vector')
    if not isinstance(n_A, (int, np.integer)) or not 1 <= n_A <= n:
        raise ValueError('n_A must lie in [1,n]')
    if (isinstance(max_controls, (bool, np.bool_)) or not isinstance(max_controls, (int, np.integer))
            or max_controls < 0):
        raise ValueError('max_controls must be a nonnegative integer')
    m = n-n_A
    table, public_info = public_calibrate_table(int(n), int(d), float(epsilon))
    permutation = np.random.Generator(np.random.PCG64(tie_seed)).permutation(d)
    rank = np.empty(d, dtype=np.int64)
    rank[permutation] = np.arange(d)
    H = np.lexsort((rank, -r))[:min(int(max_controls), d)] if m else np.empty(0, dtype=np.int64)
    if not m:
        arrays = {'H': H, 'beta': np.zeros((d, 0)), 'mu_tilde': np.zeros(d),
                  'k_tilde': np.zeros(d, dtype=np.int64), 'tie_rank': rank}
        return arrays, {'kernel': None, 'public_calibrate': public_info,
                        'coefficient_information': 'A_and_public_only', 'm': 0}
    latent = np.minimum(m-1, np.maximum(0, np.floor(m*r+0.5))).astype(np.int64)
    mu, kernel_info, grouped = expected_forward_differences(counts_A, latent, m, table,
        epsilon, tail_tolerance=tail_tolerance)
    beta = m*c['delta']*mu[:, None]*r[H][None, :] / (c['v']+2*(r[:, None]+r[H][None, :]))
    if len(H):
        beta[H, np.arange(len(H))] = 0.0
    if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(mu)):
        raise FloatingPointError('Nonfinite control coefficients')
    arrays = {'H': H, 'beta': beta, 'mu_tilde': mu, 'k_tilde': latent,
              'tie_rank': rank, **grouped}
    return arrays, {'kernel': kernel_info, 'public_calibrate': public_info,
                    'coefficient_information': 'A_and_public_only', 'm': int(m)}


def column_intersections(column_packed, H):
    """Actual B-bit intersections, with users packed along each category row."""
    column_packed = np.asarray(column_packed)
    H = np.asarray(H, dtype=np.int64)
    if column_packed.ndim != 2 or column_packed.dtype != np.uint8:
        raise ValueError('Column-packed matrix must be uint8')
    if np.any(H < 0) or np.any(H >= len(column_packed)):
        raise ValueError('Invalid control column')
    result = np.empty((len(column_packed), len(H)), dtype=np.int64)
    for index, k in enumerate(H):
        result[:, index] = np.bitwise_count(np.bitwise_and(column_packed, column_packed[k])).sum(axis=1, dtype=np.int64)
    return result


def decode(packed, n, d, epsilon, *, train_fraction=0.2, max_controls=32,
           split_seed=20261109, tie_seed=20261110, tail_tolerance=1e-14):
    """Return (estimate dictionary, JSON diagnostics, mechanism-state arrays)."""
    started = time.perf_counter()
    c = parameters(n, d, epsilon)
    packed = np.asarray(packed)
    if packed.dtype != np.uint8 or packed.shape != (n, (d+7)//8):
        raise ValueError('Expected complete n by ceil(d/8) uint8 OUE bitpacked report')
    A, B = public_split(int(n), train_fraction, split_seed)
    n_A, m = len(A), len(B)
    bits_A = np.unpackbits(packed[A], axis=1, count=d, bitorder='little')
    counts_A = bits_A.sum(axis=0, dtype=np.int64)
    del bits_A
    if m:
        bits_B = np.unpackbits(packed[B], axis=1, count=d, bitorder='little')
        counts_B = bits_B.sum(axis=0, dtype=np.int64)
        column_packed = np.ascontiguousarray(np.packbits(bits_B, axis=0, bitorder='little').T)
        del bits_B
    else:
        counts_B = np.zeros(d, dtype=np.int64)
        column_packed = np.empty((d, 0), dtype=np.uint8)
    counts = counts_A+counts_B
    full_table, full_info = public_calibrate_table(int(n), int(d), float(epsilon))
    pilot_table, pilot_info = public_calibrate_table(int(n_A), int(d), float(epsilon))
    r = simplex_projection(pilot_table[counts_A])
    coefficients, coefficient_info = build_coefficients(counts_A, r, n_A, n, d, epsilon,
        max_controls=max_controls, tie_seed=tie_seed, tail_tolerance=tail_tolerance)
    H, beta = coefficients['H'], coefficients['beta']
    intersections = column_intersections(column_packed, H)
    if m:
        T = (intersections-c['q']*(counts_B[:, None]+counts_B[H][None, :])+m*c['q']**2)/(m*c['delta']**2)
    else:
        T = np.zeros((d, 0))
    correction = np.sum(beta*T, axis=1)
    G = full_table[counts].copy()
    W = G-correction
    estimates = {'mecc': simplex_projection(W), 'calibrate_public': simplex_projection(G),
                 'mecc_pre': W, 'calibrate_pre': G}
    if not all(np.all(np.isfinite(a)) for a in estimates.values()):
        raise FloatingPointError('Nonfinite MECC estimate')
    simplex_residual = max(abs(float(estimates[name].sum())-1) for name in ('mecc','calibrate_public'))
    if simplex_residual > 1e-10 or min(estimates['mecc'].min(), estimates['calibrate_public'].min()) < -1e-12:
        raise FloatingPointError('Projected estimate failed simplex numerical check')
    diagnostics = {'parameters': c, 'train_fraction': float(train_fraction), 'n_A': n_A, 'm': m,
                   'max_controls': int(max_controls), 'active_controls': len(H),
                   'split_seed': int(split_seed), 'tie_seed': int(tie_seed),
                   'split_rule': 'PCG64 permutation, first ceil(n*train_fraction)',
                   'tie_rule': 'independent public PCG64 category rank',
                   'full_public_calibrate': full_info, 'pilot_public_calibrate': pilot_info,
                   'coefficient_info': coefficient_info,
                   'diagonal_controls': 'excluded_by_exact_zero_beta',
                   'pre_projection_mass': {'base': float(G.sum()), 'corrected': float(W.sum())},
                   'projection_mass_residual': {'base': float(estimates['calibrate_public'].sum()-1),
                                                'corrected': float(estimates['mecc'].sum()-1)},
                   'correction_l1': float(np.abs(correction).sum()),
                   'correction_max_abs': float(np.abs(correction).max()),
                   'numerical_valid': True,
                   'numerical_checks': 'finite_table_coefficients_outputs; PMF_mass; simplex_mass_nonnegativity',
                   'seconds': time.perf_counter()-started}
    arrays = {'train_indices': A, 'validation_indices': B, 'counts_A': counts_A,
              'counts_B': counts_B, 'counts': counts, 'pilot_r': r, **coefficients,
              'intersections_B': intersections, 'T': T, 'correction': correction,
              'public_g': full_table.copy()}
    return estimates, diagnostics, arrays
