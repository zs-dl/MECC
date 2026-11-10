"""Full per-user OUE report mixture MLE; only reports/public inputs are read.

The concave objective is mean(log(1 + expm1(epsilon) * B @ theta)).
Active-set fully corrective optimization uses bounded limited-memory BFGS on
normalized nonnegative weights, then checks the *full-domain* simplex gap.
No outcome-pattern enumeration or d-by-d Hessian is constructed.
"""
import math
import time

import numpy as np
from scipy.optimize import minimize, minimize_scalar, brentq


ADD_COLUMNS = 32
INNER_MAX_ITER = 300
BLOCK_BYTES = 8 * 1024 * 1024


def _positive_integer(value, name, allow_zero=False):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or value < (0 if allow_zero else 1)):
        raise ValueError(name + ' must be an integer in its allowed range')
    return int(value)


class _Reports:
    def __init__(self, packed, n, d):
        self.n, self.d = n, d
        self.packed = np.asarray(packed)
        if self.packed.dtype != np.uint8 or self.packed.shape != (n, (d + 7)//8):
            raise ValueError('packed must be uint8 with shape (n, ceil(d/8))')
        # A block is at most about 8 MiB after float64 conversion.
        self.byte_block = max(1, BLOCK_BYTES // (64*n))
        self.gradient_calls = 0

    def columns(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        return ((self.packed[:, indices//8] >> (indices % 8)) & 1).astype(float)

    def transpose(self, weights):
        self.gradient_calls += 1
        out = np.empty(self.d)
        for start in range(0, self.packed.shape[1], self.byte_block):
            stop = min(start+self.byte_block, self.packed.shape[1])
            width = min(8*(stop-start), self.d-8*start)
            block = np.unpackbits(self.packed[:, start:stop], axis=1,
                                  bitorder='little')[:, :width].astype(float)
            out[8*start:8*start+width] = block.T @ weights
        return out


def _active_set_estimate(packed, n, d, epsilon, *, max_iter=200, tol=1e-7):
    """Return (simplex estimate, JSON-safe numerical diagnostics).

    max_iter counts outer full-gradient/correction rounds; zero is allowed for
    diagnostic initialization. Success means full-domain gap <= tol, never just
    an inner solver success flag. Padding bits above d are ignored.
    epsilon=0 returns uniform; finite epsilon in (0, 700] is supported.
    """
    started = time.perf_counter()
    n, d = _positive_integer(n, 'n'), _positive_integer(d, 'd')
    max_iter = _positive_integer(max_iter, 'max_iter', True)
    if not np.isscalar(epsilon) or not math.isfinite(float(epsilon)) or not 0 <= epsilon <= 700:
        raise ValueError('epsilon must be finite and in [0, 700]')
    epsilon = float(epsilon)
    if not np.isscalar(tol) or not math.isfinite(float(tol)) or tol <= 0:
        raise ValueError('tol must be positive and finite')
    reports = _Reports(packed, n, d)
    t, scale = math.exp(-epsilon), -math.expm1(-epsilon)
    theta = np.zeros(d)
    if epsilon == 0:
        theta[:] = 1/d
        support, weights = np.arange(d), theta.copy()
        scores = np.zeros(n)
    else:
        # At uniform theta gradient and column counts need not rank identically;
        # this is simply a public, report-only starting vertex.
        first = int(np.argmax(reports.transpose(np.ones(n))))
        support, weights = np.array([first]), np.array([1.0])
        scores = reports.columns(support)[:, 0]
        theta[first] = 1.0

    def objective(s):
        return float(epsilon + np.log(t + scale*s).mean())

    history, gap_history, inner_runs = [objective(scores)], [], []
    fallback_steps, rounds, status = 0, 0, 'max_iter'
    gradient = np.zeros(d)
    for outer in range(max_iter + 1):
        gradient = scale * reports.transpose(1/(n*(t + scale*scores)))
        gap = max(0.0, float(gradient.max() - gradient @ theta))
        gap_history.append(gap)
        if gap <= tol:
            status = 'converged'
            break
        if outer == max_iter:
            break
        # Fixed public batch size, stable category-index tie breaking.
        threshold = float(gradient @ theta)
        ranked = np.argsort(-gradient, kind='stable')
        add = ranked[gradient[ranked] > threshold][:ADD_COLUMNS]
        new_support = np.union1d(support[weights > 0], add)
        matrix = reports.columns(new_support)
        initial = theta[new_support].copy()

        def loss_and_grad(w):
            mass = float(w.sum())
            if mass <= 0:
                return 1e100, np.zeros_like(w)
            normalized = w/mass
            z = t + scale*(matrix @ normalized)
            value = float(epsilon + np.log(z).mean())
            g = scale * (matrix.T @ (1/(n*z)))
            return -value, -(g - g @ normalized)/mass

        result = minimize(loss_and_grad, initial, jac=True, method='L-BFGS-B',
                          bounds=[(0.0, None)]*len(initial),
                          options={'maxiter': INNER_MAX_ITER, 'maxls': 40,
                                   'ftol': 1e-14, 'gtol': min(tol*.05, 1e-9),
                                   'maxcor': 20})
        inner_runs.append({'success': bool(result.success), 'status': int(result.status),
                           'message': str(result.message), 'nit': int(result.nit),
                           'nfev': int(result.nfev), 'support_size': len(initial)})
        candidate_weights = np.maximum(result.x, 0)
        mass = float(candidate_weights.sum())
        candidate_scores = None
        if np.isfinite(mass) and mass > 0:
            candidate_weights /= mass
            candidate_scores = matrix @ candidate_weights
        candidate_value = objective(candidate_scores) if candidate_scores is not None else -math.inf
        if np.isfinite(candidate_value) and candidate_value > history[-1]:
            support, weights, scores = new_support, candidate_weights, candidate_scores
            theta[:] = 0
            theta[support] = weights
        else:
            # Concave line search toward a maximal-gradient simplex vertex.
            # Retain only nondecreasing actual objectives.
            vertex = int(np.argmax(gradient))
            direction = reports.columns([vertex])[:, 0] - scores
            search = minimize_scalar(lambda x: -objective(scores+x*direction),
                                     bounds=(0, 1), method='bounded',
                                     options={'xatol': 1e-14})
            candidates = [(history[-1], 0.0), (objective(scores+direction), 1.0),
                          (objective(scores+float(search.x)*direction), float(search.x))]
            candidate_value, step = max(candidates)
            if candidate_value <= history[-1]:
                status = 'stalled'
                break
            theta *= 1-step
            theta[vertex] += step
            scores = scores + step*direction
            support = np.flatnonzero(theta > 0)
            weights = theta[support].copy()
            fallback_steps += 1
        rounds += 1
        history.append(objective(scores))

    diagnostics = {
        'solver': 'full-oue-active-lbfgsb-v1', 'status': status,
        'success': status == 'converged', 'n': n, 'd': d, 'epsilon': epsilon,
        'max_iter': max_iter, 'tol': float(tol), 'outer_iterations': rounds,
        'inner_max_iter': INNER_MAX_ITER, 'add_columns': ADD_COLUMNS,
        'objective': history[-1], 'duality_gap': gap_history[-1],
        'gradient_max': float(gradient.max()), 'gradient_theta': float(gradient @ theta),
        'objective_history': history, 'gap_history': gap_history,
        'inner_runs': inner_runs, 'fallback_steps': fallback_steps,
        'full_gradient_calls': reports.gradient_calls,
        'support_size': int(np.count_nonzero(theta)),
        'simplex_sum': float(theta.sum()), 'minimum_estimate': float(theta.min()),
        'monotone': bool(np.all(np.diff(history) >= 0)),
        'elapsed_seconds': time.perf_counter()-started,
        'padding': 'ignored', 'likelihood': 'full_per_user_mixture',
    }
    return theta, diagnostics


def coordinate_refine(packed, n, d, epsilon, initial_theta, *, tol=1e-7, max_steps=5000,
           objective_tolerance=1e-13):
    """Maximize the unchanged concave likelihood by simplex coordinate swaps."""
    started = time.perf_counter()
    if n < 1 or d < 1 or not np.isfinite(epsilon) or not 0 < epsilon <= 700:
        raise ValueError('Invalid public likelihood parameters')
    if tol <= 0 or max_steps < 0 or objective_tolerance < 0:
        raise ValueError('Invalid polishing settings')
    theta = np.asarray(initial_theta, dtype=float).copy()
    if theta.shape != (d,) or not np.all(np.isfinite(theta)) or theta.min() < 0 or abs(theta.sum()-1)>1e-10:
        raise ValueError('Initial estimate is not a simplex vector')
    initial_mass = float(theta.sum())
    theta /= initial_mass
    reports = _Reports(packed, n, d)
    decay, scale = math.exp(-epsilon), -math.expm1(-epsilon)

    def recompute_scores():
        active = np.flatnonzero(theta > 0)
        return reports.columns(active) @ theta[active]

    scores = recompute_scores()
    objective = lambda z: float(epsilon + np.log(z).mean())
    z = decay + scale*scores
    original_objective = objective(z)
    trace = []
    initial_gap = None
    status = 'max_steps'
    objective_roundoff_steps = 0
    for iteration in range(max_steps+1):
        # Rebuild scores periodically to control accumulated incremental rounding.
        if iteration % 32 == 0:
            scores = recompute_scores(); z = decay + scale*scores
        gradient = scale*reports.transpose(1/(n*z))
        gap = max(0.0, float(gradient.max()-gradient@theta))
        if initial_gap is None:
            initial_gap = gap
        obj = objective(z)
        if gap <= tol:
            status = 'converged'; break
        if iteration == max_steps:
            break
        receiver = int(np.argmax(gradient))
        active = np.flatnonzero(theta > 0)
        donor = int(active[np.argmin(gradient[active])])
        direction = reports.columns([receiver, donor]) @ np.array([1., -1.])
        upper = float(theta[donor])
        def derivative(step):
            # Direct directional derivative avoids subtracting two full gradients.
            return float(np.mean(scale*direction/(z+scale*step*direction), dtype=np.longdouble))
        slope0, slope_end = derivative(0.), derivative(upper)
        if receiver == donor or upper <= 0 or slope0 <= 0:
            status = 'stalled'; break
        step = upper if slope_end >= 0 else float(brentq(
            derivative, 0., upper, xtol=np.nextafter(0., 1.), rtol=8*np.finfo(float).eps, maxiter=200))
        if not 0 < step <= upper:
            status = 'stalled'; break
        relative_change = scale*step*direction/z
        stable_gain = float(np.mean(np.log1p(relative_change), dtype=np.longdouble))
        candidate_scores = scores+step*direction
        candidate_z = decay+scale*candidate_scores
        candidate_obj = objective(candidate_z)
        actual_gain = candidate_obj-obj
        if stable_gain < -objective_tolerance or actual_gain < -objective_tolerance:
            raise FloatingPointError('Pairwise likelihood step decreased objective beyond tolerance')
        theta[receiver] += step
        theta[donor] -= step
        if theta.min() < 0 or abs(theta.sum()-1)>1e-10:
            raise FloatingPointError('Pairwise step violated simplex feasibility')
        objective_roundoff_steps += int(actual_gain <= 0)
        trace.append({'iteration':iteration+1, 'gap_before':gap, 'receiver':receiver, 'donor':donor,
                      'step':step, 'directional_derivative_before':slope0,
                      'directional_derivative_after':derivative(step),
                      'stable_log1p_gain':stable_gain, 'rounded_objective_gain':actual_gain,
                      'objective_after':candidate_obj})
        scores, z = candidate_scores, candidate_z
    # Final certificate uses freshly reconstructed scores and a full-domain gradient.
    scores = recompute_scores(); z = decay+scale*scores
    gradient = scale*reports.transpose(1/(n*z))
    gap = max(0.0,float(gradient.max()-gradient@theta))
    if gap <= tol: status = 'converged'
    elif status == 'converged': status = 'certificate_failed'
    final_objective = objective(z)
    if final_objective < original_objective-objective_tolerance:
        raise FloatingPointError('Final objective below initial objective tolerance')
    return theta, {'solver':'full-oue-pairwise-polish-v1','status':status,'success':status=='converged',
        'tol':float(tol),'max_steps':int(max_steps),'steps':len(trace),
        'objective_tolerance':float(objective_tolerance),'initial_mass_before_normalization':initial_mass,
        'initial_objective':original_objective,'objective':final_objective,
        'initial_duality_gap':initial_gap,
        'objective_change':final_objective-original_objective,'duality_gap':gap,
        'gradient_max':float(gradient.max()),'gradient_theta':float(gradient@theta),
        'simplex_sum':float(theta.sum()),'minimum_estimate':float(theta.min()),
        'max_abs_parameter_change':float(np.max(np.abs(theta-initial_theta))),
        'objective_roundoff_steps':objective_roundoff_steps,'full_gradient_calls':reports.gradient_calls,
        'trace':trace,'seconds':time.perf_counter()-started,
        'selection':'likelihood derivatives and full-domain gap only; no truth or error metrics'}

def estimate(packed, n, d, epsilon, *, max_iter=200, tol=1e-7, max_coordinate_steps=5000):
    """Full-report likelihood estimate with a full-domain convergence certificate."""
    theta, active = _active_set_estimate(packed, n, d, epsilon, max_iter=max_iter, tol=tol)
    if active["success"] or epsilon == 0:
        return theta, active
    theta, coordinate = coordinate_refine(packed, n, d, epsilon, theta, tol=tol, max_steps=max_coordinate_steps)
    return theta, {**coordinate, "active_set": active, "solver": "full-oue-active-set-coordinate"}
