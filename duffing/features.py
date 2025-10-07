"""Feature extraction and simple chaos/periodicity checks for Duffing solutions."""
from typing import Tuple, Dict
import numpy as np
from .solver import solve_duffing


def poincare_section(t: np.ndarray, x: np.ndarray, omega: float, phase=0.0):
    """Sample (x, v) at times t = (2*pi/omega)*n + phase.

    Returns samples as array shape (N,)
    """
    period = 2 * np.pi / omega
    t0 = t[0]
    n_max = int(np.floor((t[-1] - t0) / period))
    sample_times = t0 + np.arange(n_max) * period + phase
    sample_times = sample_times[(sample_times >= t[0]) & (sample_times <= t[-1])]
    # use linear interpolation
    x_samples = np.interp(sample_times, t, x)
    return sample_times, x_samples


def is_periodic(x: np.ndarray, threshold=1e-3, min_cycles=5) -> bool:
    """Very simple periodicity check using autocorrelation peaks.

    This is heuristic: compute FFT and look for a dominant frequency with stable phase.
    Returns True if signal appears periodic.
    """
    N = len(x)
    if N < 10:
        return True
    x = x - np.mean(x)
    # autocorrelation via FFT
    f = np.fft.rfft(x)
    ps = np.abs(f)**2
    freqs = np.fft.rfftfreq(N)
    if len(ps) < 3:
        return True
    # ignore zero-frequency
    ps[0] = 0
    idx = np.argmax(ps)
    peak_power = ps[idx]
    total_power = np.sum(ps)
    if total_power == 0:
        return True
    dominance = peak_power / total_power
    # require a strong peak and at least a few cycles in the data
    cycles = freqs[idx] * N
    return (dominance > 0.5) and (cycles >= min_cycles)



############## DO I NEED THIS???##################################

def extract_features(t: np.ndarray, x: np.ndarray, v: np.ndarray) -> Dict[str, float]:
    """Basic feature vector from time series: RMS, max, min, variance, spectral entropy"""
    feats = {}
    feats['x_mean'] = float(np.mean(x))
    feats['x_rms'] = float(np.sqrt(np.mean(x**2)))
    feats['x_max'] = float(np.max(x))
    feats['x_min'] = float(np.min(x))
    feats['x_std'] = float(np.std(x))
    # approximate dominant frequency
    N = len(x)
    if N >= 4:
        f = np.fft.rfft(x - np.mean(x))
        ps = np.abs(f)**2
        freqs = np.fft.rfftfreq(N, d=(t[1]-t[0]))
        idx = np.argmax(ps[1:]) + 1
        feats['dom_freq'] = float(freqs[idx])
    else:
        feats['dom_freq'] = 0.0
    return feats


def estimate_lyapunov_benettin(params: dict,
                               transient_cycles: int = 50,
                               measure_cycles: int = 100,
                               eps: float = 1e-8,
                               dt_cycles: int = 1) -> float:
    """Estimate the largest Lyapunov exponent using the Benettin algorithm.

    Args:
        params: dict with keys delta, alpha, beta, gamma, omega
        transient_cycles: number of driving cycles to discard before measurement
        measure_cycles: number of driving cycles over which to measure growth
        eps: initial separation magnitude in x (perturbation applied to x only)
        dt_cycles: renormalization interval in number of driving cycles (usually 1)

    Returns:
        Approximate largest Lyapunov exponent (float, 1/time units used by params)

    Notes:
        This is a simple implementation: it evolves two nearby trajectories,
        periodically rescales the separation back to eps, and accumulates the
        logarithmic growth. The exponent is the time-average of those growths.
    """
    omega = float(params['omega'])
    period = 2 * np.pi / omega

    # run transient to get into attractor
    t_trans = transient_cycles * period
    if t_trans > 0:
        sol_tr = solve_duffing((0.0, t_trans), (0.1, 0.0), params,
                                t_eval=np.linspace(0.0, t_trans, max(200, transient_cycles*10)))
        y_ref = (float(sol_tr.y[0, -1]), float(sol_tr.y[1, -1]))
    else:
        y_ref = (0.1, 0.0)

    # prepare perturbed initial condition
    y_pert = (y_ref[0] + eps, y_ref[1])

    # renormalize interval and count
    dt = dt_cycles * period
    n_steps = int(np.maximum(1, measure_cycles // max(1, dt_cycles)))

    sum_log = 0.0
    total_time = 0.0

    t0 = 0.0
    # start measurement immediately after transient
    for i in range(n_steps):
        t1 = t0 + dt
        # integrate both trajectories from t0 to t1
        sol1 = solve_duffing((t0, t1), y_ref, params, t_eval=[t1])
        sol2 = solve_duffing((t0, t1), y_pert, params, t_eval=[t1])
        y1 = np.array([float(sol1.y[0, -1]), float(sol1.y[1, -1])])
        y2 = np.array([float(sol2.y[0, -1]), float(sol2.y[1, -1])])
        delta = y2 - y1
        dist = np.linalg.norm(delta)
        if dist <= 0:
            # numerical underflow; skip
            y_ref = (float(y1[0]), float(y1[1]))
            y_pert = (y_ref[0] + eps, y_ref[1])
            t0 = t1
            continue
        sum_log += np.log(dist / eps)
        total_time += dt
        # renormalize perturbation to eps along direction delta
        delta_unit = delta / dist
        y_ref = (float(y1[0]), float(y1[1]))
        y_pert = (float(y_ref[0] + eps * delta_unit[0]), float(y_ref[1] + eps * delta_unit[1]))
        t0 = t1

    if total_time <= 0:
        return float('nan')
    lyap = sum_log / total_time
    return float(lyap)


def classify_by_lyapunov(params: dict,
                         transient_cycles: int = 50,
                         measure_cycles: int = 100,
                         eps: float = 1e-8,
                         dt_cycles: int = 1,
                         chaotic_thresh: float = 0.01,
                         periodic_thresh: float = -1e-4) -> Dict:
    """Classify dynamics as 'chaotic', 'quasi-periodic', or 'periodic' using LLE.

    Returns dict: {'lyap': value, 'label': str}
    Thresholds are tunable. By default:
      - lyap > chaotic_thresh -> 'chaotic'
      - lyap < periodic_thresh -> 'periodic'
      - otherwise -> 'quasi-periodic'
    """
    lyap = estimate_lyapunov_benettin(params, transient_cycles=transient_cycles,
                                      measure_cycles=measure_cycles, eps=eps,
                                      dt_cycles=dt_cycles)
    if np.isnan(lyap):
        label = 'unknown'
    else:
        if lyap > chaotic_thresh:
            label = 'chaotic'
        elif lyap < periodic_thresh:
            label = 'periodic'
        else:
            label = 'quasi-periodic'
    return {'lyap': float(lyap), 'label': label}
