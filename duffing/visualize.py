"""Visualization helpers: Poincaré and semilog delta-x plotting.

Includes utilities to compute the divergence between two nearby trajectories
(useful to check whether small perturbations grow or decay).
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import pandas as pd
from .features import poincare_section
from .solver import solve_duffing


def plot_poincare(t, x, v, omega, ax=None):
    sample_t, sample_x = poincare_section(t, x, omega)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(sample_t % (2*np.pi/omega), sample_x, 'o', markersize=3)
    ax.set_xlabel('phase (mod period)')
    ax.set_ylabel('x (Poincaré)')
    return ax


def semilog_delta_x(t, x, ax=None):
    # compute successive differences and plot semilog of their RMS
    if ax is None:
        fig, ax = plt.subplots()
    dx = np.abs(np.diff(x))
    # compute running RMS on log spaced windows
    N = len(dx)
    window = max(1, N // 200)
    rms = np.array([np.sqrt(np.mean(dx[i:i+window]**2)) for i in range(0, N-window+1, window)])
    ax.semilogy(rms, label='successive |dx| RMS')
    ax.set_ylabel('RMS(|dx|)')
    ax.set_xlabel('window index')
    ax.legend()
    return ax


def compute_perturbed_delta(params: dict, t_span=(0.0, 200.0), y0=(0.1, 0.0), eps: float = 1e-6,
                            t_eval: np.ndarray = None):
    """Simulate two nearby trajectories and return absolute difference in x over time.

    Returns (t, delta_x) where delta_x = |x_nominal - x_perturbed| sampled at t.
    """
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 5000)
    # nominal trajectory
    sol1 = solve_duffing(t_span, y0, params, t_eval=t_eval)
    # perturbed initial condition in x
    y0p = (y0[0] + eps, y0[1])
    sol2 = solve_duffing(t_span, y0p, params, t_eval=t_eval)
    x1 = sol1.y[0]
    x2 = sol2.y[0]
    delta = np.abs(x1 - x2)
    return sol1.t, delta


def plot_delta_comparison(t, x, delta, ax=None):
    """Plot comparison of successive-dx RMS and perturbed-trajectory delta on semilog scale."""
    if ax is None:
        fig, ax = plt.subplots()
    # plot raw delta (downsample for plotting performance)
    ax.semilogy(t, np.clip(delta, 1e-16, None), alpha=0.8, label='|x - x_perturbed|')
    ax.set_xlabel('t')
    ax.set_ylabel('|x - x_perturbed| (log scale)')
    ax.legend()
    return ax


def compute_poincare_for_gamma(params: Dict, t_transient_cycles: int = 50,
                               n_samples: int = 200, cycles_buffer: int = 0):
    """Simulate and return Poincaré samples for given params.

    Returns an array of samples (possibly many) sampled once per driving period.
    We drop an initial transient measured in driving cycles (t_transient_cycles).
    """
    omega = float(params['omega'])
    period = 2 * np.pi / omega
    # determine total cycles: transient + samples + small buffer
    total_cycles = int(t_transient_cycles + n_samples + cycles_buffer)
    t_span = (0.0, period * total_cycles)
    t_eval = np.linspace(t_span[0], t_span[1], max(2000, total_cycles * 20))
    sol = solve_duffing(t_span, (0.1, 0.0), params, t_eval=t_eval)
    t = sol.t
    x = sol.y[0]
    sample_t, sample_x = poincare_section(t, x, omega)
    # drop transient cycles
    sample_times_from_start = (sample_t - sample_t[0]) / period
    keep_mask = sample_times_from_start >= t_transient_cycles
    samples_kept = sample_x[keep_mask]
    # limit to n_samples
    if len(samples_kept) > n_samples:
        samples_kept = samples_kept[:n_samples]
    return samples_kept


def bifurcation_from_csv(csv_path: str, out_prefix: str = 'bifurcation_base_',
                         base_ids: List[int] = None, t_transient_cycles: int = 50,
                         n_samples: int = 200, show: bool = False,
                         mode: str = 'points'):
    df = pd.read_csv(csv_path)
    if 'base_id' not in df.columns:
        # try to infer grouping by non-gamma params
        df['base_id'] = df.groupby(['delta','alpha','beta','omega']).ngroup()

    available = sorted(df['base_id'].unique())
    if base_ids is None:
        base_ids = available

    results = {}
    for bid in base_ids:
        rows = df[df['base_id'] == bid].sort_values('gamma')
        if rows.empty:
            print(f'base_id {bid} not found, skipping')
            continue
        gammas = rows['gamma'].values.astype(float)
        all_x = []
        all_g = []
        # For 'lines' mode we'll collect a matrix of samples (num_gammas x n_samples)
        samples_matrix = []
        print(f'Processing base_id={bid} with {len(gammas)} gamma values...')
        # pick base params from first row (delta,alpha,beta,omega fixed across group)
        base_params = rows.iloc[0][['delta','alpha','beta','omega']].to_dict()
        for g in gammas:
            params = {**base_params, 'gamma': float(g)}
            samples = compute_poincare_for_gamma(params, t_transient_cycles=t_transient_cycles,
                                                n_samples=n_samples)
            # store each sample as a point at this gamma
            all_x.extend(samples.tolist())
            all_g.extend([g] * len(samples))
            # keep a fixed-length vector per gamma for line plotting (pad with nan if needed)
            if mode == 'lines':
                arr = np.array(samples, dtype=float)
                if len(arr) < n_samples:
                    pad = np.full((n_samples - len(arr),), np.nan)
                    arr = np.concatenate([arr, pad])
                else:
                    arr = arr[:n_samples]
                samples_matrix.append(arr)

        results[bid] = {'gamma': np.array(all_g), 'x': np.array(all_x)}

        # plot
        fig, ax = plt.subplots(figsize=(8, 5))
        if mode == 'points':
            ax.plot(results[bid]['gamma'], results[bid]['x'], ',k', alpha=0.6)
        elif mode == 'lines':
            # samples_matrix shape: (num_gammas, n_samples)
            samp_mat = np.array(samples_matrix, dtype=float)
            # For each sample index (column) plot a continuous line across gamma values
            for j in range(samp_mat.shape[1]):
                ax.plot(gammas, samp_mat[:, j], '-', lw=0.8, alpha=0.8)
        else:
            raise ValueError("mode must be 'points' or 'lines'")

        ax.set_xlabel('gamma')
        ax.set_ylabel('Poincaré x')
        ax.set_title(f'Bifurcation diagram (base_id={bid})')
        out_path = f'{out_prefix}{bid}.png'
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        print(f'Saved {out_path} ({len(results[bid]["x"])} points)')
        if show:
            plt.show()
        plt.close(fig)

    return results
