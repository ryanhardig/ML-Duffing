"""Generate synthetic dataset by solving Duffing equation for random parameters.

This generator now sweeps gamma over a specified range for each random set of
the other Duffing parameters. The CSV will therefore contain (n_param_sets *
gamma_steps) rows.
"""
from typing import Dict
import time
import itertools
import numpy as np
import pandas as pd
from .solver import solve_duffing
from .features import extract_features, estimate_lyapunov_benettin


def generate_sample(params: Dict, t_span=(0, 200), y0=(0.1, 0.0), compute_lyapunov: bool = False) -> Dict:
    t_eval = np.linspace(t_span[0], t_span[1], 5000)
    sol = solve_duffing(t_span, y0, params, t_eval=t_eval)
    x = sol.y[0]
    v = sol.y[1]
    feats = extract_features(sol.t, x, v)
    row = {**params}
    row.update(feats)
    row['label_gamma'] = float(params['gamma'])
    # optionally compute Lyapunov exponent (may be slow)
    lyap = float('nan')
    if compute_lyapunov:
        try:
            lyap = estimate_lyapunov_benettin(params)
        except Exception:
            lyap = float('nan')
    row['lyapunov'] = float(lyap)

    # Determine periodic/chaotic label using Lyapunov only.
    # Rule: lyap > 0 => chaotic (periodic=False), lyap < 0 => periodic (periodic=True)
    # If lyap is NaN or exactly 0, set periodic to NaN (do not fall back to is_periodic)
    if np.isnan(lyap):
        row['periodic'] = float('nan')
    else:
        if lyap > 0:
            row['periodic'] = False
        elif lyap < 0:
            row['periodic'] = True
        else:
            row['periodic'] = float('nan')
    return row


def generate_dataset(n_param_sets: int, out_csv: str, rng_seed: int = 0,
                     compute_lyapunov: bool = False, run_minutes: float = 0.0):
    """Generate a dataset by sampling parameters randomly.

    Args:
        n_param_sets: number of random base parameter sets to draw (ignored if run_minutes>0).
        out_csv: output CSV path.
        rng_seed: RNG seed for reproducibility.
        compute_lyapunov: whether to compute LLE for each sample (slow).
        run_minutes: if >0, run the generator for approximately this many minutes
                     instead of a fixed number of param sets.

    Returns:
        pandas.DataFrame saved to out_csv
    """
    rng = np.random.default_rng(rng_seed)
    rows = []
    # gamma will be sampled randomly per parameter set (treated like delta/alpha/beta/omega)
    if run_minutes and run_minutes > 0:
        # timed mode: ignore n_param_sets and run until time expires
        deadline = time.time() + float(run_minutes) * 60.0
        try:
            from tqdm import tqdm
            iterator = tqdm(itertools.count(), desc=f'param sets (timed {run_minutes}m)')
            progress_tqdm = True
        except Exception:
            iterator = itertools.count()
            progress_tqdm = False
        timed = True
    else:
        try:
            from tqdm import tqdm
            iterator = tqdm(range(n_param_sets), desc='param sets')
            progress_tqdm = True
        except Exception:
            iterator = range(n_param_sets)
            progress_tqdm = False
        timed = False

    # reporting state: samples/sec (converted to samples/min)
    start_time = time.time()
    last_report = start_time
    report_interval = 10.0  # seconds between reports
    samples_generated = 0

    for i in iterator:
        if timed and time.time() > deadline:
            break
        # sample parameter ranges (reasonable defaults) for the non-gamma params
        # include gamma as a tuple range in base_params per the requested API
        base_params = {
            'delta': float(rng.uniform(0.01, 0.1)),
            'alpha': float(rng.uniform(-2.0, 2.0)),
            'beta': float(rng.uniform(0.4, 2.0)),
            'omega': float(rng.uniform(0.4, 2.0)),
            'gamma': (0.1, 5.0),
        }
        # give each base param set an id so we can group later
        base_id = i
        # sample gamma from the range stored in base_params['gamma']
        gm = base_params.get('gamma', (0.1, 5.0))
        gamma = float(rng.uniform(gm[0], gm[1]))
        params = {**base_params, 'gamma': gamma}
        row = generate_sample(params, compute_lyapunov=compute_lyapunov)
        row['base_id'] = base_id
        rows.append(row)
        samples_generated += 1

        # periodic progress reporting (samples per minute)
        now = time.time()
        if (now - last_report) >= report_interval:
            elapsed = max(1e-6, now - start_time)
            s_per_min = (samples_generated / elapsed) * 60.0
            last_report = now
            if progress_tqdm:
                try:
                    iterator.set_postfix({'s/min': f'{s_per_min:.1f}'})
                except Exception:
                    # fall back to printing if set_postfix fails
                    print(f'Samples: {samples_generated}  |  {s_per_min:.1f} samples/min')
            else:
                print(f'Samples: {samples_generated}  |  {s_per_min:.1f} samples/min')

    df = pd.DataFrame(rows)

    # write full dataframe to csv
    df.to_csv(out_csv, index=False)
    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10, help='number of random param sets')
    parser.add_argument('--out', type=str, default='duffing_dataset.csv')
    parser.add_argument('--seed', type=int, default=0)
    # gamma range is defined inside the generator's base parameter template
    parser.add_argument('--compute-lyapunov', action='store_true', help='compute Lyapunov exponent for each sample (can be slow)')
    parser.add_argument('--base-out', type=str, default='', help='optional CSV path to save unique base parameter sets')
    parser.add_argument('--minutes', type=float, default=0.0, help='run for approximately this many minutes instead of using n param sets')
    args = parser.parse_args()
    print('Generating dataset...')
    df = generate_dataset(args.n, args.out, rng_seed=args.seed,
                          compute_lyapunov=args.compute_lyapunov,
                          run_minutes=args.minutes)
    # if requested, save the unique base parameter sets
    if args.base_out:
        base_cols = ['base_id', 'delta', 'alpha', 'beta', 'omega']
        if not df.empty:
            bases = df[base_cols].drop_duplicates(subset=['base_id']).reset_index(drop=True)
            bases.to_csv(args.base_out, index=False)
            print('Saved base parameter sets to', args.base_out)
    print('Saved to', args.out)
