"""Generate synthetic dataset by solving Duffing equation for random parameters.

This generator now sweeps gamma over a specified range for each random set of
the other Duffing parameters. The CSV will therefore contain (n_param_sets *
gamma_steps) rows.
"""
from typing import Dict
import numpy as np
import pandas as pd
from .solver import solve_duffing
from .features import extract_features, is_periodic


def generate_sample(params: Dict, t_span=(0, 200), y0=(0.1, 0.0)) -> Dict:
    t_eval = np.linspace(t_span[0], t_span[1], 5000)
    sol = solve_duffing(t_span, y0, params, t_eval=t_eval)
    x = sol.y[0]
    v = sol.y[1]
    feats = extract_features(sol.t, x, v)
    periodic = is_periodic(x)
    row = {**params}
    row.update(feats)
    row['periodic'] = bool(periodic)
    row['label_gamma'] = float(params['gamma'])
    return row


def generate_dataset(n_param_sets: int, out_csv: str, rng_seed: int = 0,
                     gamma_start: float = 0.1, gamma_end: float = 1.5, gamma_steps: int = 15,
                     only_transitions: bool = False):
    """Generate a dataset by sampling other parameters randomly and sweeping gamma.

    Args:
        n_param_sets: number of random (delta, alpha, beta, omega) sets to draw.
        out_csv: output CSV path.
        rng_seed: RNG seed for reproducibility.
        gamma_start/gamma_end/gamma_steps: gamma sweep bounds and number of steps.

    Returns:
        pandas.DataFrame saved to out_csv
    """
    rng = np.random.default_rng(rng_seed)
    rows = []
    gamma_values = np.linspace(gamma_start, gamma_end, gamma_steps)
    total = n_param_sets * len(gamma_values)
    try:
        from tqdm import tqdm
        iterator = tqdm(range(n_param_sets), desc='param sets')
    except Exception:
        iterator = range(n_param_sets)

    for i in iterator:
        # sample parameter ranges (reasonable defaults) for the non-gamma params
        base_params = {
            'delta': float(rng.uniform(0.05, 0.5)),
            'alpha': float(rng.uniform(-1.0, 1.0)),
            'beta': float(rng.uniform(0.0, 1.0)),
            'omega': float(rng.uniform(0.5, 1.5)),
        }
        # give each base param set an id so we can group later
        base_id = i
        for gamma in gamma_values:
            params = {**base_params, 'gamma': float(gamma)}
            row = generate_sample(params)
            row['base_id'] = base_id
            rows.append(row)

    df = pd.DataFrame(rows)

    if only_transitions:
        # group by base_id and keep only those groups where 'periodic' changes value
        groups = df.groupby('base_id')
        keep_ids = []
        for gid, g in groups:
            vals = g['periodic'].unique()
            if len(vals) > 1:
                keep_ids.append(gid)
        if len(keep_ids) == 0:
            # nothing to keep; return empty DataFrame but still write file
            df_filtered = df.iloc[0:0]
        else:
            df_filtered = df[df['base_id'].isin(keep_ids)].reset_index(drop=True)
        df_filtered.to_csv(out_csv, index=False)
        return df_filtered

    df.to_csv(out_csv, index=False)
    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10, help='number of random param sets')
    parser.add_argument('--out', type=str, default='duffing_dataset.csv')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gamma-start', type=float, default=0.1)
    parser.add_argument('--gamma-end', type=float, default=1.5)
    parser.add_argument('--gamma-steps', type=int, default=15)
    parser.add_argument('--only-transitions', action='store_true', help='save only parameter-sets that change periodic/chaotic across the gamma sweep')
    parser.add_argument('--base-out', type=str, default='', help='optional CSV path to save unique base parameter sets that show transitions')
    args = parser.parse_args()
    print('Generating dataset...')
    df = generate_dataset(args.n, args.out, rng_seed=args.seed,
                          gamma_start=args.gamma_start, gamma_end=args.gamma_end,
                          gamma_steps=args.gamma_steps, only_transitions=args.only_transitions)
    # if requested, save the unique base parameter sets that transitioned
    if args.only_transitions and args.base_out:
        # extract unique base params for the kept base_id groups
        base_cols = ['base_id', 'delta', 'alpha', 'beta', 'omega']
        if not df.empty:
            bases = df[base_cols].drop_duplicates(subset=['base_id']).reset_index(drop=True)
            bases.to_csv(args.base_out, index=False)
            print('Saved base parameter sets to', args.base_out)
    print('Saved to', args.out)
