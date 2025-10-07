"""Plot a Duffing solution for a given row of features/parameters.

Usage examples:
  python -m duffing.plot_sample --row "0.21,-0.78,0.62,1.427,1.2, ..."
  python -m duffing.plot_sample --csv duffing_dataset.csv --index 10
"""
from typing import Dict
import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv

from .solver import solve_duffing
from .visualize import plot_poincare, compute_perturbed_delta, plot_delta_comparison


HEADER = ['delta','alpha','beta','omega','gamma',
          'x_mean','x_rms','x_max','x_min','x_std','dom_freq','periodic','label_gamma']


def parse_row_string(row_str: str) -> Dict:
    parts = [p.strip() for p in row_str.split(',')]
    if len(parts) != len(HEADER):
        raise ValueError(f'expected {len(HEADER)} values, got {len(parts)}')
    vals = []
    for p in parts:
        if p.lower() in ('true','false'):
            vals.append(p.lower() == 'true')
        else:
            try:
                vals.append(float(p))
            except Exception:
                vals.append(p)
    return dict(zip(HEADER, vals))


def load_row_from_csv(path: str, index: int) -> Dict:
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    row = rows[index]
    # convert types
    out = {}
    for k in HEADER:
        v = row[k]
        if v.lower() in ('true','false'):
            out[k] = v.lower() == 'true'
        else:
            try:
                out[k] = float(v)
            except Exception:
                out[k] = v
    return out


def plot_from_params(params: Dict, t_span=(0,200), y0=(0.1,0.0)):
    # ensure numeric params exist
    req = ['delta','alpha','beta','gamma','omega']
    for r in req:
        if r not in params:
            raise KeyError(f'missing param {r}')

    sol = solve_duffing(t_span, y0, params)
    t = sol.t
    x = sol.y[0]
    v = sol.y[1]

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2)
    ax_ts = fig.add_subplot(gs[0, :])
    ax_phase = fig.add_subplot(gs[1, 0])
    ax_poin = fig.add_subplot(gs[1, 1])

    ax_ts.plot(t, x, label='x(t)')
    ax_ts.set_xlabel('t')
    ax_ts.set_ylabel('x')
    ax_ts.set_title(f"Duffing time-series (gamma={params['gamma']})")

    ax_phase.plot(x, v, linewidth=0.5)
    ax_phase.set_xlabel('x')
    ax_phase.set_ylabel('v')
    ax_phase.set_title('Phase portrait')

    plot_poincare(t, x, v, params['omega'], ax=ax_poin)
    ax_poin.set_title('Poincaré (stroboscopic)')

    # compute perturbed delta and show semilog plot (stability/divergence test)
    tvals, delta = compute_perturbed_delta(params, t_span=t_span, y0=y0)
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 3))
    plot_delta_comparison(tvals, None, delta, ax=ax2)
    ax2.set_title('Perturbation |x - x_perturbed| (semilog)')

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--row', type=str, help='CSV-style row string matching header')
    parser.add_argument('--csv', type=str, help='CSV file to load')
    parser.add_argument('--index', type=int, default=0, help='row index when using --csv')
    parser.add_argument('--t0', type=float, default=0.0)
    parser.add_argument('--tf', type=float, default=200.0)
    args = parser.parse_args()

    if args.row:
        params = parse_row_string(args.row)
    elif args.csv:
        params = load_row_from_csv(args.csv, args.index)
    else:
        parser.error('provide --row or --csv')

    plot_from_params(params, t_span=(args.t0, args.tf))


if __name__ == '__main__':
    main()
