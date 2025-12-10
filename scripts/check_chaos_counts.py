"""Check chaotic counts in a generated Duffing dataset CSV.

Usage:
    python scripts/check_chaos_counts.py [--csv PATH] [--by-base] [--save PATH]

If the CSV contains a 'lyapunov' column, chaos is determined by lyapunov > 0.
Otherwise, if the CSV contains a 'periodic' column, rows with periodic == False
(or 'False') are counted as chaotic.

Options:
    --csv PATH    Path to CSV file (default: ./01122025_dataset.csv)
    --by-base     Report counts by unique base_id groups as well as rows
    --save PATH   Save chaotic rows to PATH (CSV)
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description='Count chaotic rows in a Duffing dataset CSV')
    p.add_argument('--csv', type=str, default='01122025_dataset.csv')
    p.add_argument('--by-base', action='store_true', help='Also report counts by unique base_id groups')
    p.add_argument('--save', type=str, default='', help='Optional path to save chaotic rows CSV')
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        sys.exit(2)

    df = pd.read_csv(csv_path)
    n = len(df)
    print(f"Loaded '{csv_path}' with {n} rows")

    results = {}

    # Prefer lyapunov if present
    if 'lyapunov' in df.columns:
        lyap = pd.to_numeric(df['lyapunov'], errors='coerce')
        chaotic_mask = lyap > 0
        results['by_lyapunov_rows'] = int(chaotic_mask.sum())
        results['by_lyapunov_frac'] = float(chaotic_mask.sum()) / max(1, n)
        print(f"Chaotic (lyapunov>0): {results['by_lyapunov_rows']} / {n} ({results['by_lyapunov_frac']:.3%})")
    else:
        print("No 'lyapunov' column found in CSV.")

    # Fall back to periodic column if present
    if 'periodic' in df.columns:
        # interpret various truthy/falsy representations
        col = df['periodic'].astype(object)
        # create mask for explicit False values
        false_mask = col.isin([False, 'False', 'false', 0, '0'])
        # also handle pandas boolean nulls
        # count only explicit False as chaotic (per project convention)
        chaotic_mask_p = false_mask.fillna(False)
        results['by_periodic_rows'] = int(chaotic_mask_p.sum())
        results['by_periodic_frac'] = float(chaotic_mask_p.sum()) / max(1, n)
        print(f"Chaotic (periodic==False): {results['by_periodic_rows']} / {n} ({results['by_periodic_frac']:.3%})")
    else:
        print("No 'periodic' column found in CSV.")

    # By unique base_id groups
    if args.by_base:
        if 'base_id' not in df.columns:
            print("No 'base_id' column present; cannot compute by-base stats.")
        else:
            bases = df.groupby('base_id')
            total_bases = bases.ngroups
            print(f"Total unique base_id groups: {total_bases}")
            # define chaotic per row using lyapunov if available, else periodic if available
            if 'lyapunov' in df.columns:
                base_chaotic = bases.apply(lambda g: (pd.to_numeric(g['lyapunov'], errors='coerce') > 0).any())
            elif 'periodic' in df.columns:
                base_chaotic = bases.apply(lambda g: g['periodic'].isin([False, 'False', 'false', 0, '0']).any())
            else:
                print('No way to determine chaos by base_id (no lyapunov or periodic columns).')
                base_chaotic = None

            if base_chaotic is not None:
                n_chaotic_bases = int(base_chaotic.sum())
                print(f"Chaotic base groups: {n_chaotic_bases} / {total_bases} ({n_chaotic_bases/ max(1,total_bases):.3%})")

    # Optionally save chaotic rows
    if args.save:
        outp = Path(args.save)
        if 'lyapunov' in df.columns:
            save_mask = pd.to_numeric(df['lyapunov'], errors='coerce') > 0
        elif 'periodic' in df.columns:
            save_mask = df['periodic'].isin([False, 'False', 'false', 0, '0'])
        else:
            print('No lyapunov or periodic column; nothing to save.')
            save_mask = None
        if save_mask is not None:
            df[save_mask].to_csv(outp, index=False)
            print(f"Saved {save_mask.sum()} chaotic rows to {outp}")


if __name__ == '__main__':
    main()
