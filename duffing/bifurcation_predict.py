"""Visualize model predictions alongside a bifurcation diagram sweep.

This utility overlays model-predicted periodic/chaotic labels on the Poincaré section points of a bifurcation sweep.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from duffing.visualize import compute_poincare_for_gamma
from duffing.model import load_dataset


def plot_bifurcation_with_predictions(
    base_params: dict,
    gammas: np.ndarray,
    model,
    n_samples: int = 150,
    t_transient_cycles: int = 30,
    ax=None,
    periodic_color: str = 'green',
    chaotic_color: str = 'red',
    marker: str = '.',
    alpha: float = 0.6,
    show: bool = True,
    legend: bool = True,
    **kwargs
):
    """
    For each gamma, compute Poincaré samples and use the model to predict periodic/chaotic.
    Paint x points green (periodic) or red (chaotic) according to the model prediction.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    all_g = []
    all_x = []
    all_labels = []
    for g in gammas:
        params = {**base_params, 'gamma': float(g)}
        # Model expects features: alpha, beta, delta, gamma, omega
        features = np.array([[params['alpha'], params['beta'], params['delta'], params['gamma'], params['omega']]])
        pred = model.predict(features)[0]
        # Compute Poincaré samples for this gamma
        samples = compute_poincare_for_gamma(params, t_transient_cycles=t_transient_cycles, n_samples=n_samples)
        all_g.extend([g] * len(samples))
        all_x.extend(samples.tolist())
        all_labels.extend([pred] * len(samples))
    all_g = np.array(all_g)
    all_x = np.array(all_x)
    all_labels = np.array(all_labels)
    # Plot periodic (1/True) in green, chaotic (0/False) in red
    mask_periodic = (all_labels == 1) | (all_labels == True)
    mask_chaotic = (all_labels == 0) | (all_labels == False)
    ax.plot(all_g[mask_periodic], all_x[mask_periodic], marker, color=periodic_color, alpha=alpha, label='Periodic (pred)')
    ax.plot(all_g[mask_chaotic], all_x[mask_chaotic], marker, color=chaotic_color, alpha=alpha, label='Chaotic (pred)')
    ax.set_xlabel('gamma')
    ax.set_ylabel('Poincaré x')
    ax.set_title('Bifurcation diagram with model predictions')
    if legend:
        ax.legend()
    if show:
        plt.show()
    return ax


def sweep_and_plot_from_csv(
    csv_path: str,
    model,
    base_id: int = 0,
    n_samples: int = 150,
    t_transient_cycles: int = 30,
    gamma_col: str = 'gamma',
    out_path: str = None,
    show: bool = True
):
    """
    Load a CSV (as used for bifurcation), extract base params for base_id, sweep gamma, and plot with model predictions.
    """
    df = pd.read_csv(csv_path)
    if 'base_id' not in df.columns:
        df['base_id'] = df.groupby(['delta','alpha','beta','omega']).ngroup()
    rows = df[df['base_id'] == base_id].sort_values(gamma_col)
    if rows.empty:
        raise ValueError(f'No rows found for base_id={base_id}')
    base = rows.iloc[0][['delta','alpha','beta','omega']].to_dict()
    gammas = rows[gamma_col].values.astype(float)
    ax = plot_bifurcation_with_predictions(base, gammas, model, n_samples=n_samples, t_transient_cycles=t_transient_cycles, show=show)
    if out_path:
        ax.figure.savefig(out_path, dpi=200)
    return ax
