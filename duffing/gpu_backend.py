"""GPU-backed Duffing solver and batched Benettin Lyapunov estimator.

This module provides a scaffold for running many Duffing ODE solves and
estimating Lyapunov exponents in parallel on CUDA using PyTorch and
torchdiffeq. It is intended as a drop-in, batched replacement for the
per-sample SciPy solver approach used elsewhere in this project.

Requirements:
- PyTorch with CUDA (match your system / supercomputer build)
- torchdiffeq (pip install torchdiffeq)

Notes:
- The Benettin estimator is implemented by integrating the tangent (variational)
  equations together with the base trajectory in segments and renormalizing the
  perturbation between segments. This avoids needing an ODE integrator that
  supports event-based renormalization.
- For best GPU utilization, batch many parameter sets (hundreds to thousands)
  per call. Tune `batch_size` for your hardware.
"""
from __future__ import annotations

from typing import Dict, Tuple
import math
import warnings

try:
    import torch
except Exception as e:
    raise ImportError("PyTorch is required for gpu_backend. Install torch with CUDA support.") from e

try:
    from torchdiffeq import odeint
except Exception as e:
    raise ImportError("torchdiffeq is required. Install with `pip install torchdiffeq`.") from e


def _duffing_rhs(t, y, params):
    """Compute RHS for a batched Duffing system.

    y: Tensor shape (B, 2) where columns are [x, v]
    params: dict of tensors each shape (B,)
    returns dy/dt with shape (B, 2)
    """
    # t is scalar tensor; some funcs expect it but we only use it for forcing
    x = y[:, 0]
    v = y[:, 1]
    delta = params['delta']
    alpha = params['alpha']
    beta = params['beta']
    gamma = params['gamma']
    omega = params['omega']

    dxdt = v
    dvdt = -delta * v - alpha * x - beta * x ** 3 + gamma * torch.cos(omega * t)
    return torch.stack([dxdt, dvdt], dim=1)


def _duffing_augmented_rhs(t, y_aug, params):
    """RHS for augmented system [x, v, dx, dv] where (dx,dv) is tangent vector.

    y_aug: (B,4)
    params: dict of tensors (B,)
    returns (B,4)
    """
    x = y_aug[:, 0]
    v = y_aug[:, 1]
    dx = y_aug[:, 2]
    dv = y_aug[:, 3]

    delta = params['delta']
    alpha = params['alpha']
    beta = params['beta']
    gamma = params['gamma']
    omega = params['omega']

    # base dynamics
    dxdt = v
    dvdt = -delta * v - alpha * x - beta * x ** 3 + gamma * torch.cos(omega * t)

    # Jacobian of base dynamics wrt (x,v): [[0,1],[-alpha-3*beta*x^2, -delta]]
    a11 = torch.zeros_like(x)
    a12 = torch.ones_like(x)
    a21 = -(alpha + 3.0 * beta * x ** 2)
    a22 = -delta

    # tangent evolution d/dt [dx,dv] = J * [dx,dv]
    d_dxdt = dv
    d_dvdt = a21 * dx + a22 * dv

    return torch.stack([dxdt, dvdt, d_dxdt, d_dvdt], dim=1)


def estimate_lyapunov_benettin_batched(
    params: Dict[str, torch.Tensor],
    y0: torch.Tensor = None,
    t0: float = 0.0,
    t1: float = 200.0,
    segments: int = 50,
    steps_per_segment: int = 100,
    eps: float = 1e-6,
    device: str = 'cuda',
) -> torch.Tensor:
    """Estimate largest Lyapunov exponent (LLE) for a batch of parameter sets.

    Args:
        params: dict with keys 'delta','alpha','beta','gamma','omega', values are
            1D tensors of length B on CPU or device.
        y0: initial state tensor shape (B,2). If None uses small common initial condition.
        t0,t1: integration interval.
        segments: number of renormalization segments (Benettin steps).
        steps_per_segment: number of internal integrator steps per segment.
        eps: initial perturbation norm.
        device: device string like 'cuda' or 'cpu'.

    Returns:
        Tensor of shape (B,) with estimated LLE (float), NaN for failed estimates.
    """
    # move params to device and verify shapes
    dev = torch.device(device if torch.cuda.is_available() and 'cuda' in device else 'cpu')
    B = None
    for k in ('delta', 'alpha', 'beta', 'gamma', 'omega'):
        if k not in params:
            raise ValueError(f"params dict must include '{k}'")
        params[k] = params[k].to(dev).contiguous()
        if B is None:
            B = params[k].shape[0]

    if y0 is None:
        y0 = torch.zeros((B, 2), dtype=torch.get_default_dtype(), device=dev)
        y0[:, 0] = 0.1  # small displacement

    # prepare initial tangent vectors (random small vectors, normalized to eps)
    rng = torch.manual_seed(0)
    delta0 = torch.randn((B, 2), device=dev)
    delta0 = delta0 / torch.norm(delta0, dim=1, keepdim=True).clamp_min(1e-12) * eps

    # segment timing
    total_steps = segments * steps_per_segment
    total_time = float(t1 - t0)
    if total_time <= 0:
        raise ValueError('t1 must be greater than t0')
    dt = total_time / total_steps

    # accumulate log norms per batch
    sum_logs = torch.zeros(B, device=dev)
    valid_mask = torch.ones(B, dtype=torch.bool, device=dev)

    # current base state and tangent
    x_curr = y0.clone().to(dev)
    delta_curr = delta0.clone()

    # pack params for closure
    p_for_rhs = {k: params[k] for k in params}

    for seg in range(segments):
        t_start = t0 + seg * steps_per_segment * dt
        t_end = t_start + steps_per_segment * dt
        t_span = torch.linspace(t_start, t_end, steps_per_segment + 1, device=dev)

        # integrate augmented system from x_curr, delta_curr
        y_aug0 = torch.cat([x_curr, delta_curr], dim=1)  # (B,4)

        try:
            sol = odeint(lambda tt, yy: _duffing_augmented_rhs(tt, yy, p_for_rhs), y_aug0, t_span, method='dopri5')
        except Exception as e:
            warnings.warn(f'ODE integration failed on segment {seg}: {e}')
            # mark all as invalid and break
            valid_mask[:] = False
            break

        y_end = sol[-1]  # (B,4)
        x_end = y_end[:, :2]
        delta_end = y_end[:, 2:]

        # norms of perturbations
        norms = torch.norm(delta_end, dim=1)
        # avoid zeros
        small = norms < 1e-16
        norms[small] = 1e-16

        # accumulate log growth: ln(norm / eps)
        sum_logs = sum_logs + torch.log(norms / eps)

        # renormalize to eps for next segment
        delta_curr = delta_end / norms.unsqueeze(1) * eps
        x_curr = x_end

    # compute LLE per batch: sum_logs / total_time
    lle = sum_logs / total_time

    # set invalid entries to NaN
    lle = lle.cpu()
    lle[~valid_mask.cpu()] = float('nan')
    return lle


def params_dict_to_tensors(param_list: list[Dict], device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """Convert a list of param dicts (each with scalar values) to a batched tensor dict.

    param_list: list of dicts length B
    returns dict with keys 'delta','alpha','beta','gamma','omega' mapping to tensors shape (B,)
    """
    keys = ['delta', 'alpha', 'beta', 'gamma', 'omega']
    B = len(param_list)
    out = {k: torch.empty(B, device=device, dtype=torch.get_default_dtype()) for k in keys}
    for i, p in enumerate(param_list):
        for k in keys:
            out[k][i] = float(p[k])
    return out


def integrate_batched_trajectories(params: Dict[str, torch.Tensor],
                                   y0: torch.Tensor = None,
                                   t_eval=None,
                                   device: str = 'cuda',
                                   method: str = 'dopri5'):
    """Integrate the Duffing base system for a batch of parameter sets.

    Args:
        params: dict of tensors (B,)
        y0: initial states shape (B,2) or None
        t_eval: torch tensor of times shape (T,) or numpy array; if None uses t0=0,t1=200 with 5000 points
        device: device string
        method: solver method for torchdiffeq

    Returns:
        sol: torch tensor shape (T, B, 2)
    """
    dev = torch.device(device if torch.cuda.is_available() and 'cuda' in device else 'cpu')
    # move params to device
    p_for_rhs = {}
    B = None
    for k in ('delta', 'alpha', 'beta', 'gamma', 'omega'):
        if k not in params:
            raise ValueError(f"params dict must include '{k}'")
        p_for_rhs[k] = params[k].to(dev).contiguous()
        if B is None:
            B = p_for_rhs[k].shape[0]

    if y0 is None:
        y0 = torch.zeros((B, 2), device=dev, dtype=torch.get_default_dtype())
        y0[:, 0] = 0.1

    # prepare t tensor
    if t_eval is None:
        t_eval = torch.linspace(0.0, 200.0, 5000, device=dev)
    else:
        if isinstance(t_eval, torch.Tensor):
            t_eval = t_eval.to(dev)
        else:
            try:
                import numpy as _np
                t_eval = torch.from_numpy(_np.asarray(t_eval)).to(dev)
            except Exception:
                t_eval = torch.linspace(0.0, 200.0, 5000, device=dev)

    try:
        sol = odeint(lambda tt, yy: _duffing_rhs(tt, yy, p_for_rhs), y0, t_eval, method=method)
    except Exception as e:
        raise
    return sol


__all__ = ['estimate_lyapunov_benettin_batched', 'params_dict_to_tensors', 'integrate_batched_trajectories']
