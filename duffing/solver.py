"""Duffing equation solver utilities.

We solve: x'' + delta x' + alpha x + beta x^3 = gamma cos(omega t)

This module provides a simple RK45 integrator wrapper using scipy.solve_ivp.
"""
from typing import Tuple
import numpy as np
from scipy.integrate import solve_ivp


def duffing_rhs(t, y, delta, alpha, beta, gamma, omega):
    x, v = y
    dxdt = v
    dvdt = -delta * v - alpha * x - beta * x**3 + gamma * np.cos(omega * t)
    return [dxdt, dvdt]


def solve_duffing(t_span: Tuple[float, float], y0: Tuple[float, float],
                  params: dict, t_eval: np.ndarray = None):
    """Solve the Duffing equation.

    params: dict with keys delta, alpha, beta, gamma, omega

    Returns: Bunch with t, y (2 x N array)
    """
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 5000)

    sol = solve_ivp(duffing_rhs, t_span, y0, t_eval=t_eval,
                    args=(params['delta'], params['alpha'], params['beta'], params['gamma'], params['omega']),
                    rtol=1e-9, atol=1e-12)
    return sol
