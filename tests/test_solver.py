import numpy as np
from duffing.solver import solve_duffing


def test_solve_duffing_basic():
    params = {'delta': 0.1, 'alpha': -1.0, 'beta': 1.0, 'gamma': 0.3, 'omega': 1.0}
    sol = solve_duffing((0, 10), (0.1, 0.0), params)
    assert sol.t[0] == 0
    assert sol.y.shape[0] == 2
    assert len(sol.t) > 1
