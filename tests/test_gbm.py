"""Tests for src.gbm."""

import numpy as np
import pytest

from src import gbm
from src.gbm_numba import NUMBA_AVAILABLE, simulate_paths_numba


def test_output_shape():
    Z = gbm.generate_normals(n_paths=200, n_steps=50, seed=0)
    S = gbm.simulate_paths(100.0, 0.05, 0.2, 1.0, Z)
    assert S.shape == (200, 51)


def test_initial_column_is_S0():
    S0 = 100.0
    Z = gbm.generate_normals(n_paths=200, n_steps=50, seed=0)
    S = gbm.simulate_paths(S0, 0.05, 0.2, 1.0, Z)
    assert np.all(S[:, 0] == S0)


def test_zero_volatility_is_deterministic():
    S0, r, T, n_steps = 100.0, 0.05, 2.0, 100
    Z = gbm.generate_normals(n_paths=10, n_steps=n_steps, seed=0)
    S = gbm.simulate_paths(S0, r, 0.0, T, Z)

    times = np.linspace(0.0, T, n_steps + 1)
    expected = S0 * np.exp(r * times)
    np.testing.assert_allclose(S, np.broadcast_to(expected, S.shape), rtol=1e-12)


def test_same_Z_gives_identical_paths():
    Z = gbm.generate_normals(n_paths=100, n_steps=20, seed=1)
    S1 = gbm.simulate_paths(100.0, 0.05, 0.2, 1.0, Z)
    S2 = gbm.simulate_paths(100.0, 0.05, 0.2, 1.0, Z)
    np.testing.assert_array_equal(S1, S2)


def test_generate_normals_same_seed_is_reproducible():
    Z1 = gbm.generate_normals(n_paths=100, n_steps=20, seed=123)
    Z2 = gbm.generate_normals(n_paths=100, n_steps=20, seed=123)
    assert Z1.shape == (100, 20)
    np.testing.assert_array_equal(Z1, Z2)


def test_1d_Z_raises():
    with pytest.raises(ValueError):
        gbm.simulate_paths(100.0, 0.05, 0.2, 1.0, np.zeros(10))


def test_vectorized_matches_loop():
    Z = gbm.generate_normals(n_paths=500, n_steps=50, seed=2)
    loop = gbm.simulate_paths(100.0, 0.05, 0.2, 1.0, Z)
    vectorized = gbm.simulate_paths_vectorized(100.0, 0.05, 0.2, 1.0, Z)
    np.testing.assert_allclose(vectorized, loop, rtol=1e-10)


def test_terminal_matches_loop_terminal_column():
    n_steps = 50
    Z = gbm.generate_normals(n_paths=500, n_steps=n_steps, seed=3)
    loop_terminal = gbm.simulate_paths(100.0, 0.05, 0.2, 1.0, Z)[:, -1]
    # The sum of n independent standard normals divided by sqrt(n) is itself
    # standard normal, and it reproduces the same terminal price exactly
    # because the drift terms sum to (r - sigma**2/2) * T.
    Z_terminal = Z.sum(axis=1) / np.sqrt(n_steps)
    terminal = gbm.simulate_terminal(100.0, 0.05, 0.2, 1.0, Z_terminal)
    np.testing.assert_allclose(terminal, loop_terminal, rtol=1e-10)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba is not installed")
def test_numba_matches_loop():
    Z = gbm.generate_normals(n_paths=500, n_steps=50, seed=4)
    loop = gbm.simulate_paths(100.0, 0.05, 0.2, 1.0, Z)
    compiled = simulate_paths_numba(100.0, 0.05, 0.2, 1.0, Z)
    np.testing.assert_allclose(compiled, loop, rtol=1e-10)
