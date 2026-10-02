"""Tests for src.gbm."""

import numpy as np
import pytest

from src import gbm


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
