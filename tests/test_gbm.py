"""Tests for src.gbm."""

import numpy as np
import pytest

from src import gbm


def test_output_shape():
    S = gbm.simulate_paths(100.0, 0.05, 0.2, 1.0, n_steps=50, n_paths=200, seed=0)
    assert S.shape == (200, 51)


def test_initial_column_is_S0():
    S0 = 100.0
    S = gbm.simulate_paths(S0, 0.05, 0.2, 1.0, n_steps=50, n_paths=200, seed=0)
    assert np.all(S[:, 0] == S0)


def test_zero_volatility_is_deterministic():
    S0, r, T, n_steps = 100.0, 0.05, 2.0, 100
    S = gbm.simulate_paths(S0, r, 0.0, T, n_steps=n_steps, n_paths=10, seed=0)

    times = np.linspace(0.0, T, n_steps + 1)
    expected = S0 * np.exp(r * times)
    np.testing.assert_allclose(S, np.broadcast_to(expected, S.shape), rtol=1e-12)
    assert S[:, -1] == pytest.approx(S0 * np.exp(r * T), rel=1e-12)
