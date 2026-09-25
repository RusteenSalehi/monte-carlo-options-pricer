"""Tests for src.black_scholes."""

import numpy as np
import pytest

from src import black_scholes


@pytest.mark.parametrize(
    "S, K, T, r, sigma",
    [
        (100.0, 100.0, 1.0, 0.05, 0.2),
        (100.0, 120.0, 0.5, 0.03, 0.3),
        (80.0, 70.0, 2.0, 0.01, 0.15),
        (50.0, 55.0, 0.25, 0.0, 0.5),
    ],
)
def test_put_call_parity(S, K, T, r, sigma):
    C = black_scholes.call_price(S, K, T, r, sigma)
    P = black_scholes.put_price(S, K, T, r, sigma)
    assert C - P == pytest.approx(S - K * np.exp(-r * T), abs=1e-10)
