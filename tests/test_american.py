"""Tests for src.binomial and src.american."""

import pytest

from src import black_scholes
from src.american import longstaff_schwartz_price
from src.binomial import crr_price

S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

# Longstaff-Schwartz (2001) American put benchmark.
P_S0, P_K, P_T, P_r, P_sigma = 36.0, 40.0, 1.0, 0.06, 0.2


@pytest.mark.parametrize(
    "option_type, bs_price",
    [("call", black_scholes.call_price), ("put", black_scholes.put_price)],
)
def test_binomial_european_matches_black_scholes(option_type, bs_price):
    tree = crr_price(S0, K, T, r, sigma, 1000, option_type, american=False)
    assert abs(tree - bs_price(S0, K, T, r, sigma)) < 0.01


def test_binomial_american_call_equals_european_call():
    american = crr_price(S0, K, T, r, sigma, 1000, "call", american=True)
    european = crr_price(S0, K, T, r, sigma, 1000, "call", american=False)
    assert abs(american - european) < 1e-6


def test_binomial_american_put_exceeds_european_put():
    american = crr_price(P_S0, P_K, P_T, P_r, P_sigma, 1000, "put", american=True)
    european = crr_price(P_S0, P_K, P_T, P_r, P_sigma, 1000, "put", american=False)
    assert american > european


def test_lsm_american_put_matches_binomial():
    lsm = longstaff_schwartz_price(P_S0, P_K, P_T, P_r, P_sigma, 50, 100_000, "put", seed=42)
    tree = crr_price(P_S0, P_K, P_T, P_r, P_sigma, 2000, "put", american=True)
    # The extra allowance covers LSM's known low bias from its suboptimal
    # exercise rule, which more paths do not remove.
    assert abs(lsm.price - tree) < 3 * lsm.std_error + 0.03


def test_lsm_american_put_at_least_european_put():
    lsm = longstaff_schwartz_price(P_S0, P_K, P_T, P_r, P_sigma, 50, 100_000, "put", seed=42)
    european = black_scholes.put_price(P_S0, P_K, P_T, P_r, P_sigma)
    assert lsm.price >= european - 3 * lsm.std_error
