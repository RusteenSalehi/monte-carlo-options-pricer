"""Tests for src.monte_carlo."""

import pytest

from src import black_scholes, monte_carlo

S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
N_STEPS = 12
N_PATHS = 200_000
SEED = 42


@pytest.mark.parametrize(
    "pricer",
    [monte_carlo.european_call_price, monte_carlo.antithetic_call_price],
)
def test_call_price_within_3_std_errors_of_black_scholes(pricer):
    bs = black_scholes.call_price(S0, K, T, r, sigma)
    mc = pricer(S0, K, T, r, sigma, N_STEPS, N_PATHS, seed=SEED)
    assert mc.n_paths == N_PATHS
    assert abs(mc.price - bs) < 3 * mc.std_error


def test_antithetic_reduces_std_error_for_atm_call():
    plain = monte_carlo.european_call_price(S0, K, T, r, sigma, N_STEPS, N_PATHS, seed=SEED)
    anti = monte_carlo.antithetic_call_price(S0, K, T, r, sigma, N_STEPS, N_PATHS, seed=SEED)
    assert anti.std_error < plain.std_error


@pytest.mark.parametrize(
    "pricer",
    [monte_carlo.antithetic_call_price, monte_carlo.antithetic_put_price],
)
def test_antithetic_odd_n_paths_raises(pricer):
    with pytest.raises(ValueError):
        pricer(S0, K, T, r, sigma, N_STEPS, 1_001, seed=SEED)
