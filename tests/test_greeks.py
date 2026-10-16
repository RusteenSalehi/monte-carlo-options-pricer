"""Tests for src.greeks."""

import pytest

from src import black_scholes, greeks

S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
N_STEPS = 12
N_PATHS = 200_000
SEED = 42


def test_call_minus_put_delta_is_one():
    call = greeks.bs_delta(S0, K, T, r, sigma, "call")
    put = greeks.bs_delta(S0, K, T, r, sigma, "put")
    assert call - put == pytest.approx(1.0)


def test_analytical_delta_matches_finite_difference_of_call_price():
    h = 0.01
    fd = (black_scholes.call_price(S0 + h, K, T, r, sigma)
          - black_scholes.call_price(S0 - h, K, T, r, sigma)) / (2 * h)
    assert abs(greeks.bs_delta(S0, K, T, r, sigma, "call") - fd) < 1e-4


def test_mc_greeks_match_analytical_for_atm_call():
    mc = greeks.mc_greeks(S0, K, T, r, sigma, N_STEPS, N_PATHS, "call", seed=SEED)
    assert mc.delta == pytest.approx(greeks.bs_delta(S0, K, T, r, sigma, "call"), rel=0.02)
    assert mc.vega == pytest.approx(greeks.bs_vega(S0, K, T, r, sigma), rel=0.02)
    assert mc.rho == pytest.approx(greeks.bs_rho(S0, K, T, r, sigma, "call"), rel=0.02)
    assert mc.gamma == pytest.approx(greeks.bs_gamma(S0, K, T, r, sigma), rel=0.05)


def test_pathwise_call_delta_within_3_std_errors_of_analytical():
    bs = greeks.bs_delta(S0, K, T, r, sigma, "call")
    pw = greeks.pathwise_call_delta(S0, K, T, r, sigma, N_STEPS, N_PATHS, seed=SEED)
    assert abs(pw.price - bs) < 3 * pw.std_error
