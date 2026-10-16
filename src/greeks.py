"""Black-Scholes Greeks: analytical, finite-difference Monte Carlo, and pathwise."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import norm

from src.black_scholes import _d1_d2
from src.gbm import generate_normals, simulate_paths
from src.monte_carlo import MCResult, _discounted_payoffs, _summarize


def _check_option_type(option_type: str) -> None:
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def bs_delta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: Literal["call", "put"]
) -> float:
    """Black-Scholes delta: sensitivity of the option price to S.

    Args:
        S: Current price of the underlying asset.
        K: Strike price of the option.
        T: Time to expiration, in years.
        r: Risk-free interest rate (annualized).
        sigma: Volatility of the underlying asset (annualized).
        option_type: "call" or "put".

    Returns:
        The option delta.
    """
    _check_option_type(option_type)
    d1, _ = _d1_d2(S, K, T, r, sigma)
    call_delta = norm.cdf(d1)
    return float(call_delta if option_type == "call" else call_delta - 1.0)


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes gamma: second derivative of the price in S.

    Identical for calls and puts.

    Args:
        S: Current price of the underlying asset.
        K: Strike price of the option.
        T: Time to expiration, in years.
        r: Risk-free interest rate (annualized).
        sigma: Volatility of the underlying asset (annualized).

    Returns:
        The option gamma.
    """
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes vega: sensitivity of the price to sigma.

    Identical for calls and puts. Expressed per unit change in sigma
    (per 1.00, i.e. 100 volatility points), not per percentage point.

    Args:
        S: Current price of the underlying asset.
        K: Strike price of the option.
        T: Time to expiration, in years.
        r: Risk-free interest rate (annualized).
        sigma: Volatility of the underlying asset (annualized).

    Returns:
        The option vega per unit change in sigma.
    """
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return float(S * norm.pdf(d1) * np.sqrt(T))


def bs_rho(
    S: float, K: float, T: float, r: float, sigma: float, option_type: Literal["call", "put"]
) -> float:
    """Black-Scholes rho: sensitivity of the price to r.

    Expressed per unit change in r (per 1.00, i.e. 100 percentage points),
    not per percentage point.

    Args:
        S: Current price of the underlying asset.
        K: Strike price of the option.
        T: Time to expiration, in years.
        r: Risk-free interest rate (annualized).
        sigma: Volatility of the underlying asset (annualized).
        option_type: "call" or "put".

    Returns:
        The option rho per unit change in r.
    """
    _check_option_type(option_type)
    _, d2 = _d1_d2(S, K, T, r, sigma)
    discounted_strike_T = K * T * np.exp(-r * T)
    if option_type == "call":
        return float(discounted_strike_T * norm.cdf(d2))
    return float(-discounted_strike_T * norm.cdf(-d2))


@dataclass(frozen=True)
class MCGreeks:
    """Monte Carlo finite-difference Greeks (vega and rho per unit change)."""

    delta: float
    gamma: float
    vega: float
    rho: float


def _price_from_Z(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    Z: np.ndarray,
    option_type: Literal["call", "put"],
) -> float:
    S_T = simulate_paths(S0, r, sigma, T, Z)[:, -1]
    return float(_discounted_payoffs(S_T, K, r, T, option_type).mean())


def mc_greeks(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    n_paths: int,
    option_type: Literal["call", "put"],
    seed: int | None = None,
) -> MCGreeks:
    """Estimate Greeks by central finite differences of Monte Carlo prices.

    Bumps are dS = 0.01 * S0, dsigma = 0.001, and dr = 0.0001. Vega and
    rho are per unit change in sigma and r.

    Args:
        S0: Initial price of the underlying asset.
        K: Strike price of the option.
        T: Time to expiration, in years.
        r: Risk-free interest rate (annualized).
        sigma: Volatility of the underlying asset (annualized).
        n_steps: Number of time steps in each simulated path.
        n_paths: Number of simulated paths.
        option_type: "call" or "put".
        seed: Optional seed for the random number generator.

    Returns:
        An MCGreeks with delta, gamma, vega, and rho.
    """
    # Every bumped price reuses the same Z. With independent random numbers,
    # the noise in each bumped price would dominate the tiny difference
    # between them, and the finite-difference estimate would blow up as the
    # bump shrinks.
    Z = generate_normals(n_paths, n_steps, seed=seed)

    def price(S0=S0, r=r, sigma=sigma):
        return _price_from_Z(S0, K, T, r, sigma, Z, option_type)

    dS, dsigma, dr = 0.01 * S0, 0.001, 0.0001
    v_up, v_mid, v_down = price(S0=S0 + dS), price(), price(S0=S0 - dS)
    return MCGreeks(
        delta=(v_up - v_down) / (2 * dS),
        gamma=(v_up - 2 * v_mid + v_down) / dS**2,
        vega=(price(sigma=sigma + dsigma) - price(sigma=sigma - dsigma)) / (2 * dsigma),
        rho=(price(r=r + dr) - price(r=r - dr)) / (2 * dr),
    )


def pathwise_call_delta(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> MCResult:
    """Estimate a European call delta with the pathwise derivative method.

    Args:
        S0: Initial price of the underlying asset.
        K: Strike price of the option.
        T: Time to expiration, in years.
        r: Risk-free interest rate (annualized).
        sigma: Volatility of the underlying asset (annualized).
        n_steps: Number of time steps in each simulated path.
        n_paths: Number of simulated paths.
        seed: Optional seed for the random number generator.

    Returns:
        An MCResult with the delta estimate (price field), its standard
        error, and the number of paths used.
    """
    Z = generate_normals(n_paths, n_steps, seed=seed)
    S_T = simulate_paths(S0, r, sigma, T, Z)[:, -1]
    # This differentiates the payoff along each path instead of re-pricing,
    # so it has its own standard error and needs no bump size.
    samples = np.exp(-r * T) * (S_T > K) * S_T / S0
    return _summarize(samples, n_paths)
