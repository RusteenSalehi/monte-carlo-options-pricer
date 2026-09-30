"""Monte Carlo pricing for European options using simulated GBM paths."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.gbm import generate_normals, simulate_paths


@dataclass(frozen=True)
class MCResult:
    """Result of a Monte Carlo option pricing run.

    Attributes:
        price: Discounted mean payoff across all simulated paths.
        std_error: Standard error of the price estimate (sample std of the
            i.i.d. samples, ddof=1, divided by sqrt(number of samples)). The
            samples are the discounted payoffs for plain Monte Carlo and the
            antithetic pair averages for antithetic variates.
        n_paths: Total number of simulated paths used.
    """

    price: float
    std_error: float
    n_paths: int


def _summarize(samples: np.ndarray, n_paths: int) -> MCResult:
    """Build an MCResult from i.i.d. samples of the discounted payoff."""
    return MCResult(
        price=float(samples.mean()),
        std_error=float(samples.std(ddof=1) / np.sqrt(samples.size)),
        n_paths=n_paths,
    )


def _discounted_payoffs(
    S_T: np.ndarray,
    K: float,
    r: float,
    T: float,
    option_type: Literal["call", "put"],
) -> np.ndarray:
    """Discounted European payoffs for an array of terminal prices."""
    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    elif option_type == "put":
        payoffs = np.maximum(K - S_T, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    return np.exp(-r * T) * payoffs


def _plain_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    n_paths: int,
    seed: int | None,
    option_type: Literal["call", "put"],
) -> MCResult:
    Z = generate_normals(n_paths, n_steps, seed=seed)
    S_T = simulate_paths(S0, r, sigma, T, Z)[:, -1]
    return _summarize(_discounted_payoffs(S_T, K, r, T, option_type), n_paths)


def _antithetic_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    n_paths: int,
    seed: int | None,
    option_type: Literal["call", "put"],
) -> MCResult:
    if n_paths % 2 != 0:
        raise ValueError(f"n_paths must be even for antithetic variates, got {n_paths}")

    Z = generate_normals(n_paths // 2, n_steps, seed=seed)
    S_T_pos = simulate_paths(S0, r, sigma, T, Z)[:, -1]
    S_T_neg = simulate_paths(S0, r, sigma, T, -Z)[:, -1]
    payoff_pos = _discounted_payoffs(S_T_pos, K, r, T, option_type)
    payoff_neg = _discounted_payoffs(S_T_neg, K, r, T, option_type)

    # The payoffs from Z and -Z are deliberately negatively correlated, so the
    # n_paths individual payoffs are NOT independent samples. Treating them as
    # i.i.d. would ignore that correlation and give the wrong standard error.
    # The pair averages, however, are i.i.d. across pairs (each uses its own
    # independent row of Z), so the SE is computed from the n_paths // 2
    # pair averages.
    pair_averages = (payoff_pos + payoff_neg) / 2
    return _summarize(pair_averages, n_paths)


def european_call_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> MCResult:
    """Estimate the price of a European call option by Monte Carlo simulation.

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
        An MCResult with the discounted average call payoff across all
        simulated paths (price), its standard error (std_error), and the
        number of paths used (n_paths).
    """
    return _plain_price(S0, K, T, r, sigma, n_steps, n_paths, seed, "call")


def european_put_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> MCResult:
    """Estimate the price of a European put option by Monte Carlo simulation.

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
        An MCResult with the discounted average put payoff across all
        simulated paths (price), its standard error (std_error), and the
        number of paths used (n_paths).
    """
    return _plain_price(S0, K, T, r, sigma, n_steps, n_paths, seed, "put")


def antithetic_call_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> MCResult:
    """Estimate a European call price by Monte Carlo with antithetic variates.

    Each normal draw Z is paired with its mirror -Z, so n_paths // 2 shock
    grids produce n_paths total paths.

    Args:
        S0: Initial price of the underlying asset.
        K: Strike price of the option.
        T: Time to expiration, in years.
        r: Risk-free interest rate (annualized).
        sigma: Volatility of the underlying asset (annualized).
        n_steps: Number of time steps in each simulated path.
        n_paths: Total number of simulated paths (must be even).
        seed: Optional seed for the random number generator.

    Returns:
        An MCResult with the mean of the antithetic pair-averaged discounted
        call payoffs (price), its standard error computed over the
        n_paths // 2 pair averages (std_error), and the total number of
        paths used (n_paths).

    Raises:
        ValueError: If n_paths is odd.
    """
    return _antithetic_price(S0, K, T, r, sigma, n_steps, n_paths, seed, "call")


def antithetic_put_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> MCResult:
    """Estimate a European put price by Monte Carlo with antithetic variates.

    Each normal draw Z is paired with its mirror -Z, so n_paths // 2 shock
    grids produce n_paths total paths.

    Args:
        S0: Initial price of the underlying asset.
        K: Strike price of the option.
        T: Time to expiration, in years.
        r: Risk-free interest rate (annualized).
        sigma: Volatility of the underlying asset (annualized).
        n_steps: Number of time steps in each simulated path.
        n_paths: Total number of simulated paths (must be even).
        seed: Optional seed for the random number generator.

    Returns:
        An MCResult with the mean of the antithetic pair-averaged discounted
        put payoffs (price), its standard error computed over the
        n_paths // 2 pair averages (std_error), and the total number of
        paths used (n_paths).

    Raises:
        ValueError: If n_paths is odd.
    """
    return _antithetic_price(S0, K, T, r, sigma, n_steps, n_paths, seed, "put")
