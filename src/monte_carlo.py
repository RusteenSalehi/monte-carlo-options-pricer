"""Monte Carlo pricing for European options using simulated GBM paths."""

import numpy as np

from src.gbm import simulate_paths


def european_call_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> float:
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
        The discounted average call payoff across all simulated paths.
    """
    S_T = simulate_paths(S0, r, sigma, T, n_steps, n_paths, seed=seed)[:, -1]
    payoffs = np.maximum(S_T - K, 0.0)
    return float(np.exp(-r * T) * payoffs.mean())


def european_put_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> float:
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
        The discounted average put payoff across all simulated paths.
    """
    S_T = simulate_paths(S0, r, sigma, T, n_steps, n_paths, seed=seed)[:, -1]
    payoffs = np.maximum(K - S_T, 0.0)
    return float(np.exp(-r * T) * payoffs.mean())
