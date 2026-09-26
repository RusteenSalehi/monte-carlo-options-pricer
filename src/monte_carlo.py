"""Monte Carlo pricing for European options using simulated GBM paths."""

from dataclasses import dataclass

import numpy as np

from src.gbm import simulate_paths


@dataclass(frozen=True)
class MCResult:
    """Result of a Monte Carlo option pricing run.

    Attributes:
        price: Discounted mean payoff across all simulated paths.
        std_error: Standard error of the price estimate (sample std of the
            discounted payoffs, ddof=1, divided by sqrt(n_paths)).
        n_paths: Number of simulated paths used.
    """

    price: float
    std_error: float
    n_paths: int


def _summarize(discounted_payoffs: np.ndarray) -> MCResult:
    n_paths = discounted_payoffs.size
    return MCResult(
        price=float(discounted_payoffs.mean()),
        std_error=float(discounted_payoffs.std(ddof=1) / np.sqrt(n_paths)),
        n_paths=n_paths,
    )


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
    S_T = simulate_paths(S0, r, sigma, T, n_steps, n_paths, seed=seed)[:, -1]
    payoffs = np.maximum(S_T - K, 0.0)
    return _summarize(np.exp(-r * T) * payoffs)


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
    S_T = simulate_paths(S0, r, sigma, T, n_steps, n_paths, seed=seed)[:, -1]
    payoffs = np.maximum(K - S_T, 0.0)
    return _summarize(np.exp(-r * T) * payoffs)
