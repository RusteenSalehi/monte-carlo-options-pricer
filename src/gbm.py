"""Geometric Brownian Motion path simulation for the underlying asset."""

import numpy as np


def simulate_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
) -> np.ndarray:
    """Simulate asset price paths under Geometric Brownian Motion.

    Args:
        S0: Initial price of the underlying asset.
        r: Risk-free interest rate (annualized).
        sigma: Volatility of the underlying asset (annualized).
        T: Time horizon, in years.
        n_steps: Number of time steps in each path.
        n_paths: Number of simulated paths to generate.

    Returns:
        An array of shape (n_paths, n_steps + 1) containing the simulated
        price paths, including the initial price at index 0.
    """
    dt = T / n_steps
    Z = np.random.standard_normal((n_paths, n_steps))

    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    for t in range(n_steps):
        S[:, t + 1] = S[:, t] * np.exp(
            (r - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * Z[:, t]
        )

    return S
