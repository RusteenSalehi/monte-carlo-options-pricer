"""Geometric Brownian Motion path simulation for the underlying asset."""

import numpy as np


def generate_normals(
    n_paths: int,
    n_steps: int,
    seed: int | None = None,
) -> np.ndarray:
    """Draw the standard normal shocks used to drive GBM paths.

    Args:
        n_paths: Number of simulated paths.
        n_steps: Number of time steps in each path.
        seed: Optional seed for the random number generator.

    Returns:
        An array of shape (n_paths, n_steps) of independent standard
        normal draws.
    """
    return np.random.default_rng(seed).standard_normal((n_paths, n_steps))


def simulate_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    Z: np.ndarray,
) -> np.ndarray:
    """Simulate asset price paths under Geometric Brownian Motion.

    The simulation is fully deterministic given its inputs: all randomness
    comes from the pre-generated shocks in Z.

    Args:
        S0: Initial price of the underlying asset.
        r: Risk-free interest rate (annualized).
        sigma: Volatility of the underlying asset (annualized).
        T: Time horizon, in years.
        Z: Standard normal shocks of shape (n_paths, n_steps); column t
            drives the step from time t to t + 1.

    Returns:
        An array of shape (n_paths, n_steps + 1) containing the simulated
        price paths, including the initial price at index 0.

    Raises:
        ValueError: If Z is not 2-dimensional.
    """
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2-dimensional, got shape {Z.shape}")

    n_paths, n_steps = Z.shape
    dt = T / n_steps
    drift = (r - sigma**2 / 2) * dt
    vol = sigma * np.sqrt(dt)

    S = np.empty((n_paths, n_steps + 1))
    S[:, 0] = S0
    for t in range(n_steps):
        S[:, t + 1] = S[:, t] * np.exp(drift + vol * Z[:, t])

    return S
