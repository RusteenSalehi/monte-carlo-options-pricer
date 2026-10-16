"""Numba-compiled GBM path simulation, matching src.gbm.simulate_paths."""

import numpy as np

# Fall back to plain Python if numba is missing or broken, so importing this
# module never breaks the rest of the project. Callers that care about speed
# check NUMBA_AVAILABLE.
try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(func):
        return func


@njit
def _simulate_paths_loop(S0, r, sigma, T, Z):
    n_paths, n_steps = Z.shape
    dt = T / n_steps
    drift = (r - sigma**2 / 2) * dt
    vol = sigma * np.sqrt(dt)

    S = np.empty((n_paths, n_steps + 1))
    for i in range(n_paths):
        S[i, 0] = S0
        for t in range(n_steps):
            S[i, t + 1] = S[i, t] * np.exp(drift + vol * Z[i, t])
    return S


def simulate_paths_numba(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    Z: np.ndarray,
) -> np.ndarray:
    """Simulate GBM paths like src.gbm.simulate_paths, compiled with numba.

    The first call compiles the function, so it is much slower than later
    calls.

    Args:
        S0: Initial price of the underlying asset.
        r: Risk-free interest rate (annualized).
        sigma: Volatility of the underlying asset (annualized).
        T: Time horizon, in years.
        Z: Standard normal shocks of shape (n_paths, n_steps).

    Returns:
        An array of shape (n_paths, n_steps + 1) containing the simulated
        price paths, including the initial price at index 0.

    Raises:
        ValueError: If Z is not 2-dimensional.
    """
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2-dimensional, got shape {Z.shape}")
    return _simulate_paths_loop(float(S0), float(r), float(sigma), float(T), Z)
