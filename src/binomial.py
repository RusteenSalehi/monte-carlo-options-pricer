"""Cox-Ross-Rubinstein binomial tree pricing for European and American options."""

from typing import Literal

import numpy as np

from src.monte_carlo import _payoffs


def crr_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    option_type: Literal["call", "put"],
    american: bool = True,
) -> float:
    """Price an option on a Cox-Ross-Rubinstein binomial tree.

    Args:
        S0: Initial price of the underlying asset.
        K: Strike price of the option.
        T: Time to expiration, in years.
        r: Risk-free interest rate (annualized).
        sigma: Volatility of the underlying asset (annualized).
        n_steps: Number of time steps in the tree.
        option_type: "call" or "put".
        american: If True, allow early exercise at every node.

    Returns:
        The option price at time 0.
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Node j at level n has j up-moves: S = S0 * u**j * d**(n - j).
    j = np.arange(n_steps + 1)
    V = _payoffs(S0 * u**j * d ** (n_steps - j), K, option_type)
    for n in range(n_steps - 1, -1, -1):
        V = disc * (p * V[1:] + (1 - p) * V[:-1])
        if american:
            S = S0 * u ** j[: n + 1] * d ** (n - j[: n + 1])
            V = np.maximum(V, _payoffs(S, K, option_type))
    return float(V[0])
