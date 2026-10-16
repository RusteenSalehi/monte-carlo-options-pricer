"""Longstaff-Schwartz least-squares Monte Carlo pricing for American options."""

from typing import Literal

import numpy as np

from src.gbm import generate_normals, simulate_paths
from src.monte_carlo import MCResult, _payoffs


def longstaff_schwartz_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    n_paths: int,
    option_type: Literal["call", "put"] = "put",
    seed: int | None = None,
    degree: int = 2,
) -> MCResult:
    """Estimate an American option price with the Longstaff-Schwartz method.

    Exercise is allowed at each of the n_steps time steps. The continuation
    value is estimated by regressing discounted future cashflows on a
    polynomial in S / K of the given degree.

    The estimate is biased slightly low: the exercise rule comes from a
    regression fit, so it is suboptimal, and any suboptimal rule is worth
    less than the optimal one.

    Args:
        S0: Initial price of the underlying asset.
        K: Strike price of the option.
        T: Time to expiration, in years.
        r: Risk-free interest rate (annualized).
        sigma: Volatility of the underlying asset (annualized).
        n_steps: Number of time steps (exercise dates) in each path.
        n_paths: Number of simulated paths.
        option_type: "call" or "put".
        seed: Optional seed for the random number generator.
        degree: Degree of the polynomial regression basis.

    Returns:
        An MCResult with the price, the standard error of the discounted
        path cashflows, and the number of paths used.
    """
    dt = T / n_steps
    disc = np.exp(-r * dt)

    Z = generate_normals(n_paths, n_steps, seed=seed)
    S = simulate_paths(S0, r, sigma, T, Z)
    cashflow = _payoffs(S[:, -1], K, option_type)

    for t in range(n_steps - 1, 0, -1):
        cashflow *= disc
        immediate = _payoffs(S[:, t], K, option_type)
        # Only in-the-money paths face a real exercise decision; including
        # out-of-the-money paths would fit the regression to states where
        # exercising pays nothing and distort it where it matters.
        itm = np.flatnonzero(immediate > 0)
        if itm.size < degree + 1:
            continue
        # Scaling by K keeps the powers of x near 1, so the regression is
        # well-conditioned.
        basis = np.vander(S[itm, t] / K, degree + 1, increasing=True)
        coef, *_ = np.linalg.lstsq(basis, cashflow[itm], rcond=None)
        continuation = basis @ coef
        exercise = itm[immediate[itm] > continuation]
        cashflow[exercise] = immediate[exercise]

    discounted = cashflow * disc
    price = max(float(discounted.mean()), float(_payoffs(np.array(S0), K, option_type)))
    return MCResult(
        price=price,
        std_error=float(discounted.std(ddof=1) / np.sqrt(n_paths)),
        n_paths=n_paths,
    )
