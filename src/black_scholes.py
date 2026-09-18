"""Black-Scholes closed-form pricing for European options."""


def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Price a European call option under the Black-Scholes model.

    Args:
        S: Current price of the underlying asset.
        K: Strike price of the option.
        T: Time to expiration, in years.
        r: Risk-free interest rate (annualized).
        sigma: Volatility of the underlying asset (annualized).

    Returns:
        The theoretical price of the call option.
    """
    raise NotImplementedError


def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Price a European put option under the Black-Scholes model.

    Args:
        S: Current price of the underlying asset.
        K: Strike price of the option.
        T: Time to expiration, in years.
        r: Risk-free interest rate (annualized).
        sigma: Volatility of the underlying asset (annualized).

    Returns:
        The theoretical price of the put option.
    """
    raise NotImplementedError
