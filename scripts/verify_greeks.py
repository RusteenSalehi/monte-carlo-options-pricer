"""Compare analytical and Monte Carlo Greeks for at-the-money options.

Run from the repo root: python -m scripts.verify_greeks
"""

from src.greeks import (
    bs_delta,
    bs_gamma,
    bs_rho,
    bs_vega,
    mc_greeks,
    pathwise_call_delta,
)

S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.2
N_STEPS = 12
N_PATHS = 200_000
SEED = 42


def main() -> None:
    print(f"S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}, "
          f"n_steps={N_STEPS}, n_paths={N_PATHS:,}, seed={SEED}")
    print("Vega and rho are per unit change in sigma and r.")

    for option_type in ("call", "put"):
        mc = mc_greeks(S0, K, T, r, sigma, N_STEPS, N_PATHS, option_type, seed=SEED)
        rows = [
            ("delta", bs_delta(S0, K, T, r, sigma, option_type), mc.delta),
            ("gamma", bs_gamma(S0, K, T, r, sigma), mc.gamma),
            ("vega", bs_vega(S0, K, T, r, sigma), mc.vega),
            ("rho", bs_rho(S0, K, T, r, sigma, option_type), mc.rho),
        ]
        print(f"\n{option_type}")
        print(f"{'greek':>6}  {'analytical':>11}  {'MC':>11}  {'abs err':>10}  {'rel err':>9}")
        for name, exact, estimate in rows:
            abs_err = abs(estimate - exact)
            print(f"{name:>6}  {exact:>11.6f}  {estimate:>11.6f}  "
                  f"{abs_err:>10.6f}  {abs_err / abs(exact):>9.4%}")

    pw = pathwise_call_delta(S0, K, T, r, sigma, N_STEPS, N_PATHS, seed=SEED)
    print(f"\npathwise call delta: {pw.price:.6f} (SE {pw.std_error:.6f}), "
          f"analytical {bs_delta(S0, K, T, r, sigma, 'call'):.6f}")


if __name__ == "__main__":
    main()
