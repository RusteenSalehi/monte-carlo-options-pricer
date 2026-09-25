"""Compare Monte Carlo call prices against Black-Scholes as n_paths grows."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.black_scholes import call_price
from src.monte_carlo import european_call_price

S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.2
N_STEPS = 12  # full paths are stored, so keep this small at 1M paths
SEED = 42
N_PATHS_LEVELS = [1_000, 10_000, 100_000, 1_000_000]


def main() -> None:
    bs = call_price(S0, K, T, r, sigma)
    print(f"S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}, n_steps={N_STEPS}")
    print(f"Black-Scholes call price: {bs:.6f}\n")
    print(f"{'n_paths':>10}  {'MC price':>10}  {'abs error':>10}")

    for n_paths in N_PATHS_LEVELS:
        mc = european_call_price(S0, K, T, r, sigma, N_STEPS, n_paths, seed=SEED)
        print(f"{n_paths:>10,}  {mc:>10.6f}  {abs(mc - bs):>10.6f}")


if __name__ == "__main__":
    main()
