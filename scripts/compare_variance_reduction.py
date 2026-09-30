"""Compare plain and antithetic Monte Carlo call pricing as n_paths grows."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.monte_carlo import antithetic_call_price, european_call_price

S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.2
N_STEPS = 12
SEED = 42
N_PATHS_LEVELS = [1_000, 10_000, 100_000, 1_000_000]


def main() -> None:
    print(f"S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}, n_steps={N_STEPS}\n")
    print(
        f"{'n_paths':>10}  {'plain':>10}  {'plain SE':>10}  "
        f"{'anti':>10}  {'anti SE':>10}  {'VR factor':>10}"
    )

    # The variance reduction factor (plain SE / antithetic SE) ** 2 is how
    # many times more plain paths you would need to match the antithetic
    # precision, since SE shrinks like 1 / sqrt(n_paths).
    for n_paths in N_PATHS_LEVELS:
        plain = european_call_price(S0, K, T, r, sigma, N_STEPS, n_paths, seed=SEED)
        anti = antithetic_call_price(S0, K, T, r, sigma, N_STEPS, n_paths, seed=SEED)
        vr_factor = (plain.std_error / anti.std_error) ** 2
        print(
            f"{n_paths:>10,}  {plain.price:>10.6f}  {plain.std_error:>10.6f}  "
            f"{anti.price:>10.6f}  {anti.std_error:>10.6f}  {vr_factor:>10.3f}"
        )


if __name__ == "__main__":
    main()
