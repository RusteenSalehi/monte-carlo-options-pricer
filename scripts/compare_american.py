"""Compare binomial and Longstaff-Schwartz American put prices across S0.

Run from the repo root: python -m scripts.compare_american
"""

from src.american import longstaff_schwartz_price
from src.binomial import crr_price
from src.black_scholes import put_price

K = 40.0
T = 1.0
r = 0.06
sigma = 0.2
S0_LEVELS = [36.0, 38.0, 40.0, 42.0, 44.0]
TREE_STEPS = 2000
LSM_STEPS = 50
LSM_PATHS = 100_000
SEED = 42


def main() -> None:
    print(f"K={K}, T={T}, r={r}, sigma={sigma}; tree {TREE_STEPS} steps; "
          f"LSM {LSM_STEPS} steps, {LSM_PATHS:,} paths, seed {SEED}\n")
    print(f"{'S0':>5}  {'binomial':>9}  {'LSM':>9}  {'LSM SE':>8}  "
          f"{'European':>9}  {'EE prem':>8}")

    for S0 in S0_LEVELS:
        tree = crr_price(S0, K, T, r, sigma, TREE_STEPS, "put", american=True)
        lsm = longstaff_schwartz_price(S0, K, T, r, sigma, LSM_STEPS, LSM_PATHS, "put", seed=SEED)
        european = put_price(S0, K, T, r, sigma)
        print(f"{S0:>5.0f}  {tree:>9.4f}  {lsm.price:>9.4f}  {lsm.std_error:>8.4f}  "
              f"{european:>9.4f}  {tree - european:>8.4f}")


if __name__ == "__main__":
    main()
