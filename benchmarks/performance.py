"""Time the GBM simulators and check they produce the same European price.

Run from the repo root: python -m benchmarks.performance
"""

import csv
import time
from pathlib import Path

import numpy as np

from src.gbm import (
    generate_normals,
    simulate_paths,
    simulate_paths_vectorized,
    simulate_terminal,
)
from src.gbm_numba import NUMBA_AVAILABLE, simulate_paths_numba
from src.monte_carlo import _discounted_payoffs

S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.2
SEED = 42
CONFIGS = [(100_000, 252), (1_000_000, 12)]
PRICE_CONFIG = (100_000, 252)
N_REPEATS = 5
CSV_PATH = Path("results/benchmarks.csv")


def best_of(func):
    """Return (best wall time in ms over N_REPEATS runs, last result)."""
    best = float("inf")
    for _ in range(N_REPEATS):
        start = time.perf_counter()
        result = func()
        best = min(best, time.perf_counter() - start)
    return best * 1000, result


def main() -> None:
    print(f"S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}, seed={SEED}, "
          f"best of {N_REPEATS} runs")
    if not NUMBA_AVAILABLE:
        print("numba not available; skipping the numba method")

    rows = []
    for n_paths, n_steps in CONFIGS:
        Z = generate_normals(n_paths, n_steps, seed=SEED)
        # Built from the same Z so the terminal-only price matches the path
        # simulators to rounding; its timing does not depend on where Z came from.
        Z_terminal = Z.sum(axis=1) / np.sqrt(n_steps)

        methods = [
            ("loop", lambda: simulate_paths(S0, r, sigma, T, Z)[:, -1]),
            ("vectorized", lambda: simulate_paths_vectorized(S0, r, sigma, T, Z)[:, -1]),
        ]
        if NUMBA_AVAILABLE:
            simulate_paths_numba(S0, r, sigma, T, Z[:1])  # compile outside the timing
            methods.append(("numba", lambda: simulate_paths_numba(S0, r, sigma, T, Z)[:, -1]))
        methods.append(("terminal", lambda: simulate_terminal(S0, r, sigma, T, Z_terminal)))

        loop_ms = None
        for name, func in methods:
            ms, S_T = best_of(func)
            loop_ms = loop_ms or ms
            price = None
            if (n_paths, n_steps) == PRICE_CONFIG:
                price = float(_discounted_payoffs(S_T, K, r, T, "call").mean())
            rows.append((name, n_paths, n_steps, ms, loop_ms / ms, price))

    print(f"\n{'method':>10}  {'config':>15}  {'best ms':>10}  {'speedup':>8}  {'call price':>11}")
    for name, n_paths, n_steps, ms, speedup, price in rows:
        config = f"{n_paths:,}x{n_steps}"
        price_str = f"{price:.6f}" if price is not None else ""
        print(f"{name:>10}  {config:>15}  {ms:>10.1f}  {speedup:>7.1f}x  {price_str:>11}")

    print(f"\n{'config':>15}  {'full paths MB':>14}  {'terminal MB':>12}")
    for n_paths, n_steps in CONFIGS:
        full_mb = n_paths * (n_steps + 1) * 8 / 1e6
        terminal_mb = n_paths * 8 / 1e6
        print(f"{f'{n_paths:,}x{n_steps}':>15}  {full_mb:>14.1f}  {terminal_mb:>12.1f}")

    CSV_PATH.parent.mkdir(exist_ok=True)
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "n_paths", "n_steps", "best_ms", "speedup_vs_loop"])
        for name, n_paths, n_steps, ms, speedup, _ in rows:
            writer.writerow([name, n_paths, n_steps, f"{ms:.3f}", f"{speedup:.3f}"])
    print(f"\nWrote {CSV_PATH}")


if __name__ == "__main__":
    main()
