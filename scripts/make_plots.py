"""Generate the project figures into results/ as PNG files.

Run from the repo root: python -m scripts.make_plots
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files only; the script must run with no display

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm

from src import black_scholes
from src.american import longstaff_schwartz_price
from src.binomial import crr_price
from src.gbm import generate_normals, simulate_paths, simulate_terminal
from src.greeks import bs_delta, bs_gamma, bs_rho, bs_vega, mc_greeks
from src.monte_carlo import (
    antithetic_call_price,
    control_variate_call_price,
    european_call_price,
)

RESULTS = Path("results")
SEED = 42
DPI = 150
S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
N_STEPS = 12  # European pricing only needs S_T, so a coarse grid is enough

# Fixed categorical order (slots 1-3), plus neutral inks for reference marks.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#8a8984"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#e6e5e1",
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "legend.frameon": False,
    "lines.linewidth": 2,
})


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(RESULTS / name, dpi=DPI)
    plt.close(fig)
    print(f"wrote {RESULTS / name}")


def strike_line(ax, vertical: bool = False) -> None:
    line = ax.axvline if vertical else ax.axhline
    line(K, color=MUTED, linestyle="--", linewidth=1.5, label=f"Strike K = {K:.0f}")


def sample_paths() -> None:
    n_steps = 252
    S = simulate_paths(S0, r, sigma, T, generate_normals(50, n_steps, seed=SEED))
    times = np.linspace(0, T, n_steps + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(times, S.T, color=BLUE, linewidth=0.8, alpha=0.5)
    strike_line(ax)
    ax.set(title=f"50 simulated GBM paths (S0 = {S0:.0f}, r = {r}, sigma = {sigma})",
           xlabel="Time (years)", ylabel="Asset price ($)")
    ax.legend(loc="upper left")
    save(fig, "sample_paths.png")


def terminal_distribution() -> None:
    S_T = simulate_terminal(S0, r, sigma, T, generate_normals(200_000, 1, seed=SEED)[:, 0])
    # Under GBM, log S_T is normal with mean log S0 + (r - sigma^2/2) T and sd sigma sqrt(T).
    density = lognorm(s=sigma * np.sqrt(T), scale=S0 * np.exp((r - sigma**2 / 2) * T))
    grid = np.linspace(S_T.min(), S_T.max(), 400)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(S_T, bins=120, density=True, color="#a3c5ef", edgecolor="white",
            linewidth=0.3, label="Simulated (200,000 paths)")
    ax.plot(grid, density.pdf(grid), color=ORANGE, label="Theoretical lognormal")
    strike_line(ax, vertical=True)
    ax.set(title=f"Terminal price distribution at T = {T:.0f} year",
           xlabel="Terminal price S_T ($)", ylabel="Probability density (1/$)")
    ax.legend()
    save(fig, "terminal_distribution.png")


def convergence_runs():
    # Even counts so antithetic pairing works at every level.
    n_paths = (np.logspace(3, 6, 10) / 2).round().astype(int) * 2
    pricers = {
        "Plain": european_call_price,
        "Antithetic": antithetic_call_price,
        "Control variate": control_variate_call_price,
    }
    results = {name: [pricer(S0, K, T, r, sigma, N_STEPS, int(n), seed=SEED) for n in n_paths]
               for name, pricer in pricers.items()}
    return n_paths, results


def convergence(n_paths, results) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for (name, res), color in zip(results.items(), (BLUE, ORANGE, AQUA)):
        ax.loglog(n_paths, [x.std_error for x in res], "o-", color=color, markersize=5, label=name)
    # Anchored at the plain method's first point so it sits beside the data.
    ref = results["Plain"][0].std_error * np.sqrt(n_paths[0] / n_paths)
    ax.loglog(n_paths, ref, "--", color=MUTED, linewidth=1.5, label="Slope -1/2 reference")
    ax.set(title="Standard error vs paths: lines are parallel because variance\n"
                 "reduction lowers the constant, not the rate",
           xlabel="Number of paths", ylabel="Standard error ($)")
    ax.legend()
    save(fig, "convergence.png")


def price_convergence(n_paths, results) -> None:
    plain = results["Plain"]
    price = np.array([x.price for x in plain])
    half_width = 1.96 * np.array([x.std_error for x in plain])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(n_paths, price - half_width, price + half_width, color=BLUE, alpha=0.18,
                    linewidth=0, label="95% band (+/- 1.96 SE)")
    ax.plot(n_paths, price, "o-", color=BLUE, markersize=5, label="Plain Monte Carlo")
    ax.axhline(black_scholes.call_price(S0, K, T, r, sigma), color=INK, linestyle="--",
               linewidth=1.5, label="Black-Scholes")
    ax.set_xscale("log")
    ax.set(title="At-the-money call price converges to Black-Scholes",
           xlabel="Number of paths", ylabel="Call price ($)")
    ax.legend()
    save(fig, "price_convergence.png")


def greeks_vs_spot() -> None:
    spots = np.linspace(60, 140, 161)
    mc_spots = np.linspace(60, 140, 9)
    mc = [mc_greeks(s, K, T, r, sigma, N_STEPS, 50_000, "call", seed=SEED) for s in mc_spots]
    panels = [
        ("Delta", lambda s: bs_delta(s, K, T, r, sigma, "call"), "delta", "Delta (per $1 in S)"),
        ("Gamma", lambda s: bs_gamma(s, K, T, r, sigma), "gamma", "Gamma (per $1 in S, squared)"),
        ("Vega", lambda s: bs_vega(s, K, T, r, sigma), "vega", "Vega ($ per 1.00 in sigma)"),
        ("Rho", lambda s: bs_rho(s, K, T, r, sigma, "call"), "rho", "Rho ($ per 1.00 in r)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, (title, exact, field, ylabel) in zip(axes.flat, panels):
        ax.plot(spots, [exact(s) for s in spots], color=BLUE, label="Analytical")
        ax.plot(mc_spots, [getattr(g, field) for g in mc], "o", color=ORANGE, markersize=7,
                markeredgecolor="white", label="Monte Carlo (50,000 paths)")
        ax.axvline(K, color=MUTED, linestyle="--", linewidth=1)
        ax.set(title=title, ylabel=ylabel)
    for ax in axes[1]:
        ax.set_xlabel("Spot price S0 ($)")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle(f"Call Greeks vs spot (K = {K:.0f}, dashed line), analytical vs finite-difference MC")
    save(fig, "greeks_vs_spot.png")


def american_vs_european() -> None:
    k, t, rate, vol = 40.0, 1.0, 0.06, 0.2
    spots = np.linspace(30, 50, 41)
    lsm_spots = np.linspace(30, 50, 11)
    tree = [crr_price(s, k, t, rate, vol, 2000, "put", american=True) for s in spots]
    european = [black_scholes.put_price(s, k, t, rate, vol) for s in spots]
    lsm = [longstaff_schwartz_price(s, k, t, rate, vol, 50, 50_000, "put", seed=SEED)
           for s in lsm_spots]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(spots, tree, color=BLUE, label="American, binomial (2,000 steps)")
    ax.errorbar(lsm_spots, [x.price for x in lsm], yerr=[x.std_error for x in lsm], fmt="o",
                color=ORANGE, markersize=6, capsize=3,
                label="American, Longstaff-Schwartz (+/- 1 SE)")
    ax.plot(spots, european, color=AQUA, label="European, Black-Scholes")
    ax.plot(spots, np.maximum(k - spots, 0), "--", color=MUTED, linewidth=1.5,
            label="Immediate exercise max(K - S, 0)")
    ax.set(title=f"American vs European put (K = {k:.0f}, T = {t:.0f}, r = {rate}, sigma = {vol})",
           xlabel="Spot price S0 ($)", ylabel="Put price ($)")
    ax.legend()
    save(fig, "american_vs_european.png")


def benchmarks() -> None:
    path = RESULTS / "benchmarks.csv"
    if not path.exists():
        print(f"{path} not found; run python -m benchmarks.performance first. "
              "Skipping benchmarks.png.")
        return
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    configs = list(dict.fromkeys((row["n_paths"], row["n_steps"]) for row in rows))
    # One panel per config: the configs differ in total work, so a shared
    # axis would squash the faster config's bars.
    fig, axes = plt.subplots(1, len(configs), figsize=(10, 3.5), squeeze=False)
    for ax, (n_paths, n_steps) in zip(axes[0], configs):
        subset = [row for row in rows if (row["n_paths"], row["n_steps"]) == (n_paths, n_steps)]
        times = [float(row["best_ms"]) for row in subset]
        ax.barh([row["method"] for row in subset], times, color=BLUE, height=0.6)
        for y, ms in enumerate(times):
            ax.text(ms, y, f" {ms:,.1f}", va="center", color=INK, fontsize=9)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.2)
        ax.set(title=f"{int(n_paths):,} paths x {n_steps} steps", xlabel="Best time (ms)")
    fig.suptitle("GBM simulation time by method (best of 5)")
    save(fig, "benchmarks.png")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    sample_paths()
    terminal_distribution()
    n_paths, results = convergence_runs()
    convergence(n_paths, results)
    price_convergence(n_paths, results)
    greeks_vs_spot()
    american_vs_european()
    benchmarks()


if __name__ == "__main__":
    main()
