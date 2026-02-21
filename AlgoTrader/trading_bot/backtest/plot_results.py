"""
backtest/plot_results.py
Visualises backtest equity curves and performance metrics.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from config.settings import RESULTS_DIR, INITIAL_CAPITAL


def plot_all(symbols: list, save: bool = True):
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig)

    # ── Equity Curves ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title("Portfolio Equity Curves", fontsize=14, fontweight="bold")
    ax1.axhline(INITIAL_CAPITAL, color="grey", linestyle="--", linewidth=0.8, label="Initial Capital")

    metrics_all = []
    for sym in symbols:
        safe = sym.replace("/", "_")
        eq_path = Path(RESULTS_DIR) / f"{safe}_equity.csv"
        mt_path = Path(RESULTS_DIR) / f"{safe}_backtest.json"
        if not eq_path.exists():
            continue
        eq = pd.read_csv(eq_path)["equity"]
        ax1.plot(eq.values, label=sym, linewidth=1.5)
        if mt_path.exists():
            with open(mt_path) as f:
                metrics_all.append(json.load(f))

    ax1.set_xlabel("Bars (hourly)")
    ax1.set_ylabel("Portfolio Value (USD)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    if metrics_all:
        # ── Bar: Total Returns ────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[1, 0])
        syms    = [m["symbol"] for m in metrics_all]
        returns = [m["total_return"] * 100 for m in metrics_all]
        colors  = ["green" if r > 0 else "red" for r in returns]
        ax2.bar(syms, returns, color=colors, alpha=0.8)
        ax2.set_title("Total Return (%)")
        ax2.set_ylabel("%")
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.grid(axis="y", alpha=0.3)

        # ── Bar: Sharpe Ratios ────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 1])
        sharpes = [m["sharpe_ratio"] for m in metrics_all]
        colors  = ["green" if s > 1 else "orange" if s > 0 else "red" for s in sharpes]
        ax3.bar(syms, sharpes, color=colors, alpha=0.8)
        ax3.set_title("Annualised Sharpe Ratio")
        ax3.axhline(1.0, color="green",  linestyle="--", linewidth=0.8, label="Good (1.0)")
        ax3.axhline(2.0, color="darkgreen", linestyle="--", linewidth=0.8, label="Excellent (2.0)")
        ax3.legend(fontsize=8)
        ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = Path(RESULTS_DIR) / "backtest_summary.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Chart saved -> {out}")
    plt.show()
    plt.close()