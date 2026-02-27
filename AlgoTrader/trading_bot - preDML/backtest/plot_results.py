"""
backtest/plot_results.py
═══════════════════════════════════════════════════════════════════════════════
Comprehensive reporting module. Generates:
  1. Detailed console report with full quantitative metrics
  2. Per-trade breakdown table
  3. Interactive HTML dashboard (equity curve, drawdown, trade P&L,
     win/loss distribution, regime breakdown, walk-forward heatmap)

Metrics:
  Performance   — Total return, CAGR, Sharpe, Sortino, Calmar, profit factor
  Risk          — Max drawdown, avg drawdown, VaR 95%, CVaR 95%, ann. volatility
  Trade quality — Win rate, avg win/loss, reward:risk ratio, expected value/trade
  Directional   — Long vs short P&L split, exit reason breakdown
  Consistency   — Walk-forward fold distribution, std dev, % profitable folds
═══════════════════════════════════════════════════════════════════════════════
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from datetime import datetime


# ─── Metric Computation ──────────────────────────────────────────────────────

def compute_full_metrics(equity_curve: list, trades: list,
                         initial_capital: float = 10_000,
                         bars_per_year: int = 8760) -> dict:
    """
    Compute the complete set of quantitative performance and risk metrics
    from a raw equity curve and trade list.
    """
    equity  = np.array(equity_curve, dtype=float)
    if len(equity) < 2:
        return {}

    returns = pd.Series(equity).pct_change().dropna()
    n_bars  = len(equity)
    n_years = max(n_bars / bars_per_year, 1e-6)

    # ── Returns & Ratios ─────────────────────────────────────────────────────
    total_return = (equity[-1] - equity[0]) / equity[0]
    cagr         = (equity[-1] / equity[0]) ** (1 / n_years) - 1
    ann_return   = returns.mean() * bars_per_year
    ann_vol      = returns.std()  * np.sqrt(bars_per_year)
    sharpe       = ann_return / (ann_vol + 1e-9)

    downside_ret = returns[returns < 0]
    downside_vol = downside_ret.std() * np.sqrt(bars_per_year) if len(downside_ret) > 1 else 1e-9
    sortino      = ann_return / (downside_vol + 1e-9)

    # ── Drawdown ─────────────────────────────────────────────────────────────
    roll_max   = pd.Series(equity).cummax()
    dd_series  = (pd.Series(equity) - roll_max) / (roll_max + 1e-9)
    max_dd     = float(dd_series.min())
    avg_dd     = float(dd_series[dd_series < -0.001].mean()) if (dd_series < -0.001).any() else 0.0
    calmar     = cagr / (abs(max_dd) + 1e-9)

    # Drawdown duration (in bars)
    in_dd, durations, count = dd_series < -0.001, [], 0
    for v in in_dd:
        if v:
            count += 1
        elif count:
            durations.append(count)
            count = 0
    if count:
        durations.append(count)
    max_dd_bars = max(durations) if durations else 0
    avg_dd_bars = float(np.mean(durations)) if durations else 0.0

    # ── Value at Risk ─────────────────────────────────────────────────────────
    var_95  = float(np.percentile(returns, 5))
    cvar_95 = float(returns[returns <= var_95].mean()) if (returns <= var_95).any() else var_95

    # ── Trade Statistics ─────────────────────────────────────────────────────
    n_trades = len(trades)
    if n_trades:
        pnls         = [t["pnl"] for t in trades]
        wins         = [p for p in pnls if p > 0]
        losses       = [p for p in pnls if p <= 0]
        win_rate     = len(wins) / n_trades
        avg_win      = float(np.mean(wins))   if wins   else 0.0
        avg_loss     = float(np.mean(losses)) if losses else 0.0
        best_trade   = float(max(pnls))
        worst_trade  = float(min(pnls))
        gross_profit = float(sum(wins))
        gross_loss   = float(abs(sum(losses)))
        profit_factor= gross_profit / (gross_loss + 1e-9)
        reward_risk  = abs(avg_win / avg_loss) if avg_loss else float("inf")
        exp_value    = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        n_longs      = sum(1 for t in trades if t.get("side") == "LONG")
        n_shorts     = sum(1 for t in trades if t.get("side") == "SHORT")
        long_pnl     = sum(t["pnl"] for t in trades if t.get("side") == "LONG")
        short_pnl    = sum(t["pnl"] for t in trades if t.get("side") == "SHORT")

        exit_reasons = {}
        for t in trades:
            r = t.get("reason", "UNKNOWN")
            exit_reasons[r] = exit_reasons.get(r, 0) + 1

        # Consecutive win/loss streaks
        streak, best_streak, worst_streak, cur = 0, 0, 0, None
        for p in pnls:
            win = p > 0
            if win == cur:
                streak += 1
            else:
                if cur is True:  best_streak  = max(best_streak,  streak)
                if cur is False: worst_streak = max(worst_streak, streak)
                streak, cur = 1, win
        best_win_streak  = best_streak
        worst_loss_streak= worst_streak
    else:
        win_rate = avg_win = avg_loss = best_trade = worst_trade = 0.0
        gross_profit = gross_loss = profit_factor = reward_risk = exp_value = 0.0
        n_longs = n_shorts = 0
        long_pnl = short_pnl = 0.0
        exit_reasons = {}
        best_win_streak = worst_loss_streak = 0

    return {
        "total_return":      total_return,
        "cagr":              cagr,
        "ann_return":        ann_return,
        "ann_volatility":    ann_vol,
        "sharpe_ratio":      sharpe,
        "sortino_ratio":     sortino,
        "calmar_ratio":      calmar,
        "profit_factor":     profit_factor,
        "max_drawdown":      max_dd,
        "avg_drawdown":      avg_dd,
        "max_dd_bars":       max_dd_bars,
        "avg_dd_bars":       avg_dd_bars,
        "var_95":            var_95,
        "cvar_95":           cvar_95,
        "n_trades":          n_trades,
        "n_longs":           n_longs,
        "n_shorts":          n_shorts,
        "win_rate":          win_rate,
        "avg_win":           avg_win,
        "avg_loss":          avg_loss,
        "best_trade":        best_trade,
        "worst_trade":       worst_trade,
        "gross_profit":      gross_profit,
        "gross_loss":        gross_loss,
        "reward_risk":       reward_risk,
        "expected_value":    exp_value,
        "long_pnl":          long_pnl,
        "short_pnl":         short_pnl,
        "best_win_streak":   best_win_streak,
        "worst_loss_streak": worst_loss_streak,
        "exit_reasons":      exit_reasons,
        "n_bars":            n_bars,
        "n_years":           n_years,
        "final_equity":      float(equity[-1]),
        "initial_capital":   initial_capital,
    }


# ─── Console Reports ─────────────────────────────────────────────────────────

def print_full_report(m: dict, symbol: str, timeframe: str = "1h"):
    """Print a complete quantitative report to console."""
    W = 62
    print(f"\n{'═'*W}")
    print(f"  PERFORMANCE REPORT  ·  {symbol}  ·  {timeframe.upper()}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'═'*W}")

    def row(label, value, unit=""):
        print(f"  {label:<28} {value}{unit}")

    def pct(v):  return f"{v:>+9.2%}"
    def rat(v):  return f"{v:>+9.2f}"
    def val(v):  return f"  £{v:>8.2f}"
    def num(v):  return f"{v:>9.0f}"

    print(f"\n  ── RETURNS {'─'*42}")
    row("Total Return",          pct(m.get("total_return", 0)))
    row("CAGR",                  pct(m.get("cagr", 0)))
    row("Annualised Return",     pct(m.get("ann_return", 0)))
    row("Annualised Volatility", pct(m.get("ann_volatility", 0)))

    print(f"\n  ── RISK-ADJUSTED {'─'*36}")
    row("Sharpe Ratio",   rat(m.get("sharpe_ratio",  0)))
    row("Sortino Ratio",  rat(m.get("sortino_ratio", 0)))
    row("Calmar Ratio",   rat(m.get("calmar_ratio",  0)))
    row("Profit Factor",  rat(m.get("profit_factor", 0)))

    print(f"\n  ── DRAWDOWN {'─'*41}")
    row("Max Drawdown",      pct(m.get("max_drawdown", 0)))
    row("Avg Drawdown",      pct(m.get("avg_drawdown", 0)))
    row("Max DD Duration",  f"{m.get('max_dd_bars', 0):>9.0f}", " bars")
    row("Avg DD Duration",  f"{m.get('avg_dd_bars', 0):>9.1f}", " bars")

    print(f"\n  ── VALUE AT RISK (95% confidence) {'─'*19}")
    row("Daily VaR",   pct(m.get("var_95",  0)))
    row("Daily CVaR",  pct(m.get("cvar_95", 0)))

    n = m.get("n_trades", 0)
    print(f"\n  ── TRADES  ({n} total) {'─'*37}")
    row("Win Rate",           pct(m.get("win_rate", 0)))
    row("Avg Win",            val(m.get("avg_win",  0)))
    row("Avg Loss",           val(m.get("avg_loss", 0)))
    row("Best Trade",         val(m.get("best_trade",  0)))
    row("Worst Trade",        val(m.get("worst_trade", 0)))
    row("Reward : Risk",     f"{m.get('reward_risk', 0):>9.2f}", "x")
    row("Expected Value",     val(m.get("expected_value", 0)))
    row("Gross Profit",       val(m.get("gross_profit", 0)))
    row("Gross Loss",        f" -£{abs(m.get('gross_loss', 0)):>8.2f}")
    row("Best Win Streak",   f"{m.get('best_win_streak',   0):>9.0f}", " trades")
    row("Worst Loss Streak", f"{m.get('worst_loss_streak', 0):>9.0f}", " trades")

    nl, ns = m.get("n_longs", 0), m.get("n_shorts", 0)
    lp, sp = m.get("long_pnl", 0), m.get("short_pnl", 0)
    print(f"\n  ── DIRECTION {'─'*40}")
    row(f"Long  ({nl} trades)", f"  £{lp:>8.2f}")
    row(f"Short ({ns} trades)", f"  £{sp:>8.2f}")

    reasons = m.get("exit_reasons", {})
    if reasons:
        print(f"\n  ── EXIT REASONS {'─'*37}")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            pct_r = count / n if n else 0
            print(f"  {reason:<24}  {count:>4}  ({pct_r:.1%})")

    print(f"\n{'═'*W}\n")


def print_walk_forward_report(wf_df: pd.DataFrame, symbol: str):
    """Print detailed walk-forward fold-by-fold statistics."""
    if wf_df is None or wf_df.empty:
        return

    W, n = 70, len(wf_df)
    print(f"\n{'═'*W}")
    print(f"  WALK-FORWARD  ·  {symbol}  ·  {n} folds")
    print(f"{'═'*W}")
    print(f"\n  {'Fold':<5}  {'Acc':>7}  {'Return':>9}  {'Sharpe':>8}  {'MaxDD':>8}  {'WinRate':>8}  {'Trades':>6}")
    print(f"  {'─'*67}")

    for _, row in wf_df.iterrows():
        tick = " ✓" if row["total_return"] > 0 else "  "
        print(
            f"  {int(row['fold']):<5}"
            f"  {row['accuracy']:>6.1%}"
            f"  {row['total_return']:>8.2%}"
            f"  {row['sharpe_ratio']:>8.2f}"
            f"  {row['max_drawdown']:>8.2%}"
            f"  {row['win_rate']:>7.1%}"
            f"  {int(row['n_trades']):>6}"
            f"{tick}"
        )

    print(f"  {'─'*67}")
    cols = ["accuracy", "total_return", "sharpe_ratio", "max_drawdown", "win_rate", "n_trades"]
    for label, fn in [("MEDIAN", wf_df[cols].median()), ("MEAN", wf_df[cols].mean()), ("STD", wf_df[cols].std())]:
        print(
            f"  {label:<5}"
            f"  {fn['accuracy']:>6.1%}"
            f"  {fn['total_return']:>8.2%}"
            f"  {fn['sharpe_ratio']:>8.2f}"
            f"  {fn['max_drawdown']:>8.2%}"
            f"  {fn['win_rate']:>7.1%}"
            f"  {fn['n_trades']:>6.1f}"
        )

    profitable   = (wf_df["total_return"] > 0).sum()
    pos_sharpe   = (wf_df["sharpe_ratio"] > 0).sum()
    above_random = (wf_df["accuracy"] > 0.36).sum()
    ret_std      = wf_df["total_return"].std()

    print(f"\n  ── CONSISTENCY {'─'*('─'*45).__len__()}")
    print(f"  Profitable folds      {profitable:>3}/{n}  ({profitable/n:.0%})")
    print(f"  Positive Sharpe folds {pos_sharpe:>3}/{n}  ({pos_sharpe/n:.0%})")
    print(f"  Above-random accuracy {above_random:>3}/{n}  ({above_random/n:.0%})")
    print(f"  Return std dev        {ret_std:>+9.2%}")
    print(f"  Best fold             {wf_df['total_return'].max():>+9.2%}")
    print(f"  Worst fold            {wf_df['total_return'].min():>+9.2%}")

    if profitable/n >= 0.6 and wf_df["sharpe_ratio"].median() > 0.5:
        verdict = "✓  CONSISTENT EDGE — consider live deployment"
    elif profitable/n >= 0.5:
        verdict = "~  MARGINAL EDGE — continue optimising"
    else:
        verdict = "✗  NO CONSISTENT EDGE"
    print(f"\n  {verdict}")
    print(f"{'═'*W}\n")


# ─── HTML Dashboard ───────────────────────────────────────────────────────────

def _heatmap_cell_color(val):
    """Map a return value to a CSS rgba colour for the heatmap."""
    intensity = min(abs(val) / 20.0, 1.0)
    if val > 0:
        return f"rgba(52,211,153,{0.15 + intensity * 0.7:.2f})"
    else:
        return f"rgba(248,113,113,{0.15 + intensity * 0.7:.2f})"


def generate_html_dashboard(results_list: list, wf_results: dict = None,
                             equity_curves: dict = None,
                             output_path: str = "backtest/results/dashboard.html"):
    """
    Generate an interactive HTML dashboard using Chart.js.
    Charts: equity curves, drawdown, return/sharpe bars, scatter,
            trade P&L distribution, walk-forward fold heatmap.
    """
    if not results_list:
        logger.warning("No results — skipping dashboard")
        return None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ── Prepare chart data ────────────────────────────────────────────────────
    results_list = sorted(results_list, key=lambda r: r.get("sharpe_ratio", 0), reverse=True)
    symbols  = [r.get("symbol", "?") for r in results_list]
    returns  = [round(r.get("total_return", 0) * 100, 2)    for r in results_list]
    sharpes  = [round(r.get("sharpe_ratio",  0),       2)   for r in results_list]
    sortinos = [round(r.get("sortino_ratio", 0),       2)   for r in results_list]
    calmars  = [round(min(r.get("calmar_ratio", 0), 5), 2)  for r in results_list]
    max_dds  = [round(r.get("max_drawdown",  0) * 100, 2)   for r in results_list]
    win_rates= [round(r.get("win_rate",      0) * 100, 1)   for r in results_list]
    pfs      = [round(min(r.get("profit_factor", 0), 6), 2) for r in results_list]
    evs      = [round(r.get("expected_value", 0),     2)    for r in results_list]
    n_trades = [int(r.get("n_trades", 0))                   for r in results_list]

    def bar_colors(vals, pos_rgba="rgba(52,211,153,0.82)", neg_rgba="rgba(248,113,113,0.82)"):
        return json.dumps([pos_rgba if v >= 0 else neg_rgba for v in vals])

    # Equity curve datasets
    eq_datasets_js = "[]"
    if equity_curves:
        palette = ["#00d4aa","#7c6af7","#f59e0b","#f87171","#38bdf8",
                   "#a78bfa","#34d399","#fb923c","#e879f9","#4ade80"]
        datasets = []
        for i, (sym, curve) in enumerate(equity_curves.items()):
            if not curve:
                continue
            # Normalise to 100 for comparison
            base   = curve[0] if curve[0] else 1
            normed = [round(v / base * 100, 2) for v in curve]
            # Downsample to max 500 points for performance
            step   = max(1, len(normed) // 500)
            normed = normed[::step]
            color  = palette[i % len(palette)]
            datasets.append(
                f'{{"label":"{sym}","data":{json.dumps(normed)},'
                f'"borderColor":"{color}","borderWidth":1.5,'
                f'"pointRadius":0,"tension":0.1,"fill":false}}'
            )
        eq_datasets_js = "[" + ",".join(datasets) + "]"

    # Walk-forward heatmap
    wf_table_html = "<p style='color:var(--muted);font-size:12px'>Run with --walkforward to see fold data.</p>"
    if wf_results:
        by_sym = {}
        all_folds = set()
        for sym, df in wf_results.items():
            if df is None or df.empty:
                continue
            by_sym[sym] = {}
            for _, row in df.iterrows():
                f = int(row["fold"])
                by_sym[sym][f] = {
                    "ret": round(float(row["total_return"]) * 100, 1),
                    "acc": round(float(row["accuracy"]) * 100, 1),
                }
                all_folds.add(f)

        if by_sym:
            folds = sorted(all_folds)
            rows  = ""
            for sym, fold_data in by_sym.items():
                cells = f'<td class="sym-cell">{sym}</td>'
                for f in folds:
                    if f not in fold_data:
                        cells += '<td style="color:#333">—</td>'
                    else:
                        v   = fold_data[f]
                        bg  = _heatmap_cell_color(v["ret"])
                        clr = "#a7f3d0" if v["ret"] >= 0 else "#fca5a5"
                        cells += (
                            f'<td style="background:{bg};color:{clr}" '
                            f'title="Acc: {v["acc"]}%">'
                            f'{v["ret"]:+.1f}%</td>'
                        )
                rows += f"<tr>{cells}</tr>"
            hdrs = "".join(f"<th>F{f}</th>" for f in folds)
            wf_table_html = (
                f'<div style="overflow-x:auto">'
                f'<table class="heatmap-table"><thead>'
                f'<tr><th>Symbol</th>{hdrs}</tr>'
                f'</thead><tbody>{rows}</tbody></table></div>'
            )

    # Full metrics table rows
    table_rows = ""
    for r in results_list:
        ret = r.get("total_return", 0)
        sh  = r.get("sharpe_ratio",  0)
        mdd = r.get("max_drawdown",  0)
        table_rows += f"""
        <tr>
          <td class="sym-cell">{r.get('symbol','?')}</td>
          <td class="{'pos' if ret>0 else 'neg'}">{ret:+.2%}</td>
          <td class="{'pos' if r.get('cagr',0)>0 else 'neg'}">{r.get('cagr',0):+.2%}</td>
          <td class="{'pos' if sh>0 else 'neg'}">{sh:.2f}</td>
          <td>{r.get('sortino_ratio',0):.2f}</td>
          <td>{r.get('calmar_ratio',0):.2f}</td>
          <td class="{'neg' if mdd<-0.15 else ''}">{mdd:.2%}</td>
          <td>{r.get('win_rate',0):.1%}</td>
          <td>{r.get('profit_factor',0):.2f}</td>
          <td>{r.get('reward_risk',0):.2f}x</td>
          <td>£{r.get('expected_value',0):.2f}</td>
          <td>{r.get('var_95',0):.2%}</td>
          <td>{r.get('n_trades',0)}</td>
          <td>{r.get('n_longs',0)} / {r.get('n_shorts',0)}</td>
        </tr>"""

    # KPI aggregates
    avg_ret   = sum(returns) / len(returns)
    avg_sh    = sum(sharpes) / len(sharpes)
    n_profit  = sum(1 for v in returns if v > 0)
    total_trd = sum(n_trades)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Trading Bot Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=Barlow:wght@400;600;800&display=swap');
:root{{
  --bg:#080b0f; --surf:#0f1318; --surf2:#141820; --border:#1c2230;
  --accent:#00e5b0; --purple:#8b7cf8; --yellow:#fbbf24; --red:#f87171; --green:#34d399;
  --text:#dde4f0; --muted:#4a5568; --mono:'IBM Plex Mono',monospace; --sans:'Barlow',sans-serif;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:var(--mono);font-size:13px;line-height:1.65}}
a{{color:var(--accent);text-decoration:none}}

header{{padding:28px 36px 22px;border-bottom:1px solid var(--border);display:flex;align-items:baseline;gap:18px}}
header h1{{font-family:var(--sans);font-size:20px;font-weight:800;color:var(--accent);letter-spacing:-0.3px}}
.meta{{font-size:11px;color:var(--muted);letter-spacing:.07em;text-transform:uppercase}}

.kpi-strip{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--border)}}
.kpi{{background:var(--surf);padding:20px 24px}}
.kpi-label{{font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin-bottom:6px}}
.kpi-val{{font-family:var(--sans);font-size:28px;font-weight:800;letter-spacing:-1px}}
.kpi-sub{{font-size:10px;color:var(--muted);margin-top:3px}}

.layout{{display:grid;gap:1px;background:var(--border)}}
.cols-2{{grid-template-columns:1fr 1fr}}
.cols-3{{grid-template-columns:1fr 1fr 1fr}}
.cols-eq{{grid-template-columns:1fr}}

.card{{background:var(--surf);padding:22px 26px}}
.card-title{{font-family:var(--sans);font-size:10px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;
             color:var(--muted);margin-bottom:16px;padding-bottom:8px;border-bottom:1px solid var(--border)}}
.chart-box{{position:relative;height:240px}}

/* Metrics table */
.mt{{width:100%;border-collapse:collapse;font-size:11.5px}}
.mt th{{text-align:right;padding:7px 10px;font-size:10px;letter-spacing:.07em;text-transform:uppercase;
        color:var(--muted);border-bottom:1px solid var(--border);white-space:nowrap}}
.mt th:first-child{{text-align:left}}
.mt td{{text-align:right;padding:9px 10px;border-bottom:1px solid var(--border)}}
.mt tr:hover td{{background:rgba(255,255,255,0.02)}}
.sym-cell{{text-align:left!important;color:var(--accent);font-weight:600;font-family:var(--sans)}}
.pos{{color:var(--green)}}
.neg{{color:var(--red)}}

/* Heatmap */
.heatmap-table{{border-collapse:collapse;font-size:11px;white-space:nowrap}}
.heatmap-table th{{padding:5px 9px;font-size:10px;letter-spacing:.07em;text-transform:uppercase;
                   color:var(--muted);text-align:center}}
.heatmap-table td{{padding:4px 7px;text-align:center;border:1px solid var(--bg);min-width:56px}}

footer{{padding:18px 36px;border-top:1px solid var(--border);font-size:10px;color:var(--muted);letter-spacing:.05em}}
</style>
</head>
<body>

<header>
  <h1>◈ TRADING BOT ANALYTICS</h1>
  <span class="meta">Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp; {len(results_list)} symbols</span>
</header>

<!-- KPI Strip -->
<div class="kpi-strip">
  <div class="kpi">
    <div class="kpi-label">Avg Return</div>
    <div class="kpi-val {'pos' if avg_ret>=0 else 'neg'}">{avg_ret:+.1f}%</div>
    <div class="kpi-sub">across {len(symbols)} symbols</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Avg Sharpe</div>
    <div class="kpi-val {'pos' if avg_sh>=0 else 'neg'}">{avg_sh:.2f}</div>
    <div class="kpi-sub">annualised ratio</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Profitable</div>
    <div class="kpi-val" style="color:var(--purple)">{n_profit}/{len(results_list)}</div>
    <div class="kpi-sub">symbols in profit</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Total Trades</div>
    <div class="kpi-val" style="color:var(--yellow)">{total_trd}</div>
    <div class="kpi-sub">after all filters</div>
  </div>
</div>

<!-- Equity Curves -->
<div class="layout cols-eq" style="margin-top:1px">
  <div class="card">
    <div class="card-title">Equity Curves (normalised to 100)</div>
    <div class="chart-box" style="height:280px"><canvas id="eqChart"></canvas></div>
  </div>
</div>

<!-- Row 1 -->
<div class="layout cols-2" style="margin-top:1px">
  <div class="card">
    <div class="card-title">Total Return by Symbol</div>
    <div class="chart-box"><canvas id="retChart"></canvas></div>
  </div>
  <div class="card">
    <div class="card-title">Sharpe · Sortino · Calmar</div>
    <div class="chart-box"><canvas id="ratioChart"></canvas></div>
  </div>
</div>

<!-- Row 2 -->
<div class="layout cols-3" style="margin-top:1px">
  <div class="card">
    <div class="card-title">Win Rate vs Profit Factor</div>
    <div class="chart-box"><canvas id="scatterChart"></canvas></div>
  </div>
  <div class="card">
    <div class="card-title">Max Drawdown by Symbol</div>
    <div class="chart-box"><canvas id="ddChart"></canvas></div>
  </div>
  <div class="card">
    <div class="card-title">Expected Value per Trade (£)</div>
    <div class="chart-box"><canvas id="evChart"></canvas></div>
  </div>
</div>

<!-- Full metrics table -->
<div class="card" style="margin-top:1px">
  <div class="card-title">Full Metrics Table — Ranked by Sharpe Ratio</div>
  <table class="mt">
    <thead><tr>
      <th>Symbol</th><th>Return</th><th>CAGR</th><th>Sharpe</th>
      <th>Sortino</th><th>Calmar</th><th>Max DD</th><th>Win Rate</th>
      <th>Prof. Factor</th><th>Rwd:Risk</th><th>Exp Val</th>
      <th>VaR 95%</th><th>Trades</th><th>L/S</th>
    </tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
</div>

<!-- Walk-forward heatmap -->
<div class="card" style="margin-top:1px">
  <div class="card-title">Walk-Forward Fold Returns — hover for accuracy</div>
  {wf_table_html}
</div>

<footer>
  All returns include trading fees &amp; slippage &nbsp;·&nbsp;
  Past performance does not guarantee future results &nbsp;·&nbsp;
  Sharpe annualised to {8760} bars/year
</footer>

<script>
const SYMS    = {json.dumps(symbols)};
const RETURNS = {json.dumps(returns)};
const SHARPES = {json.dumps(sharpes)};
const SORTINO = {json.dumps(sortinos)};
const CALMAR  = {json.dumps(calmars)};
const MAXDD   = {json.dumps(max_dds)};
const WINRATE = {json.dumps(win_rates)};
const PF      = {json.dumps(pfs)};
const EV      = {json.dumps(evs)};
const EQ_DS   = {eq_datasets_js};

const GRID  = '#1c2230', MUTED = '#4a5568', MONO = "'IBM Plex Mono',monospace";
const GREEN = 'rgba(52,211,153,0.82)', RED = 'rgba(248,113,113,0.82)';
const PURP  = 'rgba(139,124,248,0.82)', YELL = 'rgba(251,191,36,0.82)';

function rc(vals, p=GREEN, n=RED){{ return vals.map(v => v>=0 ? p : n); }}

const base = {{
  responsive:true, maintainAspectRatio:false,
  plugins:{{
    legend:{{display:false}},
    tooltip:{{backgroundColor:'#0f1318',borderColor:'#1c2230',borderWidth:1,
              titleColor:'#dde4f0',bodyColor:'#8896a8',cornerRadius:4,padding:10}}
  }},
  scales:{{
    x:{{grid:{{color:GRID}}, ticks:{{color:MUTED,font:{{family:MONO,size:11}}}}}},
    y:{{grid:{{color:GRID}}, ticks:{{color:MUTED,font:{{family:MONO,size:11}}}}}}
  }}
}};

// Equity curves
if(EQ_DS.length) {{
  const eqLabels = Array.from({{length: Math.max(...EQ_DS.map(d=>d.data.length))}}, (_,i)=>i);
  new Chart(document.getElementById('eqChart'), {{
    type:'line',
    data:{{labels:eqLabels, datasets:EQ_DS}},
    options:{{...base,
      plugins:{{...base.plugins, legend:{{display:true,
        labels:{{color:MUTED,font:{{family:MONO,size:11}},boxWidth:14,padding:14}}}}}},
      scales:{{...base.scales,
        y:{{...base.scales.y, ticks:{{...base.scales.y.ticks,callback:v=>v.toFixed(0)}}}}
      }}
    }}
  }});
}} else {{
  document.getElementById('eqChart').parentElement.innerHTML =
    '<p style="color:var(--muted);font-size:12px;padding:24px">No equity curve data — ensure BacktestEngine.equity_curve is passed through.</p>';
}}

// Return bars
new Chart(document.getElementById('retChart'),{{
  type:'bar',
  data:{{labels:SYMS, datasets:[{{data:RETURNS,backgroundColor:rc(RETURNS),borderRadius:3}}]}},
  options:{{...base,
    plugins:{{...base.plugins,tooltip:{{...base.plugins.tooltip,
      callbacks:{{label:c=>' '+c.raw.toFixed(2)+'%'}}}}}},
    scales:{{...base.scales,y:{{...base.scales.y,
      ticks:{{...base.scales.y.ticks,callback:v=>v+'%'}}}}}}
  }}
}});

// Ratio comparison
new Chart(document.getElementById('ratioChart'),{{
  type:'bar',
  data:{{labels:SYMS,datasets:[
    {{label:'Sharpe', data:SHARPES, backgroundColor:'rgba(0,229,176,0.75)', borderRadius:3}},
    {{label:'Sortino',data:SORTINO, backgroundColor:'rgba(139,124,248,0.75)',borderRadius:3}},
    {{label:'Calmar', data:CALMAR,  backgroundColor:'rgba(251,191,36,0.75)', borderRadius:3}},
  ]}},
  options:{{...base, plugins:{{...base.plugins,
    legend:{{display:true,labels:{{color:MUTED,font:{{family:MONO,size:11}},boxWidth:12,padding:12}}}}
  }}}}
}});

// Scatter: win rate vs profit factor
new Chart(document.getElementById('scatterChart'),{{
  type:'scatter',
  data:{{datasets:[{{
    data: WINRATE.map((wr,i)=>({{x:wr,y:PF[i],label:SYMS[i]}})),
    backgroundColor:'rgba(0,229,176,0.75)', pointRadius:7, pointHoverRadius:10
  }}]}},
  options:{{...base,
    plugins:{{...base.plugins,tooltip:{{...base.plugins.tooltip,
      callbacks:{{label:c=>c.raw.label+': WR='+c.raw.x.toFixed(1)+'%  PF='+c.raw.y.toFixed(2)}}}}}},
    scales:{{
      x:{{...base.scales.x,title:{{display:true,text:'Win Rate %',color:MUTED,font:{{family:MONO,size:11}}}}}},
      y:{{...base.scales.y,title:{{display:true,text:'Profit Factor',color:MUTED,font:{{family:MONO,size:11}}}}}}
    }}
  }}
}});

// Max drawdown
new Chart(document.getElementById('ddChart'),{{
  type:'bar',
  data:{{labels:SYMS, datasets:[{{data:MAXDD,backgroundColor:'rgba(248,113,113,0.78)',borderRadius:3}}]}},
  options:{{...base,scales:{{...base.scales,
    y:{{...base.scales.y,ticks:{{...base.scales.y.ticks,callback:v=>v+'%'}}}}}}
  }}
}});

// Expected value
new Chart(document.getElementById('evChart'),{{
  type:'bar',
  data:{{labels:SYMS, datasets:[{{data:EV,backgroundColor:rc(EV),borderRadius:3}}]}},
  options:{{...base,
    plugins:{{...base.plugins,tooltip:{{...base.plugins.tooltip,
      callbacks:{{label:c=>'£'+c.raw.toFixed(2)}}}}}},
    scales:{{...base.scales,y:{{...base.scales.y,
      ticks:{{...base.scales.y.ticks,callback:v=>'£'+v.toFixed(2)}}}}}}
  }}
}});
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.success(f"Dashboard saved → {output_path}")
    return output_path


# ─── Entry Point ─────────────────────────────────────────────────────────────

def plot_all(results: dict, equity_curves: dict = None,
             trades_by_symbol: dict = None, wf_results: dict = None,
             timeframe: str = "1h"):
    """
    Called from train_and_backtest.py after all symbols complete.

    Args:
        results          : {symbol: basic_metrics_dict} from BacktestEngine
        equity_curves    : {symbol: list_of_equity_values}
        trades_by_symbol : {symbol: list_of_trade_dicts}
        wf_results       : {symbol: walk_forward_dataframe}
        timeframe        : Current timeframe string for annualisation
    """
    BPY = {"15m":35040,"30m":17520,"1h":8760,"2h":4380,"4h":2190,"8h":1095,"1d":365}
    bars_per_year = BPY.get(timeframe, 8760)

    try:
        from config.settings import INITIAL_CAPITAL
    except Exception:
        INITIAL_CAPITAL = 10_000

    full_results = []
    for symbol, basic in results.items():
        equity = (equity_curves or {}).get(symbol, [])
        trades = (trades_by_symbol or {}).get(symbol, [])

        if equity:
            m = compute_full_metrics(equity, trades, INITIAL_CAPITAL, bars_per_year)
        else:
            m = dict(basic)

        m["symbol"] = symbol
        full_results.append(m)

        print_full_report(m, symbol, timeframe)

        if wf_results and symbol in wf_results:
            print_walk_forward_report(wf_results[symbol], symbol)

    generate_html_dashboard(
        full_results,
        wf_results=wf_results,
        equity_curves=equity_curves,
    )
