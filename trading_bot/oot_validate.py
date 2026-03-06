"""
oot_validate.py
───────────────
Out-of-time validation using a 9-model ensemble.

Splits data into 4 windows:
  [OOT (historical)] | [TRAIN] | [VAL] | [TEST]

Trains ensemble on TRAIN+VAL, then backtests on both TEST and OOT.
Runs N_RUNS independent times to measure remaining variance.

If OOT and TEST are both consistently positive across runs,
the signal is structural — not just regime-specific.

Usage:
    python oot_validate.py --symbol DOGE/USD
    python oot_validate.py --symbol DOGE/USD LINK/USD
    python oot_validate.py --symbol DOGE/USD --runs 5
"""
import os, sys, argparse, csv
from datetime import datetime
from pathlib import Path

_here   = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
sys.path.insert(0, _here)
sys.path.insert(0, _parent)

import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler

from timeframe_comparison import fetch_ohlcv_timeframe
from data.feature_engineer import build_features, get_feature_columns
from backtest.engine import BacktestEngine
from backtest.filters import apply_filters
from models.ensemble import generate_signals
from models.lstm_ensemble import train_ensemble, predict_proba_ensemble

# ── Config ────────────────────────────────────────────────────────────────────

TIMEFRAME    = "4h"   # overridden by --timeframe arg
HISTORY_DAYS = {
    "1h":  365,
    "4h":  730,
    "12h": 730,
    "1d":  1095,
}
OOT_RATIO    = 0.20   # oldest 20% = OOT window
TRAIN_RATIO  = 0.48   # next 48%  = train
VAL_RATIO    = 0.16   # next 16%  = val
TEST_RATIO   = 0.16   # newest 16% = test
N_MODELS     = 9
N_RUNS       = 5      # repeat full ensemble training N times to measure residual variance
TOP_PCT      = 0.15


# ── Core ──────────────────────────────────────────────────────────────────────

def run_once(symbol: str, run_idx: int, timeframe: str = "4h") -> dict | None:
    """One full OOT validation run."""
    days = HISTORY_DAYS.get(timeframe, 1825)
    raw = fetch_ohlcv_timeframe(symbol, timeframe, history_days=days)
    if raw is None or len(raw) < 2000:
        logger.error(f"{symbol}: insufficient data ({len(raw) if raw else 0} bars)")
        return None

    feat_df   = build_features(raw, timeframe=timeframe)
    feat_cols = get_feature_columns(feat_df)
    X = feat_df[feat_cols].values.astype(np.float32)
    y = feat_df["label"].values.astype(int)
    n = len(X)

    # Split
    i0 = int(n * OOT_RATIO)
    i1 = i0 + int(n * TRAIN_RATIO)
    i2 = i1 + int(n * VAL_RATIO)
    # test is i2:end

    X_oot,   y_oot   = X[:i0],   y[:i0]
    X_train, y_train = X[i0:i1], y[i0:i1]
    X_val,   y_val   = X[i1:i2], y[i1:i2]
    X_test,  y_test  = X[i2:],   y[i2:]
    feat_oot  = feat_df.iloc[:i0].reset_index(drop=True)
    feat_test = feat_df.iloc[i2:].reset_index(drop=True)

    oot_days  = int((feat_oot["timestamp"].iloc[-1]  - feat_oot["timestamp"].iloc[0]).days)  \
                if "timestamp" in feat_oot.columns else i0 // 6
    test_days = int((feat_test["timestamp"].iloc[-1] - feat_test["timestamp"].iloc[0]).days) \
                if "timestamp" in feat_test.columns else (n - i2) // 6

    logger.info(f"\nRun {run_idx+1}/{N_RUNS} | {symbol}")
    logger.info(f"  OOT:   {len(X_oot):>5} bars ({oot_days} days)")
    logger.info(f"  Train: {len(X_train):>5} bars")
    logger.info(f"  Val:   {len(X_val):>5} bars")
    logger.info(f"  Test:  {len(X_test):>5} bars ({test_days} days)")

    # Scale — fit on train only, apply everywhere
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    X_oot_s   = scaler.transform(X_oot)

    # Train 9-model ensemble
    models = train_ensemble(
        X_train_s, y_train, X_val_s, y_val,
        symbol=symbol, n_models=N_MODELS
    )

    hours_per_bar = {
        "15m": 0.25, "30m": 0.5, "1h": 1.0, "4h": 4.0,
        "12h": 12.0, "1d": 24.0,
    }

    def backtest_window(X_s, feat_window, label, n_bars):
        proba    = predict_proba_ensemble(models, X_s)
        signals  = generate_signals(proba, use_percentile=True, top_pct=TOP_PCT)
        filtered = apply_filters(feat_window, signals, timeframe=timeframe)
        m        = BacktestEngine().run(feat_window, filtered, timeframe=timeframe)
        n_trades = int(m.get("n_trades", 0))
        net      = float(m.get("total_return", 0.0))
        wr       = float(m.get("win_rate", 0.0))
        sharpe   = float(m.get("sharpe_ratio", 0.0))
        dd       = float(m.get("max_drawdown", 0.0))
        best_p   = np.maximum(proba[:, 2], proba[:, 0])
        # Annualised return
        test_years = max(n_bars * hours_per_bar.get(timeframe, 1.0) / 8760, 1/365)
        ann = (1 + net) ** (1 / test_years) - 1
        logger.info(f"  {label:<5}: net={net:+.1%} ann={ann:+.1%} WR={wr:.1%} "
                    f"Sharpe={sharpe:.2f} trades={n_trades} MaxDD={dd:.1%}")
        return {
            "window": label, "n_trades": n_trades,
            "win_rate": round(wr, 4), "net_return": round(net, 4),
            "ann_return": round(ann, 4),
            "sharpe": round(sharpe, 3), "max_dd": round(dd, 4),
            "mean_conf": round(float(best_p.mean()), 4),
        }

    test_result = backtest_window(X_test_s, feat_test, "TEST", len(X_test))
    oot_result  = backtest_window(X_oot_s,  feat_oot,  "OOT ", len(X_oot))

    return {"run": run_idx + 1, "test": test_result, "oot": oot_result}


def validate_symbol(symbol: str, n_runs: int, timeframe: str = "4h") -> list:
    logger.info(f"\n{'='*60}\n  OOT VALIDATION: {symbol} {timeframe} x{n_runs} runs\n{'='*60}")
    results = []
    for i in range(n_runs):
        r = run_once(symbol, i, timeframe=timeframe)
        if r:
            results.append(r)
    return results


def print_summary(symbol: str, results: list):
    if not results:
        return

    test_nets  = [r["test"]["net_return"] for r in results]
    oot_nets   = [r["oot"]["net_return"]  for r in results]
    test_anns  = [r["test"]["ann_return"] for r in results]
    oot_anns   = [r["oot"]["ann_return"]  for r in results]
    test_wrs   = [r["test"]["win_rate"]   for r in results]
    oot_wrs    = [r["oot"]["win_rate"]    for r in results]
    test_sh    = [r["test"]["sharpe"]     for r in results]
    oot_sh     = [r["oot"]["sharpe"]      for r in results]

    print(f"\n{'='*60}")
    print(f"  {symbol} — {len(results)} runs x {N_MODELS} models each")
    print(f"{'='*60}")
    print(f"  {'':22} {'TEST':>12} {'OOT':>12}")
    print(f"  {'-'*48}")
    print(f"  {'Net return  mean':<22} {np.mean(test_nets):>+11.1%} {np.mean(oot_nets):>+11.1%}")
    print(f"  {'Net return  std':<22} {np.std(test_nets):>11.1%} {np.std(oot_nets):>11.1%}")
    print(f"  {'Net return  min':<22} {np.min(test_nets):>+11.1%} {np.min(oot_nets):>+11.1%}")
    print(f"  {'Net return  max':<22} {np.max(test_nets):>+11.1%} {np.max(oot_nets):>+11.1%}")
    print(f"  {'Ann return  mean':<22} {np.mean(test_anns):>+11.1%} {np.mean(oot_anns):>+11.1%}")
    print(f"  {'Ann return  std':<22} {np.std(test_anns):>11.1%} {np.std(oot_anns):>11.1%}")
    print(f"  {'Win rate    mean':<22} {np.mean(test_wrs):>11.1%} {np.mean(oot_wrs):>11.1%}")
    print(f"  {'Sharpe      mean':<22} {np.mean(test_sh):>11.2f} {np.mean(oot_sh):>11.2f}")
    print(f"  {'% runs positive':<22} {sum(1 for x in test_nets if x>0)/len(results):>11.0%} "
          f"{sum(1 for x in oot_nets if x>0)/len(results):>11.0%}")

    corr = np.corrcoef(test_nets, oot_nets)[0, 1] if len(results) > 2 else float("nan")
    print(f"\n  TEST vs OOT correlation: {corr:+.2f}")

    both_pos = sum(1 for r in results if r["test"]["net_return"] > 0 and r["oot"]["net_return"] > 0)
    both_neg = sum(1 for r in results if r["test"]["net_return"] < 0 and r["oot"]["net_return"] < 0)
    t_p_o_n  = sum(1 for r in results if r["test"]["net_return"] > 0 and r["oot"]["net_return"] < 0)
    t_n_o_p  = sum(1 for r in results if r["test"]["net_return"] < 0 and r["oot"]["net_return"] > 0)

    n = len(results)
    print(f"\n  Outcome breakdown ({n} runs):")
    print(f"    Both positive:   {both_pos}/{n}  <- genuine signal if high")
    print(f"    Test+ OOT-:      {t_p_o_n}/{n}  <- regime-specific")
    print(f"    Test- OOT+:      {t_n_o_p}/{n}")
    print(f"    Both negative:   {both_neg}/{n}  <- no signal")

    # Verdict
    print(f"\n  VERDICT:")
    mean_oot = np.mean(oot_nets)
    std_oot  = np.std(oot_nets)
    if both_pos >= int(n * 0.7) and mean_oot > 0.03:
        print(f"  STRONG SIGNAL — {both_pos}/{n} both-positive runs, OOT mean {mean_oot:+.1%}")
        print(f"  Proceed to live deployment with this symbol.")
    elif mean_oot > 0 and std_oot < 0.10:
        print(f"  WEAK SIGNAL — positive mean OOT ({mean_oot:+.1%}) but low consistency")
        print(f"  Consider deploying with reduced position sizing.")
    elif std_oot < 0.08:
        print(f"  NO SIGNAL — variance reduced but mean near zero ({mean_oot:+.1%})")
        print(f"  Drop this symbol from the paper trade.")
    else:
        print(f"  INCONCLUSIVE — high variance remains (std={std_oot:.1%})")
        print(f"  The ensemble hasn't fully stabilised. Consider more runs.")
    print()


def save_results(symbol: str, results: list, ts: str, timeframe: str = "4h"):
    out_dir = Path("backtest/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe    = symbol.replace("/", "_")
    path    = out_dir / f"oot_ensemble_{safe}_{timeframe}_{ts}.csv"

    rows = []
    for r in results:
        for window, d in [("TEST", r["test"]), ("OOT", r["oot"])]:
            rows.append({"run": r["run"], "symbol": symbol, **d})

    if rows:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        logger.info(f"Saved -> {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",    nargs="+", default=["DOGE/USD"],
                        help="Symbol(s) to validate")
    parser.add_argument("--timeframe", default="4h",
                        choices=["1h", "4h", "12h", "1d"],
                        help="Timeframe to validate (default: 4h)")
    parser.add_argument("--runs",      type=int, default=N_RUNS,
                        help="Number of independent ensemble runs")
    args   = parser.parse_args()
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")

    for symbol in args.symbol:
        results = validate_symbol(symbol, args.runs, timeframe=args.timeframe)
        print_summary(symbol, results)
        save_results(symbol, results, ts, timeframe=args.timeframe)


if __name__ == "__main__":
    main()