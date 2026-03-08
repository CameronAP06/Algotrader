"""
oot_validate.py
───────────────
Walk-forward out-of-time validation using a 9-model ensemble.

Instead of a single fixed OOT window (which only tests one historical regime),
this runs N_WF_FOLDS walk-forward windows that each cover a different calendar
period. A signal that survives across all windows is genuinely structural.

Architecture per fold:
  [  OOT (oldest 12%, fixed) | TRAIN (grows) | VAL | TEST (steps forward)  ]

The OOT window is always the same oldest slice (never touched by training).
The TRAIN/VAL/TEST windows step forward for each fold so the test period
covers a different regime each time.

Usage:
    python oot_validate.py --symbol DOGE/USD
    python oot_validate.py --symbol EOS/USD --timeframe 30m
    python oot_validate.py --symbol DOGE/USD LINK/USD AAVE/USD --timeframe 4h
    python oot_validate.py --symbol DOGE/USD --folds 4
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

HISTORY_DAYS = {
    "30m": 180,
    "1h":  365,
    "4h":  730,
    "12h": 730,
    "1d":  1095,
}

OOT_RATIO   = 0.12   # oldest 12% reserved as OOT — never used for training
N_MODELS    = 9
N_WF_FOLDS  = 3      # walk-forward test windows (different calendar regimes)
TOP_PCT     = 0.15

# Walk-forward fold sizing as fraction of total data
FOLD_VAL    = 0.14
FOLD_TEST   = 0.10   # each fold's test window — steps backward from end

# Minimum trades to produce a meaningful verdict per timeframe
MIN_TRADES_TF = {
    "30m": 12,
    "1h":  12,
    "4h":   6,
    "12h":  4,
    "1d":   4,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_date(series) -> str:
    try:
        return f"{series.iloc[0].strftime('%Y-%m-%d')}→{series.iloc[-1].strftime('%Y-%m-%d')}"
    except Exception:
        return "?"


def _backtest_window(models, X_s, feat_window, label, timeframe):
    proba    = predict_proba_ensemble(models, X_s)
    signals  = generate_signals(proba, use_percentile=True, top_pct=TOP_PCT)
    filtered = apply_filters(feat_window, signals, timeframe=timeframe)
    m        = BacktestEngine().run(feat_window, filtered, timeframe=timeframe)
    net      = float(m.get("total_return",  0.0))
    ann      = float(m.get("ann_return",    0.0))
    wr       = float(m.get("win_rate",      0.0))
    sharpe   = float(m.get("sharpe_ratio",  0.0))
    calmar   = float(m.get("calmar_ratio",  0.0))
    dd       = float(m.get("max_drawdown",  0.0))
    n_trades = int(m.get("n_trades",        0))
    date_str = _fmt_date(feat_window["timestamp"]) if "timestamp" in feat_window.columns else "?"
    logger.info(f"  {label:<18} [{date_str}]: net={net:+.1%} ann={ann:+.1%} "
                f"WR={wr:.1%} Sh={sharpe:.2f} Cal={calmar:.2f} trades={n_trades}")
    return {
        "window":     label,
        "date_range": date_str,
        "n_trades":   n_trades,
        "win_rate":   round(wr,     4),
        "net_return": round(net,    4),
        "ann_return": round(ann,    4),
        "sharpe":     round(sharpe, 3),
        "calmar":     round(calmar, 3),
        "max_dd":     round(dd,     4),
    }


# ── Core ──────────────────────────────────────────────────────────────────────

def validate_symbol(symbol: str, timeframe: str = "4h", n_folds: int = N_WF_FOLDS) -> list:
    """
    Run walk-forward OOT validation. Returns list of fold result dicts,
    each containing 'oot', 'test', and 'fold' keys.
    """
    logger.info(f"\n{'='*65}\n  OOT VALIDATION: {symbol} {timeframe} — {n_folds} walk-forward folds\n{'='*65}")

    days = HISTORY_DAYS.get(timeframe, 730)
    raw  = fetch_ohlcv_timeframe(symbol, timeframe, history_days=days)
    if raw is None or len(raw) < 1000:
        logger.error(f"{symbol}: insufficient data ({len(raw) if raw else 0} bars)")
        return []

    feat_df   = build_features(raw, timeframe=timeframe)
    feat_cols = get_feature_columns(feat_df)
    X = feat_df[feat_cols].values.astype(np.float32)
    y = feat_df["label"].values.astype(int)
    n = len(X)

    # Fixed OOT window — always the oldest OOT_RATIO of data
    oot_end   = int(n * OOT_RATIO)
    X_oot_raw = X[:oot_end]
    feat_oot  = feat_df.iloc[:oot_end].reset_index(drop=True)

    oot_date = _fmt_date(feat_oot["timestamp"]) if "timestamp" in feat_oot.columns else "?"
    logger.info(f"  OOT window (fixed): {oot_end} bars [{oot_date}]")
    logger.info(f"  Remaining for WF:   {n - oot_end} bars")
    logger.info(f"  Walk-forward folds: {n_folds}\n")

    fold_step = int(n * FOLD_TEST)
    fold_val  = int(n * FOLD_VAL)

    all_fold_results = []

    for fold in range(n_folds):
        # Step test window backward from the end of data
        test_end   = n - fold * fold_step
        test_start = test_end - fold_step
        val_start  = test_start - fold_val
        train_end  = val_start   # train from oot_end up to val_start

        if train_end <= oot_end or test_start <= val_start or test_end > n:
            logger.warning(f"  Fold {fold+1}: invalid window boundaries — skipping")
            continue

        train_size = train_end - oot_end
        if train_size < 200:
            logger.warning(f"  Fold {fold+1}: train too small ({train_size} bars) — skipping")
            continue

        logger.info(f"\n  ── Fold {fold+1}/{n_folds} ──────────────────────────────────────────")
        logger.info(f"  Train: [{oot_end}:{train_end}] = {train_size} bars")
        logger.info(f"  Val:   [{val_start}:{test_start}] = {fold_val} bars")
        logger.info(f"  Test:  [{test_start}:{test_end}] = {fold_step} bars")

        X_train   = X[oot_end:train_end]
        y_train   = y[oot_end:train_end]
        X_val     = X[val_start:test_start]
        y_val     = y[val_start:test_start]
        X_test    = X[test_start:test_end]
        feat_test = feat_df.iloc[test_start:test_end].reset_index(drop=True)

        # Scale — fit on this fold's train only
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_val_s   = scaler.transform(X_val)
        X_test_s  = scaler.transform(X_test)
        X_oot_s   = scaler.transform(X_oot_raw)

        # Train ensemble
        models = train_ensemble(
            X_train_s, y_train, X_val_s, y_val,
            symbol=f"{symbol}_{timeframe}_fold{fold+1}", n_models=N_MODELS
        )

        test_result = _backtest_window(models, X_test_s, feat_test,
                                       f"TEST fold {fold+1}", timeframe)
        oot_result  = _backtest_window(models, X_oot_s,  feat_oot,
                                       f"OOT  fold {fold+1}", timeframe)

        all_fold_results.append({
            "fold":  fold + 1,
            "test":  test_result,
            "oot":   oot_result,
        })

    return all_fold_results


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(symbol: str, timeframe: str, results: list) -> str:
    if not results:
        logger.warning(f"No results for {symbol}")
        return "NO DATA"

    n = len(results)
    test_nets = [r["test"]["net_return"] for r in results]
    oot_nets  = [r["oot"]["net_return"]  for r in results]
    test_anns = [r["test"]["ann_return"] for r in results]
    oot_anns  = [r["oot"]["ann_return"]  for r in results]
    test_wrs  = [r["test"]["win_rate"]   for r in results]
    oot_wrs   = [r["oot"]["win_rate"]    for r in results]
    test_sh   = [r["test"]["sharpe"]     for r in results]
    oot_sh    = [r["oot"]["sharpe"]      for r in results]
    test_cal  = [r["test"]["calmar"]     for r in results]
    oot_cal   = [r["oot"]["calmar"]      for r in results]
    test_tr   = [r["test"]["n_trades"]   for r in results]
    oot_tr    = [r["oot"]["n_trades"]    for r in results]

    print(f"\n{'='*70}")
    print(f"  {symbol} {timeframe} — {n} walk-forward folds × {N_MODELS} models")
    print(f"{'='*70}")

    # Per-fold detail table
    print(f"\n  {'Fold':<6} {'Window':<23} {'Net':>7} {'Ann':>7} {'WR':>6} "
          f"{'Sh':>6} {'Cal':>6} {'Trades':>7}")
    print(f"  {'─'*70}")
    for r in results:
        for wk, lbl in [("test", "TEST"), ("oot", "OOT")]:
            d = r[wk]
            pos = "+" if d["net_return"] >= 0 else ""
            print(f"  {r['fold']:<6} {lbl+' '+d['date_range']:<23} "
                  f"{d['net_return']:>+6.1%} {d['ann_return']:>+6.1%} "
                  f"{d['win_rate']:>5.1%} {d['sharpe']:>6.2f} "
                  f"{d['calmar']:>6.2f} {d['n_trades']:>7}")

    # Aggregate stats
    print(f"\n  {'Metric':<24} {'TEST mean':>10} {'TEST std':>9} {'OOT mean':>10} {'OOT std':>9}")
    print(f"  {'─'*64}")
    for label, t_vals, o_vals in [
        ("Net return",  test_nets, oot_nets),
        ("Ann return",  test_anns, oot_anns),
        ("Win rate",    test_wrs,  oot_wrs),
        ("Sharpe",      test_sh,   oot_sh),
        ("Calmar",      test_cal,  oot_cal),
    ]:
        fmt = ".1%" if "return" in label.lower() or "rate" in label.lower() else ".2f"
        print(f"  {label:<24} "
              f"{np.mean(t_vals):>+9{fmt}} {np.std(t_vals):>8{fmt}} "
              f"{np.mean(o_vals):>+9{fmt}} {np.std(o_vals):>8{fmt}}")
    print(f"  {'Trades (avg)':<24} {np.mean(test_tr):>9.1f} {'':>9} {np.mean(oot_tr):>9.1f}")

    both_pos  = sum(1 for r in results if r["test"]["net_return"] > 0 and r["oot"]["net_return"] > 0)
    both_neg  = sum(1 for r in results if r["test"]["net_return"] < 0 and r["oot"]["net_return"] < 0)
    t_p_o_n   = sum(1 for r in results if r["test"]["net_return"] > 0 and r["oot"]["net_return"] < 0)
    t_n_o_p   = sum(1 for r in results if r["test"]["net_return"] < 0 and r["oot"]["net_return"] > 0)
    oot_pos   = sum(1 for x in oot_nets if x > 0)

    print(f"\n  Fold outcomes ({n} folds):")
    print(f"    Both positive:   {both_pos}/{n}  ← deploy signal if high")
    print(f"    Test+ OOT-:      {t_p_o_n}/{n}  ← regime-specific overfitting")
    print(f"    Test- OOT+:      {t_n_o_p}/{n}  ← old signal, not recent")
    print(f"    Both negative:   {both_neg}/{n}  ← no signal")

    min_trades = MIN_TRADES_TF.get(timeframe, 6)
    low_trade_folds = sum(1 for r in results
                          if r["test"]["n_trades"] < min_trades
                          or r["oot"]["n_trades"]  < min_trades)
    if low_trade_folds:
        print(f"\n  ⚠  {low_trade_folds} fold(s) below min trades threshold ({min_trades}) "
              f"— treat those folds with caution")

    # ── Verdict ──────────────────────────────────────────────────────────────
    mean_oot_ann = np.mean(oot_anns)
    std_oot_ann  = np.std(oot_anns)

    print(f"\n  VERDICT:")
    if both_pos == n and mean_oot_ann > 0.08 and std_oot_ann < 0.15:
        verdict = "✓ DEPLOY"
        detail  = (f"All {n}/{n} folds both-positive. OOT ann={mean_oot_ann:+.1%} "
                   f"std={std_oot_ann:.1%}. Robust across regimes.")
    elif both_pos >= n - 1 and mean_oot_ann > 0.05:
        verdict = "✓ DEPLOY (weak)"
        detail  = (f"{both_pos}/{n} both-positive folds. OOT ann={mean_oot_ann:+.1%}. "
                   f"Mostly consistent — deploy with normal sizing.")
    elif oot_pos == n and mean_oot_ann > 0.03:
        verdict = "~ MONITOR"
        detail  = (f"OOT positive in all {n} folds but TEST inconsistent. "
                   f"Signal exists historically but model struggles on recent data. "
                   f"Re-check after next data refresh.")
    elif mean_oot_ann > 0 and both_pos >= n // 2:
        verdict = "~ WEAK"
        detail  = (f"OOT mean positive ({mean_oot_ann:+.1%}) but only {both_pos}/{n} "
                   f"folds both-positive. Too inconsistent for deployment.")
    elif both_neg >= n - 1:
        verdict = "✗ DROP"
        detail  = f"Both TEST and OOT negative in {both_neg}/{n} folds. No signal."
    else:
        verdict = "✗ DROP"
        detail  = (f"Inconsistent with no clear positive pattern. "
                   f"OOT ann={mean_oot_ann:+.1%}, {both_pos}/{n} both-positive.")

    print(f"  {verdict}")
    print(f"  {detail}")
    print()
    return verdict


# ── Save ──────────────────────────────────────────────────────────────────────

def save_results(symbol: str, timeframe: str, results: list, ts: str) -> Path:
    out_dir = Path("backtest/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = symbol.replace("/", "_")
    path = out_dir / f"oot_ensemble_{safe}_{timeframe}_{ts}.csv"

    rows = []
    for r in results:
        base = {"fold": r["fold"], "symbol": symbol, "timeframe": timeframe}
        for wk in ["test", "oot"]:
            rows.append({**base, **r[wk]})

    if rows:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        logger.info(f"Saved -> {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Walk-forward OOT validation")
    parser.add_argument("--symbol",    nargs="+", default=["DOGE/USD"])
    parser.add_argument("--timeframe", default="4h",
                        choices=["30m", "1h", "4h", "12h", "1d"])
    parser.add_argument("--folds",     type=int, default=N_WF_FOLDS,
                        help=f"Walk-forward folds (default: {N_WF_FOLDS})")
    args = parser.parse_args()
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")

    verdicts = {}
    for symbol in args.symbol:
        results = validate_symbol(symbol, timeframe=args.timeframe, n_folds=args.folds)
        verdict = print_summary(symbol, args.timeframe, results)
        save_results(symbol, args.timeframe, results, ts)
        verdicts[symbol] = verdict

    if len(args.symbol) > 1:
        print(f"\n{'='*50}")
        print(f"  BATCH SUMMARY")
        print(f"{'='*50}")
        for sym, v in verdicts.items():
            print(f"  {sym:<14} {v or 'no result'}")
        print()


if __name__ == "__main__":
    main()