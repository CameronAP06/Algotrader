"""
lstm_scan.py
────────────
9-model LSTM ensemble scan across all 4h symbols.

Each symbol trains a full 9-model ensemble — averaging probabilities
across models reduces run-to-run variance by ~1/sqrt(9) vs single model.

Usage:
    python lstm_scan.py
    python lstm_scan.py --symbols DOGE/USD LINK/USD BTC/USD
    python lstm_scan.py --timeframe 4h --min-bars 3000
"""
import os, sys, argparse, csv, time
from datetime import datetime
from pathlib import Path

_here   = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
sys.path.insert(0, _here)
sys.path.insert(0, _parent)

import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler

from config.settings import TRADING_PAIRS, HISTORY_DAYS
from timeframe_comparison import fetch_ohlcv_timeframe
from data.feature_engineer import build_features, get_feature_columns
from backtest.engine import BacktestEngine
from backtest.filters import apply_filters
from models.ensemble import generate_signals
from models.lstm_ensemble import train_ensemble, predict_proba_ensemble

# ── Config ────────────────────────────────────────────────────────────────────

TIMEFRAME     = "4h"
TRAIN_RATIO   = 0.60
VAL_RATIO     = 0.20
TEST_RATIO    = 0.20
TOP_PCT       = 0.15
MIN_BARS      = 3000
MIN_TRADES    = 5
N_MODELS      = 9

_HDR = (
    f"{'Symbol':<14} {'Bars':>6} {'Trades':>7} {'WR':>6} "
    f"{'Net':>7} {'Sharpe':>7} {'MaxDD':>7} {'MeanConf':>9} {'Status'}"
)
_SEP = "─" * 80


# ── Core ──────────────────────────────────────────────────────────────────────

def run_symbol(symbol: str, timeframe: str) -> dict | None:
    t0 = time.time()
    try:
        # 1. Fetch + engineer
        raw = fetch_ohlcv_timeframe(symbol, timeframe, history_days=HISTORY_DAYS)
        if raw is None or len(raw) < MIN_BARS:
            logger.warning(f"{symbol}: {len(raw) if raw is not None else 0} bars — skipping")
            return None

        feat_df   = build_features(raw, timeframe=timeframe)
        feat_cols = get_feature_columns(feat_df)
        X = feat_df[feat_cols].values.astype(np.float32)
        y = feat_df["label"].values.astype(int)

        # 2. Split + scale
        n  = len(X)
        t1 = int(n * TRAIN_RATIO)
        t2 = int(n * (TRAIN_RATIO + VAL_RATIO))

        scaler   = StandardScaler()
        X_scaled = X.copy()
        X_scaled[:t1]   = scaler.fit_transform(X[:t1])
        X_scaled[t1:t2] = scaler.transform(X[t1:t2])
        X_scaled[t2:]   = scaler.transform(X[t2:])

        X_train, y_train = X_scaled[:t1],  y[:t1]
        X_val,   y_val   = X_scaled[t1:t2], y[t1:t2]
        X_test,  y_test  = X_scaled[t2:],   y[t2:]
        feat_test = feat_df.iloc[t2:].reset_index(drop=True)

        label_counts = {int(k): int(v)
                        for k, v in zip(*np.unique(y_train, return_counts=True))}

        # 3. Train 9-model ensemble
        models = train_ensemble(
            X_train, y_train, X_val, y_val,
            symbol=symbol, n_models=N_MODELS
        )

        # 4. Ensemble predict — averaged probabilities across all 9 models
        proba = predict_proba_ensemble(models, X_test)

        # 5. Signals + backtest
        signals  = generate_signals(proba, use_percentile=True, top_pct=TOP_PCT)
        filtered = apply_filters(feat_test, signals, timeframe=timeframe)
        engine   = BacktestEngine()
        m        = engine.run(feat_test, filtered, timeframe=timeframe)

        n_trades = int(m.get("n_trades", 0))
        net      = float(m.get("total_return", 0.0))
        wr       = float(m.get("win_rate", 0.0))
        sharpe   = float(m.get("sharpe_ratio", 0.0))
        dd       = float(m.get("max_drawdown", 0.0))

        best_p    = np.maximum(proba[:, 2], proba[:, 0])
        mean_conf = float(best_p.mean())
        max_conf  = float(best_p.max())

        pred_cls    = proba.argmax(axis=1)
        pct_down    = (pred_cls == 0).mean()
        pct_neutral = (pred_cls == 1).mean()
        pct_up      = (pred_cls == 2).mean()

        sig_arr = np.array(signals["signal"])
        n_buy   = int((sig_arr == "BUY").sum())
        n_sell  = int((sig_arr == "SELL").sum())

        elapsed = time.time() - t0

        # 6. Flag
        if n_trades < MIN_TRADES:
            status = "TOO_FEW_TRADES"
        elif net > 0.08 and wr >= 0.42:
            status = "** STRONG"
        elif net > 0.04 and wr >= 0.38:
            status = "* GOOD"
        elif net > 0.0 and wr >= 0.35:
            status = "+ MARGINAL"
        elif net > 0.0:
            status = "~ WEAK+"
        else:
            status = "- LOSS"

        return {
            "symbol":           symbol,
            "timeframe":        timeframe,
            "n_models":         N_MODELS,
            "n_bars":           len(feat_df),
            "n_train":          len(X_train),
            "n_val":            len(X_val),
            "n_test":           len(X_test),
            "label_down":       label_counts.get(0, 0),
            "label_neutral":    label_counts.get(1, 0),
            "label_up":         label_counts.get(2, 0),
            "n_buy":            n_buy,
            "n_sell":           n_sell,
            "n_trades":         n_trades,
            "win_rate":         round(wr, 4),
            "net_return":       round(net, 4),
            "sharpe":           round(sharpe, 3),
            "max_drawdown":     round(dd, 4),
            "mean_conf":        round(mean_conf, 4),
            "max_conf":         round(max_conf, 4),
            "pct_pred_down":    round(float(pct_down), 3),
            "pct_pred_neutral": round(float(pct_neutral), 3),
            "pct_pred_up":      round(float(pct_up), 3),
            "status":           status,
            "elapsed_s":        round(elapsed, 1),
        }

    except Exception as e:
        logger.error(f"{symbol} failed: {e}")
        import traceback; traceback.print_exc()
        return None


def print_row(r: dict):
    print(
        f"  {r['symbol']:<13} {r['n_bars']:>6}  {r['n_trades']:>6}  "
        f"{r['win_rate']:>5.1%}  {r['net_return']:>+6.1%}  "
        f"{r['sharpe']:>6.2f}  {r['max_drawdown']:>6.1%}  "
        f"{r['mean_conf']:>8.3f}  {r['status']}"
    )


def save_csv(results, path):
    if not results:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)


def print_summary(results):
    valid  = [r for r in results if r["n_trades"] >= MIN_TRADES]
    if not valid:
        print("\nNo valid results.")
        return

    ranked     = sorted(valid, key=lambda r: r["net_return"], reverse=True)
    profitable = [r for r in ranked if r["net_return"] > 0]

    print("\n" + "=" * 80)
    print(f"  9-MODEL ENSEMBLE SCAN — {len(ranked)} symbols with >=5 trades")
    print("=" * 80)
    print(f"  {_HDR}")
    print("  " + _SEP)
    for r in ranked:
        print_row(r)
    print("  " + _SEP)
    print(f"\n  Profitable: {len(profitable)}/{len(ranked)} symbols")

    if profitable:
        best = profitable[0]
        print(f"  Best: {best['symbol']} -- {best['net_return']:+.1%} net, "
              f"{best['win_rate']:.1%} WR, {best['sharpe']:.2f} Sharpe")
        print(f"  Avg net (profitable): {np.mean([r['net_return'] for r in profitable]):+.1%}")

    shortlist = [r for r in ranked if r["net_return"] > 0.04 and r["win_rate"] >= 0.38]
    print()
    if shortlist:
        print("  SHORTLIST for OOT validation (net >4%, WR >=38%):")
        for r in shortlist:
            print(f"    {r['symbol']} -- {r['net_return']:+.1%} net | "
                  f"{r['win_rate']:.1%} WR | {r['sharpe']:.2f} Sharpe")
    else:
        print("  No symbols meet shortlist threshold.")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--symbols",   nargs="+", default=None)
    parser.add_argument("--min-bars",  type=int, default=MIN_BARS)
    args = parser.parse_args()

    symbols   = args.symbols or TRADING_PAIRS
    timeframe = args.timeframe

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir  = Path("backtest/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"ensemble_scan_{timeframe}_{ts}.csv"

    print(f"\n{'='*80}")
    print(f"  9-MODEL ENSEMBLE SCAN — {timeframe} — {len(symbols)} symbols")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Output:  {csv_path}")
    print(f"{'='*80}")
    print(f"  {_HDR}")
    print("  " + _SEP)

    results, failed = [], []

    for i, symbol in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] {symbol}")
        r = run_symbol(symbol, timeframe)
        if r is None:
            failed.append(symbol)
            continue
        results.append(r)
        print_row(r)
        save_csv(results, csv_path)  # incremental save

    print("  " + _SEP)
    print_summary(results)

    if failed:
        print(f"  Failed/skipped ({len(failed)}): {', '.join(failed)}")
    print(f"  Results saved -> {csv_path}\n")


if __name__ == "__main__":
    main()