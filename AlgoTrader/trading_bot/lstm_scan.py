"""
lstm_scan.py
────────────
LSTM-only scan across all 4h symbols to identify which have genuine signal.

For each symbol it:
  1. Fetches + engineers features (from cache if available)
  2. Trains a single LSTM on train+val split
  3. Backtests on the held-out test set
  4. Records key metrics

Outputs:
  - Live ranked table printed to console as each symbol completes
  - backtest/results/lstm_scan_4h_<timestamp>.csv

Usage:
    python lstm_scan.py
    python lstm_scan.py --timeframe 4h      # default
    python lstm_scan.py --symbols BTC/USD ETH/USD DOGE/USD  # subset
    python lstm_scan.py --min-bars 4000     # skip thin symbols
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
from models import lstm_model

# ── Constants ─────────────────────────────────────────────────────────────────

TIMEFRAME   = "4h"
TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
TEST_RATIO  = 0.20
FEE_PER_TRADE = 0.001   # 0.1% per side — Kraken taker
TOP_PCT     = 0.15      # Top 15% most confident bars fire signals
MIN_BARS    = 3000      # Skip symbols with less data than this
MIN_TRADES  = 5         # Skip results with too few trades to be meaningful

# Column widths for live console table
_HDR = (
    f"{'Symbol':<14} {'Bars':>6} {'Trades':>7} {'WR':>6} "
    f"{'Net':>7} {'Sharpe':>7} {'MaxDD':>7} {'MeanConf':>9} {'Status'}"
)
_SEP = "─" * 80


# ── Helpers ───────────────────────────────────────────────────────────────────

def split_data(X, y, feat_df):
    n  = len(X)
    t1 = int(n * TRAIN_RATIO)
    t2 = int(n * (TRAIN_RATIO + VAL_RATIO))
    return (
        X[:t1],  y[:t1],
        X[t1:t2], y[t1:t2],
        X[t2:],   y[t2:],
        feat_df.iloc[t2:].reset_index(drop=True),
    )


def run_symbol(symbol: str, timeframe: str) -> dict | None:
    """Train LSTM + backtest for one symbol. Returns metrics dict or None on failure."""
    t0 = time.time()
    try:
        # ── 1. Fetch & feature engineer ───────────────────────────────────────
        raw = fetch_ohlcv_timeframe(symbol, timeframe, history_days=HISTORY_DAYS)
        if raw is None or len(raw) < MIN_BARS:
            logger.warning(f"{symbol}: only {len(raw) if raw is not None else 0} bars — skipping")
            return None

        feat_df   = build_features(raw, timeframe=timeframe)
        feat_cols = get_feature_columns(feat_df)

        X = feat_df[feat_cols].values.astype(np.float32)
        y = feat_df["label"].values.astype(int)

        # ── 2. Scale ──────────────────────────────────────────────────────────
        n  = len(X)
        t1 = int(n * TRAIN_RATIO)
        t2 = int(n * (TRAIN_RATIO + VAL_RATIO))

        scaler = StandardScaler()
        X_scaled = X.copy()
        X_scaled[:t1]  = scaler.fit_transform(X[:t1])
        X_scaled[t1:t2] = scaler.transform(X[t1:t2])
        X_scaled[t2:]   = scaler.transform(X[t2:])

        X_train, y_train, X_val, y_val, X_test, y_test, feat_test = split_data(
            X_scaled, y, feat_df
        )

        label_counts = {int(k): int(v)
                        for k, v in zip(*np.unique(y_train, return_counts=True))}

        # ── 3. Train LSTM ─────────────────────────────────────────────────────
        model = lstm_model.train(X_train, y_train, X_val, y_val, symbol=symbol)

        # ── 4. Predict ────────────────────────────────────────────────────────
        proba = lstm_model.predict_proba(model, X_test)

        # ── 5. Signals + backtest ─────────────────────────────────────────────
        signals  = generate_signals(proba, use_percentile=True, top_pct=TOP_PCT)
        filtered = apply_filters(feat_test, signals, timeframe=timeframe)
        engine   = BacktestEngine()
        m        = engine.run(feat_test, filtered, timeframe=timeframe)

        n_trades = int(m.get("n_trades", 0))
        net      = float(m.get("total_return", 0.0))   # engine already nets out fees
        wr       = float(m.get("win_rate", 0.0))
        sharpe   = float(m.get("sharpe_ratio", 0.0))
        dd       = float(m.get("max_drawdown", 0.0))

        # Confidence stats
        best_p    = np.maximum(proba[:, 2], proba[:, 0])
        mean_conf = float(best_p.mean())
        max_conf  = float(best_p.max())

        # Prediction bias — flag if model heavily leans one direction
        pred_cls   = proba.argmax(axis=1)
        pct_down   = (pred_cls == 0).mean()
        pct_neutral= (pred_cls == 1).mean()
        pct_up     = (pred_cls == 2).mean()

        # Signal counts
        sig_arr = np.array(signals["signal"])
        n_buy   = int((sig_arr == "BUY").sum())
        n_sell  = int((sig_arr == "SELL").sum())

        elapsed = time.time() - t0

        # ── 6. Flag ───────────────────────────────────────────────────────────
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
            "symbol":       symbol,
            "timeframe":    timeframe,
            "n_bars":       len(feat_df),
            "n_train":      len(X_train),
            "n_val":        len(X_val),
            "n_test":       len(X_test),
            "label_down":   label_counts.get(0, 0),
            "label_neutral":label_counts.get(1, 0),
            "label_up":     label_counts.get(2, 0),
            "n_buy":        n_buy,
            "n_sell":       n_sell,
            "n_trades":     n_trades,
            "win_rate":     round(wr, 4),
            "net_return":   round(net, 4),
            "sharpe":       round(sharpe, 3),
            "max_drawdown": round(dd, 4),
            "mean_conf":    round(mean_conf, 4),
            "max_conf":     round(max_conf, 4),
            "pct_pred_down":  round(pct_down, 3),
            "pct_pred_neutral": round(pct_neutral, 3),
            "pct_pred_up":  round(pct_up, 3),
            "status":       status,
            "elapsed_s":    round(elapsed, 1),
        }

    except Exception as e:
        logger.error(f"{symbol} failed: {e}")
        import traceback; traceback.print_exc()
        return None


def print_row(r: dict):
    """Print a single result row to console."""
    print(
        f"  {r['symbol']:<13} {r['n_bars']:>6}  {r['n_trades']:>6}  "
        f"{r['win_rate']:>5.1%}  {r['net_return']:>+6.1%}  "
        f"{r['sharpe']:>6.2f}  {r['max_drawdown']:>6.1%}  "
        f"{r['mean_conf']:>8.3f}  {r['status']}"
    )


def save_csv(results: list, path: Path):
    if not results:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)


def print_summary(results: list):
    """Print ranked summary table at end of scan."""
    valid = [r for r in results if r["n_trades"] >= MIN_TRADES]
    if not valid:
        print("\nNo valid results with sufficient trades.")
        return

    ranked = sorted(valid, key=lambda r: r["net_return"], reverse=True)

    print("\n" + "=" * 80)
    print(f"  LSTM 4H SCAN SUMMARY — {len(ranked)} symbols with ≥{MIN_TRADES} trades")
    print("=" * 80)
    print(f"  {_HDR}")
    print("  " + _SEP)

    profitable = [r for r in ranked if r["net_return"] > 0]
    for r in ranked:
        print_row(r)

    print("  " + _SEP)
    print(f"\n  Profitable: {len(profitable)}/{len(ranked)} symbols")
    if profitable:
        best = profitable[0]
        print(f"  Best:       {best['symbol']} -- {best['net_return']:+.1%} net, "
              f"{best['win_rate']:.1%} WR, {best['sharpe']:.2f} Sharpe")
        avg_net = np.mean([r["net_return"] for r in profitable])
        print(f"  Avg net (profitable only): {avg_net:+.1%}")
    print()

    # Shortlist — candidates for walk-forward
    shortlist = [r for r in ranked if r["net_return"] > 0.04 and r["win_rate"] >= 0.38]
    if shortlist:
        print("  SHORTLIST (net >4%, WR >=38% -- run walk-forward on these):")
        for r in shortlist:
            print(f"    {r['symbol']} -- {r['net_return']:+.1%} net | "
                  f"{r['win_rate']:.1%} WR | {r['sharpe']:.2f} Sharpe")
    else:
        print("  No symbols meet shortlist threshold (net >4%, WR ≥38%)")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LSTM-only 4h signal scan")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--symbols",   nargs="+", default=None,
                        help="Override symbol list (default: all TRADING_PAIRS)")
    parser.add_argument("--min-bars",  type=int, default=MIN_BARS)
    args = parser.parse_args()

    symbols   = args.symbols or TRADING_PAIRS
    timeframe = args.timeframe

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir  = Path("backtest/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"lstm_scan_{timeframe}_{ts}.csv"

    print(f"\n{'='*80}")
    print(f"  LSTM SIGNAL SCAN — {timeframe} — {len(symbols)} symbols")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Output:  {csv_path}")
    print(f"{'='*80}")
    print(f"  {_HDR}")
    print("  " + _SEP)

    results = []
    failed  = []

    for i, symbol in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] {symbol}")
        r = run_symbol(symbol, timeframe)
        if r is None:
            failed.append(symbol)
            continue

        results.append(r)
        print_row(r)

        # Save incrementally — don't lose progress if it crashes mid-scan
        save_csv(results, csv_path)

    print("  " + _SEP)
    print_summary(results)

    if failed:
        print(f"  Failed/skipped ({len(failed)}): {', '.join(failed)}")
    print(f"  Full results saved → {csv_path}\n")


if __name__ == "__main__":
    main()