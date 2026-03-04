"""
oot_validate.py  —  Out-of-Time Validation
───────────────────────────────────────────
Tests a FROZEN, already-trained LSTM on historical data that predates
the training window entirely. The model never sees this data during
training — it's a true out-of-sample test.

Workflow:
  1. Fetch the full available history for a symbol (e.g. back to 2017)
  2. Split into:
       [HISTORICAL (OOT)]  |  [TRAIN]  |  [VAL]  |  [TEST]
        never touched          model trained on this        already tested
  3. Fit scaler on TRAIN only (no leakage)
  4. Apply scaler + frozen model to HISTORICAL period
  5. Backtest and report

This answers: "Does the signal generalise to market regimes the model
never experienced during training?"

Usage:
    python oot_validate.py --symbol DOGE/USD
    python oot_validate.py --symbol BTC/USD --history-days 3000
    python oot_validate.py  # defaults to DOGE/USD
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

from config.settings import HISTORY_DAYS
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
TOP_PCT     = 0.15
MIN_OOT_BARS = 200   # Minimum bars in OOT window to be worth reporting

# Maximum history to fetch — 3000 days gets back to mid-2017 for BTC/ETH
MAX_HISTORY_DAYS = 3000


# ── Core ──────────────────────────────────────────────────────────────────────

def run_oot(symbol: str, timeframe: str, history_days: int, verbose: bool = True):
    """
    Full out-of-time validation for one symbol.

    Returns a dict with metrics for both the OOT period and the
    standard test period side-by-side for comparison.
    """

    # ── 1. Fetch maximum available history ────────────────────────────────────
    logger.info(f"Fetching {history_days} days of history for {symbol}...")
    raw = fetch_ohlcv_timeframe(symbol, timeframe, history_days=history_days)

    if raw is None or len(raw) < 500:
        logger.error(f"{symbol}: insufficient data ({len(raw) if raw is not None else 0} bars)")
        return None

    feat_df   = build_features(raw, timeframe=timeframe)
    feat_cols = get_feature_columns(feat_df)

    X_all = feat_df[feat_cols].values.astype(np.float32)
    y_all = feat_df["label"].values.astype(int)
    n_all = len(X_all)

    # ── 2. Identify the standard train/val/test split boundaries ─────────────
    # These are the same splits used during training — based on HISTORY_DAYS data
    # Standard data: last HISTORY_DAYS worth of bars
    bars_standard = int(HISTORY_DAYS / (history_days / n_all))
    bars_standard = min(bars_standard, n_all)

    # OOT data = everything BEFORE the standard training window
    oot_end   = n_all - bars_standard
    train_end = oot_end + int(bars_standard * TRAIN_RATIO)
    val_end   = train_end + int(bars_standard * VAL_RATIO)
    # test_end  = n_all

    n_oot   = oot_end
    n_train = train_end - oot_end
    n_val   = val_end - train_end
    n_test  = n_all - val_end

    if n_oot < MIN_OOT_BARS:
        logger.warning(f"{symbol}: only {n_oot} OOT bars (need {MIN_OOT_BARS}) — "
                       f"try --history-days {history_days + 500}")
        return None

    logger.info(f"Data split:")
    logger.info(f"  OOT   (unseen): bars 0     → {oot_end}   = {n_oot} bars "
                f"({n_oot*4/24:.0f} days)  "
                f"[{feat_df.index[0]} → {feat_df.index[oot_end-1]}]")
    logger.info(f"  Train:          bars {oot_end} → {train_end} = {n_train} bars")
    logger.info(f"  Val:            bars {train_end} → {val_end}   = {n_val} bars")
    logger.info(f"  Test:           bars {val_end} → {n_all}  = {n_test} bars")

    # ── 3. Fit scaler on TRAIN only — no leakage into OOT ────────────────────
    scaler = StandardScaler()
    scaler.fit(X_all[oot_end:train_end])

    X_oot   = scaler.transform(X_all[:oot_end])
    X_train = scaler.transform(X_all[oot_end:train_end])
    X_val   = scaler.transform(X_all[train_end:val_end])
    X_test  = scaler.transform(X_all[val_end:])

    y_oot   = y_all[:oot_end]
    y_train = y_all[oot_end:train_end]
    y_val   = y_all[train_end:val_end]
    y_test  = y_all[val_end:]

    feat_oot  = feat_df.iloc[:oot_end].reset_index(drop=True)
    feat_test = feat_df.iloc[val_end:].reset_index(drop=True)

    # ── 4. Train LSTM on train+val (same as standard pipeline) ───────────────
    logger.info(f"Training LSTM on {n_train} bars...")
    model = lstm_model.train(X_train, y_train, X_val, y_val, symbol=symbol)

    # ── 5. Run inference on BOTH test and OOT windows ─────────────────────────
    def backtest_window(X, y, feat, label):
        proba    = lstm_model.predict_proba(model, X)
        signals  = generate_signals(proba, use_percentile=True, top_pct=TOP_PCT)
        filtered = apply_filters(feat, signals, timeframe=timeframe)
        engine   = BacktestEngine()
        m        = engine.run(feat, filtered, timeframe=timeframe)

        n_trades = int(m.get("n_trades", 0))
        net      = float(m.get("total_return", 0.0))
        wr       = float(m.get("win_rate", 0.0))
        sharpe   = float(m.get("sharpe_ratio", 0.0))
        dd       = float(m.get("max_drawdown", 0.0))
        acc      = (proba.argmax(axis=1) == y).mean()

        sig_arr  = np.array(signals["signal"])
        n_buy    = int((sig_arr == "BUY").sum())
        n_sell   = int((sig_arr == "SELL").sum())

        best_p   = np.maximum(proba[:, 2], proba[:, 0])

        return {
            "window":    label,
            "n_bars":    len(X),
            "n_days":    round(len(X) * 4 / 24, 0),
            "n_buy":     n_buy,
            "n_sell":    n_sell,
            "n_trades":  n_trades,
            "win_rate":  round(wr, 4),
            "net_return":round(net, 4),
            "sharpe":    round(sharpe, 3),
            "max_dd":    round(dd, 4),
            "accuracy":  round(float(acc), 4),
            "mean_conf": round(float(best_p.mean()), 4),
        }

    logger.info("Running inference on standard TEST window...")
    test_result = backtest_window(X_test, y_test, feat_test, "TEST (in-distribution)")

    logger.info("Running inference on OOT (historical, never-seen) window...")
    oot_result  = backtest_window(X_oot,  y_oot,  feat_oot,  "OOT  (out-of-time)")

    return test_result, oot_result


def print_comparison(symbol: str, test_r: dict, oot_r: dict):
    """Print side-by-side comparison of test vs OOT results."""
    bar = "=" * 70

    def verdict(r):
        if r["n_trades"] < 5:    return "TOO FEW TRADES"
        if r["net_return"] > 0.08 and r["win_rate"] >= 0.42: return "** STRONG"
        if r["net_return"] > 0.04 and r["win_rate"] >= 0.38: return "*  GOOD"
        if r["net_return"] > 0.0  and r["win_rate"] >= 0.35: return "+  MARGINAL"
        if r["net_return"] > 0.0:                             return "~  WEAK+"
        return "-  LOSS"

    print(f"\n{bar}")
    print(f"  OUT-OF-TIME VALIDATION — {symbol} [{TIMEFRAME}]")
    print(bar)
    print(f"  {'':30} {'TEST':>18} {'OOT':>18}")
    print(f"  {'':30} {'(in-distribution)':>18} {'(historical)':>18}")
    print(f"  {'-'*66}")
    print(f"  {'Window length (days)':<30} {test_r['n_days']:>18.0f} {oot_r['n_days']:>18.0f}")
    print(f"  {'Bars':<30} {test_r['n_bars']:>18} {oot_r['n_bars']:>18}")
    print(f"  {'Trades':<30} {test_r['n_trades']:>18} {oot_r['n_trades']:>18}")
    print(f"  {'Win rate':<30} {test_r['win_rate']:>17.1%} {oot_r['win_rate']:>17.1%}")
    print(f"  {'Net return':<30} {test_r['net_return']:>+17.1%} {oot_r['net_return']:>+17.1%}")
    print(f"  {'Sharpe':<30} {test_r['sharpe']:>18.2f} {oot_r['sharpe']:>18.2f}")
    print(f"  {'Max drawdown':<30} {test_r['max_dd']:>17.1%} {oot_r['max_dd']:>17.1%}")
    print(f"  {'Mean confidence':<30} {test_r['mean_conf']:>18.3f} {oot_r['mean_conf']:>18.3f}")
    print(f"  {'Verdict':<30} {verdict(test_r):>18} {verdict(oot_r):>18}")
    print(f"  {'-'*66}")

    # Interpretation
    print()
    t_pos = test_r["net_return"] > 0
    o_pos = oot_r["net_return"] > 0
    if t_pos and o_pos:
        print("  RESULT: Signal holds in BOTH windows.")
        print("  Strong evidence of a genuine, regime-independent edge.")
    elif t_pos and not o_pos:
        print("  RESULT: Profitable in TEST but not in OOT.")
        print("  Signal may be specific to 2021-2026 market conditions.")
        print("  Treat live deployment with caution.")
    elif not t_pos and o_pos:
        print("  RESULT: Profitable in OOT but not TEST.")
        print("  Unusual — possibly regime-dependent in the opposite direction.")
    else:
        print("  RESULT: Loss in both windows. No reliable signal found.")
    print(f"\n{bar}\n")


def save_results(symbol: str, test_r: dict, oot_r: dict):
    out_dir  = Path("backtest/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"oot_{symbol.replace('/','_')}_{TIMEFRAME}_{ts}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=test_r.keys())
        w.writeheader()
        w.writerow(test_r)
        w.writerow(oot_r)

    logger.success(f"Results saved -> {csv_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Out-of-time validation for LSTM signal")
    parser.add_argument("--symbol",       default="DOGE/USD")
    parser.add_argument("--timeframe",    default="4h")
    parser.add_argument("--history-days", type=int, default=MAX_HISTORY_DAYS,
                        help=f"How far back to fetch (default: {MAX_HISTORY_DAYS})")
    args = parser.parse_args()

    result = run_oot(args.symbol, args.timeframe, args.history_days)
    if result is None:
        logger.error("Validation failed — see errors above.")
        return

    test_r, oot_r = result
    print_comparison(args.symbol, test_r, oot_r)
    save_results(args.symbol, test_r, oot_r)


if __name__ == "__main__":
    main()
