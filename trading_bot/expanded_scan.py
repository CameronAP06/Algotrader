"""
expanded_scan.py
────────────────
Full ensemble scan across crypto, forex, and multiple timeframes.

Covers:
  - Crypto:   37 pairs (existing list)
  - Forex:    EUR/USD, GBP/USD, JPY/USD, CHF/USD, AUD/USD, CAD/USD
  - Timeframes: 4h, 8h, 1d

Runs a 9-model ensemble per symbol/timeframe combination.
Saves results incrementally — safe to interrupt and resume.

Usage:
    python expanded_scan.py                          # all symbols, all timeframes
    python expanded_scan.py --timeframes 4h 8h       # skip 1d
    python expanded_scan.py --asset-class forex      # forex only
    python expanded_scan.py --asset-class crypto     # crypto only
    python expanded_scan.py --resume                 # skip already-completed rows
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

from timeframe_comparison import fetch_ohlcv_timeframe
from data.feature_engineer import build_features, get_feature_columns
from backtest.engine import BacktestEngine
from backtest.filters import apply_filters
from models.ensemble import generate_signals
from models.lstm_ensemble import train_ensemble, predict_proba_ensemble

# ── Symbol Lists ──────────────────────────────────────────────────────────────

CRYPTO_PAIRS = [
    # Large caps
    "BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "ADA/USD", "AVAX/USD",
    "DOT/USD", "POL/USD", "LTC/USD", "BCH/USD", "LINK/USD",
    # Mid caps
    "DOGE/USD", "ETC/USD", "XLM/USD", "ATOM/USD", "ALGO/USD",
    "NEAR/USD", "S/USD", "XTZ/USD", "EOS/USD",  # VET/USD not in Kraken CSV
    # DeFi / alts
    "UNI/USD", "AAVE/USD", "MKR/USD", "COMP/USD", "SNX/USD",
    "CRV/USD", "GRT/USD", "INJ/USD", "ARB/USD", "OP/USD",
    "RUNE/USD", "FIL/USD",
    # Privacy / other
    "XMR/USD", "ZEC/USD", "DASH/USD",
]

# Kraken forex pairs — quoted as foreign/USD (e.g. how many USD per EUR)
FOREX_PAIRS = [
    "EUR/USD",   # Kraken spot only
    "GBP/USD",
    "AUD/USD",
    # CAD, CHF, JPY only available as derivatives on Kraken
]

ALL_PAIRS = CRYPTO_PAIRS + FOREX_PAIRS

TIMEFRAMES = ["15m", "30m", "1h", "4h", "12h", "1d"]

# Minimum bars per timeframe (need enough for seq_len + train/val/test split)
MIN_BARS = {
    "15m": 10000,
    "30m": 6000,
    "1h":  5000,
    "4h":  3000,
    "12h": 1000,
    "1d":   500,
}

# History to fetch per timeframe (used by API fallback only — CSV loader uses HISTORY_CAPS)
HISTORY_DAYS = {
    "15m": 180,
    "30m": 180,
    "1h":  365,
    "4h":  730,
    "12h": 730,
    "1d":  1095,
}

TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
TOP_PCT     = 0.15
N_MODELS    = 9
MIN_TRADES  = 5

_HDR = (
    f"{'Symbol':<12} {'TF':>4} {'Class':>6} {'Bars':>6} {'Trades':>7} "
    f"{'WR':>6} {'Net':>7} {'Ann':>7} {'Sharpe':>7} {'MaxDD':>7} {'Status'}"
)
_SEP = "─" * 95


# ── Core ──────────────────────────────────────────────────────────────────────

def asset_class(symbol: str) -> str:
    forex = {s.split("/")[0] for s in FOREX_PAIRS}
    return "forex" if symbol.split("/")[0] in forex else "crypto"


def run_symbol(symbol: str, timeframe: str) -> dict | None:
    t0 = time.time()
    try:
        days = HISTORY_DAYS.get(timeframe, 730)
        raw = fetch_ohlcv_timeframe(symbol, timeframe, history_days=days)

        min_b = MIN_BARS.get(timeframe, 1000)
        if raw is None or len(raw) < min_b:
            logger.warning(f"{symbol} {timeframe}: {len(raw) if raw is not None else 0} bars — skipping")
            return None

        feat_df   = build_features(raw, timeframe=timeframe)
        feat_cols = get_feature_columns(feat_df)
        # Need enough usable rows for seq_len + train/val split
        from config.settings import LSTM_PARAMS
        seq_len = LSTM_PARAMS["sequence_length"]
        min_usable = seq_len * 3   # at bare minimum 3x the sequence length
        if len(feat_df) < min_usable:
            logger.warning(f"{symbol} {timeframe}: only {len(feat_df)} usable rows after features (need {min_usable}) — skipping")
            return None

        X = feat_df[feat_cols].values.astype(np.float32)
        y = feat_df["label"].values.astype(int)
        n = len(X)

        t1 = int(n * TRAIN_RATIO)
        t2 = int(n * (TRAIN_RATIO + VAL_RATIO))

        scaler   = StandardScaler()
        X_scaled = X.copy()
        X_scaled[:t1]   = scaler.fit_transform(X[:t1])
        X_scaled[t1:t2] = scaler.transform(X[t1:t2])
        X_scaled[t2:]   = scaler.transform(X[t2:])

        X_train, y_train = X_scaled[:t1],   y[:t1]
        X_val,   y_val   = X_scaled[t1:t2], y[t1:t2]
        X_test,  y_test  = X_scaled[t2:],   y[t2:]
        feat_test = feat_df.iloc[t2:].reset_index(drop=True)

        label_counts = {int(k): int(v)
                        for k, v in zip(*np.unique(y_train, return_counts=True))}

        # 9-model ensemble
        models = train_ensemble(
            X_train, y_train, X_val, y_val,
            symbol=f"{symbol}_{timeframe}", n_models=N_MODELS
        )

        proba    = predict_proba_ensemble(models, X_test)
        signals  = generate_signals(proba, use_percentile=True, top_pct=TOP_PCT)
        filtered = apply_filters(feat_test, signals, timeframe=timeframe)
        m        = BacktestEngine().run(feat_test, filtered, timeframe=timeframe)

        n_trades = int(m.get("n_trades", 0))
        net      = float(m.get("total_return", 0.0))
        wr       = float(m.get("win_rate", 0.0))
        sharpe   = float(m.get("sharpe_ratio", 0.0))
        dd       = float(m.get("max_drawdown", 0.0))

        # Annualised return — normalises cumulative return by test window length
        # so results are comparable across timeframes and history lengths
        hours_per_bar = {
            "15m": 0.25, "30m": 0.5, "1h": 1.0, "4h": 4.0,
            "12h": 12.0, "1d": 24.0, "1w": 168.0, "2w": 336.0,
        }
        test_hours = len(X_test) * hours_per_bar.get(timeframe, 1.0)
        test_years = max(test_hours / 8760, 1/365)   # floor at 1 day to avoid div/0
        ann_return = (1 + net) ** (1 / test_years) - 1

        best_p    = np.maximum(proba[:, 2], proba[:, 0])
        mean_conf = float(best_p.mean())

        sig_arr = np.array(signals["signal"])
        n_buy   = int((sig_arr == "BUY").sum())
        n_sell  = int((sig_arr == "SELL").sum())

        elapsed = time.time() - t0

        if n_trades < MIN_TRADES:
            status = "TOO_FEW_TRADES"
        elif ann_return > 0.20 and wr >= 0.42:
            status = "** STRONG"
        elif ann_return > 0.10 and wr >= 0.38:
            status = "* GOOD"
        elif ann_return > 0.0 and wr >= 0.35:
            status = "+ MARGINAL"
        elif ann_return > 0.0:
            status = "~ WEAK+"
        else:
            status = "- LOSS"

        return {
            "symbol":           symbol,
            "timeframe":        timeframe,
            "asset_class":      asset_class(symbol),
            "n_models":         N_MODELS,
            "n_bars":           n,
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
            "ann_return":       round(ann_return, 4),
            "sharpe":           round(sharpe, 3),
            "max_drawdown":     round(dd, 4),
            "mean_conf":        round(mean_conf, 4),
            "status":           status,
            "elapsed_s":        round(elapsed, 1),
        }

    except Exception as e:
        logger.error(f"{symbol} {timeframe} failed: {e}")
        import traceback; traceback.print_exc()
        return None


def print_row(r: dict):
    print(
        f"  {r['symbol']:<12} {r['timeframe']:>4}  {r['asset_class']:>6}  "
        f"{r['n_bars']:>5}  {r['n_trades']:>6}  "
        f"{r['win_rate']:>5.1%}  {r['net_return']:>+6.1%}  {r['ann_return']:>+6.1%}  "
        f"{r['sharpe']:>6.2f}  {r['max_drawdown']:>6.1%}  {r['status']}"
    )


def save_csv(results, path):
    if not results:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)


def load_completed(path: Path) -> set:
    """Load already-completed symbol/timeframe pairs for resume support."""
    if not path.exists():
        return set()
    completed = set()
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            completed.add((row["symbol"], row["timeframe"]))
    return completed


def print_summary(results: list):
    valid  = [r for r in results if r["n_trades"] >= MIN_TRADES]
    if not valid:
        print("\nNo valid results.")
        return

    ranked     = sorted(valid, key=lambda r: r["net_return"], reverse=True)
    profitable = [r for r in ranked if r["net_return"] > 0]

    print(f"\n{'='*85}")
    print(f"  EXPANDED ENSEMBLE SCAN — FINAL SUMMARY")
    print(f"{'='*85}")
    print(f"  {_HDR}")
    print("  " + _SEP)
    for r in ranked[:30]:   # Top 30
        print_row(r)
    print("  " + _SEP)

    print(f"\n  Profitable: {len(profitable)}/{len(ranked)} symbol/timeframe combos")

    # Best per asset class
    for cls in ["crypto", "forex"]:
        cls_results = [r for r in profitable if r["asset_class"] == cls]
        if cls_results:
            best = max(cls_results, key=lambda r: r["sharpe"])
            print(f"  Best {cls}: {best['symbol']} {best['timeframe']} — "
                  f"{best['net_return']:+.1%} net | {best['win_rate']:.1%} WR | "
                  f"{best['sharpe']:.2f} Sharpe")

    # Shortlist
    shortlist = [r for r in ranked
                 if r["ann_return"] > 0.10 and r["win_rate"] >= 0.38]
    print(f"\n  SHORTLIST for OOT validation ({len(shortlist)} combos):")
    for r in shortlist:
        print(f"    {r['symbol']:<12} {r['timeframe']:>4} | "
              f"{r['net_return']:+.1%} net | {r['ann_return']:+.1%} ann | "
              f"{r['win_rate']:.1%} WR | {r['sharpe']:.2f} Sharpe | {r['asset_class']}")

    # Timeframe comparison for symbols appearing in multiple TFs
    tf_groups = {}
    for r in profitable:
        tf_groups.setdefault(r["symbol"], []).append(r)
    multi_tf = {s: rs for s, rs in tf_groups.items() if len(rs) > 1}
    if multi_tf:
        print(f"\n  MULTI-TIMEFRAME WINNERS (profitable across >1 TF):")
        for sym, rs in sorted(multi_tf.items()):
            tfs = " | ".join(
                f"{r['timeframe']} {r['net_return']:+.1%} ({r['sharpe']:.2f} Sh)"
                for r in sorted(rs, key=lambda x: x["timeframe"])
            )
            print(f"    {sym:<12}  {tfs}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframes",   nargs="+", default=TIMEFRAMES)
    parser.add_argument("--asset-class",  choices=["all", "crypto", "forex"],
                        default="all")
    parser.add_argument("--symbols",      nargs="+", default=None)
    parser.add_argument("--resume",       action="store_true",
                        help="Skip symbol/TF combos already in output CSV")
    args = parser.parse_args()

    # Symbol list
    if args.symbols:
        symbols = args.symbols
    elif args.asset_class == "crypto":
        symbols = CRYPTO_PAIRS
    elif args.asset_class == "forex":
        symbols = FOREX_PAIRS
    else:
        symbols = ALL_PAIRS

    timeframes = args.timeframes
    total      = len(symbols) * len(timeframes)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir  = Path("backtest/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"expanded_scan_{ts}.csv"

    # Resume support — find existing output file if --resume
    completed = set()
    if args.resume:
        existing = sorted(out_dir.glob("expanded_scan_*.csv"), reverse=True)
        if existing:
            csv_path  = existing[0]
            completed = load_completed(csv_path)
            logger.info(f"Resuming from {csv_path} — {len(completed)} combos already done")

    print(f"\n{'='*85}")
    print(f"  EXPANDED ENSEMBLE SCAN")
    print(f"  Symbols: {len(symbols)} | Timeframes: {timeframes} | Combos: {total}")
    print(f"  Asset classes: {args.asset_class} | Models per combo: {N_MODELS}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Output:  {csv_path}")
    print(f"{'='*85}")
    print(f"  {_HDR}")
    print("  " + _SEP)

    results, failed, skipped = [], [], 0

    # Load existing results if resuming
    if completed and csv_path.exists():
        with open(csv_path, encoding="utf-8-sig") as f:
            results = list(csv.DictReader(f))
            # Convert numeric fields back
            for r in results:
                for k in ["n_bars","n_train","n_val","n_test","n_trades",
                          "n_buy","n_sell","n_models",
                          "label_down","label_neutral","label_up"]:
                    if k in r:
                        r[k] = int(r[k])
                for k in ["win_rate","net_return","sharpe","max_drawdown",
                          "mean_conf","elapsed_s"]:
                    if k in r:
                        r[k] = float(r[k])

    idx = 0
    for symbol in symbols:
        for timeframe in timeframes:
            idx += 1

            if (symbol, timeframe) in completed:
                skipped += 1
                continue

            logger.info(f"[{idx}/{total}] {symbol} {timeframe}")
            r = run_symbol(symbol, timeframe)

            if r is None:
                failed.append(f"{symbol} {timeframe}")
                continue

            results.append(r)
            print_row(r)
            save_csv(results, csv_path)

    print("  " + _SEP)
    print_summary(results)

    if skipped:
        print(f"  Skipped (already complete): {skipped}")
    if failed:
        print(f"  Failed/skipped ({len(failed)}): {', '.join(failed[:10])}"
              + ("..." if len(failed) > 10 else ""))
    print(f"  Results saved -> {csv_path}\n")


if __name__ == "__main__":
    main()