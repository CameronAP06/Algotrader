#!/usr/bin/env python3
"""
timeframe_comparison.py
═══════════════════════════════════════════════════════════════════════════════
Loops through multiple timeframes, runs the full training + backtest pipeline
for each, and produces a comparison table showing which timeframe has the
most learnable signal.

Timeframes tested: 15m, 1h, 4h, 1d
For each timeframe x symbol combination:
  - Fetches OHLCV data at that resolution
  - Engineers features
  - Trains TFT + CNN + LSTM ensemble
  - Runs walk-forward validation (if --walkforward)
  - OR runs single backtest
  - Records accuracy, return, Sharpe, win rate, trades

Usage:
  python timeframe_comparison.py
  python timeframe_comparison.py --symbol INJ/USD
  python timeframe_comparison.py --walkforward
  python timeframe_comparison.py --timeframes 1h 4h 1d
═══════════════════════════════════════════════════════════════════════════════
"""
import argparse, os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
import pandas as pd
import numpy as np
from pathlib import Path

from config.settings import TRADING_PAIRS, LOG_DIR, RESULTS_DIR
from data.alt_data import fetch_all_alt_data, merge_alt_data
from data.feature_engineer import build_features, get_feature_columns
from utils.splitter import time_split, save_scaler
from models import catboost_model, cnn_model, lstm_model
from models.ensemble import (
    weighted_ensemble, optimise_weights,
    generate_signals, save_weights
)
from backtest.engine import BacktestEngine
from backtest.filters import apply_filters

os.makedirs(LOG_DIR, exist_ok=True)
logger.add(f"{LOG_DIR}/tf_compare_{{time}}.log", rotation="10 MB", level="INFO")


# ─── Checkpoint / Resume ────────────────────────────────────────────

def checkpoint_path(symbol: str, timeframe: str) -> Path:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return Path(RESULTS_DIR) / f"{symbol.replace('/',  '_')}_{timeframe}_walkforward.csv"


def load_checkpoint(symbol: str, timeframe: str):
    """Load completed walk-forward results if they exist, to skip retraining."""
    path = checkpoint_path(symbol, timeframe)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df.empty or len(df) < 2:
            return None
        result = {
            "symbol":       symbol,
            "timeframe":    timeframe,
            "accuracy":     float(df["accuracy"].median()),
            "total_return": float(df["total_return"].median()),
            "sharpe_ratio": float(df["sharpe_ratio"].median()),
            "max_drawdown": float(df["max_drawdown"].min()),
            "win_rate":     float(df["win_rate"].median()),
            "n_trades":     float(df["n_trades"].median()),
            "n_bars":       0,
        }
        n = len(df)
        acc = result["accuracy"]
        ret = result["total_return"]
        logger.info(f"[CHECKPOINT] Skipping {symbol} {timeframe} -- already done ({n} folds, acc={acc:.1%}, ret={ret:.2%})")
        return result
    except Exception as e:
        logger.warning(f"Failed to load checkpoint {symbol} {timeframe}: {e}")
        return None


# ─── GPU Monitoring ───────────────────────────────────────────────────

def log_gpu_temp():
    """Log GPU temperature and utilisation. Warns at >=80C."""
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode != 0:
            return
        parts = [p.strip() for p in r.stdout.strip().split(",")]
        temp, util, mem_used, mem_total = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        msg = f"GPU: {temp}C | {util}% util | {mem_used}/{mem_total} MiB VRAM"
        if temp >= 85:
            logger.warning(f"HIGH GPU TEMP: {msg} -- consider pausing")
        elif temp >= 75:
            logger.warning(f"GPU warm: {msg}")
        else:
            logger.info(msg)
    except Exception:
        pass  # nvidia-smi unavailable -- non-fatal


# ─── Timeframe Config ────────────────────────────────────────────────────────

TIMEFRAME_CONFIG = {
    "15m": {
        "ccxt_tf":    "15m",
        "history_days": 365,      # 1 year — 15m bars fill up fast
        "label_horizon": 16,      # 16 bars = 4 hours ahead
        "min_bars":   5000,
        "description": "15 minutes"
    },
    "1h": {
        "ccxt_tf":    "1h",
        "history_days": 1825,     # 5 years
        "label_horizon": 8,       # 8 bars = 8 hours ahead
        "min_bars":   3000,
        "description": "1 hour"
    },
    "4h": {
        "ccxt_tf":    "4h",
        "history_days": 1825,
        "label_horizon": 6,       # 6 bars = 24 hours ahead
        "min_bars":   1000,
        "description": "4 hours"
    },
        
    "8h": {
        "ccxt_tf":    "8h",
        "history_days": 1825,
        "label_horizon": 6,       # 6 bars = 24 hours ahead
        "min_bars":   1000,
        "description": "8 hours"
    },
    "1d": {
        "ccxt_tf":    "1d",
        "history_days": 1825,
        "label_horizon": 5,       # 5 bars = 5 days ahead
        "min_bars":   300,
        "description": "1 day"
    },
}

SYMBOL_MAP = {
    "BTC/USD":  "BTC/USDT",
    "ETH/USD":  "ETH/USDT",
    "XRP/USD":  "XRP/USDT",
    "SOL/USD":  "SOL/USDT",
    "AVAX/USD":  "AVAX/USDT",
    "LINK/USD":  "LINK/USDT",
    "MATIC/USD":  "MATIC/USDT",
    "INJ/USD":  "INJ/USDT",
    "ARB/USD":  "ARB/USDT",
    "OP/USD":   "OP/USDT",
    "RUNE/USD":  "RUNE/USDT",
    "DOGE/USD":  "DOGE/USDT",
    "ADA/USD":  "ADA/USDT",
}


# ─── Data Fetching ───────────────────────────────────────────────────────────

def fetch_ohlcv_timeframe(symbol: str, timeframe: str, history_days: int) -> pd.DataFrame:
    """Fetch OHLCV data at a specific timeframe from Binance."""
    import ccxt
    from datetime import datetime, timedelta

    binance_symbol = SYMBOL_MAP.get(symbol, symbol.replace("/", "/"))
    cache_path = Path(f"data/raw/{symbol.replace('/','_')}_{timeframe}.csv")

    if cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=["timestamp"])
        latest = df["timestamp"].max()
        # Use cache if less than 4h old for intraday, 24h for daily
        max_age = timedelta(hours=4 if timeframe in ["15m","1h"] else 24)
        if pd.Timestamp.utcnow().tz_localize(None) - latest.replace(tzinfo=None) < max_age:
            logger.info(f"Loaded {symbol} {timeframe} from cache ({len(df)} rows)")
            return df

    logger.info(f"Fetching {symbol} [{timeframe}] — {history_days} days...")
    exchange = ccxt.binance({"enableRateLimit": True})

    since_ms  = int((datetime.utcnow() - timedelta(days=history_days)).timestamp() * 1000)
    all_ohlcv = []
    batch     = 1000

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(binance_symbol, timeframe, since=since_ms, limit=batch)
        except Exception as e:
            logger.error(f"Fetch error {symbol} {timeframe}: {e}")
            break

        if not ohlcv:
            break

        all_ohlcv.extend(ohlcv)
        if len(ohlcv) < batch:
            break
        since_ms = ohlcv[-1][0] + 1
        time.sleep(0.1)

    if not all_ohlcv:
        logger.error(f"No data returned for {symbol} {timeframe}")
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.success(f"{symbol} {timeframe}: {len(df)} candles ({df['timestamp'].iloc[0].date()} -> {df['timestamp'].iloc[-1].date()})")
    return df


# ─── Single Pipeline Run ─────────────────────────────────────────────────────

def run_pipeline_for_timeframe(symbol: str, raw_df: pd.DataFrame,
                                timeframe: str, horizon: int,
                                alt_df: pd.DataFrame) -> dict:
    """Run full pipeline for one symbol + timeframe combination."""

    # Merge alt data (daily — still useful even for 4h/1d)
    enriched_df = merge_alt_data(raw_df, alt_df)

    # Override horizon for this timeframe
    import config.settings as s
    original_horizon = s.PREDICTION_HORIZON
    s.PREDICTION_HORIZON = horizon

    feat_df      = build_features(enriched_df, symbol)
    feature_cols = get_feature_columns(feat_df)

    # Restore
    s.PREDICTION_HORIZON = original_horizon

    if len(feat_df) < 500:
        logger.warning(f"Insufficient data: {symbol} {timeframe} ({len(feat_df)} rows)")
        return {}

    logger.info(f"{symbol} {timeframe}: {len(feat_df)} rows, {len(feature_cols)} features")

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = time_split(feat_df, feature_cols)
    save_scaler(scaler, f"{symbol}_{timeframe}")

    # Train
    cat  = catboost_model.train(X_train, y_train, X_val, y_val, f"{symbol}_{timeframe}")
    cnn  = cnn_model.train(X_train, y_train, X_val, y_val, f"{symbol}_{timeframe}")
    lstm = lstm_model.train(X_train, y_train, X_val, y_val, f"{symbol}_{timeframe}")

    # Ensemble
    cat_val_p  = catboost_model.predict_proba(cat,  X_val)
    cnn_val_p  = cnn_model.predict_proba(cnn,       X_val)
    lstm_val_p = lstm_model.predict_proba(lstm,      X_val)
    weights    = optimise_weights(cat_val_p, cnn_val_p, lstm_val_p, y_val)

    cat_test_p  = catboost_model.predict_proba(cat,  X_test)
    cnn_test_p  = cnn_model.predict_proba(cnn,       X_test)
    lstm_test_p = lstm_model.predict_proba(lstm,      X_test)

    blended  = weighted_ensemble(cat_test_p, cnn_test_p, lstm_test_p, weights)
    signals  = generate_signals(blended, symbol=symbol)

    preds    = blended.argmax(axis=1)
    accuracy = (preds == y_test[-len(preds):]).mean()

    test_df  = feat_df.tail(len(preds)).reset_index(drop=True)
    filtered = apply_filters(test_df, signals)

    engine  = BacktestEngine()
    metrics = engine.run(test_df, filtered, symbol)

    return {
        "symbol":       symbol,
        "timeframe":    timeframe,
        "accuracy":     accuracy,
        "n_bars":       len(feat_df),
        **{k: metrics[k] for k in ["total_return","sharpe_ratio","max_drawdown","win_rate","n_trades"]},
    }


# ─── Comparison Table ────────────────────────────────────────────────────────

def print_comparison(results: list):
    """Print a ranked comparison table across all timeframe x symbol combos."""
    if not results:
        print("No results to display.")
        return

    df = pd.DataFrame(results)

    print("\n" + "="*85)
    print("TIMEFRAME COMPARISON RESULTS")
    print("="*85)
    print(f"{'Symbol':<12} {'TF':<6} {'Accuracy':>9} {'Return':>9} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>9} {'Trades':>7}")
    print("-"*85)

    # Sort by accuracy descending
    df_sorted = df.sort_values("accuracy", ascending=False)
    for _, row in df_sorted.iterrows():
        print(
            f"{row['symbol']:<12} {row['timeframe']:<6} "
            f"{row['accuracy']:>8.1%} "
            f"{row['total_return']:>8.2%} "
            f"{row['sharpe_ratio']:>8.2f} "
            f"{row['max_drawdown']:>7.2%} "
            f"{row['win_rate']:>8.1%} "
            f"{int(row['n_trades']):>7}"
        )

    print("="*85)

    # Per-timeframe summary
    print("\nPER-TIMEFRAME AVERAGES:")
    print(f"{'Timeframe':<10} {'Avg Acc':>9} {'Avg Return':>11} {'Avg Sharpe':>11} {'Profitable':>11}")
    print("-"*50)
    for tf, group in df.groupby("timeframe"):
        profitable = (group["total_return"] > 0).sum()
        print(
            f"{tf:<10} "
            f"{group['accuracy'].mean():>8.1%} "
            f"{group['total_return'].mean():>10.2%} "
            f"{group['sharpe_ratio'].mean():>11.2f} "
            f"  {profitable}/{len(group)}"
        )

    # Best combo
    best = df_sorted.iloc[0]
    print(f"\n→ Best combination: {best['symbol']} on {best['timeframe']} "
          f"(acc={best['accuracy']:.1%}, return={best['total_return']:.2%})")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = Path(RESULTS_DIR) / "timeframe_comparison.csv"
    df_sorted.to_csv(out_path, index=False)
    logger.info(f"Comparison saved -> {out_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-timeframe comparison")
    parser.add_argument("--symbol",     type=str, default=None,
                        help="Run single symbol (e.g. INJ/USD)")
    parser.add_argument("--timeframes", nargs="+", default=["15m","1h","4h","1d"],
                        help="Timeframes to test (e.g. --timeframes 1h 4h 1d)")
    parser.add_argument("--walkforward", action="store_true",
                        help="Use walk-forward validation instead of single backtest")
    parser.add_argument("--refresh",    action="store_true",
                        help="Force re-download all data")
    args = parser.parse_args()

    symbols    = [args.symbol] if args.symbol else TRADING_PAIRS
    timeframes = args.timeframes

    # Validate timeframes
    invalid = [tf for tf in timeframes if tf not in TIMEFRAME_CONFIG]
    if invalid:
        print(f"Invalid timeframes: {invalid}. Choose from: {list(TIMEFRAME_CONFIG.keys())}")
        sys.exit(1)

    logger.info(f"Testing {len(symbols)} symbols x {len(timeframes)} timeframes = "
                f"{len(symbols)*len(timeframes)} combinations")

    # Fetch alt data once
    alt_df = fetch_all_alt_data(force_refresh=args.refresh)

    all_results = []

    for timeframe in timeframes:
        tf_config = TIMEFRAME_CONFIG[timeframe]
        logger.info(f"\n{'='*60}\nTimeframe: {tf_config['description']} ({timeframe})\n{'='*60}")

        for symbol in symbols:
            log_gpu_temp()
            logger.info(f"\nProcessing: {symbol} [{timeframe}]")

            raw_df = fetch_ohlcv_timeframe(
                symbol,
                tf_config["ccxt_tf"],
                tf_config["history_days"]
            )

            if raw_df.empty or len(raw_df) < tf_config["min_bars"]:
                logger.warning(f"Skipping {symbol} {timeframe} — insufficient data ({len(raw_df)} bars)")
                continue

            # ── Resume: skip if already completed ──────────────────
            if args.walkforward and not args.refresh:
                cached = load_checkpoint(symbol, timeframe)
                if cached:
                    all_results.append(cached)
                    continue

            try:
                if args.walkforward:
                    from utils.walk_forward import walk_forward_validate, save_wf_results
                    import config.settings as s
                    orig = s.PREDICTION_HORIZON
                    s.PREDICTION_HORIZON = tf_config["label_horizon"]
                    enriched = merge_alt_data(raw_df, alt_df)
                    wf = walk_forward_validate(enriched, f"{symbol}_{timeframe}")
                    s.PREDICTION_HORIZON = orig
                    if not wf.empty:
                        save_wf_results(wf, f"{symbol}_{timeframe}")
                        all_results.append({
                            "symbol":       symbol,
                            "timeframe":    timeframe,
                            "accuracy":     wf["accuracy"].median(),
                            "total_return": wf["total_return"].median(),
                            "sharpe_ratio": wf["sharpe_ratio"].median(),
                            "max_drawdown": wf["max_drawdown"].min(),
                            "win_rate":     wf["win_rate"].median(),
                            "n_trades":     wf["n_trades"].median(),
                            "n_bars":       len(enriched),
                        })
                else:
                    result = run_pipeline_for_timeframe(
                        symbol, raw_df,
                        timeframe, tf_config["label_horizon"],
                        alt_df
                    )
                    if result:
                        all_results.append(result)

            except Exception as e:
                logger.error(f"Failed {symbol} {timeframe}: {e}")
                import traceback; traceback.print_exc()

    print_comparison(all_results)


if __name__ == "__main__":
    main()
