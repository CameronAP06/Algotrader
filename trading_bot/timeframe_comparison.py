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
try:
    from data.alt_data import fetch_all_alt_data, merge_alt_data, compute_btc_dom_proxy
    _ALT_DATA_AVAILABLE = True
except ImportError:
    _ALT_DATA_AVAILABLE = False
    # expanded_scan.py imports fetch_ohlcv_timeframe from this module but never
    # calls fetch_all_alt_data / merge_alt_data. Silently degrade so that
    # expanded_scan can import cleanly even when alt_data.py is not present.
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


# ─── Checkpoint / Resume ─────────────────────────────────────────────────────

def checkpoint_path(symbol: str, timeframe: str) -> Path:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return Path(RESULTS_DIR) / f"{symbol.replace('/','_')}_{timeframe}_walkforward.csv"


def load_checkpoint(symbol: str, timeframe: str, edge_override: dict = None):
    """
    Load completed walk-forward results if they exist, to skip retraining.
    If edge_override is provided, invalidates any checkpoint that was NOT
    built with matching edge params (horizon/threshold) — forcing a retrain
    with the correct label configuration.
    """
    path = checkpoint_path(symbol, timeframe)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df.empty or len(df) < 2:
            return None

        # If edge params are specified, check if checkpoint was built with them
        # We store horizon/threshold in the CSV if present; if not, assume stale
        if edge_override:
            if "horizon_bars" in df.columns and "threshold" in df.columns:
                cp_horizon = float(df["horizon_bars"].iloc[0])
                cp_thresh  = float(df["threshold"].iloc[0])
                if (abs(cp_horizon - edge_override["horizon_bars"]) > 0.1 or
                        abs(cp_thresh - edge_override["threshold"]) > 0.0001):
                    logger.info(
                        f"[CHECKPOINT] Invalidating {symbol} {timeframe} -- "
                        f"edge params changed "
                        f"(h={cp_horizon:.0f}→{edge_override['horizon_bars']}, "
                        f"t={cp_thresh:.3f}→{edge_override['threshold']:.3f})"
                    )
                    path.unlink()   # delete stale checkpoint
                    return None
            else:
                # Old checkpoint has no edge metadata — always invalidate
                logger.info(
                    f"[CHECKPOINT] Invalidating {symbol} {timeframe} -- "
                    f"no edge metadata in checkpoint, forcing retrain with edge params"
                )
                path.unlink()
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
        logger.info(
            f"[CHECKPOINT] Skipping {symbol} {timeframe} -- "
            f"already done ({n} folds, acc={acc:.1%}, ret={ret:.2%})"
        )
        return result
    except Exception as e:
        logger.warning(f"Failed to load checkpoint {symbol} {timeframe}: {e}")
        return None


# ─── GPU Monitoring ───────────────────────────────────────────────────────────

def log_gpu_temp():
    """Log GPU temperature and utilisation. Warns at >=75C."""
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
        temp, util, mem_used, mem_total = (
            int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        )
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
    "2h": {
        "ccxt_tf":    "2h",
        "history_days": 1825,
        "label_horizon": 6,       # 6 bars = 12 hours ahead
        "min_bars":   1000,
        "description": "2 hours"
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
    """Fetch OHLCV data at a specific timeframe from Kraken.

    Priority:
      1. Full historical CSVs via kraken_history (25GB dataset — all native timeframes
         including 1h, 4h, 12h, 1d, 30m — NO resampling for these)
      2. Local cache (data/raw/) — used when CSV data is fresh enough
      3. API fallback — 720-bar limit, only used when CSV data genuinely absent

    NOTE: 12h is a native Kraken timeframe (_720.csv files exist). Do NOT resample
    from 4h — always load the native CSV via this function.
    """
    import ccxt
    from datetime import datetime, timedelta

    # Try full history loader first — covers ALL native timeframes.
    # Inject the directory containing this file so kraken_history.py is always
    # findable regardless of the caller's working directory.
    try:
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        if _this_dir not in sys.path:
            sys.path.insert(0, _this_dir)
        from kraken_history import fetch_ohlcv_full, get_history_dir
        history_dir = get_history_dir()
        if history_dir:
            df = fetch_ohlcv_full(symbol, timeframe, history_dir)
            if df is not None and len(df) > 0:
                return df
            else:
                logger.warning(
                    f"{symbol} {timeframe}: kraken_history returned empty — "
                    f"CSV may be missing from {history_dir}. Falling back to API (720 bar limit)."
                )
        else:
            logger.warning(
                f"{symbol} {timeframe}: Kraken history directory not found. "
                f"Set KRAKEN_HISTORY_DIR env var to point to your data directory. "
                f"Falling back to API (720 bar limit — will likely be skipped as too few bars)."
            )
    except ImportError:
        logger.error(
            f"kraken_history.py not found. Expected alongside timeframe_comparison.py. "
            f"Falling back to API for {symbol} {timeframe}."
        )

    kraken_symbol = symbol  # Kraken uses USD pairs natively, no remapping needed
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
    exchange = ccxt.kraken({"enableRateLimit": True, "rateLimit": 3000})

    since_ms  = int((datetime.utcnow() - timedelta(days=history_days)).timestamp() * 1000)
    all_ohlcv = []
    batch     = 720  # Kraken max candles per request

    now_ms   = int(datetime.utcnow().timestamp() * 1000)
    end_ms   = now_ms - 2 * 60 * 60 * 1000   # stop 2h before now
    retries  = 0

    while since_ms < end_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(kraken_symbol, timeframe, since=since_ms, limit=batch)
            retries = 0
        except Exception as e:
            logger.error(f"Fetch error {symbol} {timeframe}: {e}")
            retries += 1
            if retries >= 3:
                break
            time.sleep(5)
            continue

        if not ohlcv:
            break

        all_ohlcv.extend(ohlcv)
        last_ts = ohlcv[-1][0]
        if last_ts >= end_ms:
            break
        # Always advance by the last returned timestamp — never rely on
        # batch size to detect end of history (Kraken returns partial batches)
        since_ms = last_ts + 1
        time.sleep(0.5)

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




def resample_to_8h(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 4h OHLCV candles into 8h candles.
    Kraken doesn't support 8h natively — we synthesise it from 4h data.
    Groups every 2 consecutive 4h bars: open=first, high=max, low=min,
    close=last, volume=sum. Timestamp = start of the 8h window.
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    # Align to 8h boundaries (00:00, 08:00, 16:00 UTC)
    # Each 8h window contains exactly 2 x 4h bars
    ts = df["timestamp"]
    if hasattr(ts.iloc[0], 'tzinfo') and ts.iloc[0].tzinfo is not None:
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
    else:
        epoch = pd.Timestamp("1970-01-01")

    hours_since_epoch = (ts - epoch).dt.total_seconds() / 3600
    df["_8h_bucket"] = (hours_since_epoch // 8).astype(int)

    agg = df.groupby("_8h_bucket").agg(
        timestamp=("timestamp", "first"),
        open=("open",   "first"),
        high=("high",   "max"),
        low=("low",     "min"),
        close=("close", "last"),
        volume=("volume","sum"),
    ).reset_index(drop=True)

    return agg


def resample_to_12h(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 4h OHLCV candles into 12h candles.
    Kraken's ccxt interface doesn't accept '12h' as a timeframe string.
    Groups every 3 consecutive 4h bars: open=first, high=max, low=min,
    close=last, volume=sum. Timestamp = start of the 12h window.
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    # Align to 12h boundaries (00:00, 12:00 UTC)
    ts = df["timestamp"]
    if hasattr(ts.iloc[0], 'tzinfo') and ts.iloc[0].tzinfo is not None:
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
    else:
        epoch = pd.Timestamp("1970-01-01")

    hours_since_epoch = (ts - epoch).dt.total_seconds() / 3600
    df["_12h_bucket"] = (hours_since_epoch // 12).astype(int)

    agg = df.groupby("_12h_bucket").agg(
        timestamp=("timestamp", "first"),
        open=("open",   "first"),
        high=("high",   "max"),
        low=("low",     "min"),
        close=("close", "last"),
        volume=("volume","sum"),
    ).reset_index(drop=True)

    return agg


def fetch_ohlcv_12h(symbol: str, history_days: int) -> pd.DataFrame:
    """Load native 12h OHLCV data from the Kraken CSV dataset.

    12h is a native Kraken timeframe — files are named e.g. XBTUSD_720.csv.
    We load these directly via fetch_ohlcv_timeframe (which calls kraken_history).

    Resampling from 4h is only used as a last resort when the native CSV is
    genuinely absent — this should be rare given the 25GB dataset.
    """
    # Prefer native 12h CSV — fast and accurate
    df = fetch_ohlcv_timeframe(symbol, "12h", history_days)
    if df is not None and len(df) > 10:
        return df

    # Fallback: resample from 4h only if native CSV missing
    logger.warning(
        f"{symbol}: no native 12h CSV found — resampling from 4h as fallback. "
        f"Check that your Kraken dataset includes *_720.csv files."
    )
    df_4h = fetch_ohlcv_timeframe(symbol, "4h", history_days)
    if df_4h is None or len(df_4h) < 2:
        return pd.DataFrame()
    df_12h = resample_to_12h(df_4h)
    logger.info(f"{symbol} 12h (resampled from 4h fallback): {len(df_12h)} candles")
    return df_12h





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
    filtered = apply_filters(test_df, signals, timeframe=timeframe)

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
    parser.add_argument("--timeframes", nargs="+", default=["1h","2h","4h","8h","1d"],
                        help="Timeframes to test (e.g. --timeframes 1h 4h 1d)")
    parser.add_argument("--walkforward", action="store_true",
                        help="Use walk-forward validation instead of single backtest")
    parser.add_argument("--refresh",    action="store_true",
                        help="Force re-download all data")
    parser.add_argument("--use-edges",  type=str, default=None,
                        metavar="PATH",
                        help="Path to best_edges.json from edge_scanner.py — "
                             "overrides horizon/threshold per symbol/timeframe")
    args = parser.parse_args()

    symbols    = [args.symbol] if args.symbol else TRADING_PAIRS
    timeframes = args.timeframes

    # Load discovered edges if provided
    edge_configs = {}
    if args.use_edges:
        import json
        with open(args.use_edges) as f:
            edge_configs = json.load(f)
        logger.info(
            f"Loaded {len(edge_configs)} edge configs from {args.use_edges}: "
            + ", ".join(edge_configs.keys())
        )

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

        # Pre-fetch all symbols for this timeframe so we can compute cross-symbol features
        all_raw: dict = {}
        for sym in symbols:
            df = fetch_ohlcv_timeframe(sym, tf_config["ccxt_tf"], tf_config["history_days"])
            if not df.empty and len(df) >= tf_config["min_bars"]:
                all_raw[sym] = df
            else:
                logger.warning(f"Skipping {sym} {timeframe} — insufficient data ({len(df)} bars)")

        for symbol in symbols:
            if symbol not in all_raw:
                continue

            raw_df = all_raw[symbol]
            logger.info(f"\nProcessing: {symbol} [{timeframe}]")
            log_gpu_temp()

            try:
                if args.walkforward:
                    from utils.walk_forward import walk_forward_validate, save_wf_results
                    import config.settings as s
                    orig = s.PREDICTION_HORIZON
                    s.PREDICTION_HORIZON = tf_config["label_horizon"]
                    enriched = merge_alt_data(raw_df, alt_df)
                    # If btc_dominance was dropped (constant fallback), add proxy from OHLCV
                    if "btc_dominance" not in enriched.columns:
                        proxy = compute_btc_dom_proxy(raw_df, all_raw)
                        if not proxy.empty:
                            enriched = enriched.copy()
                            enriched["btc_dom_proxy"] = proxy.values
                            enriched["btc_dom_proxy_change"] = enriched["btc_dom_proxy"].diff(24)
                            logger.info(f"Added BTC dominance proxy for {symbol} [{timeframe}]")

                    # Apply discovered edge params if available for this symbol/timeframe
                    edge_key = f"{symbol}_{timeframe}"
                    edge_override = edge_configs.get(edge_key)
                    if edge_override:
                        import data.feature_engineer as fe
                        _orig_get_label_params = fe.get_label_params
                        def _edge_label_params(tf, _h=edge_override["horizon_bars"],
                                               _t=edge_override["threshold"],
                                               _orig=_orig_get_label_params):
                            return _h, _t
                        fe.get_label_params = _edge_label_params
                        logger.info(
                            f"Using discovered edge params for {edge_key}: "
                            f"horizon={edge_override['horizon_hours']}h "
                            f"({edge_override['horizon_bars']} bars), "
                            f"threshold={edge_override['threshold']:.1%}"
                        )

                    wf = walk_forward_validate(enriched, f"{symbol}_{timeframe}", timeframe=timeframe)

                    # Restore original label params
                    if edge_override:
                        fe.get_label_params = _orig_get_label_params
                    s.PREDICTION_HORIZON = orig
                    if not wf.empty:
                        # Tag the results with edge params so checkpoint validation works
                        if edge_override:
                            wf["horizon_bars"] = edge_override["horizon_bars"]
                            wf["threshold"]    = edge_override["threshold"]
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