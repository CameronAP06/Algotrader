"""
kraken_history.py
─────────────────
Loads full Kraken OHLCVT history from the official downloadable CSV files,
then tops up to the present using the live API (max 720 bars).

Kraken CSV filename format:  {PAIR}_{MINUTES}.csv
  e.g.  XBTUSD_240.csv   → BTC/USD 4h
        ETHUSD_1440.csv  → ETH/USD 1d
        DOGEUSD_60.csv   → DOGE/USD 1h

CSV columns (no header): timestamp, open, high, low, close, volume, trades

Supported timeframes and their minute values:
  1h  → 60
  4h  → 240
  1d  → 1440
  8h  → resample from 4h (240)
  1w  → resample from 1d (1440)
  2w  → resample from 1d (1440)

Usage:
    from kraken_history import fetch_ohlcv_full

    df = fetch_ohlcv_full("DOGE/USD", "4h", history_dir=r"C:\\KrakenData")
    df = fetch_ohlcv_full("BTC/USD",  "1w", history_dir=r"C:\\KrakenData")
"""

import os
import time
import glob
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from loguru import logger


# ── Symbol name mapping (ccxt → Kraken CSV filename base) ────────────────────

# Kraken uses legacy pair names in their CSV exports
# e.g. BTC/USD → XBTUSD, ETH/USD → ETHUSD
# Maps ccxt-style symbol → Kraken CSV filename base
# Most pairs follow {BASE}USD pattern; a few use legacy names
SYMBOL_TO_KRAKEN = {
    # Legacy Kraken names
    "BTC/USD":  "XBTUSD",
    "ETH/USD":  "ETHUSD",
    "XRP/USD":  "XRPUSD",
    "LTC/USD":  "LTCUSD",
    "XMR/USD":  "XMRUSD",
    "ZEC/USD":  "ZECUSD",
    "ETC/USD":  "ETCUSD",
    "DOGE/USD": "XDGUSD",
    "DASH/USD": "DASHUSD",
    # Standard {BASE}USD names — auto-resolved by fallback logic below
    # Listed here explicitly for clarity
    "ADA/USD":  "ADAUSD",
    "SOL/USD":  "SOLUSD",
    "DOT/USD":  "DOTUSD",
    "AVAX/USD": "AVAXUSD",
    "LINK/USD": "LINKUSD",
    "POL/USD":"POLUSD",
    "ATOM/USD": "ATOMUSD",
    "UNI/USD":  "UNIUSD",
    "AAVE/USD": "AAVEUSD",
    "INJ/USD":  "INJUSD",
    "ARB/USD":  "ARBUSD",
    "OP/USD":   "OPUSD",
    "ALGO/USD": "ALGOUSD",
    "BCH/USD":  "BCHUSD",
    "FIL/USD":  "FILUSD",
    "GRT/USD":  "GRTUSD",
    "NEAR/USD": "NEARUSD",
    "RUNE/USD": "RUNEUSD",
    "S/USD":  "SUSD",
    "XLM/USD":  "XLMUSD",
    "XTZ/USD":  "XTZUSD",
    "EOS/USD":  "EOSUSD",
    "MKR/USD":  "MKRUSD",
    "COMP/USD": "COMPUSD",
    "SNX/USD":  "SNXUSD",
    "CRV/USD":  "CRVUSD",
    "EUR/USD":  "EURUSD",
    "GBP/USD":  "GBPUSD",
    "AUD/USD":  "AUDUSD",
    # VET not available in Kraken CSV
}

# Timeframe → minute interval in CSV filename
TF_TO_MINUTES = {
    "15m": 15,
    "30m": 30,
    "1h":  60,
    "4h":  240,
    "12h": 720,
    "1d":  1440,
    "1w":  1440,  # load 1d, then resample
    "2w":  1440,  # load 1d, then resample
}

NEEDS_RESAMPLE = {"8h", "1w", "2w"}

# Minimum bars required per timeframe for auto-discovery
# Filters out newly listed pairs with insufficient history
MIN_BARS_FOR_DISCOVERY = {
    "15m": 8000,   # ~83 days
    "30m": 5000,   # ~104 days
    "1h":  4000,   # ~167 days
    "4h":  2000,   # ~333 days
    "12h": 700,    # ~350 days
    "1d":  365,    # 1 year
}

# Known legacy Kraken CSV name → ccxt-style symbol
# Used to reverse-map filenames during auto-discovery
KRAKEN_TO_SYMBOL = {v: k for k, v in SYMBOL_TO_KRAKEN.items()}

# Additional legacy prefixes Kraken uses (X prefix for some metals/crypto)
_KRAKEN_LEGACY_PREFIX = {
    "XBT": "BTC",
    "XDG": "DOGE",
    "XET": "ETH",   # rare
    "XLT": "LTC",   # rare
}

# ── History caps (max days of data to use per timeframe) ─────────────────────
# Older data trains across too many market regimes, hurting generalisation.
# Caps are intentionally tighter for shorter timeframes where regime drift
# is faster, and looser for weekly/daily which need more bars to be useful.
HISTORY_CAPS = {
    "15m": 180,    # 6 months  — intraday patterns change fast
    "30m": 180,    # 6 months
    "1h":  365,    # 1 year    — one full market cycle
    "4h":  730,    # 2 years   — multiple cycles, avoids 2017-era noise
    "12h": 730,    # 2 years
    "1d":  1095,   # 3 years   — needs more bars to be statistically meaningful
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_csv(history_dir: str, kraken_name: str, minutes: int) -> str | None:
    """
    Locate the CSV file in the flat OHLC_Kraken directory.
    Files are named {PAIR}_{MINUTES}.csv e.g. XBTUSD_240.csv
    """
    candidates = [
        f"{kraken_name}_{minutes}.csv",
        f"{kraken_name.upper()}_{minutes}.csv",
    ]
    for fname in candidates:
        path = os.path.join(history_dir, fname)
        if os.path.exists(path):
            return path
    return None


def _load_kraken_csv(path: str) -> pd.DataFrame:
    """Load a Kraken OHLCVT CSV into a standard OHLCV DataFrame."""
    df = pd.read_csv(
        path,
        header=None,
        names=["timestamp", "open", "high", "low", "close", "volume", "trades"],
        dtype={
            "timestamp": np.int64,
            "open": np.float64, "high": np.float64,
            "low": np.float64,  "close": np.float64,
            "volume": np.float64, "trades": np.int64,
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_localize(None)
    df = df.drop(columns=["trades"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def _topup_from_api(symbol: str, timeframe: str, after: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch the most recent bars from Kraken API to bridge the gap between
    the end of the historical CSV and today.
    Only fetches if gap > 1 bar worth of time.
    """
    import ccxt

    if timeframe in ("1w", "2w"):
        logger.debug(f"  Skipping API top-up for {timeframe} — CSV history is sufficient")
        return pd.DataFrame()

    tf_hours = {"15m": 0.25, "30m": 0.5, "1h": 1, "4h": 4, "12h": 12, "1d": 24}
    gap_hours = (datetime.utcnow() - after.to_pydatetime()).total_seconds() / 3600
    bar_hours = tf_hours.get(timeframe, 1)

    if gap_hours < bar_hours * 2:
        logger.info(f"  Top-up not needed — gap is only {gap_hours:.1f}h")
        return pd.DataFrame()

    logger.info(f"  Topping up {symbol} {timeframe} from {after.date()} via API...")
    exchange = ccxt.kraken({"enableRateLimit": True, "rateLimit": 3000})
    since_ms = int(after.timestamp() * 1000)

    all_ohlcv = []
    for attempt in range(3):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=720)
            all_ohlcv = ohlcv
            break
        except Exception as e:
            logger.warning(f"  API top-up attempt {attempt+1} failed: {e}")
            time.sleep(3)

    if not all_ohlcv:
        logger.warning(f"  API top-up returned no data for {symbol}")
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    # Only keep bars after our cutoff
    df = df[df["timestamp"] > after].reset_index(drop=True)
    logger.info(f"  Top-up: {len(df)} new bars")
    return df


def _resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample a lower timeframe DataFrame to a higher one."""
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    if timeframe == "8h":
        # Group 4h → 8h: every 2 bars
        ts = df["timestamp"]
        epoch = pd.Timestamp("1970-01-01")
        hours = (ts - epoch).dt.total_seconds() / 3600
        df["_bucket"] = (hours // 8).astype(int)

    elif timeframe == "12h":
        # Group 4h → 12h: every 3 bars; align to 00:00 / 12:00 UTC boundaries
        ts = df["timestamp"]
        epoch = pd.Timestamp("1970-01-01")
        hours = (ts - epoch).dt.total_seconds() / 3600
        df["_bucket"] = (hours // 12).astype(int)

    elif timeframe == "1w":
        # Align to Monday boundaries
        df["_bucket"] = df["timestamp"].dt.to_period("W").apply(lambda p: p.start_time)
        df["_bucket"] = df["_bucket"].astype(np.int64) // 10**9 // (7 * 24 * 3600)

    elif timeframe == "2w":
        ts = df["timestamp"]
        epoch = pd.Timestamp("1970-01-01")
        days = (ts - epoch).dt.total_seconds() / 86400
        df["_bucket"] = (days // 14).astype(int)

    agg = df.groupby("_bucket").agg(
        timestamp=("timestamp", "first"),
        open=("open",    "first"),
        high=("high",    "max"),
        low=("low",      "min"),
        close=("close",  "last"),
        volume=("volume","sum"),
    ).reset_index(drop=True)

    return agg


# ── Main public function ──────────────────────────────────────────────────────

def fetch_ohlcv_full(
    symbol: str,
    timeframe: str,
    history_dir: str,
    cache_dir: str = "data/raw",
) -> pd.DataFrame:
    """
    Load full OHLCV history for a symbol/timeframe.

    Strategy:
      1. Check local cache (data/raw/) — use if fresh enough
      2. Load Kraken historical CSV from history_dir
      3. Top up with live API (max 720 bars) to get recent data
      4. Resample if needed (8h, 1w, 2w)
      5. Save combined result to cache

    Args:
        symbol:      ccxt-style symbol e.g. "DOGE/USD"
        timeframe:   "1h", "4h", "8h", "1d", "1w", "2w"
        history_dir: Path to folder containing Kraken CSV files
        cache_dir:   Where to save/load cached results

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    # 1. Check cache
    safe_sym  = symbol.replace("/", "_")
    cache_path = Path(cache_dir) / f"{safe_sym}_{timeframe}.csv"

    if cache_path.exists():
        df_cached = pd.read_csv(cache_path, parse_dates=["timestamp"])
        latest    = df_cached["timestamp"].max()
        max_age   = timedelta(hours=4 if timeframe in ("1h", "4h", "8h") else 24)
        if pd.Timestamp.utcnow().tz_localize(None) - latest.replace(tzinfo=None) < max_age:
            logger.info(f"Loaded {symbol} {timeframe} from cache ({len(df_cached)} rows)")
            return df_cached

    # 2. Find and load historical CSV
    # Try explicit mapping first, then auto-derive as {BASE}USD
    base = symbol.split("/")[0]
    kraken_name = SYMBOL_TO_KRAKEN.get(symbol, f"{base}USD")

    minutes = TF_TO_MINUTES[timeframe]
    csv_path = _find_csv(history_dir, kraken_name, minutes)

    if csv_path is None:
        logger.warning(f"No CSV found for {kraken_name}_{minutes} in {history_dir} — falling back to API only")
        return _api_only_fallback(symbol, timeframe)

    logger.info(f"Loading {symbol} {timeframe} from {csv_path}...")
    df_hist = _load_kraken_csv(csv_path)
    logger.info(f"  Loaded {len(df_hist)} bars ({df_hist['timestamp'].iloc[0].date()} → {df_hist['timestamp'].iloc[-1].date()})")

    # Apply history cap — slice to most recent N days to avoid training across
    # too many market regimes (older data hurts generalisation on recent prices)
    cap_days = HISTORY_CAPS.get(timeframe)
    if cap_days:
        cutoff = df_hist["timestamp"].max() - pd.Timedelta(days=cap_days)
        before = len(df_hist)
        df_hist = df_hist[df_hist["timestamp"] >= cutoff].reset_index(drop=True)
        logger.info(f"  History cap {cap_days}d: {before} → {len(df_hist)} bars (from {df_hist['timestamp'].iloc[0].date()})")

    # 3. Top up with API to get recent bars
    last_ts  = df_hist["timestamp"].max()
    df_topup = _topup_from_api(symbol, timeframe if timeframe not in NEEDS_RESAMPLE else "1d", last_ts)

    if not df_topup.empty:
        df_hist = pd.concat([df_hist, df_topup], ignore_index=True)
        df_hist = df_hist.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    # 4. Resample if needed
    if timeframe in NEEDS_RESAMPLE:
        logger.info(f"  Resampling to {timeframe}...")
        df_hist = _resample(df_hist, timeframe)

    logger.success(f"{symbol} {timeframe}: {len(df_hist)} bars total ({df_hist['timestamp'].iloc[0].date()} → {df_hist['timestamp'].iloc[-1].date()})")

    # 5. Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    df_hist.to_csv(cache_path, index=False)

    return df_hist


def _api_only_fallback(symbol: str, timeframe: str) -> pd.DataFrame:
    """Last resort — just use the API (720 bar limit applies)."""
    import ccxt
    # 12h is not accepted by Kraken's ccxt API — resample from 4h
    if timeframe == "12h":
        logger.warning(f"API-only fetch for {symbol} 12h: resampling from 4h (Kraken API doesn't support 12h)")
        df_4h = _api_only_fallback(symbol, "4h")
        if df_4h.empty:
            return pd.DataFrame()
        return _resample(df_4h, "12h")

    api_tf_map = {"8h": "4h", "1w": "1d", "2w": "1d"}
    api_tf = api_tf_map.get(timeframe, timeframe)

    logger.warning(f"API-only fetch for {symbol} {api_tf} (max 720 bars)")
    exchange = ccxt.kraken({"enableRateLimit": True, "rateLimit": 3000})
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, api_tf, limit=720)
    except Exception as e:
        logger.error(f"API fallback failed: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    if timeframe in NEEDS_RESAMPLE:
        df = _resample(df, timeframe)

    return df


# ── Auto-discovery ────────────────────────────────────────────────────────────

def auto_discover_pairs(
    history_dir: str,
    timeframe: str = "4h",
    quote: str = "USD",
    min_bars: int | None = None,
    exclude: set | None = None,
) -> list[str]:
    """
    Scan the Kraken CSV directory and return all tradeable pairs that have
    enough history for the given timeframe.

    Args:
        history_dir:  Path to OHLC_Kraken directory
        timeframe:    Timeframe to check bar count against (default: "4h")
        quote:        Quote currency filter — only return {BASE}/{quote} pairs
        min_bars:     Minimum bar count required. Defaults to MIN_BARS_FOR_DISCOVERY[timeframe]
        exclude:      Set of ccxt symbols to exclude e.g. {"BTC/USD"}

    Returns:
        Sorted list of ccxt-style symbols e.g. ["AAVE/USD", "ADA/USD", ...]

    Example:
        pairs = auto_discover_pairs("data/OHLC_Kraken", timeframe="4h")
        # Returns all USD pairs with 2000+ 4h bars
    """
    if min_bars is None:
        min_bars = MIN_BARS_FOR_DISCOVERY.get(timeframe, 2000)

    minutes = TF_TO_MINUTES.get(timeframe)
    if minutes is None:
        raise ValueError(f"Unknown timeframe: {timeframe}")

    # For resampled timeframes, find the source timeframe's CSV
    if timeframe in NEEDS_RESAMPLE:
        src_minutes = TF_TO_MINUTES[timeframe]
    else:
        src_minutes = minutes

    quote_suffix = quote  # e.g. "USD"
    pattern = os.path.join(history_dir, f"*{quote_suffix}_{src_minutes}.csv")
    csv_files = glob.glob(pattern)

    # Also search recursively in subdirectories
    if not csv_files:
        csv_files = glob.glob(
            os.path.join(history_dir, "**", f"*{quote_suffix}_{src_minutes}.csv"),
            recursive=True
        )

    if not csv_files:
        logger.warning(f"auto_discover_pairs: no *{quote_suffix}_{src_minutes}.csv files found in {history_dir}")
        return []

    exclude = exclude or set()
    discovered = []
    skipped_short = 0
    skipped_exclude = 0

    for csv_path in sorted(csv_files):
        fname = os.path.basename(csv_path)
        # Extract Kraken name: e.g. "XBTUSD_240.csv" → "XBTUSD"
        kraken_name = fname.replace(f"_{src_minutes}.csv", "")

        # Convert to ccxt symbol
        if kraken_name in KRAKEN_TO_SYMBOL:
            # Known legacy mapping
            symbol = KRAKEN_TO_SYMBOL[kraken_name]
        else:
            # Auto-derive: strip quote suffix → base → add "/USD"
            if kraken_name.endswith(quote_suffix):
                base = kraken_name[:-len(quote_suffix)]
                # Handle X-prefixed legacy names
                for legacy, modern in _KRAKEN_LEGACY_PREFIX.items():
                    if base == legacy:
                        base = modern
                        break
                symbol = f"{base}/{quote}"
            else:
                continue  # Not a USD pair

        if symbol in exclude:
            skipped_exclude += 1
            continue

        # Check bar count — count lines in CSV (fast, no full parse)
        try:
            with open(csv_path, "rb") as f:
                n_bars = sum(1 for _ in f)
            if n_bars < min_bars:
                skipped_short += 1
                continue
        except Exception:
            continue

        discovered.append(symbol)

    discovered = sorted(set(discovered))
    logger.info(
        f"auto_discover_pairs [{timeframe} {quote}]: {len(discovered)} pairs found "
        f"(skipped {skipped_short} short history, {skipped_exclude} excluded) "
        f"from {len(csv_files)} CSV files"
    )
    return discovered


def discover_all_timeframes(
    history_dir: str,
    timeframes: list[str] | None = None,
    quote: str = "USD",
    exclude: set | None = None,
) -> dict[str, list[str]]:
    """
    Run auto_discover_pairs for multiple timeframes.
    Returns dict of {timeframe: [symbols]} — useful for seeing which pairs
    have enough history at each timeframe.

    Example:
        available = discover_all_timeframes("data/OHLC_Kraken")
        # available["4h"] → all pairs with 2000+ 4h bars
        # available["1d"] → all pairs with 365+ 1d bars
    """
    if timeframes is None:
        timeframes = ["30m", "1h", "4h", "12h", "1d"]

    result = {}
    for tf in timeframes:
        result[tf] = auto_discover_pairs(
            history_dir, timeframe=tf, quote=quote, exclude=exclude
        )

    # Summary
    logger.info("Discovery summary:")
    for tf, pairs in result.items():
        logger.info(f"  {tf:>4}: {len(pairs):>4} pairs")

    return result



    """
    Returns the Kraken OHLC history directory.
    Structure: trading_bot/data/OHLC_Kraken/{TICKERNAME}/
    Returns the OHLC_Kraken root — _find_csv searches subdirs recursively.
    """
    candidates = [
        os.environ.get("KRAKEN_HISTORY_DIR", ""),
        "data/OHLC_Kraken",                          # relative — standard location
        os.path.join(os.path.dirname(__file__), "data", "OHLC_Kraken"),  # absolute
    ]
    for path in candidates:
        if path and os.path.isdir(path):
            csvs = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
            if csvs:
                logger.info(f"Found Kraken OHLC data at: {path} ({len(csvs)} CSV files)")
                return path
    logger.warning("Kraken OHLC directory not found — falling back to API (720 bar limit)")
    return None


if __name__ == "__main__":
    # Quick test
    history_dir = get_history_dir()
    if not history_dir:
        print("No Kraken history directory found.")
        print("Set KRAKEN_HISTORY_DIR env var or place files in C:\\KrakenData")
    else:
        for sym, tf in [("BTC/USD", "4h"), ("DOGE/USD", "1d"), ("ETH/USD", "1w")]:
            df = fetch_ohlcv_full(sym, tf, history_dir)
            if not df.empty:
                print(f"{sym} {tf}: {len(df)} bars, {df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()}")