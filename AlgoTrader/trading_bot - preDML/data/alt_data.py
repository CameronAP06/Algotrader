"""
data/alt_data.py
Fetches alternative/sentiment data that gives models genuinely new information
beyond OHLCV. All sources are free, no API key required.

Sources:
  1. Fear & Greed Index (alternative.me) — daily sentiment 0-100
  2. BTC Dominance (CoinGecko) — cross-asset signal
  3. Google Trends (pytrends) — retail search interest for crypto terms
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
from config.settings import DATA_DIR

ALT_DATA_PATH = Path(DATA_DIR) / "alt_data.csv"


# ─── Fear & Greed Index ──────────────────────────────────────────────────────

def fetch_fear_greed(days: int = 1825) -> pd.DataFrame:
    """
    Fetch Bitcoin Fear & Greed Index from alternative.me.
    Returns daily values 0 (extreme fear) to 100 (extreme greed).
    This is a genuine leading/coincident indicator — extreme fear often
    precedes recoveries, extreme greed often precedes corrections.
    """
    try:
        resp = requests.get(
            f"https://api.alternative.me/fng/?limit={days}&format=json",
            timeout=10
        )
        data = resp.json()["data"]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
        df["fear_greed"] = df["value"].astype(float)
        df = df[["timestamp", "fear_greed"]].sort_values("timestamp").reset_index(drop=True)
        logger.success(f"Fear & Greed: {len(df)} records ({df['timestamp'].iloc[0].date()} -> {df['timestamp'].iloc[-1].date()})")
        return df
    except Exception as e:
        logger.warning(f"Fear & Greed fetch failed: {e}")
        return pd.DataFrame()


# ─── BTC Dominance ───────────────────────────────────────────────────────────

def fetch_btc_dominance(days: int = 1825) -> pd.DataFrame:
    """
    Fetch BTC market dominance from CoinGecko.
    When BTC dominance rises, altcoins tend to underperform and vice versa.
    This is a useful cross-asset signal for ETH/SOL/XRP.
    """
    # CoinGecko changed their response structure — the chart data is now
    # nested under a "data" key. We try two endpoints with fallback.
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/global/market_cap_chart",
            params={"days": days},
            timeout=15,
            headers={"Accept": "application/json"}
        )
        raw = resp.json()

        # API v3 wraps response under "data" key (changed ~2024)
        data = raw.get("data", raw)

        # The field is market_cap_percentage at this level
        btc_pct = data.get("market_cap_percentage", {}).get("btc", [])

        # Older format had it as a flat list of [ts_ms, value] pairs
        # Newer format may have it as a dict {btc: [[ts, val], ...]}
        if not btc_pct:
            raise ValueError("Empty BTC dominance series from chart endpoint")

        df = pd.DataFrame(btc_pct, columns=["ts_ms", "btc_dominance"])
        df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        df = df[["timestamp", "btc_dominance"]].sort_values("timestamp").reset_index(drop=True)
        logger.success(f"BTC Dominance: {len(df)} records")
        return df

    except Exception as e:
        logger.warning(f"BTC dominance chart endpoint failed: {e} — trying /global fallback")

    # Fallback: /global gives current dominance only — synthesise a flat series
    # This is less useful (no history) but better than nothing
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/global",
            timeout=10,
            headers={"Accept": "application/json"}
        )
        raw  = resp.json()
        dom  = raw.get("data", raw).get("market_cap_percentage", {}).get("btc")
        if dom is None:
            raise ValueError("No BTC dominance in /global response")

        # Build a flat series — models will see a constant but won't error
        end   = pd.Timestamp.utcnow()
        start = end - pd.Timedelta(days=days)
        idx   = pd.date_range(start=start, end=end, freq="D", tz="UTC")
        df    = pd.DataFrame({"timestamp": idx, "btc_dominance": float(dom)})
        logger.warning(
            f"BTC Dominance: using current-only value ({dom:.1f}%) "
            f"— historical chart unavailable. Feature will be constant."
        )
        return df

    except Exception as e2:
        logger.warning(f"BTC dominance fetch failed entirely: {e2} — skipping feature")
        return pd.DataFrame()


# ─── Google Trends ───────────────────────────────────────────────────────────

def fetch_google_trends(days: int = 1825) -> pd.DataFrame:
    """
    Fetch Google Trends for 'bitcoin' search interest.
    High search interest often precedes or accompanies price peaks.
    Requires pytrends: pip install pytrends
    """
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl="en-US", tz=0)

        # Split into chunks — Google Trends only gives weekly data for >3 months
        end   = datetime.utcnow()
        start = end - timedelta(days=days)

        timeframe = f"{start.strftime('%Y-%m-%d')} {end.strftime('%Y-%m-%d')}"
        pytrends.build_payload(["bitcoin"], cat=0, timeframe=timeframe, geo="", gprop="")
        df = pytrends.interest_over_time()

        if df.empty:
            raise ValueError("Empty trends response")

        df = df.reset_index()[["date", "bitcoin"]]
        df.columns = ["timestamp", "google_trends_btc"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["google_trends_btc"] = df["google_trends_btc"].astype(float)
        logger.success(f"Google Trends: {len(df)} records")
        return df
    except ImportError:
        logger.warning("pytrends not installed — skipping Google Trends")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"Google Trends fetch failed: {e} — skipping")
        return pd.DataFrame()


# ─── Combine & Cache ─────────────────────────────────────────────────────────

def fetch_all_alt_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch and merge all alternative data sources into a single daily DataFrame.
    Caches to CSV — refreshes if cache is older than 24 hours or forced.
    """
    if ALT_DATA_PATH.exists() and not force_refresh:
        df = pd.read_csv(ALT_DATA_PATH, parse_dates=["timestamp"])
        # Check freshness
        latest = pd.to_datetime(df["timestamp"]).max()
        if latest.tz_localize(None) > datetime.utcnow() - timedelta(hours=24):
            logger.info(f"Loaded cached alt data ({len(df)} rows)")
            return df
        logger.info("Alt data cache stale — refreshing")

    logger.info("Fetching alternative data...")

    # Fear & Greed (most important — always fetch)
    fg_df = fetch_fear_greed(days=1825)

    # BTC Dominance
    dom_df = fetch_btc_dominance(days=1825)
    time.sleep(1)  # CoinGecko rate limit

    # Google Trends (optional)
    gt_df = fetch_google_trends(days=1825)

    if fg_df.empty:
        logger.warning("No alt data available — returning empty DataFrame")
        return pd.DataFrame()

    # Start with Fear & Greed as base (daily)
    # Normalize timestamps to date only for merging
    fg_df["date"] = fg_df["timestamp"].dt.date

    merged = fg_df.copy()

    if not dom_df.empty:
        dom_df["date"] = dom_df["timestamp"].dt.date
        dom_daily = dom_df.groupby("date")["btc_dominance"].mean().reset_index()
        merged = merged.merge(dom_daily, on="date", how="left")

    if not gt_df.empty:
        gt_df["date"] = gt_df["timestamp"].dt.date
        merged = merged.merge(gt_df[["date", "google_trends_btc"]], on="date", how="left")

    # Forward fill any gaps
    merged = merged.sort_values("timestamp").ffill()

    os.makedirs(DATA_DIR, exist_ok=True)
    merged.to_csv(ALT_DATA_PATH, index=False)
    logger.success(f"Alt data saved: {len(merged)} rows, columns: {list(merged.columns)}")
    return merged


# ─── BTC Dominance Proxy (from OHLCV) ────────────────────────────────────────

def compute_btc_dom_proxy(ohlcv_df: pd.DataFrame, all_ohlcv: dict) -> pd.Series:
    """
    Compute a BTC dominance proxy from OHLCV data when CoinGecko is unavailable.

    Method: rolling 30-day return of BTC minus rolling 30-day average return
    of all other symbols. Positive = BTC outperforming alts (dominance rising).

    Args:
        ohlcv_df  : The symbol's own OHLCV DataFrame (used for timestamp alignment)
        all_ohlcv : Dict of {symbol: df} for all symbols being traded

    Returns:
        Series of btc_dom_proxy aligned to ohlcv_df's timestamp index
    """
    if "BTC/USD" not in all_ohlcv and "BTC/USDT" not in all_ohlcv:
        return pd.Series(dtype=float)

    btc_key = "BTC/USD" if "BTC/USD" in all_ohlcv else "BTC/USDT"
    btc_df  = all_ohlcv[btc_key].set_index("timestamp")["close"]
    btc_ret = btc_df.pct_change(24 * 30).rename("btc_30d_ret")  # 30d rolling return (hourly)

    alt_rets = []
    for sym, df in all_ohlcv.items():
        if sym in ("BTC/USD", "BTC/USDT"):
            continue
        r = df.set_index("timestamp")["close"].pct_change(24 * 30)
        alt_rets.append(r)

    if not alt_rets:
        return pd.Series(dtype=float)

    alt_avg = pd.concat(alt_rets, axis=1).mean(axis=1)
    proxy   = (btc_ret - alt_avg).rename("btc_dom_proxy")

    # Align to ohlcv_df's timestamps
    target_idx = pd.to_datetime(ohlcv_df["timestamp"])
    proxy = proxy.reindex(target_idx, method="nearest", tolerance="1h")
    return proxy


# ─── Merge with OHLCV ────────────────────────────────────────────────────────

def merge_alt_data(ohlcv_df: pd.DataFrame, alt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge daily alt data into hourly OHLCV DataFrame via forward-fill.
    Also adds derived features from the alt data.
    """
    if alt_df.empty:
        logger.warning("No alt data to merge — continuing without it")
        return ohlcv_df

    # Normalize timestamps
    ohlcv = ohlcv_df.copy()
    if ohlcv["timestamp"].dt.tz is not None:
        ohlcv["timestamp"] = ohlcv["timestamp"].dt.tz_localize(None)

    alt = alt_df.copy()
    alt["timestamp"] = pd.to_datetime(alt["timestamp"]).dt.tz_localize(None)

    # Merge on nearest past timestamp (forward-fill daily -> hourly)
    merged = pd.merge_asof(
        ohlcv.sort_values("timestamp"),
        alt[["timestamp", "fear_greed"] +
            (["btc_dominance"] if "btc_dominance" in alt.columns else []) +
            (["google_trends_btc"] if "google_trends_btc" in alt.columns else [])
        ].sort_values("timestamp"),
        on="timestamp",
        direction="backward"
    )

    # Derived features
    if "fear_greed" in merged.columns:
        merged["fear_greed"] = merged["fear_greed"].fillna(50.0)  # neutral default
        # Rolling stats
        merged["fear_greed_7d"]  = merged["fear_greed"].rolling(168).mean()  # 7d avg (hourly)
        merged["fear_greed_momentum"] = merged["fear_greed"].diff(24)        # 24h change
        # Extreme zones — mean reversion signals
        merged["fear_greed_extreme_fear"]  = (merged["fear_greed"] < 25).astype(float)
        merged["fear_greed_extreme_greed"] = (merged["fear_greed"] > 75).astype(float)

    if "btc_dominance" in merged.columns:
        merged["btc_dominance"] = merged["btc_dominance"].ffill().fillna(50.0)
        # If the series is constant (fallback fired, no real history),
        # drop it — a constant feature adds noise, not signal
        if merged["btc_dominance"].nunique() <= 1:
            logger.warning(
                "btc_dominance is constant (API fallback) — dropping feature "
                "to avoid feeding noise to the model. Will use BTC proxy instead."
            )
            merged = merged.drop(columns=["btc_dominance"])
        else:
            merged["btc_dom_change_24h"] = merged["btc_dominance"].diff(24)

    if "google_trends_btc" in merged.columns:
        merged["google_trends_btc"] = merged["google_trends_btc"].ffill().fillna(50.0)
        merged["trends_momentum"] = merged["google_trends_btc"].diff(168)  # weekly change

    n_cols = len([c for c in merged.columns if c not in ohlcv_df.columns])
    logger.info(f"Alt data merged: added {n_cols} new columns")
    return merged
