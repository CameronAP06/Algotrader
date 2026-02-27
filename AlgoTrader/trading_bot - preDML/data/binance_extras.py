"""
data/binance_extras.py
Fetches funding rates and order book imbalance from Binance.
These are genuine leading indicators unavailable from OHLCV data alone:
  - Funding rates: futures market sentiment, mean-reverts when extreme
  - Order book imbalance: immediate buy/sell pressure before it hits price
No API key required.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
from config.settings import DATA_DIR, TIMEFRAME

BINANCE_FUTURES_URL = "https://fapi.binance.com"
BINANCE_SPOT_URL    = "https://api.binance.com"

SYMBOL_MAP = {
    "BTC/USD": "BTCUSDT",
    "ETH/USD": "ETHUSDT",
    "SOL/USD": "SOLUSDT",
    "XRP/USD": "XRPUSDT",
}


# ─── Funding Rates ───────────────────────────────────────────────────────────

def fetch_funding_rates(symbol: str, days: int = 1825) -> pd.DataFrame:
    """
    Fetch historical funding rates from Binance perpetual futures.
    Funding is paid every 8 hours — we'll resample to hourly for alignment.
    Positive funding = longs paying shorts (market overbought)
    Negative funding = shorts paying longs (market oversold)
    """
    binance_symbol = SYMBOL_MAP.get(symbol, symbol.replace("/", ""))
    since_ms = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    all_rates = []

    logger.info(f"Fetching funding rates for {symbol}...")

    while True:
        try:
            resp = requests.get(
                f"{BINANCE_FUTURES_URL}/fapi/v1/fundingRate",
                params={"symbol": binance_symbol, "startTime": since_ms, "limit": 1000},
                timeout=10
            )
            data = resp.json()
        except Exception as e:
            logger.error(f"Funding rate fetch error: {e}")
            break

        if not data or isinstance(data, dict):
            break

        all_rates.extend(data)
        if len(data) < 1000:
            break
        since_ms = int(data[-1]["fundingTime"]) + 1
        time.sleep(0.1)

    if not all_rates:
        logger.warning(f"No funding rate data for {symbol} — will use zeros")
        return pd.DataFrame()

    df = pd.DataFrame(all_rates)
    df["timestamp"] = pd.to_datetime(df["fundingTime"].astype(int), unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    df = df[["timestamp", "funding_rate"]].sort_values("timestamp").reset_index(drop=True)
    logger.success(f"{symbol}: {len(df)} funding rate records")
    return df


def fetch_order_book_snapshot(symbol: str, depth: int = 10) -> dict:
    """
    Fetch current order book snapshot.
    Returns bid/ask imbalance ratio — useful for live trading signal confirmation.
    """
    binance_symbol = SYMBOL_MAP.get(symbol, symbol.replace("/", ""))
    try:
        resp = requests.get(
            f"{BINANCE_SPOT_URL}/api/v3/depth",
            params={"symbol": binance_symbol, "limit": depth},
            timeout=5
        )
        book = resp.json()
        bid_vol = sum(float(b[1]) for b in book["bids"])
        ask_vol = sum(float(a[1]) for a in book["asks"])
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
        return {"bid_vol": bid_vol, "ask_vol": ask_vol, "imbalance": imbalance}
    except Exception as e:
        logger.warning(f"Order book fetch error: {e}")
        return {"bid_vol": 0, "ask_vol": 0, "imbalance": 0}


# ─── Merge with OHLCV ────────────────────────────────────────────────────────

def merge_funding_rates(ohlcv_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Merge funding rates into hourly OHLCV DataFrame.
    Since funding is every 8h, we forward-fill to hourly and add derived features.
    """
    funding_df = fetch_funding_rates(symbol, days=1825)

    if funding_df.empty:
        # Fill with zeros if no data available
        ohlcv_df["funding_rate"]     = 0.0
        ohlcv_df["funding_rate_8h"]  = 0.0
        ohlcv_df["funding_cumsum"]   = 0.0
        ohlcv_df["funding_extreme"]  = 0.0
        return ohlcv_df

    # Ensure timestamps are tz-aware for merging
    if ohlcv_df["timestamp"].dt.tz is None:
        ohlcv_df["timestamp"] = ohlcv_df["timestamp"].dt.tz_localize("UTC")

    # Forward-fill funding rate to every hour
    merged = pd.merge_asof(
        ohlcv_df.sort_values("timestamp"),
        funding_df.sort_values("timestamp"),
        on="timestamp",
        direction="backward"
    )
    merged["funding_rate"] = merged["funding_rate"].fillna(0.0)

    # Derived funding features
    merged["funding_rate_8h"] = merged["funding_rate"].rolling(8).mean()   # 8h avg
    merged["funding_cumsum"]  = merged["funding_rate"].rolling(24).sum()   # 24h cumulative
    # Extreme funding = potential mean reversion signal
    funding_std = merged["funding_rate"].rolling(168).std()  # 1-week std
    merged["funding_extreme"] = (merged["funding_rate"] / (funding_std + 1e-9)).clip(-3, 3)

    logger.info(f"{symbol}: Funding rates merged — avg={merged['funding_rate'].mean():.6f}")
    return merged


def save_extras(df: pd.DataFrame, symbol: str):
    os.makedirs(DATA_DIR, exist_ok=True)
    safe_name = symbol.replace("/", "_")
    path = Path(DATA_DIR) / f"{safe_name}_extras.csv"
    df[["timestamp", "funding_rate", "funding_rate_8h",
        "funding_cumsum", "funding_extreme"]].to_csv(path, index=False)
    logger.info(f"Saved extras -> {path}")


def load_or_fetch_extras(ohlcv_df: pd.DataFrame, symbol: str,
                         force_refresh: bool = False) -> pd.DataFrame:
    """Load cached extras or fetch fresh from Binance."""
    safe_name = symbol.replace("/", "_")
    path = Path(DATA_DIR) / f"{safe_name}_extras.csv"

    if path.exists() and not force_refresh:
        extras = pd.read_csv(path, parse_dates=["timestamp"])
        # Check if cache is recent enough (within 24h)
        if extras["timestamp"].max() > pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=1):
            logger.info(f"Loaded cached extras for {symbol}")
            if ohlcv_df["timestamp"].dt.tz is None:
                ohlcv_df["timestamp"] = pd.to_datetime(ohlcv_df["timestamp"])
            extras["timestamp"] = pd.to_datetime(extras["timestamp"]).dt.tz_localize(None)
            ohlcv_df["timestamp"] = ohlcv_df["timestamp"].dt.tz_localize(None)
            return pd.merge_asof(
                ohlcv_df.sort_values("timestamp"),
                extras.sort_values("timestamp"),
                on="timestamp", direction="backward"
            ).fillna(0.0)

    return merge_funding_rates(ohlcv_df, symbol)
