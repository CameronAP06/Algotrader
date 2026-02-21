"""
data/kraken_fetcher.py
Fetches OHLCV data from Binance via ccxt — no API key required.
Binance supports full historical pagination unlike Kraken's 720-candle limit.
"""
import os, time, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ccxt
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
from config.settings import TRADING_PAIRS, TIMEFRAME, HISTORY_DAYS, DATA_DIR

# Kraken -> Binance symbol mapping
SYMBOL_MAP = {
    "BTC/USD":  "BTC/USDT",
    "ETH/USD":  "ETH/USDT",
    "SOL/USD":  "SOL/USDT",
    "XRP/USD":  "XRP/USDT",
}


def get_exchange():
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "rateLimit": 100,
        "options": {"defaultType": "spot"},
    })
    logger.info(f"Connected to {exchange.name}")
    return exchange


def fetch_ohlcv(exchange, symbol, timeframe=TIMEFRAME, days=HISTORY_DAYS):
    binance_symbol = SYMBOL_MAP.get(symbol, symbol)
    since_ms  = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    until_ms  = int(datetime.utcnow().timestamp() * 1000)
    all_candles = []
    batch_size  = 1000  # Binance max per request

    logger.info(f"Fetching {binance_symbol} [{timeframe}] — {days} days of history...")

    current_since = since_ms
    while current_since < until_ms:
        try:
            candles = exchange.fetch_ohlcv(
                binance_symbol, timeframe=timeframe,
                since=current_since, limit=batch_size
            )
        except ccxt.RateLimitExceeded:
            logger.warning("Rate limit hit — sleeping 10s")
            time.sleep(10)
            continue
        except ccxt.NetworkError as e:
            logger.error(f"Network error: {e} — retrying in 5s")
            time.sleep(5)
            continue

        if not candles:
            break

        all_candles.extend(candles)
        last_ts = candles[-1][0]

        logger.debug(f"  Fetched {len(candles)} candles up to {pd.to_datetime(last_ts, unit='ms')} | total={len(all_candles)}")

        if len(candles) < batch_size:
            break

        current_since = last_ts + 1
        time.sleep(0.1)  # Binance is generous with rate limits

    if not all_candles:
        logger.error(f"No data returned for {symbol}")
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    logger.success(f"{symbol}: {len(df)} candles ({df['timestamp'].iloc[0].date()} -> {df['timestamp'].iloc[-1].date()})")
    return df


def save_data(df, symbol):
    os.makedirs(DATA_DIR, exist_ok=True)
    safe_name = symbol.replace("/", "_")
    path = Path(DATA_DIR) / f"{safe_name}_{TIMEFRAME}.csv"
    df.to_csv(path, index=False)
    logger.info(f"Saved -> {path}")
    return path


def load_data(symbol):
    safe_name = symbol.replace("/", "_")
    path = Path(DATA_DIR) / f"{safe_name}_{TIMEFRAME}.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["timestamp"])
        logger.info(f"Loaded cached data for {symbol} ({len(df)} rows)")
        return df
    logger.info(f"No cache found for {symbol} — fetching from Binance")
    exchange = get_exchange()
    df = fetch_ohlcv(exchange, symbol)
    if not df.empty:
        save_data(df, symbol)
    return df


def fetch_all_pairs(force_refresh=False):
    exchange = get_exchange()
    data = {}
    for symbol in TRADING_PAIRS:
        safe_name = symbol.replace("/", "_")
        path = Path(DATA_DIR) / f"{safe_name}_{TIMEFRAME}.csv"
        if path.exists() and not force_refresh:
            df = pd.read_csv(path, parse_dates=["timestamp"])
            logger.info(f"Loaded {symbol} from cache ({len(df)} rows)")
        else:
            df = fetch_ohlcv(exchange, symbol)
            if not df.empty:
                save_data(df, symbol)
        if not df.empty:
            data[symbol] = df
    logger.success(f"Loaded {len(data)}/{len(TRADING_PAIRS)} pairs")
    return data


if __name__ == "__main__":
    data = fetch_all_pairs(force_refresh=True)
    for sym, df in data.items():
        print(f"\n{sym}: {len(df)} rows | {df['timestamp'].iloc[0].date()} -> {df['timestamp'].iloc[-1].date()}")