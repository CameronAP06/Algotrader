"""
data/feature_engineer.py
Builds a rich feature matrix from raw OHLCV data.
Uses technical indicators + price-derived features + volatility signals.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from config.settings import (
    FEATURE_WINDOW_SIZES, PREDICTION_HORIZON,
    LABEL_THRESHOLD, TIMEFRAME, DATA_DIR, FEATURE_DIR
)


# ─── Label Generation ───────────────────────────────────────────────────────

def create_labels(df: pd.DataFrame, horizon: int = PREDICTION_HORIZON,
                  threshold: float = LABEL_THRESHOLD) -> pd.Series:
    """
    Creates 3-class labels:
      2 = UP   (future return > +threshold)
      0 = DOWN (future return < -threshold)
      1 = NEUTRAL
    """
    future_return = df["close"].pct_change(horizon).shift(-horizon)
    labels = pd.Series(1, index=df.index, name="label")  # default NEUTRAL
    labels[future_return > threshold]  = 2   # UP
    labels[future_return < -threshold] = 0   # DOWN
    return labels


# ─── Technical Indicators ───────────────────────────────────────────────────

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    for w in FEATURE_WINDOW_SIZES:
        df[f"sma_{w}"]  = df["close"].rolling(w).mean()
        df[f"ema_{w}"]  = df["close"].ewm(span=w, adjust=False).mean()
        df[f"close_vs_sma_{w}"] = df["close"] / df[f"sma_{w}"] - 1
    return df


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    for w in [1, 3, 5, 7, 14, 21]:
        df[f"roc_{w}"] = df["close"].pct_change(w)
    # RSI
    for w in [7, 14, 21]:
        delta = df["close"].diff()
        gain  = delta.clip(lower=0).rolling(w).mean()
        loss  = (-delta.clip(upper=0)).rolling(w).mean()
        rs    = gain / (loss + 1e-9)
        df[f"rsi_{w}"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    return df


def add_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    for w in [14, 20]:
        mid = df["close"].rolling(w).mean()
        std = df["close"].rolling(w).std()
        df[f"bb_upper_{w}"] = mid + 2 * std
        df[f"bb_lower_{w}"] = mid - 2 * std
        df[f"bb_width_{w}"] = (df[f"bb_upper_{w}"] - df[f"bb_lower_{w}"]) / (mid + 1e-9)
        df[f"bb_pct_{w}"]   = (df["close"] - df[f"bb_lower_{w}"]) / (df[f"bb_upper_{w}"] - df[f"bb_lower_{w}"] + 1e-9)
    return df


def add_atr(df: pd.DataFrame) -> pd.DataFrame:
    """Average True Range — measures volatility."""
    for w in [7, 14]:
        high_low   = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close  = (df["low"]  - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f"atr_{w}"] = tr.rolling(w).mean()
        df[f"atr_pct_{w}"] = df[f"atr_{w}"] / df["close"]
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    for w in [7, 14, 21]:
        df[f"vol_sma_{w}"] = df["volume"].rolling(w).mean()
        df[f"vol_ratio_{w}"] = df["volume"] / (df[f"vol_sma_{w}"] + 1e-9)
    # On-Balance Volume
    direction = np.sign(df["close"].diff())
    df["obv"] = (direction * df["volume"]).cumsum()
    df["obv_roc_7"] = df["obv"].pct_change(7)
    # VWAP (rolling)
    typical = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap_14"] = (typical * df["volume"]).rolling(14).sum() / df["volume"].rolling(14).sum()
    df["close_vs_vwap"] = df["close"] / (df["vwap_14"] + 1e-9) - 1
    return df


def add_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Numerical candle shape features."""
    body   = df["close"] - df["open"]
    range_ = df["high"]  - df["low"] + 1e-9
    df["body_pct"]       = body / df["open"]
    df["upper_shadow"]   = (df["high"] - df[["close","open"]].max(axis=1)) / range_
    df["lower_shadow"]   = (df[["close","open"]].min(axis=1) - df["low"])  / range_
    df["body_range_pct"] = body.abs() / range_
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode time cyclically — crypto has intraday seasonality."""
    ts = df["timestamp"] if "timestamp" in df.columns else df.index
    if hasattr(ts, "dt"):
        df["hour"]       = ts.dt.hour
        df["dayofweek"]  = ts.dt.dayofweek
        df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"]    = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["dow_cos"]    = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def add_lagged_features(df: pd.DataFrame, n_lags: int = 5) -> pd.DataFrame:
    """Add lagged returns and RSI — gives models temporal context."""
    for lag in range(1, n_lags + 1):
        df[f"return_lag_{lag}"] = df["close"].pct_change().shift(lag)
        df[f"volume_lag_{lag}"] = df["volume"].pct_change().shift(lag)
    return df


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """
    Run all feature engineering steps on a raw OHLCV DataFrame.
    Returns a cleaned DataFrame with features + label column.
    """
    logger.info(f"Building features for {symbol or 'unknown'} ({len(df)} rows)")

    df = df.copy()
    df = add_moving_averages(df)
    df = add_momentum(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_volume_features(df)
    df = add_candle_patterns(df)
    df = add_time_features(df)
    df = add_lagged_features(df)

    df["label"] = create_labels(df)

    # Drop NaN rows (from rolling windows + forward-looking label)
    # Replace infinities (from division by zero in feature calc) before dropping NaNs
    n_before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)
    logger.info(f"Dropped {n_before - len(df)} NaN/inf rows → {len(df)} usable rows")

    label_dist = df["label"].value_counts().to_dict()
    logger.info(f"Label distribution: DOWN={label_dist.get(0,0)} NEUTRAL={label_dist.get(1,0)} UP={label_dist.get(2,0)}")

    return df


def save_features(df: pd.DataFrame, symbol: str) -> Path:
    os.makedirs(FEATURE_DIR, exist_ok=True)
    safe_name = symbol.replace("/", "_")
    path = Path(FEATURE_DIR) / f"{safe_name}_{TIMEFRAME}_features.csv"
    df.to_csv(path, index=False)
    logger.info(f"Saved features -> {path}")
    return path


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return all feature columns (exclude OHLCV, timestamp, label)."""
    exclude = {"timestamp", "open", "high", "low", "close", "volume", "label"}
    return [c for c in df.columns if c not in exclude]


def build_all_pairs(raw_data: dict) -> dict:
    """Build features for all pairs."""
    featured = {}
    for symbol, df in raw_data.items():
        feat_df = build_features(df, symbol)
        save_features(feat_df, symbol)
        featured[symbol] = feat_df
    return featured


if __name__ == "__main__":
    from kraken_fetcher import fetch_all_pairs
    raw = fetch_all_pairs()
    featured = build_all_pairs(raw)
    for sym, df in featured.items():
        cols = get_feature_columns(df)
        print(f"\n{sym}: {len(df)} rows, {len(cols)} features")
        print(df[["timestamp","close","label"]].tail(5).to_string())
        