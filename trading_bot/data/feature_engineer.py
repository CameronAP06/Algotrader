"""
data/feature_engineer.py
Builds a rich feature matrix from raw OHLCV data.

Four accuracy levers implemented:
  1. Volatility-adjusted labels  — threshold scales with ATR, not fixed %
  2. Timeframe-aware prediction horizon — always predicts ~24h ahead regardless of bar size
  3. Regime features             — ADX, trend strength, market state classification
  4. Regime-aware label filter   — only label bars where regime is clear
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from config.settings import (
    FEATURE_WINDOW_SIZES, PREDICTION_HORIZON,
    LABEL_THRESHOLD, TIMEFRAME, DATA_DIR, FEATURE_DIR,
    USE_REGIME_FILTER, REGIME_WINDOW, REGIME_ADX_THRESHOLD
)


# ─── Lever 1 & 2: Timeframe-Aware Labels ─────────────────────────────────────

# Bars per hour for each supported timeframe
_BARS_PER_HOUR = {
    "5m":  12.0,
    "15m": 4.0,
    "30m": 2.0,
    "1h":  1.0,
    "2h":  0.5,
    "4h":  0.25,
    "8h":  0.125,
    "12h": 1/12,
    "1d":  1/24,
    "1w":  1/168,   # 1 bar = 168 hours
    "2w":  1/336,   # 1 bar = 336 hours
}

def get_label_params(timeframe: str = "1h") -> tuple:
    """
    Return (horizon_bars, label_threshold) tuned for the given timeframe.

    Horizon: targets ~24h of real time ahead so the model always predicts
    one full day forward regardless of bar size.

    Threshold: scales with typical ATR at each timeframe. Must comfortably
    exceed fees (0.1% x2) + slippage, and target genuinely tradeable moves.
    Goal: ~25% UP / ~25% DOWN / ~50% NEUTRAL label split.
    """
    bph   = _BARS_PER_HOUR.get(timeframe, 1.0)
    horizon = max(1, round(24 * bph))   # 24h ahead in bars

    thresholds = {
        "5m":  0.006,   # 0.6%  — very short-term, ultra-tight threshold
        "15m": 0.010,   # 1.0%  — intraday swing worth trading
        "30m": 0.013,   # 1.3%  — between 15m and 1h
        "1h":  0.018,   # 1.8%  — 1h crypto typically moves 0.8-1.5%, need headroom
        "2h":  0.025,   # 2.5%
        "4h":  0.035,   # 3.5%  — meaningful 4h move
        "8h":  0.045,   # 4.5%  — half-day directional move
        "12h": 0.055,   # 5.5%  — between 8h and 1d
        "1d":  0.060,   # 6.0%  — daily candle with real conviction
        "1w":  0.120,   # 12.0% — weekly crypto move with real conviction
        "2w":  0.180,   # 18.0% — two-week directional move
    }
    threshold = thresholds.get(timeframe, 0.012)

    logger.debug(
        f"Label params [{timeframe}]: horizon={horizon} bars (24h), "
        f"threshold={threshold:.1%}"
    )
    return horizon, threshold


def create_labels(df: pd.DataFrame,
                  horizon: int = PREDICTION_HORIZON,
                  threshold: float = LABEL_THRESHOLD,
                  timeframe: str = "1h") -> pd.Series:
    """
    Creates 3-class labels using TIMEFRAME-AWARE, VOLATILITY-ADJUSTED thresholds.

    When called with a timeframe, get_label_params() overrides the defaults so
    that:
      - horizon always represents ~24h of real time
      - threshold is calibrated to the typical ATR at that bar size

    ATR-scaling is also applied on top so that in high-vol periods a larger
    move is needed to classify as UP/DOWN, reducing whipsaw labels.

    Labels:
      2 = UP   (future return > dynamic threshold)
      0 = DOWN (future return < -dynamic threshold)
      1 = NEUTRAL
    """
    # If timeframe given, override horizon and base threshold
    if timeframe and timeframe != "1h":
        horizon, threshold = get_label_params(timeframe)
    elif timeframe == "1h":
        horizon, threshold = get_label_params("1h")

    future_return = df["close"].pct_change(horizon).shift(-horizon)

    # ATR-scaling: in high-vol regimes require a larger move to label UP/DOWN
    if "atr_14" in df.columns:
        atr_pct = df["atr_14"] / df["close"]
        # Add 30% of ATR on top of base threshold, capped at 2x threshold
        dynamic_threshold = (threshold + atr_pct * 0.3).clip(threshold, threshold * 2.5)
    else:
        dynamic_threshold = threshold

    labels = pd.Series(1, index=df.index, name="label")
    labels[future_return >  dynamic_threshold] = 2   # UP
    labels[future_return < -dynamic_threshold] = 0   # DOWN
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
    for w in [7, 14]:
        high_low   = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close  = (df["low"]  - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f"atr_{w}"]     = tr.rolling(w).mean()
        df[f"atr_pct_{w}"] = df[f"atr_{w}"] / df["close"]
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    for w in [7, 14, 21]:
        df[f"vol_sma_{w}"]   = df["volume"].rolling(w).mean()
        df[f"vol_ratio_{w}"] = df["volume"] / (df[f"vol_sma_{w}"] + 1e-9)
    direction = np.sign(df["close"].diff())
    df["obv"]       = (direction * df["volume"]).cumsum()
    df["obv_roc_7"] = df["obv"].pct_change(7)
    typical         = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap_14"]       = (typical * df["volume"]).rolling(14).sum() / df["volume"].rolling(14).sum()
    df["close_vs_vwap"] = df["close"] / (df["vwap_14"] + 1e-9) - 1
    return df


def add_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    body   = df["close"] - df["open"]
    range_ = df["high"]  - df["low"] + 1e-9
    df["body_pct"]       = body / df["open"]
    df["upper_shadow"]   = (df["high"] - df[["close","open"]].max(axis=1)) / range_
    df["lower_shadow"]   = (df[["close","open"]].min(axis=1) - df["low"])  / range_
    df["body_range_pct"] = body.abs() / range_
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"] if "timestamp" in df.columns else df.index
    if hasattr(ts, "dt"):
        df["hour"]      = ts.dt.hour
        df["dayofweek"] = ts.dt.dayofweek
        df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"]   = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["dow_cos"]   = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def add_lagged_features(df: pd.DataFrame, n_lags: int = 5) -> pd.DataFrame:
    for lag in range(1, n_lags + 1):
        df[f"return_lag_{lag}"] = df["close"].pct_change().shift(lag)
        df[f"volume_lag_{lag}"] = df["volume"].pct_change().shift(lag)
    return df


# ─── Lever 3 & 4: Regime Detection ──────────────────────────────────────────

def add_regime_features(df: pd.DataFrame,
                        window: int = REGIME_WINDOW,
                        adx_threshold: float = REGIME_ADX_THRESHOLD) -> pd.DataFrame:
    """
    Adds market regime features using ADX and trend structure.

    ADX > 25 = trending (models work well here)
    ADX < 20 = ranging/choppy (models perform poorly here)

    Features added:
      - adx: trend strength (directionless)
      - plus_di / minus_di: directional indicators
      - di_diff: positive = uptrend pressure, negative = downtrend
      - adx_trend: binary — is market currently trending?
      - regime: 0=ranging, 1=trending up, 2=trending down
      - hurst_approx: >0.5 trending, <0.5 mean-reverting
      - vol_regime: current volatility vs recent average
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)

    up_move   = high - high.shift()
    down_move = low.shift() - low

    plus_dm  = np.where((up_move > down_move) & (up_move > 0),   up_move,   0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_w    = tr.rolling(window).mean()
    plus_di  = 100 * pd.Series(plus_dm,  index=df.index).rolling(window).mean() / (atr_w + 1e-9)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(window).mean() / (atr_w + 1e-9)

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx = dx.rolling(window).mean()

    df["adx"]       = adx
    df["plus_di"]   = plus_di
    df["minus_di"]  = minus_di
    df["di_diff"]   = plus_di - minus_di
    df["adx_trend"] = (adx > adx_threshold).astype(float)

    regime = pd.Series(0, index=df.index)
    regime[(adx > adx_threshold) & (plus_di > minus_di)]  = 1
    regime[(adx > adx_threshold) & (minus_di > plus_di)]  = 2
    df["regime"]     = regime
    df["regime_sin"] = np.sin(2 * np.pi * regime / 3)
    df["regime_cos"] = np.cos(2 * np.pi * regime / 3)

    # Hurst exponent approximation
    returns = close.pct_change()
    var_1   = returns.rolling(window).var()
    var_4   = returns.rolling(window * 4).var() / 4
    df["hurst_approx"] = (np.log(var_4 + 1e-9) / np.log(var_1 + 1e-9)).clip(0, 2)

    # Volatility regime ratio
    atr_pct = df["atr_14"] / close if "atr_14" in df.columns else tr.rolling(14).mean() / close
    df["vol_regime"] = atr_pct / (atr_pct.rolling(168).mean() + 1e-9)

    return df


def add_alt_data_features(df: pd.DataFrame) -> pd.DataFrame:
    if "fear_greed" in df.columns and "rsi_14" in df.columns:
        df["sentiment_rsi_divergence"] = df["fear_greed"] / 100.0 - df["rsi_14"] / 100.0
    if "btc_dominance" in df.columns:
        df["btc_dom_ma7"]    = df["btc_dominance"].rolling(168).mean()
        df["btc_dom_vs_avg"] = df["btc_dominance"] - df["btc_dom_ma7"]
    return df


def add_funding_features(df: pd.DataFrame) -> pd.DataFrame:
    if "funding_rate" not in df.columns:
        return df
    df["funding_momentum"] = df["funding_rate"].diff(8)
    if "sma_20" in df.columns:
        df["funding_trend_agree"] = (
            np.sign(df["close"] - df["sma_20"]) *
            np.sign(-df["funding_rate"])
        )
    return df



def add_raw_ohlcv_sequences(df: pd.DataFrame, windows: list = [5, 10, 20]) -> pd.DataFrame:
    """
    Add normalised raw OHLCV rolling windows as features.

    Instead of hand-crafting what patterns to look for, we give the models
    the raw price/volume history and let them discover structure themselves.
    Each window computes a z-score normalised rolling snapshot:
      - Prices normalised by the window mean (removes level effect)
      - Volume normalised by rolling mean

    The TFT's VariableSelectionNetwork will learn which of these windows
    and which OHLCV dimensions actually matter per symbol.
    """
    for w in windows:
        roll_close = df["close"].rolling(w)
        roll_vol   = df["volume"].rolling(w)

        mu_c  = roll_close.mean()
        std_c = roll_close.std().clip(lower=1e-9)
        mu_v  = roll_vol.mean()
        std_v = roll_vol.std().clip(lower=1e-9)

        # Normalised OHLCV vs rolling window stats
        df[f"raw_close_{w}"]  = (df["close"]  - mu_c) / std_c
        df[f"raw_open_{w}"]   = (df["open"]   - mu_c) / std_c
        df[f"raw_high_{w}"]   = (df["high"]   - mu_c) / std_c
        df[f"raw_low_{w}"]    = (df["low"]    - mu_c) / std_c
        df[f"raw_volume_{w}"] = (df["volume"] - mu_v) / std_v

        # Price range and body within window context
        df[f"raw_range_{w}"]  = (df["high"] - df["low"]) / (std_c + 1e-9)
        df[f"raw_body_{w}"]   = (df["close"] - df["open"]) / (std_c + 1e-9)

    return df

# ─── Main Pipeline ───────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, symbol: str = "", timeframe: str = "1h") -> pd.DataFrame:
    logger.info(f"Building features for {symbol or 'unknown'} ({len(df)} rows)")

    df = df.copy()

    # ATR must come before create_labels (lever 1 needs it)
    df = add_moving_averages(df)
    df = add_momentum(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_volume_features(df)
    df = add_candle_patterns(df)
    df = add_time_features(df)
    df = add_lagged_features(df)
    df = add_raw_ohlcv_sequences(df)  # B: raw input for self-supervised learning
    df = add_regime_features(df)   # Lever 3
    df = add_funding_features(df)
    df = add_alt_data_features(df)

    # Lever 1 + 2: timeframe-aware, volatility-adjusted labels
    df["label"] = create_labels(df, timeframe=timeframe)

    n_before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)
    logger.info(f"Dropped {n_before - len(df)} NaN/inf rows → {len(df)} usable rows")

    label_dist = df["label"].value_counts().to_dict()
    logger.info(
        f"Label distribution: DOWN={label_dist.get(0,0)} "
        f"NEUTRAL={label_dist.get(1,0)} UP={label_dist.get(2,0)}"
    )

    if "regime" in df.columns:
        r = df["regime"].value_counts().to_dict()
        logger.info(f"Regime: ranging={r.get(0,0)} trending_up={r.get(1,0)} trending_down={r.get(2,0)}")

    return df


def save_features(df: pd.DataFrame, symbol: str) -> Path:
    os.makedirs(FEATURE_DIR, exist_ok=True)
    safe_name = symbol.replace("/", "_")
    path = Path(FEATURE_DIR) / f"{safe_name}_{TIMEFRAME}_features.csv"
    df.to_csv(path, index=False)
    logger.info(f"Saved features -> {path}")
    return path


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = {"timestamp", "open", "high", "low", "close", "volume", "label"}
    return [c for c in df.columns if c not in exclude]


def build_all_pairs(raw_data: dict) -> dict:
    featured = {}
    for symbol, df in raw_data.items():
        feat_df = build_features(df, symbol)
        save_features(feat_df, symbol)
        featured[symbol] = feat_df
    return featured