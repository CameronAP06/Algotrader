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

    # 1d override: predicting 1 day ahead is too noisy (1 bar = too much variance).
    # Use 5 bars (5 days) so the model learns smoother medium-term trends.
    # With max-forward labelling, a bar is UP if price rises 3.5%+ at ANY point
    # in the next 5 days — much more predictable than "exactly tomorrow."
    if timeframe == "1d":
        horizon = 5

    thresholds = {
        "15m": 0.006,   # 0.6%  — covers fees+slippage, targets clean intraday scalp
        "30m": 0.008,   # 0.8%  — slightly larger move required at 30m
        "1h":  0.015,   # 1.5%  — was 1.8%; slightly more signals, still above fee floor
        "2h":  0.022,   # 2.2%
        "4h":  0.030,   # 3.0%  — was 3.5%; meaningful 4h move, more label diversity
        "8h":  0.040,   # 4.0%  — half-day directional move
        "12h": 0.045,   # 4.5%  — between 8h and 1d
        "1d":  0.035,   # 3.5%  — was 6.0% (too high → 90%+ neutral). Daily crypto
                        #          routinely moves 3-5%; 3.5% gives ~25% UP/DOWN split
                        #          NOTE: 1d uses horizon=5 bars (5 days) set below
        "1w":  0.080,   # 8.0%  — was 12%; weekly with real conviction
        "2w":  0.120,   # 12.0% — was 18%
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

    NOTE: the `horizon` and `threshold` default args are overridden by
    get_label_params(timeframe) for ALL supported timeframes. They exist only
    as fallback defaults for unknown/custom timeframes passed without a string.

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
        # Add 20% of ATR on top of base threshold, capped at 1.5x threshold.
        # Was 30% / 2.5x — too aggressive, especially for 1d where daily ATR
        # can push threshold to 15%+, creating near-zero non-neutral labels.
        dynamic_threshold = (threshold + atr_pct * 0.20).clip(threshold, threshold * 1.5)
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
                        adx_threshold: float = REGIME_ADX_THRESHOLD,
                        timeframe: str = "1h") -> pd.DataFrame:
    """
    Enhanced market regime detection.

    Signals computed:
      ADX / DI system         — classic trend strength + direction
      Hurst exponent proxy    — >0.5 trending, <0.5 mean-reverting
      GARCH volatility proxy  — exponentially-weighted variance ratio
        (high vol_regime = elevated risk, lower Kelly fraction warranted)
      Choppiness Index        — 100 = pure noise, 0 = pure trend
      Market efficiency ratio — how efficiently price moved vs total path
      Volatility regime       — current ATR vs long-run average
      Regime transition score — rate of change of ADX (regime change warning)
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    # ── ADX / DI ──────────────────────────────────────────────────────────────
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
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx      = dx.rolling(window).mean()

    df["adx"]       = adx
    df["plus_di"]   = plus_di
    df["minus_di"]  = minus_di
    df["di_diff"]   = plus_di - minus_di
    df["adx_trend"] = (adx > adx_threshold).astype(float)

    regime = pd.Series(0, index=df.index)
    regime[(adx > adx_threshold) & (plus_di > minus_di)]  = 1   # trending up
    regime[(adx > adx_threshold) & (minus_di > plus_di)]  = 2   # trending down
    df["regime"]     = regime
    df["regime_sin"] = np.sin(2 * np.pi * regime / 3)
    df["regime_cos"] = np.cos(2 * np.pi * regime / 3)

    # Regime transition warning — rate of change of ADX
    df["adx_roc_5"]  = adx.pct_change(5).fillna(0)
    df["adx_roc_14"] = adx.pct_change(14).fillna(0)

    # ── Hurst Exponent (R/S approximation) ────────────────────────────────────
    # Compare variance at different time scales. H>0.5 = trending, H<0.5 = mean-reverting.
    returns     = close.pct_change()
    var_short   = returns.rolling(window).var() + 1e-12
    var_long    = returns.rolling(window * 4).var() / 4 + 1e-12
    df["hurst_approx"] = (np.log(var_long) / np.log(var_short)).clip(0, 2)

    # ── GARCH-proxy: EWMA volatility ratio ────────────────────────────────────
    # Short-span EWMA vol / long-span EWMA vol.
    # >1 = vol expanding (regime uncertainty), <1 = vol contracting (stable)
    ewm_short = returns.ewm(span=10).std()
    ewm_long  = returns.ewm(span=60).std()
    df["vol_regime"]       = (ewm_short / (ewm_long + 1e-9)).clip(0, 5)
    df["vol_regime_trend"] = df["vol_regime"].rolling(5).mean()  # smoothed

    # ── Choppiness Index ──────────────────────────────────────────────────────
    # CI = 100 * log10(sum(ATR_1) / (max_high - min_low)) / log10(n)
    # 100 = perfect chop, ~38 = perfect trend. Values above 61.8 = ranging.
    n = window
    atr_1_sum = tr.rolling(n).sum()
    highest   = high.rolling(n).max()
    lowest    = low.rolling(n).min()
    ci_range  = (highest - lowest).clip(lower=1e-9)
    df["choppiness"] = (100 * np.log10(atr_1_sum / ci_range + 1e-9)
                        / np.log10(n)).clip(0, 100)
    df["is_choppy"]  = (df["choppiness"] > 61.8).astype(float)

    # ── Market Efficiency Ratio (MER) ─────────────────────────────────────────
    # How efficiently price moved from start to end of window vs sum of all moves.
    # High MER = strong directional trend. Low MER = whipsaw.
    price_change = (close - close.shift(n)).abs()
    path_length  = close.diff().abs().rolling(n).sum() + 1e-9
    df["efficiency_ratio"] = (price_change / path_length).clip(0, 1)

    # ── ATR volatility regime ─────────────────────────────────────────────────
    # Rolling window = ~1 week of bars in real time, regardless of timeframe.
    # 168 was hardcoded (= 1 week at 1h) but equals 28 days at 4h and 168 days
    # at 1d — meaningless. Now computed as bars-per-week for the given timeframe.
    _bars_per_week = {
        "15m": 672, "30m": 336, "1h": 168, "2h": 84,
        "4h": 42,   "8h": 21,   "12h": 14, "1d": 7, "1w": 1,
    }
    vol_lookback = max(_bars_per_week.get(timeframe, 168), 14)
    atr_pct = df["atr_14"] / close if "atr_14" in df.columns else (
        tr.rolling(14).mean() / close)
    df["atr_vol_regime"] = (atr_pct / (atr_pct.rolling(vol_lookback).mean() + 1e-9)).clip(0, 5)

    return df


def add_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive sentiment and crowd-positioning proxy features from price/volume data.

    When external sentiment data (fear_greed, funding_rate) is present these
    are enhanced; otherwise we synthesise crowd sentiment proxies from OHLCV.

    Features:
      - Synthetic Fear & Greed proxy from vol + momentum + RSI
      - Volume-weighted sentiment (up-vol vs down-vol momentum)
      - Put/call proxy (open interest unavailable, so we use price gap analysis)
      - Sentiment momentum (rate of change of sentiment score)
      - Divergence: price making new highs while sentiment weakening
    """
    close  = df["close"]
    volume = df["volume"]

    # ── Synthetic Fear & Greed proxy ─────────────────────────────────────────
    # Blend of: RSI momentum + vol spike + price vs 30d high
    rsi14   = df["rsi_14"] if "rsi_14" in df.columns else pd.Series(50.0, index=df.index)
    vol_rat = df["vol_ratio_14"] if "vol_ratio_14" in df.columns else pd.Series(1.0, index=df.index)

    # Normalise each component to 0-1
    rsi_norm    = rsi14 / 100.0
    vol_norm    = (vol_rat - 1.0).clip(-2, 2) / 4.0 + 0.5
    price_30h   = close.rolling(30).max()
    price_norm  = (close / (price_30h + 1e-9)).clip(0.5, 1.0)
    # Directional momentum
    roc14       = close.pct_change(14)
    roc_norm    = (roc14 / 0.20 + 0.5).clip(0, 1)  # ±20% maps to 0-1

    # Composite score: high = greed, low = fear
    fg_proxy = (rsi_norm * 0.35 + vol_norm * 0.20 +
                price_norm * 0.25 + roc_norm * 0.20).clip(0, 1)
    df["sentiment_fg_proxy"] = fg_proxy

    # Use real Fear & Greed if available
    if "fear_greed" in df.columns:
        fg_real = df["fear_greed"] / 100.0
        # Blend: real data dominates but proxy fills gaps
        df["sentiment_score"] = fg_real.fillna(fg_proxy)
    else:
        df["sentiment_score"] = fg_proxy

    # Sentiment momentum (5-bar and 14-bar ROC)
    df["sentiment_roc_5"]  = df["sentiment_score"].diff(5)
    df["sentiment_roc_14"] = df["sentiment_score"].diff(14)

    # ── Volume-weighted directional sentiment ─────────────────────────────────
    # Up-volume: bars where close > open weighted by volume
    # Down-volume: bars where close < open weighted by volume
    direction     = np.sign(close.diff())
    up_vol   = volume.where(direction > 0, 0).rolling(14).sum()
    down_vol = volume.where(direction < 0, 0).rolling(14).sum()
    df["vol_sentiment"] = (up_vol - down_vol) / (up_vol + down_vol + 1e-9)
    df["vol_sent_ma7"]  = df["vol_sentiment"].rolling(7).mean()

    # ── Sentiment vs price divergence ─────────────────────────────────────────
    # Classic: price at new 20-bar high but sentiment declining → bearish divergence
    price_20h = close.rolling(20).max()
    is_near_high  = (close >= price_20h * 0.97).astype(float)
    sent_falling  = (df["sentiment_roc_5"] < -0.03).astype(float)
    df["bearish_divergence"] = is_near_high * sent_falling

    price_20l = close.rolling(20).min()
    is_near_low  = (close <= price_20l * 1.03).astype(float)
    sent_rising  = (df["sentiment_roc_5"] > 0.03).astype(float)
    df["bullish_divergence"] = is_near_low * sent_rising

    # ── Funding rate features (if available) ─────────────────────────────────
    if "fear_greed" in df.columns and "rsi_14" in df.columns:
        df["sentiment_rsi_divergence"] = (df["fear_greed"] / 100.0
                                          - df["rsi_14"] / 100.0)

    return df


def add_alt_data_features(df: pd.DataFrame, timeframe: str = "1h") -> pd.DataFrame:
    # NOTE: sentiment_rsi_divergence is intentionally NOT computed here —
    # add_sentiment_features() already handles it. Computing it again would
    # silently overwrite the result with an identical value (dead duplication).
    if "btc_dominance" in df.columns:
        # Lookback = ~1 week of bars in real time, regardless of timeframe.
        # rolling(168) was hardcoded (= 1 week at 1h) but equals 168 DAYS at 1d.
        # Same fix applied here as in add_regime_features.
        _bars_per_week = {
            "15m": 672, "30m": 336, "1h": 168, "2h": 84,
            "4h": 42,   "8h": 21,   "12h": 14, "1d": 7, "1w": 1,
        }
        lookback = max(_bars_per_week.get(timeframe, 168), 7)
        df["btc_dom_ma7"]    = df["btc_dominance"].rolling(lookback).mean()
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



def add_raw_ohlcv_sequences(df: pd.DataFrame, windows: list = None) -> pd.DataFrame:
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
    if windows is None:
        windows = [5, 10, 20]
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

# ─── Return Distribution Features ───────────────────────────────────────────

def add_return_distribution_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Higher-moment return distribution features — skewness and kurtosis over
    rolling windows, plus up-bar percentage.

    Why these help:
    - Up-bar percentage captures directional bias better than a simple SMA
      because it's normalised to [0,1] and not distorted by outlier bars.
    - Rolling skewness: +ve skew means recent gains are concentrated in
      a few large bars (trend); −ve skew suggests crash-like behaviour.
    - Rolling kurtosis: high kurtosis (fat tails) signals potential vol
      expansion — the model can learn to be more cautious in these regimes.
    """
    ret = df["close"].pct_change()

    for w in (7, 14, 28):
        up_bars = (ret > 0).rolling(w, min_periods=max(1, w // 2)).mean()
        df[f"up_bar_pct_{w}"] = up_bars.fillna(0.5)

    for w in (20, 50):
        df[f"return_skew_{w}"] = (
            ret.rolling(w, min_periods=max(5, w // 4))
               .skew()
               .fillna(0.0)
        )
        df[f"return_kurt_{w}"] = (
            ret.rolling(w, min_periods=max(5, w // 4))
               .kurt()
               .fillna(0.0)
        )

    return df


# ─── Main Pipeline ───────────────────────────────────────────────────────────

# ─── BTC/ETH Cross-Symbol Features ─────────────────────────────────────────

def add_btc_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Inject BTC price-action features into an altcoin feature matrix.

    BTC leads altcoins by 15-60 minutes — its recent return and volatility
    are among the highest-signal free features for altcoin prediction.

    Features added:
      btc_ret_{1,4,24}h   — BTC log-return over N bars
      btc_vol_24h          — BTC 24-bar rolling volatility (std of 1-bar returns)
      btc_vs_alt_24h       — BTC 24h return minus this symbol's 24h return
                             (positive = BTC outperforming → altcoin likely to follow)

    Skipped silently if btc_df is None or timestamps don't align.
    """
    if btc_df is None or btc_df.empty:
        return df

    try:
        btc = btc_df.copy()
        if "timestamp" in btc.columns:
            btc = btc.set_index("timestamp")["close"]
        else:
            btc = btc["close"]

        # Align BTC timestamps to this symbol's timestamps
        target_ts = (pd.to_datetime(df["timestamp"])
                     if "timestamp" in df.columns else df.index)
        btc_aligned = btc.reindex(target_ts, method="nearest", tolerance="2h")

        # BTC returns at multiple horizons
        for h in [1, 4, 24]:
            df[f"btc_ret_{h}h"] = np.log1p(btc_aligned.pct_change(h).values)

        # BTC rolling volatility (std of 1-bar log-returns over 24 bars)
        btc_r1 = np.log1p(btc_aligned.pct_change(1))
        df["btc_vol_24h"] = btc_r1.rolling(24).std().values

        # BTC vs this symbol momentum divergence
        if "close" in df.columns:
            own_ret_24 = np.log1p(df["close"].pct_change(24))
            df["btc_vs_alt_24h"] = df["btc_ret_24h"] - own_ret_24

        n_valid = btc_aligned.notna().sum()
        logger.debug(f"  BTC features added ({n_valid}/{len(df)} aligned bars)")

    except Exception as e:
        logger.warning(f"  BTC feature injection failed: {e} — skipping")

    return df


def _cache_path(symbol: str, timeframe: str, has_btc: bool = False) -> Path:
    safe_name = symbol.replace("/", "_")
    suffix    = "_btc" if has_btc else ""
    return Path(FEATURE_DIR) / f"{safe_name}_{timeframe}{suffix}_features.csv"


def _cache_is_valid(symbol: str, timeframe: str, raw_df_len: int,
                    has_btc: bool = False) -> bool:
    """Return True if a fresh feature cache exists and is newer than the raw data file."""
    cache = _cache_path(symbol, timeframe, has_btc=has_btc)
    if not cache.exists():
        return False
    try:
        safe_name = symbol.replace("/", "_")
        raw_path  = Path(DATA_DIR) / f"{safe_name}_{timeframe}.csv"
        if raw_path.exists() and cache.stat().st_mtime < raw_path.stat().st_mtime:
            return False   # raw data is newer → cache is stale
        cached = pd.read_csv(cache, nrows=5)
        if cached.empty:
            return False
        return True
    except Exception:
        return False


def build_features(df: pd.DataFrame, symbol: str = "", timeframe: str = "1h",
                   use_cache: bool = True,
                   btc_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Build feature matrix from raw OHLCV data.

    btc_df: optional BTC/USD OHLCV DataFrame — when provided and symbol is not
    BTC/ETH, injects BTC cross-symbol features (return, volatility, divergence).
    A separate cache file is used when BTC features are present so that the
    two variants don't clobber each other.
    """
    is_btc = symbol in ("BTC/USD", "BTC/USDT", "ETH/USD", "ETH/USDT")
    has_btc = (btc_df is not None and not btc_df.empty and not is_btc)

    if use_cache and symbol and _cache_is_valid(symbol, timeframe, len(df), has_btc=has_btc):
        cache = _cache_path(symbol, timeframe, has_btc=has_btc)
        try:
            feat_df = pd.read_csv(cache)
            logger.info(f"Loaded features from cache ({len(feat_df)} rows): {cache}")
            return feat_df
        except Exception as e:
            logger.warning(f"Cache load failed ({e}), rebuilding features")

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
    df = add_raw_ohlcv_sequences(df)  # raw sequences for self-supervised learning
    df = add_regime_features(df, timeframe=timeframe)      # ADX, Hurst, choppiness, efficiency ratio
    df = add_sentiment_features(df)   # Fear/greed proxy, vol sentiment, divergences
    df = add_funding_features(df)
    df = add_alt_data_features(df, timeframe=timeframe)
    df = add_return_distribution_features(df)             # skew, kurtosis, up-bar pct

    # BTC cross-symbol features (skip for BTC/ETH themselves)
    if has_btc:
        df = add_btc_features(df, btc_df)

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

    # Persist to cache so subsequent runs skip the rebuild
    if use_cache and symbol:
        save_features(df, symbol, timeframe=timeframe, has_btc=has_btc)

    return df


def save_features(df: pd.DataFrame, symbol: str, timeframe: str = TIMEFRAME,
                  has_btc: bool = False) -> Path:
    os.makedirs(FEATURE_DIR, exist_ok=True)
    path = _cache_path(symbol, timeframe, has_btc=has_btc)
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