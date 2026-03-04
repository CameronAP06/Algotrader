"""
src/features.py
───────────────
Minimal feature engineering for inference only.
Mirrors the training pipeline exactly — same features, same order.
No labels needed here (we're predicting, not training).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_features(df: pd.DataFrame, timeframe: str = "4h") -> pd.DataFrame:
    """
    Build feature matrix from raw OHLCV.
    Must produce identical columns to the training pipeline.
    """
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    c = df["close"]
    o = df["open"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # ── Price-derived ─────────────────────────────────────────────────────────
    df["returns"]      = c.pct_change()
    df["log_returns"]  = np.log(c / c.shift(1))
    df["hl_range"]     = (h - l) / c
    df["oc_range"]     = (c - o) / o
    df["upper_shadow"]  = (h - np.maximum(o, c)) / c
    df["lower_shadow"]  = (np.minimum(o, c) - l) / c

    # ── Moving averages & crossovers ──────────────────────────────────────────
    for w in [5, 10, 20, 50]:
        df[f"sma_{w}"]     = c.rolling(w).mean()
        df[f"sma_{w}_dist"]= (c - df[f"sma_{w}"]) / df[f"sma_{w}"]

    df["ema_12"]       = c.ewm(span=12).mean()
    df["ema_26"]       = c.ewm(span=26).mean()
    df["macd"]         = df["ema_12"] - df["ema_26"]
    df["macd_signal"]  = df["macd"].ewm(span=9).mean()
    df["macd_hist"]    = df["macd"] - df["macd_signal"]

    # ── Momentum ──────────────────────────────────────────────────────────────
    for w in [6, 12, 24]:
        df[f"mom_{w}"] = c.pct_change(w)

    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - 100 / (1 + rs)
    df["rsi_dist_50"] = (df["rsi_14"] - 50) / 50

    # Stochastic
    low14  = l.rolling(14).min()
    high14 = h.rolling(14).max()
    df["stoch_k"] = 100 * (c - low14) / (high14 - low14 + 1e-9)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # ── Volatility ────────────────────────────────────────────────────────────
    # ATR
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    df["atr_14"]     = tr.rolling(14).mean()
    df["atr_pct"]    = df["atr_14"] / c

    # Bollinger Bands
    bb_mid           = c.rolling(20).mean()
    bb_std           = c.rolling(20).std()
    df["bb_upper"]   = bb_mid + 2 * bb_std
    df["bb_lower"]   = bb_mid - 2 * bb_std
    df["bb_width"]   = (df["bb_upper"] - df["bb_lower"]) / bb_mid
    df["bb_pos"]     = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    # Historical vol
    for w in [10, 20]:
        df[f"hvol_{w}"] = df["log_returns"].rolling(w).std() * np.sqrt(w)

    # Vol regime
    atr_pct = df["atr_pct"]
    df["vol_regime"] = atr_pct / (atr_pct.rolling(168).mean() + 1e-9)

    # Hurst approx
    var_1 = df["log_returns"].rolling(4).var()
    var_4 = df["log_returns"].rolling(16).var()
    df["hurst_approx"] = (np.log(var_4 + 1e-9) / np.log(var_1 + 1e-9)).clip(0, 2)

    # ── Volume ────────────────────────────────────────────────────────────────
    df["volume_sma20"]  = v.rolling(20).mean()
    df["volume_ratio"]  = v / (df["volume_sma20"] + 1e-9)
    df["volume_change"] = v.pct_change()

    # OBV
    obv = (np.sign(df["returns"]) * v).cumsum()
    df["obv_norm"]  = obv / (obv.abs().rolling(20).mean() + 1e-9)

    # ── Trend / regime ────────────────────────────────────────────────────────
    # ADX
    plus_dm  = (h.diff()).clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    mask     = plus_dm < minus_dm
    plus_dm[mask] = 0
    mask2    = minus_dm <= plus_dm
    minus_dm[mask2] = 0

    atr14    = df["atr_14"]
    plus_di  = 100 * plus_dm.rolling(14).mean()  / (atr14 + 1e-9)
    minus_di = 100 * minus_dm.rolling(14).mean() / (atr14 + 1e-9)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    df["adx_14"]      = dx.rolling(14).mean()
    df["adx_plus_di"] = plus_di
    df["adx_minus_di"]= minus_di

    # Regime labels
    adx_thresh = 25.0
    is_trending = df["adx_14"] >= adx_thresh
    df["regime_trending_up"]   = (is_trending & (plus_di > minus_di)).astype(float)
    df["regime_trending_down"] = (is_trending & (minus_di > plus_di)).astype(float)
    df["regime_ranging"]       = (~is_trending).astype(float)

    # Price position in longer-term context
    df["dist_52w_high"] = c / h.rolling(365).max() - 1
    df["dist_52w_low"]  = c / l.rolling(365).min() - 1

    # ── Drop NaN/inf ──────────────────────────────────────────────────────────
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the feature columns in the correct order (excludes OHLCV + labels)."""
    exclude = {"timestamp", "open", "high", "low", "close", "volume", "label"}
    cols = [c for c in df.columns if c not in exclude]
    return cols


def fit_scaler(X: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def apply_scaler(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    return scaler.transform(X)
