"""
backtest/filters.py
Rule-based confirmation filters — ML signals must pass ALL filters to execute.
These act as a sanity check on top of model predictions, filtering out
trades where simple market structure disagrees with the ML signal.

Filters:
  1. Trend filter     — don't trade against the 20-SMA direction
  2. Volatility filter — skip low-volatility environments (ATR percentile)
  3. Volume filter    — require above-average volume for entry
  4. Funding filter   — avoid longs when funding is extremely positive (overextended)
"""
import numpy as np
import pandas as pd
from loguru import logger


def apply_filters(df: pd.DataFrame, signals: dict) -> dict:
    """
    Apply rule-based filters to ML-generated signals.
    Returns a new signals dict with filtered entries replaced by HOLD.

    df must contain: close, sma_20, atr_14, volume, vol_sma_14
    and optionally: funding_rate
    """
    signal_arr = signals["signal"].copy()
    n          = len(signal_arr)

    # Align df to signal length
    df_aligned = df.tail(n).reset_index(drop=True)

    filtered_count = {"trend": 0, "volatility": 0, "volume": 0, "funding": 0, "regime": 0}

    # Pre-compute ATR percentile threshold (bottom 20%)
    atr_col = df_aligned["atr_14"] if "atr_14" in df_aligned.columns else pd.Series(np.ones(n))
    atr_threshold = atr_col.quantile(0.20)

    # Pre-compute volume threshold
    vol_col     = df_aligned["volume"] if "volume" in df_aligned.columns else pd.Series(np.ones(n))
    vol_sma_col = df_aligned["vol_sma_14"] if "vol_sma_14" in df_aligned.columns else pd.Series(np.ones(n))

    # Pre-compute SMA
    sma_col = df_aligned["sma_20"] if "sma_20" in df_aligned.columns else df_aligned["close"]

    # Funding rate
    has_funding = "funding_rate" in df_aligned.columns
    funding_col = df_aligned["funding_rate"] if has_funding else pd.Series(np.zeros(n))

    # Regime (lever 4)
    from config.settings import USE_REGIME_FILTER, REGIME_ADX_THRESHOLD
    has_regime    = USE_REGIME_FILTER and "regime" in df_aligned.columns and "adx" in df_aligned.columns
    regime_col    = df_aligned["regime"] if has_regime else pd.Series(np.zeros(n))
    adx_col       = df_aligned["adx"]    if has_regime else pd.Series(np.full(n, 100.0))
    adx_threshold = REGIME_ADX_THRESHOLD

    for i in range(n):
        sig = signal_arr[i]
        if sig == "HOLD":
            continue

        price = df_aligned["close"].iloc[i]
        sma   = sma_col.iloc[i]
        atr   = atr_col.iloc[i]
        vol   = vol_col.iloc[i]
        vol_avg = vol_sma_col.iloc[i]
        funding = funding_col.iloc[i]

        # ── Filter 1: Trend ────────────────────────────────────────────
        # BUY only when price is above SMA (uptrend)
        # SELL only when price is below SMA (downtrend)
        if sig == "BUY" and price < sma:
            signal_arr[i] = "HOLD"
            filtered_count["trend"] += 1
            continue
        if sig == "SELL" and price > sma:
            signal_arr[i] = "HOLD"
            filtered_count["trend"] += 1
            continue

        # ── Filter 2: Volatility ───────────────────────────────────────
        # Skip trades in low-volatility environments — choppy and fee-heavy
        if atr < atr_threshold:
            signal_arr[i] = "HOLD"
            filtered_count["volatility"] += 1
            continue

        # ── Filter 3: Volume ───────────────────────────────────────────
        # Require volume >= 80% of average — avoid low-liquidity moves
        if vol < vol_avg * 0.80:
            signal_arr[i] = "HOLD"
            filtered_count["volume"] += 1
            continue

        # ── Filter 4: Funding rate extremes ───────────────────────────
        if has_funding:
            if sig == "BUY"  and funding >  0.001:
                signal_arr[i] = "HOLD"
                filtered_count["funding"] += 1
                continue
            if sig == "SELL" and funding < -0.001:
                signal_arr[i] = "HOLD"
                filtered_count["funding"] += 1
                continue

        # ── Filter 5: Regime filter ────────────────────────────────
        # Only trade when market is in a trending regime (ADX > threshold)
        # Ranging markets produce false breakouts that get chopped by fees
        if has_regime:
            regime_val = regime_col.iloc[i]
            adx_val    = adx_col.iloc[i]
            if adx_val < adx_threshold:
                signal_arr[i] = "HOLD"
                filtered_count["regime"] += 1
                continue
            # Also enforce directional alignment with regime
            if sig == "BUY"  and regime_val == 2:  # Don't buy in confirmed downtrend
                signal_arr[i] = "HOLD"
                filtered_count["regime"] += 1
                continue
            if sig == "SELL" and regime_val == 1:  # Don't sell in confirmed uptrend
                signal_arr[i] = "HOLD"
                filtered_count["regime"] += 1
                continue

    total_filtered = sum(filtered_count.values())
    original_signals = (signals["signal"] != "HOLD").sum()
    logger.info(
        f"Filters removed {total_filtered}/{original_signals} signals — "
        f"trend={filtered_count['trend']} vol={filtered_count['volatility']} "
        f"volume={filtered_count['volume']} funding={filtered_count['funding']} "
        f"regime={filtered_count['regime']}"
    )

    return {
        "signal":     signal_arr,
        "confidence": signals["confidence"],
        "up_prob":    signals["up_prob"],
        "down_prob":  signals["down_prob"],
    }


def filter_summary(original_signals: dict, filtered_signals: dict):
    """Log a before/after comparison of signal counts."""
    orig_buy  = (original_signals["signal"] == "BUY").sum()
    orig_sell = (original_signals["signal"] == "SELL").sum()
    filt_buy  = (filtered_signals["signal"] == "BUY").sum()
    filt_sell = (filtered_signals["signal"] == "SELL").sum()
    logger.info(
        f"Signal filter: BUY {orig_buy}→{filt_buy} | SELL {orig_sell}→{filt_sell}"
    )
