"""
backtest/filters.py
Rule-based confirmation filters applied on top of ML signals.

DIAGNOSIS (Apr 2026): choppiness + efficiency_ratio filters were killing 53%
of all signals (293/552) with no on/off switch — they always ran if the columns
existed and both incorrectly counted into the "regime" bucket, making the log
misleading. Fixed by adding USE_CHOPPINESS_FILTER / USE_EFFICIENCY_FILTER flags
and giving each filter its own counter.

DIAGNOSIS (Feb 2026): filters were removing 95.4% of all signals.
  Volume filter:  39.7% killed (threshold 80% of avg vol — too strict)
  Regime filter:  35.0% killed (ADX > 25 — too strict for crypto)
  Volatility:     20.8% killed (bottom 20% ATR — piling on)
  Total cascade:  only 4.6% of signals survived → avg 3.8 trades/fold

FIX: Loosened defaults + per-filter on/off switches in settings.py.
Longer timeframes get automatically relaxed thresholds via TF_RELAX.

Target: 15-30 trades per fold at 1h, 8-20 at 4h.
"""
import numpy as np
import pandas as pd
from loguru import logger
from config.settings import (
    USE_REGIME_FILTER,     REGIME_ADX_THRESHOLD,
    USE_VOLUME_FILTER,     VOLUME_FILTER_PCT,
    USE_VOLATILITY_FILTER, VOLATILITY_FILTER_PCT,
    USE_TREND_FILTER,      USE_FUNDING_FILTER,
    USE_CHOPPINESS_FILTER, USE_EFFICIENCY_FILTER,
    AMIHUD_FILTER_PCT,
)

# Longer timeframes get looser thresholds — fewer bars, coarser signal
TF_RELAX = {
    "15m": 1.00, "30m": 1.00,
    "1h":  1.00,
    "2h":  0.85,
    "4h":  0.70,
    "8h":  0.55,
    "1d":  0.30,
}


def apply_filters(df: pd.DataFrame, signals: dict,
                  timeframe: str = "1h") -> dict:
    signal_arr = signals["signal"].copy()
    n          = len(signal_arr)
    df_aligned = df.tail(n).reset_index(drop=True)
    filtered_count = {
        "trend": 0, "vol": 0, "volume": 0,
        "funding": 0, "regime": 0, "choppiness": 0, "efficiency": 0, "amihud": 0,
    }

    relax = TF_RELAX.get(timeframe, 1.0)

    close_col = df_aligned["close"]
    sma_col   = df_aligned["sma_20"] if "sma_20" in df_aligned.columns else close_col

    atr_col       = df_aligned.get("atr_14", pd.Series(np.ones(n)))
    atr_pct       = max(VOLATILITY_FILTER_PCT * relax, 0.05)
    atr_threshold = atr_col.quantile(atr_pct)

    vol_col     = df_aligned.get("volume",     pd.Series(np.ones(n)))
    vol_sma_col = df_aligned.get("vol_sma_14", pd.Series(np.ones(n)))
    volume_pct  = VOLUME_FILTER_PCT * relax

    # Amihud illiquidity ratio = |pct_change| / volume — high = illiquid bar
    pct_changes      = close_col.pct_change().abs().fillna(0)
    amihud_series    = pct_changes / (vol_col.replace(0, np.nan).fillna(1))
    amihud_threshold = float(amihud_series.quantile(AMIHUD_FILTER_PCT))

    has_funding = "funding_rate" in df_aligned.columns and USE_FUNDING_FILTER
    funding_col = df_aligned["funding_rate"] if has_funding else pd.Series(np.zeros(n))

    has_regime    = (USE_REGIME_FILTER
                     and "regime" in df_aligned.columns
                     and "adx"    in df_aligned.columns)
    regime_col    = df_aligned["regime"] if has_regime else pd.Series(np.zeros(n))
    adx_col       = df_aligned["adx"]    if has_regime else pd.Series(np.full(n, 100.0))
    adx_threshold = REGIME_ADX_THRESHOLD * relax

    has_choppiness = (USE_CHOPPINESS_FILTER
                      and "is_choppy" in df_aligned.columns)
    has_efficiency = (USE_EFFICIENCY_FILTER
                      and "efficiency_ratio" in df_aligned.columns
                      and timeframe != "1d")

    for i in range(n):
        sig = signal_arr[i]
        if sig == "HOLD":
            continue

        price   = close_col.iloc[i]
        sma     = sma_col.iloc[i]
        atr     = atr_col.iloc[i]
        vol     = vol_col.iloc[i]
        vol_avg = vol_sma_col.iloc[i]
        funding = funding_col.iloc[i]

        # Filter 1: Trend — don't trade against the SMA
        if USE_TREND_FILTER:
            if sig == "BUY"  and price < sma:
                signal_arr[i] = "HOLD"; filtered_count["trend"] += 1; continue
            if sig == "SELL" and price > sma:
                signal_arr[i] = "HOLD"; filtered_count["trend"] += 1; continue

        # Filter 2: Volatility — skip very low ATR bars (dead market)
        if USE_VOLATILITY_FILTER:
            if atr < atr_threshold:
                signal_arr[i] = "HOLD"; filtered_count["vol"] += 1; continue

        # Filter 3: Volume — require minimum fraction of average volume
        if USE_VOLUME_FILTER:
            if vol < vol_avg * volume_pct:
                signal_arr[i] = "HOLD"; filtered_count["volume"] += 1; continue

        # Filter 4: Funding rate extremes (perp only)
        if has_funding:
            if sig == "BUY"  and funding >  0.001:
                signal_arr[i] = "HOLD"; filtered_count["funding"] += 1; continue
            if sig == "SELL" and funding < -0.001:
                signal_arr[i] = "HOLD"; filtered_count["funding"] += 1; continue

        # Filter 5: Regime — ADX momentum gate
        if has_regime:
            adx_val    = adx_col.iloc[i]
            regime_val = regime_col.iloc[i]
            if adx_val < adx_threshold:
                signal_arr[i] = "HOLD"; filtered_count["regime"] += 1; continue
            if sig == "BUY"  and regime_val == 2:   # confirmed downtrend
                signal_arr[i] = "HOLD"; filtered_count["regime"] += 1; continue
            if sig == "SELL" and regime_val == 1:   # confirmed uptrend
                signal_arr[i] = "HOLD"; filtered_count["regime"] += 1; continue

        # Filter 6: Choppiness Index — skip pure-noise markets
        if has_choppiness:
            if df_aligned["is_choppy"].iloc[i] > 0.5:
                signal_arr[i] = "HOLD"; filtered_count["choppiness"] += 1; continue

        # Filter 7: Market Efficiency Ratio — require directional efficiency
        if has_efficiency:
            er = df_aligned["efficiency_ratio"].iloc[i]
            if er < 0.10:
                signal_arr[i] = "HOLD"; filtered_count["efficiency"] += 1; continue

        # Filter 8: Amihud illiquidity — skip very thin/illiquid bars
        if amihud_series.iloc[i] > amihud_threshold:
            signal_arr[i] = "HOLD"; filtered_count["amihud"] += 1; continue

    total_filtered   = sum(filtered_count.values())
    original_signals = int((np.array(signals["signal"]) != "HOLD").sum())
    surviving        = original_signals - total_filtered
    survival_pct     = surviving / max(original_signals, 1)

    # Only log filters that actually killed something (skip zero-count clutter)
    active = " ".join(
        f"{k}={v}" for k, v in filtered_count.items() if v > 0
    ) or "none"
    logger.info(
        f"Filters [{timeframe}] {total_filtered}/{original_signals} removed "
        f"({survival_pct:.0%} survive) — {active}"
    )

    return {
        "signal":     signal_arr,
        "confidence": signals["confidence"],
        "up_prob":    signals["up_prob"],
        "down_prob":  signals["down_prob"],
    }
