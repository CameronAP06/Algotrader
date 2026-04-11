"""
utils/walk_forward.py
Walk-forward validation for the ML trading ensemble.

Instead of a single train/test split, this rolls a training window forward
through time, testing on each subsequent period independently. This gives
a robust estimate of whether models have a consistent edge across different
market regimes.

Timeframe-aware: each timeframe gets its own bar counts that map to the
same real-world durations (e.g. always ~1 year training, ~110 days test).

Supported timeframes: 15m, 30m, 1h, 2h, 4h, 8h, 1d
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from loguru import logger
from typing import Dict

from data.feature_engineer import build_features, get_feature_columns
from utils.splitter import save_scaler
from sklearn.preprocessing import StandardScaler
from models import tft_model, cnn_model, lstm_model
from models.ensemble import weighted_ensemble, optimise_weights, generate_signals
from backtest.engine import BacktestEngine
from backtest.filters import apply_filters


# ─── Timeframe-Aware Config ──────────────────────────────────────────────────
#
# All durations map to the same real-world periods regardless of timeframe:
#   Train window : ~1 year
#   Val window   : ~55 days  (within train window, used for weight optimisation)
#   Test window  : ~55 days  (out-of-sample test per fold)
#   Step size    : ~55 days  (how far to advance each fold)
#
# Bars per timeframe for each duration:
#   15m : 4 bars/hour  → 1yr=35040, 55d=5280
#   30m : 2 bars/hour  → 1yr=17520, 55d=2640
#   1h  : 1 bar/hour   → 1yr=8760,  55d=1320
#   2h  : 0.5 bars/hr  → 1yr=4380,  55d=660
#   4h  : 6 bars/day   → 1yr=2190,  55d=330
#   8h  : 3 bars/day   → 1yr=1095,  55d=165
#   1d  : 1 bar/day    → 1yr=365,   55d=55

BARS_PER_HOUR = {
    "15m": 4,
    "30m": 2,
    "1h":  1,
    "2h":  0.5,
    "4h":  0.25,
    "8h":  0.125,
    "1d":  1/24,
}

def get_wf_config(timeframe: str = "1h") -> dict:
    """
    Return walk-forward bar counts scaled to the given timeframe.
    All timeframes map to the same real-world durations:
      ~1 year training | ~55 day val | ~55 day test | ~55 day step
    """
    bph = BARS_PER_HOUR.get(timeframe, 1.0)
    hours_per_year = 365 * 24
    hours_55d      = 55  * 24

    train_bars = max(200, round(hours_per_year * bph))   # ~1 year
    val_bars   = max(50,  round(hours_55d      * bph))   # ~55 days
    test_bars  = max(50,  round(hours_55d      * bph))   # ~55 days
    step_bars  = max(30,  round(hours_55d      * bph))   # ~55 days step

    # Minimum train samples: 10% of target train size, but never less than
    # 2× the sequence length (so the model has enough sequences to learn from).
    # For 1d: 365 * 0.5 = 182. For 1h: 8760 * 0.1 = 876 → capped at 500 min.
    min_train = max(
        round(train_bars * 0.5),   # at least 50% of target train size
        round(hours_per_year * bph * 0.1),  # at least 10% of a year
    )

    config = {
        "train_bars":    train_bars,
        "val_bars":      val_bars,
        "test_bars":     test_bars,
        "step_bars":     step_bars,
        "min_train":     min_train,
        "min_folds":     3,
        "timeframe":     timeframe,
        "use_optuna":    True,
        "optuna_trials": 30,   # trials per symbol — increase for better search
    }

    logger.debug(
        f"WF config [{timeframe}]: train={train_bars} val={val_bars} "
        f"test={test_bars} step={step_bars} "
        f"(min_data_needed={train_bars + val_bars + test_bars})"
    )
    return config


# Legacy default config (1h) — kept for backwards compatibility
WF_CONFIG = get_wf_config("1h")


def get_model_params(timeframe: str) -> dict:
    """
    Return timeframe-appropriate sequence model params.
    Daily data has far fewer bars so needs shorter sequences + smaller batches.
    """
    params = {
        "15m": {"sequence_length": 96,  "batch_size": 128, "epochs": 40},
        "30m": {"sequence_length": 48,  "batch_size": 128, "epochs": 40},
        "1h":  {"sequence_length": 48,  "batch_size": 128, "epochs": 40},
        "2h":  {"sequence_length": 24,  "batch_size": 64,  "epochs": 40},
        "4h":  {"sequence_length": 16,  "batch_size": 64,  "epochs": 40},
        "8h":  {"sequence_length": 10,  "batch_size": 32,  "epochs": 50},
        "1d":  {"sequence_length": 10,  "batch_size": 16,  "epochs": 60},
    }
    return params.get(timeframe, params["1h"])


# ─── Single Fold ─────────────────────────────────────────────────────────────

def run_fold(feat_df: pd.DataFrame, feature_cols: list,
             train_start: int, train_end: int,
             val_end: int, test_end: int,
             fold_num: int, symbol: str,
             config: dict,
             optuna_params: dict = None) -> Dict:
    """
    Run a single walk-forward fold.
    If optuna_params is provided, uses Optuna-selected features,
    TFT hyperparams, signal threshold, and filter params.
    """

    # Apply Optuna-selected feature subset if available
    active_cols = feature_cols
    if optuna_params and "feature_cols" in optuna_params:
        active_cols = [c for c in optuna_params["feature_cols"] if c in feature_cols]
        if len(active_cols) < 10:
            active_cols = feature_cols  # fallback if too few survive
        logger.debug(f"Fold {fold_num}: using {len(active_cols)}/{len(feature_cols)} Optuna-selected features")

    X = feat_df[active_cols].values
    y = feat_df["label"].values

    X_train = X[train_start:train_end]
    y_train = y[train_start:train_end]
    X_val   = X[train_end:val_end]
    y_val   = y[train_end:val_end]
    X_test  = X[val_end:test_end]
    y_test  = y[val_end:test_end]

    min_train = config.get("min_train", 500)
    if len(X_train) < min_train or len(X_test) < 30:
        logger.warning(
            f"Fold {fold_num}: skipping — train={len(X_train)} "
            f"(need {min_train}), test={len(X_test)}"
        )
        return {}

    # Scale — fit on train only, apply to val/test
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    logger.info(f"Fold {fold_num}: train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    # Apply timeframe-aware model params + Optuna overrides
    import models.tft_model as tft_mod
    import models.cnn_model as cnn_mod
    import models.lstm_model as lstm_mod

    tf_model_p = get_model_params(config.get("timeframe", "1h"))
    tft_orig  = tft_mod.TFT_PARAMS.copy()
    cnn_orig  = cnn_mod.CNN_PARAMS.copy()
    lstm_orig = lstm_mod.LSTM_PARAMS.copy()

    # Apply timeframe-aware sequence lengths
    tft_mod.TFT_PARAMS["sequence_length"] = tf_model_p["sequence_length"]
    tft_mod.TFT_PARAMS["batch_size"]      = tf_model_p["batch_size"]
    tft_mod.TFT_PARAMS["epochs"]          = tf_model_p["epochs"]
    cnn_mod.CNN_PARAMS["sequence_length"] = tf_model_p["sequence_length"]
    cnn_mod.CNN_PARAMS["batch_size"]      = tf_model_p["batch_size"]
    lstm_mod.LSTM_PARAMS["sequence_length"] = tf_model_p["sequence_length"]
    lstm_mod.LSTM_PARAMS["batch_size"]      = tf_model_p["batch_size"]

    # Apply Optuna TFT overrides on top
    if optuna_params and "tft_params" in optuna_params:
        op = optuna_params["tft_params"]
        tft_mod.TFT_PARAMS["sequence_length"] = op.get("seq_len", tf_model_p["sequence_length"])
        tft_mod.TFT_PARAMS["d_model"]         = op.get("d_model", tft_mod.TFT_PARAMS["d_model"])
        tft_mod.TFT_PARAMS["n_heads"]         = op.get("n_heads", tft_mod.TFT_PARAMS["n_heads"])
        tft_mod.TFT_PARAMS["n_lstm_layers"]   = op.get("n_lstm_layers", 2)
        tft_mod.TFT_PARAMS["dropout"]         = op.get("dropout", 0.1)
        tft_mod.TFT_PARAMS["batch_size"]      = op.get("batch_size", tf_model_p["batch_size"])
        tft_mod.TFT_PARAMS["learning_rate"]   = op.get("lr", 0.0005)

    # Train models
    try:
        tft  = tft_model.train(X_train, y_train, X_val, y_val, f"{symbol}_f{fold_num}")
        cnn  = cnn_model.train(X_train, y_train, X_val, y_val, f"{symbol}_f{fold_num}")
        lstm = lstm_model.train(X_train, y_train, X_val, y_val, f"{symbol}_f{fold_num}")
    except Exception as e:
        logger.error(f"Fold {fold_num} training failed: {e}")
        return {}

    # Ensemble weights optimised on validation set
    tft_val_p  = tft_model.predict_proba(tft,  X_val)
    cnn_val_p  = cnn_model.predict_proba(cnn,  X_val)
    lstm_val_p = lstm_model.predict_proba(lstm, X_val)
    weights    = optimise_weights(tft_val_p, cnn_val_p, lstm_val_p, y_val)

    # Test predictions
    tft_test_p  = tft_model.predict_proba(tft,  X_test)
    cnn_test_p  = cnn_model.predict_proba(cnn,  X_test)
    lstm_test_p = lstm_model.predict_proba(lstm, X_test)

    # Restore model params
    tft_mod.TFT_PARAMS.update(tft_orig)
    cnn_mod.CNN_PARAMS.update(cnn_orig)
    lstm_mod.LSTM_PARAMS.update(lstm_orig)

    blended = weighted_ensemble(tft_test_p, cnn_test_p, lstm_test_p, weights)

    # Use Optuna-tuned signal threshold if available
    sig_thresh = None
    if optuna_params and "signal_threshold" in optuna_params:
        sig_thresh = optuna_params["signal_threshold"]
    top_pct = optuna_params.get("top_pct", 0.15) if optuna_params else 0.15
    signals = generate_signals(blended, symbol=symbol.split('_')[0] + '/USD',
                               threshold=sig_thresh,
                               use_percentile=True, top_pct=top_pct)

    import config.settings as s

    # Snapshot globals that may be mutated by Optuna params
    orig_vol      = s.VOLUME_FILTER_PCT
    orig_adx      = s.REGIME_ADX_THRESHOLD
    orig_volflt   = s.VOLATILITY_FILTER_PCT
    orig_stop_mult = s.ATR_STOP_MULT
    orig_tp_mult   = s.ATR_TP_MULT

    try:
        # Apply Optuna-tuned filter params if available
        if optuna_params and "filter_params" in optuna_params:
            fp = optuna_params["filter_params"]
            s.VOLUME_FILTER_PCT     = fp.get("volume_pct",     orig_vol)
            s.REGIME_ADX_THRESHOLD  = fp.get("adx_threshold",  orig_adx)
            s.VOLATILITY_FILTER_PCT = fp.get("volatility_pct", orig_volflt)

        # Apply Optuna-tuned ATR params if available
        if optuna_params and "atr_params" in optuna_params:
            ap = optuna_params["atr_params"]
            s.ATR_STOP_MULT = ap.get("atr_stop_mult", orig_stop_mult)
            s.ATR_TP_MULT   = ap.get("atr_tp_mult",   orig_tp_mult)

        preds    = blended.argmax(axis=1)
        y_test_a = y_test[-len(preds):]
        accuracy = (preds == y_test_a).mean()

        test_df  = feat_df.iloc[val_end:test_end].reset_index(drop=True)
        test_df  = test_df.tail(len(preds)).reset_index(drop=True)
        filtered = apply_filters(test_df, signals, timeframe=config.get("timeframe", "1h"))

        engine  = BacktestEngine()
        metrics = engine.run(test_df, filtered, symbol, timeframe=config.get("timeframe", "1h"))

    finally:
        # Always restore globals — even on exception
        s.VOLUME_FILTER_PCT     = orig_vol
        s.REGIME_ADX_THRESHOLD  = orig_adx
        s.VOLATILITY_FILTER_PCT = orig_volflt
        s.ATR_STOP_MULT         = orig_stop_mult
        s.ATR_TP_MULT           = orig_tp_mult

    result = {
        "fold":         fold_num,
        "train_start":  train_start,
        "test_start":   val_end,
        "test_end":     test_end,
        "accuracy":     accuracy,
        "total_return": metrics["total_return"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "max_drawdown": metrics["max_drawdown"],
        "win_rate":     metrics["win_rate"],
        "n_trades":     metrics["n_trades"],
        "weights":      weights,
    }

    logger.info(
        f"Fold {fold_num}: acc={accuracy:.3f} | "
        f"return={metrics['total_return']:.2%} | "
        f"sharpe={metrics['sharpe_ratio']:.2f} | "
        f"trades={metrics['n_trades']}"
    )
    return result


# ─── Full Walk-Forward ────────────────────────────────────────────────────────

def walk_forward_validate(raw_df: pd.DataFrame, symbol: str,
                          config: dict = None,
                          timeframe: str = "1h") -> pd.DataFrame:
    """
    Run full walk-forward validation for one symbol.

    Args:
        raw_df    : Raw OHLCV DataFrame (before feature engineering)
        symbol    : Symbol name for logging (e.g. 'INJ/USD_4h')
        config    : Optional explicit config dict; if None, auto-derived from timeframe
        timeframe : Timeframe string used to auto-derive config ('1h', '4h', '1d', etc.)

    Returns:
        DataFrame with one row per fold containing accuracy, return, Sharpe, etc.
    """
    if config is None:
        config = get_wf_config(timeframe)

    logger.info(f"\n{'='*60}\nWalk-Forward Validation: {symbol}\n{'='*60}")
    logger.info(
        f"Config [{timeframe}]: train={config['train_bars']} bars "
        f"val={config['val_bars']} test={config['test_bars']} "
        f"step={config['step_bars']}"
    )

    feat_df      = build_features(raw_df, symbol, timeframe=timeframe)
    feature_cols = get_feature_columns(feat_df)
    n            = len(feat_df)

    train_bars = config["train_bars"]
    val_bars   = config["val_bars"]
    test_bars  = config["test_bars"]
    step_bars  = config["step_bars"]

    min_needed = train_bars + val_bars + test_bars
    if n < min_needed:
        # Try to rescue by scaling down proportionally to available data.
        # Keep the same train/val/test ratios but shrink to fit.
        ratio = n / min_needed
        if ratio >= 0.55:  # at least 55% of target data — viable but tight
            new_train = max(int(train_bars * ratio), 500)
            new_val   = max(int(val_bars   * ratio), 100)
            new_test  = max(int(test_bars  * ratio), 100)
            if n >= new_train + new_val + new_test:
                logger.warning(
                    f"Insufficient data for walk-forward [{timeframe}]: "
                    f"{n}/{min_needed} bars — scaling down proportionally "
                    f"(train={new_train} val={new_val} test={new_test})"
                )
                train_bars = new_train
                val_bars   = new_val
                test_bars  = new_test
            else:
                logger.error(
                    f"Insufficient data for walk-forward [{timeframe}]: "
                    f"{n} bars available, need at least {min_needed} "
                    f"(scaled attempt also failed — skipping)"
                )
                return pd.DataFrame()
        else:
            logger.error(
                f"Insufficient data for walk-forward [{timeframe}]: "
                f"{n} bars available, need {min_needed} "
                f"({ratio:.0%} of target — too little to scale, skipping)"
            )
            return pd.DataFrame()

    # ── Optuna search on fold 1 validation data ──────────────────────────────
    optuna_params = None
    use_optuna    = config.get("use_optuna", True)

    if use_optuna:
        try:
            from utils.optuna_search import run_optuna_search
            # Fold 1 windows
            f1_train_start = 0
            f1_train_end   = train_bars
            f1_val_end     = train_bars + val_bars
            f1_test_end    = f1_val_end + test_bars

            X_all  = feat_df[feature_cols].values
            y_all  = feat_df["label"].values
            scaler = StandardScaler()
            X_tr1  = scaler.fit_transform(X_all[f1_train_start:f1_train_end])
            X_v1   = scaler.transform(X_all[f1_train_end:f1_val_end])
            y_tr1  = y_all[f1_train_start:f1_train_end]
            y_v1   = y_all[f1_train_end:f1_val_end]
            feat_df_v1 = feat_df.iloc[f1_train_end:f1_val_end].reset_index(drop=True)

            n_trials = config.get("optuna_trials", 30)
            optuna_params = run_optuna_search(
                X_tr1, y_tr1, X_v1, y_v1,
                all_cols=feature_cols,
                feat_df_val=feat_df_v1,
                symbol=symbol,
                timeframe=timeframe,
                n_trials=n_trials,
            )
            logger.success(
                f"Optuna complete: {len(optuna_params['feature_cols'])} features selected, "
                f"best Sharpe={optuna_params['best_sharpe']:.3f}"
            )
        except Exception as e:
            logger.warning(f"Optuna search failed ({e}) — using default params")
            optuna_params = None

    results    = []
    fold       = 1
    test_start = train_bars + val_bars

    while test_start + test_bars <= n:
        train_start = max(0, test_start - train_bars - val_bars)
        train_end   = test_start - val_bars
        val_end     = test_start
        test_end    = test_start + test_bars

        result = run_fold(
            feat_df, feature_cols,
            train_start, train_end, val_end, test_end,
            fold, symbol, config,
            optuna_params=optuna_params,
        )
        if result:
            results.append(result)

        test_start += step_bars
        fold       += 1

    if not results:
        logger.error("No valid folds completed")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    _print_wf_summary(results_df, symbol)
    return results_df


# ─── Summary ─────────────────────────────────────────────────────────────────

def _print_wf_summary(df: pd.DataFrame, symbol: str):
    print(f"\n{'='*70}")
    print(f"WALK-FORWARD RESULTS: {symbol} ({len(df)} folds)")
    print(f"{'='*70}")
    print(f"{'Fold':<6} {'Accuracy':>9} {'Return':>9} {'Sharpe':>8} {'WinRate':>9} {'Trades':>7}")
    print(f"{'-'*70}")

    for _, row in df.iterrows():
        print(
            f"{int(row['fold']):<6} "
            f"{row['accuracy']:>8.1%} "
            f"{row['total_return']:>8.2%} "
            f"{row['sharpe_ratio']:>8.2f} "
            f"{row['win_rate']:>8.1%} "
            f"{int(row['n_trades']):>7}"
        )

    print(f"{'-'*70}")
    print(
        f"{'MEDIAN':<6} "
        f"{df['accuracy'].median():>8.1%} "
        f"{df['total_return'].median():>8.2%} "
        f"{df['sharpe_ratio'].median():>8.2f} "
        f"{df['win_rate'].median():>8.1%} "
        f"{df['n_trades'].median():>7.0f}"
    )
    print(
        f"{'MEAN':<6} "
        f"{df['accuracy'].mean():>8.1%} "
        f"{df['total_return'].mean():>8.2%} "
        f"{df['sharpe_ratio'].mean():>8.2f} "
        f"{df['win_rate'].mean():>8.1%} "
        f"{df['n_trades'].mean():>7.0f}"
    )
    print(f"{'='*70}")

    profitable_folds = (df["total_return"] > 0).sum()
    pos_sharpe_folds = (df["sharpe_ratio"] > 0).sum()
    above_random     = (df["accuracy"] > 0.36).sum()

    print(f"\nInterpretation:")
    print(f"  Profitable folds : {profitable_folds}/{len(df)} ({profitable_folds/len(df):.0%})")
    print(f"  Positive Sharpe  : {pos_sharpe_folds}/{len(df)} ({pos_sharpe_folds/len(df):.0%})")
    print(f"  Above-random acc : {above_random}/{len(df)} ({above_random/len(df):.0%})")

    if profitable_folds / len(df) >= 0.6 and df["sharpe_ratio"].median() > 0.5:
        print(f"\n  ✓ Model shows CONSISTENT edge — proceed to live trading")
    elif profitable_folds / len(df) >= 0.5:
        print(f"\n  ~ Model shows MARGINAL edge — needs improvement before live trading")
    else:
        print(f"\n  ✗ Model shows NO consistent edge")


def save_wf_results(results_df: pd.DataFrame, symbol: str):
    """Save walk-forward results to CSV."""
    os.makedirs("backtest/results", exist_ok=True)
    path = f"backtest/results/{symbol.replace('/','_')}_walkforward.csv"
    results_df.to_csv(path, index=False)
    logger.info(f"Walk-forward results saved -> {path}")