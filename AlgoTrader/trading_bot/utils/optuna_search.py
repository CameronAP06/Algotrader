"""
utils/optuna_search.py
Automated hyperparameter and feature search using Optuna.

What it searches over PER SYMBOL PER TIMEFRAME:
  1. Feature subsets   — which feature groups to include
  2. Model hyperparams — TFT d_model, n_heads, dropout, lr
  3. Signal threshold  — per-symbol optimal threshold
  4. Filter thresholds — volume_pct, adx_threshold relaxation

The search runs on the VALIDATION set of the first fold only
(fast proxy), then the best params are used for full walk-forward.
Typical search: 50 trials × ~2 min/trial = ~2 hrs per symbol.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import optuna
from loguru import logger

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─── Feature Groups ───────────────────────────────────────────────────────────
# Instead of always using all 80+ features, Optuna decides which groups to include.
# This prevents the models from being distracted by noisy or redundant indicators.

FEATURE_GROUPS = {
    "trend":      ["sma_", "ema_", "close_vs_sma_"],
    "momentum":   ["roc_", "rsi_"],
    "macd":       ["macd"],
    "bollinger":  ["bb_upper_", "bb_lower_", "bb_width_", "bb_pct_"],
    "atr":        ["atr_", "atr_pct_"],
    "volume":     ["vol_sma_", "vol_ratio_", "obv", "vwap", "close_vs_vwap"],
    "candles":    ["body_pct", "upper_shadow", "lower_shadow", "body_range_pct"],
    "time":       ["hour", "dayofweek", "hour_sin", "hour_cos", "dow_sin", "dow_cos"],
    "lags":       ["return_lag_", "volume_lag_"],
    "regime":     ["adx", "plus_di", "minus_di", "di_diff", "adx_trend",
                   "regime", "regime_sin", "regime_cos", "hurst_approx", "vol_regime"],
    "alt_data":   ["fear_greed", "btc_dominance", "google_trends",
                   "btc_dom_proxy", "sentiment_rsi_divergence"],
    "raw_ohlcv":  ["raw_open_", "raw_high_", "raw_low_", "raw_close_", "raw_volume_"],
}

# Groups always included (model won't work without these)
MANDATORY_GROUPS = {"atr"}  # needed for ATR stops in engine


def select_features(all_cols: list, trial: optuna.Trial) -> list:
    """
    Given all available feature columns, use Optuna trial to select
    which feature groups to include. Returns filtered column list.
    """
    selected = set()

    for group_name, prefixes in FEATURE_GROUPS.items():
        # Mandatory groups always on
        if group_name in MANDATORY_GROUPS:
            include = True
        else:
            include = trial.suggest_categorical(f"feat_{group_name}", [True, False])

        if include:
            for col in all_cols:
                for prefix in prefixes:
                    if col.startswith(prefix) or col == prefix:
                        selected.add(col)
                        break

    # Always include raw OHLCV columns that the engine needs
    for col in all_cols:
        if col in {"open", "high", "low", "close", "volume"}:
            selected.add(col)

    result = [c for c in all_cols if c in selected]
    logger.debug(f"Feature selection: {len(result)}/{len(all_cols)} columns")
    return result


def suggest_tft_params(trial: optuna.Trial) -> dict:
    """Suggest TFT hyperparameters for this trial."""
    d_model = trial.suggest_categorical("d_model", [32, 64, 128])
    return {
        "sequence_length": trial.suggest_categorical("seq_len", [12, 24, 48]),
        "d_model":         d_model,
        "n_heads":         trial.suggest_categorical("n_heads",
                               [h for h in [2, 4, 8] if d_model % h == 0]),
        "n_lstm_layers":   trial.suggest_int("n_lstm_layers", 1, 3),
        # MIOpen dropout kernel broken on gfx1100+MSVC14.39 — locked to 0.0
        # Re-enable once rocRAND headers are found: dropout = trial.suggest_float("dropout", 0.05, 0.4)
        "dropout":         0.0,
        "epochs":          50,
        "batch_size":      trial.suggest_categorical("batch_size", [32, 64, 128]),
        "learning_rate":   trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "patience":        10,
    }


def suggest_filter_params(trial: optuna.Trial) -> dict:
    """Suggest filter threshold relaxations."""
    return {
        "volume_pct":     trial.suggest_float("volume_pct",  0.20, 0.70),
        "adx_threshold":  trial.suggest_float("adx_thresh",  10.0, 25.0),
        "volatility_pct": trial.suggest_float("vol_pct",     0.05, 0.20),
    }


def suggest_signal_threshold(trial: optuna.Trial) -> float:
    # Ensemble outputs are compressed vs single-model — blending 3 models
    # pushes probabilities toward 0.5, so threshold must be lower than
    # edge scanner's 0.60 (which ran LSTM standalone)
    return trial.suggest_float("signal_thresh", 0.33, 0.45)


def suggest_atr_params(trial: optuna.Trial) -> dict:
    """Suggest ATR stop/TP multipliers — tighter stops = more trades."""
    stop_mult = trial.suggest_float("atr_stop_mult", 0.8, 2.0)
    return {
        "atr_stop_mult": stop_mult,
        "atr_tp_mult":   stop_mult * 2.0,  # always maintain 2:1 RR
    }


# ─── Objective ────────────────────────────────────────────────────────────────

def make_objective(X_train, y_train, X_val, y_val, all_cols,
                   feat_df_val, timeframe: str = "1h"):
    """
    Returns an Optuna objective function that:
    1. Selects feature subset
    2. Trains TFT with suggested hyperparams
    3. Generates signals with suggested threshold
    4. Runs backtest on val set
    5. Returns Sharpe ratio (maximised)
    """
    from models import tft_model
    from models.ensemble import generate_signals
    from backtest.engine import BacktestEngine
    from backtest.filters import apply_filters
    import config.settings as s

    def objective(trial: optuna.Trial) -> float:
        try:
            # 1. Feature selection
            selected_cols = select_features(all_cols, trial)
            if len(selected_cols) < 10:
                return -999.0  # Too few features

            col_idx = [all_cols.index(c) for c in selected_cols if c in all_cols]
            X_tr = X_train[:, col_idx]
            X_v  = X_val[:,   col_idx]

            # 2. TFT hyperparameters
            tft_p = suggest_tft_params(trial)

            # Override global TFT params for this trial
            import models.tft_model as tft_mod
            orig_params = tft_mod.TFT_PARAMS.copy()
            tft_mod.TFT_PARAMS.update(tft_p)

            model = tft_mod.train(X_tr, y_train, X_v, y_val, "optuna_trial")
            probs = tft_mod.predict_proba(model, X_v)

            # Restore params
            tft_mod.TFT_PARAMS.update(orig_params)

            # 3. Signal threshold
            threshold = suggest_signal_threshold(trial)

            # 4. Filter params
            filter_p = suggest_filter_params(trial)
            orig_vol    = s.VOLUME_FILTER_PCT
            orig_adx    = s.REGIME_ADX_THRESHOLD
            orig_volflt = s.VOLATILITY_FILTER_PCT
            s.VOLUME_FILTER_PCT     = filter_p["volume_pct"]
            s.REGIME_ADX_THRESHOLD  = filter_p["adx_threshold"]
            s.VOLATILITY_FILTER_PCT = filter_p["volatility_pct"]

            # 4b. ATR stop/TP multipliers
            atr_p = suggest_atr_params(trial)
            orig_stop_mult = s.ATR_STOP_MULT
            orig_tp_mult   = s.ATR_TP_MULT
            s.ATR_STOP_MULT = atr_p["atr_stop_mult"]
            s.ATR_TP_MULT   = atr_p["atr_tp_mult"]

            signals = generate_signals(probs, threshold=threshold,
                                        use_percentile=True, top_pct=top_pct)
            filtered = apply_filters(feat_df_val, signals, timeframe=timeframe)

            # Restore filter + ATR params
            s.VOLUME_FILTER_PCT     = orig_vol
            s.REGIME_ADX_THRESHOLD  = orig_adx
            s.VOLATILITY_FILTER_PCT = orig_volflt
            s.ATR_STOP_MULT         = orig_stop_mult
            s.ATR_TP_MULT           = orig_tp_mult

            # 5. Backtest on validation bars
            engine  = BacktestEngine()
            metrics = engine.run(feat_df_val, filtered)

            n_trades = metrics["n_trades"]
            if n_trades < 3:
                return -10.0  # Too few trades — no signal
            if n_trades > 60:
                return -5.0   # Too many trades — low-confidence noise, fee destruction

            sharpe   = metrics["sharpe_ratio"]
            win_rate = metrics["win_rate"]
            ret      = metrics["total_return"]

            # Composite score rewarding:
            # 1. Win rate above breakeven (33% at 2:1) — most important
            # 2. Positive return
            # 3. Trade count in sweet spot (3-30) — not too few, not too many
            # 4. Sharpe as tiebreaker but capped to avoid outlier games
            wr_bonus      = max(0.0, (win_rate - 0.33) * 10)   # 0 at breakeven, +1 at 43%
            ret_bonus     = 2.0 if ret > 0 else 0.0
            count_bonus   = min(2.0, n_trades / 15)             # peaks at 30 trades, no bonus beyond
            sharpe_capped = max(-3.0, min(3.0, sharpe))         # cap to ±3 to avoid outliers

            return sharpe_capped + wr_bonus + ret_bonus + count_bonus

        except Exception as e:
            logger.debug(f"Trial failed: {e}")
            return -999.0

    return objective


# ─── Search Runner ────────────────────────────────────────────────────────────

def run_optuna_search(X_train: np.ndarray, y_train: np.ndarray,
                      X_val:   np.ndarray, y_val:   np.ndarray,
                      all_cols: list,
                      feat_df_val,
                      symbol:    str  = "unknown",
                      timeframe: str  = "1h",
                      n_trials:  int  = 50) -> dict:
    """
    Run Optuna search and return best params dict.
    Best params include: selected feature indices, tft_params, filter_params, signal_threshold.
    """
    logger.info(f"Optuna search: {symbol} [{timeframe}] — {n_trials} trials")

    objective = make_objective(X_train, y_train, X_val, y_val,
                               all_cols, feat_df_val, timeframe)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    logger.success(
        f"Optuna [{symbol}]: best Sharpe={best.value:.3f} "
        f"after {len(study.trials)} trials"
    )

    # Reconstruct best feature list
    best_cols = select_features_from_params(all_cols, best.params)

    return {
        "feature_cols":      best_cols,
        "tft_params":        {k: best.params[k] for k in
                              ["seq_len","d_model","n_heads","n_lstm_layers",
                               "dropout","batch_size","lr"]
                              if k in best.params},
        "signal_threshold":  best.params.get("signal_thresh", 0.36),
        "top_pct":           best.params.get("top_pct", 0.15),
        "filter_params": {
            "volume_pct":    best.params.get("volume_pct",  0.40),
            "adx_threshold": best.params.get("adx_thresh",  18.0),
            "volatility_pct":best.params.get("vol_pct",     0.10),
        },
        "atr_params": {
            "atr_stop_mult": best.params.get("atr_stop_mult", 1.2),
            "atr_tp_mult":   best.params.get("atr_stop_mult", 1.2) * 2.0,
        },
        "best_sharpe":       best.value,
        "n_trials":          len(study.trials),
    }


def select_features_from_params(all_cols: list, params: dict) -> list:
    """Reconstruct selected feature columns from saved Optuna params."""
    selected = set()
    for group_name, prefixes in FEATURE_GROUPS.items():
        if group_name in MANDATORY_GROUPS:
            include = True
        else:
            include = params.get(f"feat_{group_name}", True)
        if include:
            for col in all_cols:
                for prefix in prefixes:
                    if col.startswith(prefix) or col == prefix:
                        selected.add(col)
                        break
    for col in all_cols:
        if col in {"open", "high", "low", "close", "volume"}:
            selected.add(col)
    return [c for c in all_cols if c in selected]