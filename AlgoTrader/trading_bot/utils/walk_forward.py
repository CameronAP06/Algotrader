"""
utils/walk_forward.py
Walk-forward validation for the ML trading ensemble.

Instead of a single train/test split, this rolls a training window forward
through time, testing on each subsequent period independently. This gives
20-30 test periods instead of one, making it possible to determine whether
the model consistently beats random chance or just got lucky once.

Walk-forward schedule (example with 1825 days of data, hourly bars):
  Fold 1:  Train on bars 0-21900,    Test on bars 21900-24528  (110 days)
  Fold 2:  Train on bars 2628-24528, Test on bars 24528-27156
  ...
  Fold N:  Train on bars N*2628-..., Test on bars ...-43800

Each fold:
  1. Trains all three models fresh
  2. Optimises ensemble weights on a val split within the training window
  3. Generates signals and backtests on the test window
  4. Records accuracy, return, Sharpe, win rate

Final output: distribution of metrics across all folds — if median Sharpe > 0
and accuracy > 36%, the models are learning something real.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from loguru import logger
from typing import List, Dict

from data.feature_engineer import build_features, get_feature_columns
from utils.splitter import save_scaler
from sklearn.preprocessing import StandardScaler
from models import tft_model, cnn_model, lstm_model
from models.ensemble import weighted_ensemble, optimise_weights, generate_signals
from backtest.engine import BacktestEngine
from backtest.filters import apply_filters


# ─── Walk-Forward Config ─────────────────────────────────────────────────────

WF_CONFIG = {
    "train_bars":  17520,   # 2 years of hourly bars for training
    "val_bars":    2628,    # ~110 days for validation (within train window)
    "test_bars":   2628,    # ~110 days for testing per fold
    "step_bars":   2628,    # Advance window by 110 days each fold
    "min_folds":   3,       # Minimum folds to report meaningful stats
}


# ─── Single Fold ─────────────────────────────────────────────────────────────

def run_fold(feat_df: pd.DataFrame, feature_cols: list,
             train_start: int, train_end: int,
             val_end: int, test_end: int,
             fold_num: int, symbol: str) -> Dict:
    """Run a single walk-forward fold."""

    X = feat_df[feature_cols].values
    y = feat_df["label"].values

    X_train = X[train_start:train_end]
    y_train = y[train_start:train_end]
    X_val   = X[train_end:val_end]
    y_val   = y[train_end:val_end]
    X_test  = X[val_end:test_end]
    y_test  = y[val_end:test_end]

    if len(X_train) < 5000 or len(X_test) < 100:
        logger.warning(f"Fold {fold_num}: insufficient data, skipping")
        return {}

    # Scale — fit on train only
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    logger.info(f"Fold {fold_num}: train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    # Train models
    try:
        tft  = tft_model.train(X_train, y_train, X_val, y_val, f"{symbol}_f{fold_num}")
        cnn  = cnn_model.train(X_train, y_train, X_val, y_val, f"{symbol}_f{fold_num}")
        lstm = lstm_model.train(X_train, y_train, X_val, y_val, f"{symbol}_f{fold_num}")
    except Exception as e:
        logger.error(f"Fold {fold_num} training failed: {e}")
        return {}

    # Ensemble
    tft_val_p  = tft_model.predict_proba(tft,  X_val)
    cnn_val_p  = cnn_model.predict_proba(cnn,  X_val)
    lstm_val_p = lstm_model.predict_proba(lstm, X_val)
    weights    = optimise_weights(tft_val_p, cnn_val_p, lstm_val_p, y_val)

    tft_test_p  = tft_model.predict_proba(tft,  X_test)
    cnn_test_p  = cnn_model.predict_proba(cnn,  X_test)
    lstm_test_p = lstm_model.predict_proba(lstm, X_test)

    blended = weighted_ensemble(tft_test_p, cnn_test_p, lstm_test_p, weights)
    signals = generate_signals(blended)

    # Accuracy
    preds    = blended.argmax(axis=1)
    y_test_a = y_test[-len(preds):]
    accuracy = (preds == y_test_a).mean()

    # Backtest
    test_df  = feat_df.iloc[val_end:test_end].reset_index(drop=True)
    test_df  = test_df.tail(len(preds)).reset_index(drop=True)
    filtered = apply_filters(test_df, signals)

    engine  = BacktestEngine()
    metrics = engine.run(test_df, filtered, symbol)

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
                          config: dict = WF_CONFIG) -> pd.DataFrame:
    """
    Run full walk-forward validation for one symbol.
    Returns a DataFrame with one row per fold.
    """
    logger.info(f"\n{'='*60}\nWalk-Forward Validation: {symbol}\n{'='*60}")

    feat_df      = build_features(raw_df, symbol)
    feature_cols = get_feature_columns(feat_df)
    n            = len(feat_df)

    train_bars = config["train_bars"]
    val_bars   = config["val_bars"]
    test_bars  = config["test_bars"]
    step_bars  = config["step_bars"]

    # First test window starts after initial training + validation
    first_test_start = train_bars + val_bars

    if n < first_test_start + test_bars:
        logger.error(f"Insufficient data for walk-forward: {n} bars, need {first_test_start + test_bars}")
        return pd.DataFrame()

    results = []
    fold    = 1

    test_start = first_test_start
    while test_start + test_bars <= n:
        train_start = max(0, test_start - train_bars - val_bars)
        train_end   = test_start - val_bars
        val_end     = test_start
        test_end    = test_start + test_bars

        result = run_fold(
            feat_df, feature_cols,
            train_start, train_end, val_end, test_end,
            fold, symbol
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
    """Print walk-forward results with statistical summary."""
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

    # Interpretation
    profitable_folds = (df["total_return"] > 0).sum()
    pos_sharpe_folds = (df["sharpe_ratio"] > 0).sum()
    above_random     = (df["accuracy"] > 0.36).sum()

    print(f"\nInterpretation:")
    print(f"  Profitable folds:    {profitable_folds}/{len(df)} ({profitable_folds/len(df):.0%})")
    print(f"  Positive Sharpe:     {pos_sharpe_folds}/{len(df)} ({pos_sharpe_folds/len(df):.0%})")
    print(f"  Above-random acc:    {above_random}/{len(df)} ({above_random/len(df):.0%})")

    if profitable_folds / len(df) >= 0.6 and df["sharpe_ratio"].median() > 0.5:
        print(f"\n  ✓ Model shows CONSISTENT edge — proceed to live trading")
    elif profitable_folds / len(df) >= 0.5:
        print(f"\n  ~ Model shows MARGINAL edge — needs improvement before live trading")
    else:
        print(f"\n  ✗ Model shows NO consistent edge — features or labels need rethinking")


def save_wf_results(results_df: pd.DataFrame, symbol: str):
    """Save walk-forward results to CSV."""
    os.makedirs("backtest/results", exist_ok=True)
    path = f"backtest/results/{symbol.replace('/','_')}_walkforward.csv"
    results_df.to_csv(path, index=False)
    logger.info(f"Walk-forward results saved -> {path}")
