#!/usr/bin/env python3
"""
train_and_backtest.py
═══════════════════════════════════════════════════════════════════════════════
MAIN ENTRY POINT — Phase 1

Steps:
  1. Fetch historical OHLCV data from Kraken (no API key needed)
  2. Engineer 80+ features
  3. Train CatBoost + 1D CNN + LSTM ensemble
  4. Optimise ensemble weights on validation set
  5. Generate trading signals on test set
  6. Run backtest with realistic fees and risk management
  7. Print + save performance metrics

Usage:
  python train_and_backtest.py
  python train_and_backtest.py --refresh   # Force re-download data
  python train_and_backtest.py --symbol BTC/USD  # Single pair
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
import pandas as pd
import numpy as np

from config.settings import TRADING_PAIRS, LOG_DIR
from data.kraken_fetcher import fetch_all_pairs
from data.feature_engineer import build_features, get_feature_columns
from utils.splitter import time_split, save_scaler
from models import catboost_model, cnn_model, lstm_model
from models.ensemble import (
    weighted_ensemble, optimise_weights,
    generate_signals, save_weights
)
from backtest.engine import BacktestEngine
from backtest.plot_results import plot_all


# ─── Setup Logging ──────────────────────────────────────────────────────────

os.makedirs(LOG_DIR, exist_ok=True)
logger.add(f"{LOG_DIR}/train_{{time}}.log", rotation="10 MB", level="INFO")


# ─── Per-Symbol Pipeline ────────────────────────────────────────────────────

def run_pipeline(symbol: str, raw_df: pd.DataFrame) -> dict:
    """Full pipeline for one trading pair."""
    logger.info(f"\n{'='*60}\nProcessing: {symbol}\n{'='*60}")

    # 1. Feature Engineering
    feat_df = build_features(raw_df, symbol)
    feature_cols = get_feature_columns(feat_df)
    logger.info(f"Features: {len(feature_cols)} columns")

    # 2. Train/Val/Test Split (time-ordered, no leakage)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = time_split(feat_df, feature_cols)
    save_scaler(scaler, symbol)

    # 3. Train Individual Models
    cat = catboost_model.train(X_train, y_train, X_val, y_val, symbol)
    catboost_model.save(cat, symbol)

    cnn = cnn_model.train(X_train, y_train, X_val, y_val, symbol)
    cnn_model.save(cnn, symbol)

    lstm = lstm_model.train(X_train, y_train, X_val, y_val, symbol)
    lstm_model.save(lstm, symbol)

    # 4. Validation Probabilities
    cat_val_p  = catboost_model.predict_proba(cat, X_val)
    cnn_val_p  = cnn_model.predict_proba(cnn,  X_val)
    lstm_val_p = lstm_model.predict_proba(lstm,  X_val)

    # 5. Optimise Ensemble Weights on Validation Set
    best_weights = optimise_weights(cat_val_p, cnn_val_p, lstm_val_p, y_val)
    save_weights(best_weights, symbol)

    # 6. Test Set Predictions
    cat_test_p  = catboost_model.predict_proba(cat, X_test)
    cnn_test_p  = cnn_model.predict_proba(cnn,  X_test)
    lstm_test_p = lstm_model.predict_proba(lstm,  X_test)

    blended = weighted_ensemble(cat_test_p, cnn_test_p, lstm_test_p, best_weights)
    signals = generate_signals(blended)

    # 7. Evaluate Signal Quality
    preds      = blended.argmax(axis=1)
    test_labels = y_test[-len(preds):]
    test_acc   = (preds == test_labels).mean()
    logger.info(f"Test accuracy: {test_acc:.4f}")

    # 8. Backtest
    test_df = feat_df.tail(len(preds)).reset_index(drop=True)
    engine  = BacktestEngine()
    metrics = engine.run(test_df, signals, symbol)
    engine.save_results(metrics, symbol)

    return metrics


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train and backtest ML trading bot")
    parser.add_argument("--refresh", action="store_true",  help="Force re-download data")
    parser.add_argument("--symbol",  type=str, default=None, help="Run single symbol (e.g. BTC/USD)")
    parser.add_argument("--no-plot", action="store_true",  help="Skip chart output")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else TRADING_PAIRS

    # Fetch data
    raw_data = fetch_all_pairs(force_refresh=args.refresh)
    all_metrics = []

    for symbol in symbols:
        if symbol not in raw_data:
            logger.warning(f"No data for {symbol}, skipping")
            continue
        try:
            metrics = run_pipeline(symbol, raw_data[symbol])
            all_metrics.append(metrics)
        except Exception as e:
            logger.error(f"Pipeline failed for {symbol}: {e}")
            import traceback; traceback.print_exc()

    # Summary Table
    if all_metrics:
        print("\n" + "="*70)
        print("BACKTEST SUMMARY")
        print("="*70)
        headers = ["Symbol", "Return", "Sharpe", "MaxDD", "WinRate", "Trades"]
        print(f"{'Symbol':<12} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>9} {'Trades':>7}")
        print("-"*70)
        for m in all_metrics:
            print(
                f"{m['symbol']:<12} "
                f"{m['total_return']:>9.2%} "
                f"{m['sharpe_ratio']:>8.2f} "
                f"{m['max_drawdown']:>7.2%} "
                f"{m['win_rate']:>8.2%} "
                f"{int(m['n_trades']):>7}"
            )
        print("="*70)

    if not args.no_plot and all_metrics:
        plot_all(symbols)


if __name__ == "__main__":
    main()