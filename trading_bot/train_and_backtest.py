#!/usr/bin/env python3
"""
train_and_backtest.py
═══════════════════════════════════════════════════════════════════════════════
MAIN ENTRY POINT — Phase 1

Steps:
  1. Fetch historical OHLCV data from Kraken (no API key needed)
  2. Engineer 120+ features
  3. Train XGBoost + CatBoost + 9-model multi-stream LSTM ensemble
  4. Optimise trio blend weights on validation set (Differential Evolution)
  5. Generate trading signals on test set using val-calibrated threshold
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

from config.settings import TRADING_PAIRS, LOG_DIR, TIMEFRAME
from data.kraken_fetcher import fetch_all_pairs
from data.feature_engineer import build_features, get_feature_columns
from utils.splitter import time_split, save_scaler
from models.xgb_model import train_xgb_model, predict_proba_xgb
from models.catboost_model import train_catboost_model, predict_proba_catboost
from models.lstm_ensemble import train_ensemble, predict_proba_ensemble
from models.ensemble import (
    blend_ensemble_trio, optimise_trio_weights,
    compute_signal_threshold, generate_signals, save_weights
)
from backtest.engine import BacktestEngine
from backtest.filters import apply_filters
from backtest.plot_results import plot_all


# ─── Setup Logging ──────────────────────────────────────────────────────────

os.makedirs(LOG_DIR, exist_ok=True)
logger.add(f"{LOG_DIR}/train_{{time}}.log", rotation="10 MB", level="INFO")


# ─── Per-Symbol Pipeline ────────────────────────────────────────────────────

def run_pipeline(symbol: str, raw_df: pd.DataFrame,
                 timeframe: str = TIMEFRAME) -> tuple:
    """
    Full pipeline for one trading pair.
    Returns (metrics, equity_curve, trades).
    """
    logger.info(f"\n{'='*60}\nProcessing: {symbol}\n{'='*60}")

    # 1. Feature Engineering (uses cache when available)
    feat_df      = build_features(raw_df, symbol, timeframe=timeframe)
    feature_cols = get_feature_columns(feat_df)
    logger.info(f"Features: {len(feature_cols)} columns")

    # 2. Train/Val/Test Split (time-ordered, no leakage)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = time_split(feat_df, feature_cols)
    save_scaler(scaler, symbol)

    # 3. Train Trio Ensemble
    lstm_models = train_ensemble(X_train, y_train, X_val, y_val,
                                  symbol=symbol, feature_cols=feature_cols)
    xgb_model   = train_xgb_model(X_train, y_train, X_val, y_val, symbol=symbol)
    cat_model   = train_catboost_model(X_train, y_train, X_val, y_val, symbol=symbol)

    # 4. Validation Probabilities
    lstm_val_p = predict_proba_ensemble(lstm_models, X_val)
    xgb_val_p  = predict_proba_xgb(xgb_model, X_val)
    cat_val_p  = predict_proba_catboost(cat_model, X_val)

    # 5. Optimise Blend Weights + Val-Calibrated Threshold
    best_weights  = optimise_trio_weights(lstm_val_p, xgb_val_p, cat_val_p, y_val)
    save_weights(best_weights, symbol)
    val_blended   = blend_ensemble_trio(lstm_val_p, xgb_val_p, cat_val_p, best_weights)
    val_threshold = compute_signal_threshold(val_blended, top_pct=0.30)
    logger.info(f"Val-calibrated threshold: {val_threshold:.4f} | "
                f"weights: lstm={best_weights[0]:.3f} xgb={best_weights[1]:.3f} "
                f"catboost={best_weights[2]:.3f}")

    # 6. Test Set Predictions
    lstm_test_p = predict_proba_ensemble(lstm_models, X_test)
    xgb_test_p  = predict_proba_xgb(xgb_model, X_test)
    cat_test_p  = predict_proba_catboost(cat_model, X_test)

    blended = blend_ensemble_trio(lstm_test_p, xgb_test_p, cat_test_p, best_weights)
    signals = generate_signals(blended, use_percentile=False, threshold=val_threshold,
                                symbol=symbol)

    # 7. Evaluate Signal Quality
    preds       = blended.argmax(axis=1)
    test_labels = y_test[-len(preds):]
    test_acc    = (preds == test_labels).mean()
    logger.info(f"Test accuracy: {test_acc:.4f}")

    # 8. Apply confirmation filters then backtest
    test_df  = feat_df.tail(len(preds)).reset_index(drop=True)
    filtered = apply_filters(test_df, signals, timeframe=timeframe)

    engine  = BacktestEngine()
    metrics = engine.run(test_df, filtered, symbol, timeframe=timeframe)
    engine.save_results(metrics, symbol)

    return metrics, engine.equity_curve, engine.trades


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train and backtest ML trading bot")
    parser.add_argument("--refresh",   action="store_true", help="Force re-download data")
    parser.add_argument("--symbol",    type=str, default=None, help="Run single symbol (e.g. BTC/USD)")
    parser.add_argument("--timeframe", type=str, default=TIMEFRAME, help="Candle timeframe (default from settings)")
    parser.add_argument("--no-plot",   action="store_true", help="Skip chart output")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else TRADING_PAIRS

    # Fetch data
    raw_data = fetch_all_pairs(force_refresh=args.refresh)
    all_metrics = []
    all_equity  = {}
    all_trades  = {}

    for symbol in symbols:
        if symbol not in raw_data:
            logger.warning(f"No data for {symbol}, skipping")
            continue
        try:
            metrics, equity, trades = run_pipeline(symbol, raw_data[symbol],
                                                   timeframe=args.timeframe)
            all_metrics.append(metrics)
            all_equity[metrics["symbol"]] = equity
            all_trades[metrics["symbol"]] = trades
        except Exception as e:
            logger.error(f"Pipeline failed for {symbol}: {e}")
            import traceback; traceback.print_exc()

    # Enrich metrics with full computed stats (Sortino, EV, streaks, etc.)
    if all_metrics:
        from backtest.plot_results import compute_full_metrics
        enriched_metrics = {}
        for m in all_metrics:
            sym    = m["symbol"]
            equity = all_equity.get(sym, [])
            trades = all_trades.get(sym, [])
            if equity:
                full = compute_full_metrics(equity, trades)
                full["symbol"] = sym
                enriched_metrics[sym] = full
            else:
                enriched_metrics[sym] = m

        # Summary table — logged AND printed so it appears in .log file
        sep  = "=" * 80
        sep2 = "-" * 80
        hdr  = (f"{'Symbol':<12} {'Return':>10} {'Sharpe':>8} {'Sortino':>8} "
                f"{'MaxDD':>8} {'WinRate':>8} {'Trades':>7} {'EV/Trade':>9}")
        rows = []
        for m in sorted(enriched_metrics.values(),
                        key=lambda x: x.get("sharpe_ratio", 0), reverse=True):
            ev  = m.get("expected_value", 0)
            row = (
                f"{m['symbol']:<12} "
                f"{m['total_return']:>9.2%} "
                f"{m['sharpe_ratio']:>8.2f} "
                f"{m.get('sortino_ratio', 0):>8.2f} "
                f"{m['max_drawdown']:>7.2%} "
                f"{m['win_rate']:>7.2%} "
                f"{int(m['n_trades']):>7} "
                f"${ev:>7.2f}"
            )
            rows.append(row)
        summary_lines = [sep, "BACKTEST SUMMARY", sep, hdr, sep2] + rows + [sep]
        summary_str = "\n".join(summary_lines)
        print("\n" + summary_str)
        logger.info("\n" + summary_str)

    if not args.no_plot and all_metrics:
        metrics_dict = {m["symbol"]: m for m in all_metrics}
        plot_all(
            results=metrics_dict,
            equity_curves=all_equity,
            trades_by_symbol=all_trades,
        )


if __name__ == "__main__":
    main()