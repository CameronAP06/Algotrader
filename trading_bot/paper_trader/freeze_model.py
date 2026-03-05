"""
freeze_model.py
───────────────
Run this LOCALLY (on your GPU machine) before deploying to Railway.
Trains a 9-model ensemble for each symbol and saves frozen weights.

Usage:
    python freeze_model.py --symbol DOGE/USD
    python freeze_model.py --symbol DOGE/USD --symbol LINK/USD
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import joblib
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import StandardScaler

from timeframe_comparison import fetch_ohlcv_timeframe
from data.feature_engineer import build_features, get_feature_columns
from models.lstm_ensemble import train_ensemble, LSTMClassifier
from config.settings import LSTM_PARAMS

TIMEFRAME    = "4h"
HISTORY_DAYS = 1825
TRAIN_RATIO  = 0.80
OUT_DIR      = Path(__file__).parent / "models"


def safe_name(symbol: str) -> str:
    return symbol.replace("/", "_").lower()


def freeze(symbol: str):
    OUT_DIR.mkdir(exist_ok=True)
    logger.info(f"\n{'='*50}\nFreezing ensemble for {symbol}\n{'='*50}")

    # 1. Fetch
    raw = fetch_ohlcv_timeframe(symbol, TIMEFRAME, history_days=HISTORY_DAYS)
    if raw is None or len(raw) < 500:
        logger.error(f"Insufficient data for {symbol}")
        return

    # 2. Features
    feat_df   = build_features(raw, timeframe=TIMEFRAME)
    feat_cols = get_feature_columns(feat_df)
    X_all     = feat_df[feat_cols].values.astype(np.float32)
    y_all     = feat_df["label"].values.astype(int)
    split     = int(len(X_all) * TRAIN_RATIO)

    X_train, X_val = X_all[:split], X_all[split:]
    y_train, y_val = y_all[:split], y_all[split:]
    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Features: {len(feat_cols)}")

    # 3. Scaler — fit on train only
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s   = scaler.transform(X_val)

    # 4. Train 9-model ensemble
    models = train_ensemble(X_train_s, y_train, X_val_s, y_val, symbol=symbol)

    # 5. Save all 9 model weights
    name = safe_name(symbol)
    for i, model in enumerate(models):
        torch.save(model.state_dict(), OUT_DIR / f"lstm_{name}_{i}.pt")

    # 6. Save shared info (architecture + feature cols)
    m = models[0]
    joblib.dump({
        "input_size":   m.lstm.input_size,
        "hidden_size":  m.lstm.hidden_size,
        "num_layers":   m.lstm.num_layers,
        "dropout":      LSTM_PARAMS["dropout"],
        "seq_len":      LSTM_PARAMS["sequence_length"],
        "n_models":     len(models),
        "symbol":       symbol,
        "timeframe":    TIMEFRAME,
        "feature_cols": feat_cols,
    }, OUT_DIR / f"lstm_{name}_info.pkl")

    # 7. Save scaler
    joblib.dump(scaler, OUT_DIR / f"scaler_{name}.pkl")

    logger.success(
        f"Saved {len(models)}-model ensemble for {symbol}:\n"
        f"  {len(models)} x lstm_{name}_N.pt\n"
        f"  lstm_{name}_info.pkl\n"
        f"  scaler_{name}.pkl"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", action="append", default=["DOGE/USD"])
    args    = parser.parse_args()
    symbols = list(dict.fromkeys(args.symbol))

    logger.info(f"Freezing ensemble for: {symbols}")
    for sym in symbols:
        freeze(sym)

    logger.info(f"\nDone. Commit all files in {OUT_DIR} to your Railway repo.")


if __name__ == "__main__":
    main()
