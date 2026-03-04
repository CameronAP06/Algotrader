"""
freeze_model.py
───────────────
Run this LOCALLY (on your GPU machine) before deploying to Railway.

This script:
  1. Fetches DOGE/USD 4h history from Binance
  2. Trains the LSTM on 80% of the data (train+val split)
  3. Saves the frozen model weights + scaler + feature info
     to paper_trader/models/ ready for deployment

The cloud service loads these frozen files — it never retrains.

Usage:
    python freeze_model.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import joblib
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import StandardScaler

# Import from your existing trading bot
from timeframe_comparison import fetch_ohlcv_timeframe
from data.feature_engineer import build_features, get_feature_columns
from models.lstm_model import train as train_lstm, LSTMClassifier
from config.settings import LSTM_PARAMS

SYMBOL       = "DOGE/USD"
TIMEFRAME    = "4h"
HISTORY_DAYS = 1825   # 5 years
TRAIN_RATIO  = 0.80   # Use 80% for training (no test split — we've already validated)
OUT_DIR      = Path(__file__).parent / "models"


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # ── 1. Fetch data ─────────────────────────────────────────────────────────
    logger.info(f"Fetching {HISTORY_DAYS} days of {SYMBOL} {TIMEFRAME}...")
    raw = fetch_ohlcv_timeframe(SYMBOL, TIMEFRAME, history_days=HISTORY_DAYS)
    if raw is None or len(raw) < 500:
        logger.error("Insufficient data")
        return

    # ── 2. Feature engineering ────────────────────────────────────────────────
    feat_df   = build_features(raw, timeframe=TIMEFRAME)
    feat_cols = get_feature_columns(feat_df)

    X_all = feat_df[feat_cols].values.astype(np.float32)
    y_all = feat_df["label"].values.astype(int)
    n     = len(X_all)

    split = int(n * TRAIN_RATIO)
    X_train, X_val = X_all[:split], X_all[split:]
    y_train, y_val = y_all[:split], y_all[split:]

    logger.info(f"Train: {len(X_train)} bars | Val: {len(X_val)} bars")
    logger.info(f"Features: {len(feat_cols)}")

    # ── 3. Fit scaler on train only ───────────────────────────────────────────
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s   = scaler.transform(X_val)

    # ── 4. Train LSTM ─────────────────────────────────────────────────────────
    logger.info("Training LSTM...")
    model = train_lstm(X_train_s, y_train, X_val_s, y_val, symbol=SYMBOL)

    # ── 5. Save everything ────────────────────────────────────────────────────
    # Model weights
    weights_path = OUT_DIR / "lstm_doge.pt"
    torch.save(model.state_dict(), weights_path)
    logger.success(f"Saved weights → {weights_path}")

    # Model architecture info
    info = {
        "input_size":  model.lstm.input_size,
        "hidden_size": model.lstm.hidden_size,
        "num_layers":  model.lstm.num_layers,
        "dropout":     LSTM_PARAMS["dropout"],
        "seq_len":     LSTM_PARAMS["sequence_length"],
        "symbol":      SYMBOL,
        "timeframe":   TIMEFRAME,
        "n_features":  len(feat_cols),
        "feature_cols":feat_cols,
    }
    info_path = OUT_DIR / "lstm_info.pkl"
    joblib.dump(info, info_path)
    logger.success(f"Saved model info → {info_path}")

    # Scaler
    scaler_path = OUT_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.success(f"Saved scaler → {scaler_path}")

    logger.info("\n" + "="*50)
    logger.info("Model frozen and ready for deployment.")
    logger.info(f"Copy the contents of {OUT_DIR} to your Railway repo.")
    logger.info("="*50)


if __name__ == "__main__":
    main()
