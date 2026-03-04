"""
freeze_model.py
───────────────
Run this LOCALLY (on your GPU machine) before deploying to Railway.

Usage:
    python freeze_model.py --symbol DOGE/USD
    python freeze_model.py --symbol LINK/USD
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
from models.lstm_model import train as train_lstm
from config.settings import LSTM_PARAMS

TIMEFRAME    = "4h"
HISTORY_DAYS = 1825
TRAIN_RATIO  = 0.80
OUT_DIR      = Path(__file__).parent / "models"


def safe_name(symbol: str) -> str:
    return symbol.replace("/", "_").lower()


def freeze(symbol: str):
    OUT_DIR.mkdir(exist_ok=True)
    logger.info(f"\n{'='*50}\nFreezing {symbol}\n{'='*50}")

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

    # 3. Scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s   = scaler.transform(X_val)

    # 4. Train
    model = train_lstm(X_train_s, y_train, X_val_s, y_val, symbol=symbol)

    # 5. Save — all files named by symbol e.g. lstm_doge_usd.pt
    name = safe_name(symbol)

    torch.save(model.state_dict(), OUT_DIR / f"lstm_{name}.pt")
    joblib.dump({
        "input_size":   model.lstm.input_size,
        "hidden_size":  model.lstm.hidden_size,
        "num_layers":   model.lstm.num_layers,
        "dropout":      LSTM_PARAMS["dropout"],
        "seq_len":      LSTM_PARAMS["sequence_length"],
        "symbol":       symbol,
        "timeframe":    TIMEFRAME,
        "feature_cols": feat_cols,
    }, OUT_DIR / f"lstm_{name}_info.pkl")
    joblib.dump(scaler, OUT_DIR / f"scaler_{name}.pkl")

    logger.success(f"Saved: lstm_{name}.pt | lstm_{name}_info.pkl | scaler_{name}.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", action="append",
                        default=["DOGE/USD"],
                        help="Symbol(s) to freeze. Can be repeated.")
    args = parser.parse_args()

    # deduplicate while preserving order
    symbols = list(dict.fromkeys(args.symbol))
    logger.info(f"Freezing {len(symbols)} symbol(s): {symbols}")

    for sym in symbols:
        freeze(sym)

    logger.info("\nAll models frozen and ready for deployment.")
    logger.info(f"Commit the contents of {OUT_DIR} to your Railway repo.")


if __name__ == "__main__":
    main()
