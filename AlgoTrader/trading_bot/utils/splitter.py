"""
utils/splitter.py
Walk-forward train/val/test split that respects time ordering.
NEVER shuffles data — future data must never leak into the past.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from loguru import logger
from config.settings import TRAIN_RATIO, VAL_RATIO, MODEL_DIR


def time_split(df: pd.DataFrame, feature_cols: list):
    """
    Split DataFrame chronologically into train/val/test.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test, scaler
    """
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    X = df[feature_cols].values
    y = df["label"].values

    X_train, y_train = X[:train_end],       y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:],          y[val_end:]

    # Fit scaler on TRAIN only — no data leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    logger.info(
        f"Split: train={len(X_train)} | val={len(X_val)} | test={len(X_test)}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def save_scaler(scaler, symbol):
    import os
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = Path(MODEL_DIR) / f"{symbol.replace('/','_')}_scaler.pkl"
    joblib.dump(scaler, path)
    logger.info(f"Saved scaler -> {path}")


def load_scaler(symbol):
    path = Path(MODEL_DIR) / f"{symbol.replace('/','_')}_scaler.pkl"
    return joblib.load(path)
