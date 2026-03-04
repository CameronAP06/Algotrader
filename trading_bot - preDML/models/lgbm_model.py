"""
models/lgbm_model.py
LightGBM classifier for directional price prediction.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from loguru import logger
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from config.settings import LGBM_PARAMS, MODEL_DIR


def train(X_train, y_train, X_val, y_val, symbol: str = "model"):
    logger.info(f"Training LightGBM for {symbol}...")
    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[],
    )
    val_acc = (model.predict(X_val) == y_val).mean()
    logger.success(f"LightGBM val accuracy: {val_acc:.4f}")
    return model


def save(model, symbol: str):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = Path(MODEL_DIR) / f"{symbol.replace('/','_')}_lgbm.pkl"
    joblib.dump(model, path)
    logger.info(f"Saved LightGBM -> {path}")
    return path


def load(symbol: str):
    path = Path(MODEL_DIR) / f"{symbol.replace('/','_')}_lgbm.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No LightGBM model at {path}")
    return joblib.load(path)


def predict_proba(model, X) -> np.ndarray:
    """Returns (n_samples, 3) probability array: [DOWN, NEUTRAL, UP]"""
    return model.predict_proba(X)
