"""
models/xgb_model.py
XGBoost classifier for directional price prediction.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib
from pathlib import Path
from loguru import logger
from xgboost import XGBClassifier
from config.settings import XGB_PARAMS, MODEL_DIR


def train(X_train, y_train, X_val, y_val, symbol: str = "model"):
    logger.info(f"Training XGBoost for {symbol}...")
    model = XGBClassifier(**XGB_PARAMS, eval_metric="mlogloss", early_stopping_rounds=30)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_acc = (model.predict(X_val) == y_val).mean()
    logger.success(f"XGBoost val accuracy: {val_acc:.4f}")
    return model


def save(model, symbol: str):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = Path(MODEL_DIR) / f"{symbol.replace('/','_')}_xgb.pkl"
    joblib.dump(model, path)
    logger.info(f"Saved XGBoost -> {path}")
    return path


def load(symbol: str):
    path = Path(MODEL_DIR) / f"{symbol.replace('/','_')}_xgb.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No XGBoost model at {path}")
    return joblib.load(path)


def predict_proba(model, X) -> np.ndarray:
    return model.predict_proba(X)
