"""
models/catboost_model.py
CatBoost classifier for directional price prediction.
Handles ordered/time-series data natively with built-in overfitting detection.
Drop-in replacement for lgbm_model.py.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib
from pathlib import Path
from loguru import logger
from catboost import CatBoostClassifier
from config.settings import MODEL_DIR


CATBOOST_PARAMS = {
    "iterations":        500,
    "learning_rate":     0.05,
    "depth":             6,
    "l2_leaf_reg":       3.0,
    "loss_function":     "MultiClass",
    "classes_count":     3,
    "eval_metric":       "Accuracy",
    "class_weights":     [2.0, 0.5, 2.0],  # UP/DOWN weighted 4x NEUTRAL
    "random_seed":       42,
    "thread_count":      -1,
    "verbose":           False,
    "early_stopping_rounds": 30,
}


def train(X_train, y_train, X_val, y_val, symbol: str = "model"):
    logger.info(f"Training CatBoost for {symbol}...")
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )
    val_acc = (model.predict(X_val).flatten() == y_val).mean()
    logger.success(f"CatBoost val accuracy: {val_acc:.4f}")
    return model


def save(model, symbol: str):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = Path(MODEL_DIR) / f"{symbol.replace('/','_')}_catboost.cbm"
    model.save_model(str(path))
    logger.info(f"Saved CatBoost -> {path}")
    return path


def load(symbol: str):
    path = Path(MODEL_DIR) / f"{symbol.replace('/','_')}_catboost.cbm"
    if not path.exists():
        raise FileNotFoundError(f"No CatBoost model at {path}")
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def predict_proba(model, X) -> np.ndarray:
    """Returns (n_samples, 3) probability array: [DOWN, NEUTRAL, UP]"""
    return model.predict_proba(X)