"""
models/lgbm_model.py
LightGBM classifier for directional price prediction.
Fast, handles tabular features natively, highly interpretable.
Drop-in replacement / addition to the ensemble alongside TFT/CNN/LSTM.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib
from pathlib import Path
from loguru import logger
from config.settings import MODEL_DIR

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("LightGBM not installed — run: pip install lightgbm")

LGBM_PARAMS = {
    "objective":        "multiclass",
    "num_class":        3,
    "metric":           "multi_logloss",
    "n_estimators":     500,
    "learning_rate":    0.05,
    "num_leaves":       31,
    "max_depth":        6,
    "min_child_samples":20,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    # No class_weight here — sklearn LGBMClassifier's class_weight dict
    # does not reliably overcome majority class dominance (best_iter=1 symptom).
    # We compute sample_weight per row in train() instead — explicit and reliable.
    "random_state":     42,
    "n_jobs":           -1,
    "verbose":          -1,
}

# Per-class multipliers applied via sample_weight
CLASS_WEIGHTS = {0: 2.0, 1: 0.5, 2: 2.0}  # DOWN, NEUTRAL, UP

EARLY_STOPPING_ROUNDS = 30


def _make_sample_weights(y: np.ndarray) -> np.ndarray:
    """Compute per-sample weights inversely proportional to class frequency.
    Dynamic weighting adapts to whatever imbalance the data actually has —
    fixed weights like 2.0/0.5/2.0 are not strong enough when NEUTRAL is 75%+.
    """
    counts = np.bincount(y.astype(int), minlength=3).astype(float)
    counts = np.where(counts == 0, 1, counts)  # avoid div/0
    # Inverse frequency — minority classes get proportionally higher weight
    inv_freq = 1.0 / counts
    # Normalise so mean weight ≈ 1.0 (keeps loss scale stable)
    inv_freq = inv_freq / inv_freq.mean()
    return np.array([inv_freq[int(label)] for label in y], dtype=np.float32)


def train(X_train, y_train, X_val, y_val, symbol: str = "model"):
    logger.info(f"Training LightGBM for {symbol} — {len(X_train)} train samples, "
                f"{len(X_val)} val samples, {X_train.shape[1]} features")

    # Log training label distribution so we can verify weighting is appropriate
    unique, counts = np.unique(y_train, return_counts=True)
    dist_str = " ".join(f"class{int(k)}={v}" for k, v in zip(unique, counts))
    logger.info(f"  Train label dist: {dist_str}")

    sample_weight = _make_sample_weights(y_train)

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=50),  # Log every 50 iters so we can track progress
        ],
    )

    val_preds = model.predict(X_val)
    val_acc   = (val_preds == y_val).mean()
    best_iter = model.best_iteration_

    # Log class distribution of predictions
    unique, counts = np.unique(val_preds, return_counts=True)
    dist = {int(k): int(v) for k, v in zip(unique, counts)}
    logger.success(
        f"LightGBM val_acc={val_acc:.4f} | best_iter={best_iter} | "
        f"pred_dist: DOWN={dist.get(0,0)} NEUTRAL={dist.get(1,0)} UP={dist.get(2,0)}"
    )
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


def feature_importance(model, feature_names: list = None) -> dict:
    """Returns feature importances sorted descending."""
    imp = model.feature_importances_
    names = feature_names or [f"f{i}" for i in range(len(imp))]
    ranked = sorted(zip(names, imp), key=lambda x: -x[1])
    return {name: int(score) for name, score in ranked}