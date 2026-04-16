"""
models/xgb_model.py
───────────────────
Per-fold XGBoost tabular model for the trio ensemble.

Design goals:
  - Trains on individual bars (no sequence window) — complementary to LSTM
  - Class-weighted to handle 50%+ NEUTRAL imbalance
  - Val-set early stopping to prevent overfitting on small folds
  - Fold-level cache (pickle, keyed on training data + params hash)
  - Returns (n_bars, 3) probability array — no sequence padding offset

Usage:
    from models.xgb_model import train_xgb_model, predict_proba_xgb

    xgb_model = train_xgb_model(X_train, y_train, X_val, y_val,
                                 symbol="BTC/USD", fold_idx=0)
    proba = predict_proba_xgb(xgb_model, X_test)   # (n_bars, 3)
"""

import os, sys, hashlib, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from loguru import logger

from config.settings import XGB_TREE_PARAMS, MODEL_DIR

CACHE_DIR = Path(MODEL_DIR) / "xgb_cache"

# Version tag — bump to invalidate all caches
_XGB_VERSION = b"xgb_v1_tabular"


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _xgb_cache_key(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> str:
    h = hashlib.md5()
    h.update(str(X_train.shape).encode())
    h.update(str(y_train.shape).encode())
    stride = max(1, len(X_train) // 200)
    h.update(X_train[::stride].astype(np.float32).tobytes())
    h.update(y_train[::max(1, len(y_train) // 200)].tobytes())
    h.update(str(sorted(XGB_TREE_PARAMS.items())).encode())
    h.update(str(seed).encode())
    h.update(_XGB_VERSION)
    return h.hexdigest()[:20]


def _try_load_xgb_cache(key: str):
    """Return loaded XGB model or None on cache miss."""
    path = CACHE_DIR / f"{key}.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        logger.warning(f"XGB cache load failed ({e}), retraining")
        return None


def _save_xgb_cache(model, key: str):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / f"{key}.pkl", "wb") as f:
        pickle.dump(model, f, protocol=4)


# ── Training ──────────────────────────────────────────────────────────────────

def train_xgb_model(X_train: np.ndarray, y_train: np.ndarray,
                    X_val:   np.ndarray, y_val:   np.ndarray,
                    symbol:  str = "model",
                    fold_idx: int = 0,
                    seed: int = 42):
    """
    Train a single XGBoost classifier on tabular bar-level features.

    Key design choices vs a naive XGB fit:
      - Inverse-frequency class weights correct for dominant NEUTRAL class
        without the WeightedRandomSampler trick needed for mini-batch LSTM.
        XGB processes all samples at once so we can weight directly.
      - Val-set early stopping: XGB sees the validation set only to decide
        when to stop adding trees, NOT to optimise hyperparameters.
      - Explicit eval_set passed as separate array — no data leakage.
      - `num_class` and `objective` set for soft-probability output so
        predict_proba returns (n, 3) like the LSTM ensemble does.

    Args:
        X_train, y_train : training features and integer labels (0=DOWN,1=NEUTRAL,2=UP)
        X_val,   y_val   : validation features and labels (for early stopping)
        symbol           : for log messages only
        fold_idx         : for log messages and cache key disambiguation
        seed             : random seed

    Returns:
        Trained XGBClassifier instance with .predict_proba() method.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("xgboost not installed — run: pip install xgboost")

    key = _xgb_cache_key(X_train, y_train, seed)
    cached = _try_load_xgb_cache(key)
    if cached is not None:
        logger.info(f"  XGB [{symbol} fold={fold_idx}] Cache HIT (key={key[:8]}…)")
        return cached

    # Inverse-frequency class weights: NEUTRAL gets weight 1.0, UP/DOWN get
    # weight proportional to how underrepresented they are.
    counts = np.bincount(y_train, minlength=3).astype(float)
    counts = np.maximum(counts, 1.0)
    inv_freq = counts.sum() / (3.0 * counts)   # normalise so mean weight ≈ 1.0
    sample_weights = inv_freq[y_train]

    params = dict(XGB_TREE_PARAMS)
    params["random_state"] = seed

    # early_stopping_rounds must be passed to constructor in newer xgboost
    early_stopping = params.pop("early_stopping_rounds", 50)

    model = XGBClassifier(**params, early_stopping_rounds=early_stopping)

    logger.info(
        f"  XGB [{symbol} fold={fold_idx} seed={seed}] Training on "
        f"{len(X_train)} bars, val on {len(X_val)} bars, "
        f"{X_train.shape[1]} features"
    )
    logger.info(
        f"  XGB class weights: DOWN={inv_freq[0]:.3f} "
        f"NEUTRAL={inv_freq[1]:.3f} UP={inv_freq[2]:.3f}"
    )

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best_iter = getattr(model, "best_iteration", None)
    logger.info(
        f"  XGB [{symbol} fold={fold_idx}] Done — "
        f"best_iteration={best_iter}"
    )

    _save_xgb_cache(model, key)
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_proba_xgb(model, X: np.ndarray) -> np.ndarray:
    """
    Return (n_bars, 3) probability array for all bars.

    Unlike LSTM, XGB operates bar-by-bar (no sequence window), so there is
    no padding offset — all n_bars get a real prediction.

    Class order is [DOWN=0, NEUTRAL=1, UP=2] because XGB is fitted on
    integer labels 0/1/2 and returns classes in sorted order.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    proba = model.predict_proba(X).astype(np.float32)

    if proba.shape[1] != 3:
        raise ValueError(
            f"predict_proba_xgb: expected 3 classes, got {proba.shape[1]}. "
            f"Check that the model was trained with num_class=3."
        )
    return proba
