"""
models/catboost_model.py
────────────────────────
Per-fold CatBoost tabular model for the trio ensemble.

Design goals:
  - Trains on individual bars (no sequence window) — complementary to LSTM
  - auto_class_weights="Balanced" handles NEUTRAL imbalance natively
  - Ordered boosting mode captures time-series structure without leaking
    future information (each tree is fitted on a subset of earlier rows)
  - Val-set early stopping prevents overfitting on small folds
  - Fold-level cache (pickle, keyed on training data + params hash)
  - Returns (n_bars, 3) probability array — no sequence padding offset

Usage:
    from models.catboost_model import train_catboost_model, predict_proba_catboost

    cat_model = train_catboost_model(X_train, y_train, X_val, y_val,
                                      symbol="BTC/USD", fold_idx=0)
    proba = predict_proba_catboost(cat_model, X_test)   # (n_bars, 3)
"""

import os, sys, hashlib, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from loguru import logger

from config.settings import CATBOOST_PARAMS, MODEL_DIR

CACHE_DIR = Path(MODEL_DIR) / "catboost_cache"

# Version tag — bump to invalidate all caches
_CAT_VERSION = b"catboost_v1_tabular"


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cat_cache_key(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> str:
    h = hashlib.md5()
    h.update(str(X_train.shape).encode())
    h.update(str(y_train.shape).encode())
    stride = max(1, len(X_train) // 200)
    h.update(X_train[::stride].astype(np.float32).tobytes())
    h.update(y_train[::max(1, len(y_train) // 200)].tobytes())
    h.update(str(sorted(CATBOOST_PARAMS.items())).encode())
    h.update(str(seed).encode())
    h.update(_CAT_VERSION)
    return h.hexdigest()[:20]


def _try_load_cat_cache(key: str):
    """Return loaded CatBoost model or None on cache miss."""
    path = CACHE_DIR / f"{key}.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        logger.warning(f"CatBoost cache load failed ({e}), retraining")
        return None


def _save_cat_cache(model, key: str):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / f"{key}.pkl", "wb") as f:
        pickle.dump(model, f, protocol=4)


# ── Training ──────────────────────────────────────────────────────────────────

def train_catboost_model(X_train: np.ndarray, y_train: np.ndarray,
                         X_val:   np.ndarray, y_val:   np.ndarray,
                         symbol:  str = "model",
                         fold_idx: int = 0,
                         seed: int = 42):
    """
    Train a single CatBoost classifier on tabular bar-level features.

    Key design choices:
      - auto_class_weights="Balanced": CatBoost re-weights internally so
        UP/DOWN classes contribute equal gradient to NEUTRAL (~50% of bars).
        This is the CatBoost-native equivalent of XGB's sample_weights trick.
      - No categorical features: all inputs are pre-engineered floats.
        CatBoost's main advantage here is its regularised symmetric trees
        and built-in overfitting detector (early stopping on eval_set).
      - random_seed per call: different seeds give different local minima,
        allowing the ensemble to benefit from diversity across tree learners.
      - Pool construction: CatBoost's Pool object avoids redundant copies and
        enables efficient eval_set tracking.

    Args:
        X_train, y_train : training features and integer labels (0=DOWN,1=NEUTRAL,2=UP)
        X_val,   y_val   : validation features and labels (for early stopping)
        symbol           : for log messages only
        fold_idx         : for log messages and cache key disambiguation
        seed             : random seed

    Returns:
        Trained CatBoostClassifier instance with .predict_proba() method.
    """
    try:
        from catboost import CatBoostClassifier, Pool
    except ImportError:
        raise ImportError("catboost not installed — run: pip install catboost")

    key = _cat_cache_key(X_train, y_train, seed)
    cached = _try_load_cat_cache(key)
    if cached is not None:
        logger.info(f"  CatBoost [{symbol} fold={fold_idx}] Cache HIT (key={key[:8]}…)")
        return cached

    params = dict(CATBOOST_PARAMS)
    params["random_seed"] = seed

    model = CatBoostClassifier(**params)

    train_pool = Pool(X_train, label=y_train)
    val_pool   = Pool(X_val,   label=y_val)

    counts = np.bincount(y_train, minlength=3).astype(int)
    logger.info(
        f"  CatBoost [{symbol} fold={fold_idx} seed={seed}] Training on "
        f"{len(X_train)} bars, val on {len(X_val)} bars, "
        f"{X_train.shape[1]} features | "
        f"DOWN={counts[0]} NEUTRAL={counts[1]} UP={counts[2]}"
    )

    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
    )

    best_iter = model.get_best_iteration()
    logger.info(
        f"  CatBoost [{symbol} fold={fold_idx}] Done — "
        f"best_iteration={best_iter}"
    )

    _save_cat_cache(model, key)
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_proba_catboost(model, X: np.ndarray) -> np.ndarray:
    """
    Return (n_bars, 3) probability array for all bars.

    Unlike LSTM, CatBoost operates bar-by-bar (no sequence window), so there
    is no padding offset — all n_bars get a real prediction.

    Class order is [DOWN=0, NEUTRAL=1, UP=2] matching the integer label
    encoding used during training.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    proba = model.predict_proba(X).astype(np.float32)

    if proba.shape[1] != 3:
        raise ValueError(
            f"predict_proba_catboost: expected 3 classes, got {proba.shape[1]}. "
            f"Check that the model was trained with loss_function=MultiClass."
        )
    return proba
