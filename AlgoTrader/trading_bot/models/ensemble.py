"""
models/ensemble.py
Combines LightGBM + XGBoost + LSTM predictions into a single signal.
Weights are optimised automatically on the validation set.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib
from pathlib import Path
from loguru import logger
from scipy.optimize import minimize
from config.settings import ENSEMBLE_WEIGHTS, SIGNAL_THRESHOLD, MODEL_DIR


def _stack_probas(lgbm_p, xgb_p, lstm_p):
    """Align all three probability arrays to the same length (LSTM has padding)."""
    n = min(len(lgbm_p), len(xgb_p), len(lstm_p))
    return lgbm_p[-n:], xgb_p[-n:], lstm_p[-n:]


def weighted_ensemble(lgbm_p, xgb_p, lstm_p, weights=None):
    """Blend three (n, 3) probability arrays."""
    if weights is None:
        w = list(ENSEMBLE_WEIGHTS.values())
    else:
        w = weights
    lgbm_p, xgb_p, lstm_p = _stack_probas(lgbm_p, xgb_p, lstm_p)
    return w[0] * lgbm_p + w[1] * xgb_p + w[2] * lstm_p


def optimise_weights(lgbm_p, xgb_p, lstm_p, y_true):
    """
    Find ensemble weights that maximise validation accuracy
    using constrained optimisation (weights sum to 1, all >= 0).
    """
    lgbm_p, xgb_p, lstm_p = _stack_probas(lgbm_p, xgb_p, lstm_p)
    y_true = y_true[-len(lgbm_p):]

    def neg_accuracy(w):
        blended = w[0]*lgbm_p + w[1]*xgb_p + w[2]*lstm_p
        preds   = blended.argmax(axis=1)
        return -(preds == y_true).mean()

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bounds = [(0.05, 0.90)] * 3
    result = minimize(neg_accuracy, x0=[1/3, 1/3, 1/3],
                      bounds=bounds, constraints=constraints, method="SLSQP")
    best_w = result.x
    best_acc = -result.fun
    logger.success(f"Optimised weights: LGBM={best_w[0]:.3f} XGB={best_w[1]:.3f} LSTM={best_w[2]:.3f} | acc={best_acc:.4f}")
    return best_w.tolist()


def generate_signals(blended_proba, threshold=SIGNAL_THRESHOLD):
    """
    Convert ensemble probabilities to trading signals.

    Returns a dict with keys:
      signal   : 'BUY', 'SELL', or 'HOLD'
      confidence: float 0–1
      up_prob  : float
      down_prob: float
    """
    up_prob   = blended_proba[:, 2]
    down_prob = blended_proba[:, 0]

    signals = np.full(len(blended_proba), "HOLD", dtype=object)
    signals[up_prob   >= threshold] = "BUY"
    signals[down_prob >= threshold] = "SELL"

    # Remove conflicting signals (both up & down above threshold)
    conflict = (up_prob >= threshold) & (down_prob >= threshold)
    signals[conflict] = "HOLD"

    confidence = np.maximum(up_prob, down_prob)

    n_buy  = (signals == "BUY").sum()
    n_sell = (signals == "SELL").sum()
    n_hold = (signals == "HOLD").sum()
    logger.info(
        f"Signal distribution (threshold={threshold}): "
        f"BUY={n_buy} ({n_buy/len(signals):.1%})  "
        f"SELL={n_sell} ({n_sell/len(signals):.1%})  "
        f"HOLD={n_hold} ({n_hold/len(signals):.1%})"
    )

    return {
        "signal":     signals,
        "confidence": confidence,
        "up_prob":    up_prob,
        "down_prob":  down_prob,
    }


def save_weights(weights, symbol):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = Path(MODEL_DIR) / f"{symbol.replace('/','_')}_ensemble_weights.pkl"
    joblib.dump(weights, path)
    logger.info(f"Saved ensemble weights -> {path}")


def load_weights(symbol):
    path = Path(MODEL_DIR) / f"{symbol.replace('/','_')}_ensemble_weights.pkl"
    if path.exists():
        return joblib.load(path)
    logger.warning(f"No saved weights for {symbol}, using defaults")
    return list(ENSEMBLE_WEIGHTS.values())