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
from config.settings import ENSEMBLE_WEIGHTS, SIGNAL_THRESHOLD, SIGNAL_THRESHOLDS, MODEL_DIR


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


def optimise_weights(p1, p2, p3, y_true):
    """
    Find ensemble weights that maximise validation accuracy.

    Uses a two-stage approach:
      Stage 1 — Coarse grid search across all weight combos (step=0.05)
                Works even on flat loss surfaces where gradient methods fail
      Stage 2 — Fine grid search around the best coarse solution (step=0.01)

    This replaces the previous SLSQP approach which always returned
    [0.333, 0.333, 0.333] because the loss surface is too flat for
    gradient-based optimisation when models have similar accuracy.
    """
    p1, p2, p3 = _stack_probas(p1, p2, p3)
    y_true = np.array(y_true[-len(p1):])

    def accuracy(w):
        blended = w[0]*p1 + w[1]*p2 + w[2]*p3
        return (blended.argmax(axis=1) == y_true).mean()

    # Stage 1: coarse grid (step=0.05) — ~231 combinations
    best_acc = -1
    best_w   = [1/3, 1/3, 1/3]
    step = 0.05
    steps = np.arange(0, 1 + step, step)

    for w0 in steps:
        for w1 in steps:
            w2 = 1.0 - w0 - w1
            if w2 < 0 or w2 > 1:
                continue
            w = [w0, w1, w2]
            acc = accuracy(w)
            if acc > best_acc:
                best_acc = acc
                best_w   = w

    # Stage 2: fine grid around best solution (step=0.01, ±0.1 window)
    fine_step = 0.01
    for dw0 in np.arange(-0.10, 0.11, fine_step):
        for dw1 in np.arange(-0.10, 0.11, fine_step):
            w0 = np.clip(best_w[0] + dw0, 0, 1)
            w1 = np.clip(best_w[1] + dw1, 0, 1)
            w2 = 1.0 - w0 - w1
            if w2 < 0 or w2 > 1:
                continue
            w   = [w0, w1, w2]
            acc = accuracy(w)
            if acc > best_acc:
                best_acc = acc
                best_w   = w

    # Normalise to sum exactly to 1
    total  = sum(best_w)
    best_w = [w / total for w in best_w]

    logger.success(
        f"Optimised weights: TFT={best_w[0]:.3f} CNN={best_w[1]:.3f} "
        f"LSTM={best_w[2]:.3f} | acc={best_acc:.4f}"
    )
    return best_w


def generate_signals(blended_proba, threshold=SIGNAL_THRESHOLD, symbol=None,
                     use_percentile: bool = True, top_pct: float = 0.15):
    """
    Convert ensemble probabilities to trading signals.
    Returns a dict with keys: signal, confidence, up_prob, down_prob.

    use_percentile=True (default): fires on top top_pct of confident bars,
    adapting to whatever probability range the ensemble actually outputs.
    This is robust to probability compression from ensemble averaging.

    use_percentile=False: uses fixed threshold (legacy behaviour).
    """
    if symbol is not None:
        threshold = SIGNAL_THRESHOLDS.get(symbol, threshold)
    up_prob   = blended_proba[:, 2]
    down_prob = blended_proba[:, 0]
    best_prob = np.maximum(up_prob, down_prob)

    if use_percentile and len(best_prob) >= 20:
        # Find the threshold that selects the top top_pct most confident bars
        # Floor at 0.34 to never fire on near-random predictions
        candidates = best_prob[best_prob > 0.34]
        if len(candidates) >= 5:
            threshold = float(np.percentile(candidates, (1 - top_pct) * 100))
            threshold = max(threshold, 0.34)
        else:
            threshold = 0.40  # fallback if very few confident bars

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