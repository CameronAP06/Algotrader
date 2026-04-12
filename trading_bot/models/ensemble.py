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


def _stack_probas(p1, p2, p3):
    """Align all three probability arrays to the same length (seq models have padding)."""
    n = min(len(p1), len(p2), len(p3))
    return p1[-n:], p2[-n:], p3[-n:]


def weighted_ensemble(p1, p2, p3, weights=None):
    """Blend three (n, 3) probability arrays. Model order: catboost, cnn, lstm."""
    if weights is None:
        w = list(ENSEMBLE_WEIGHTS.values())
    else:
        w = weights
    p1, p2, p3 = _stack_probas(p1, p2, p3)
    return w[0] * p1 + w[1] * p2 + w[2] * p3


def optimise_weights(p1, p2, p3, y_true, use_genetic: bool = True):
    """
    Find ensemble weights that maximise validation F1 (macro) via Differential
    Evolution — a genetic/evolutionary search that avoids local optima on the
    flat loss surfaces typical of ML ensemble problems.

    Strategy:
      1. Differential Evolution (scipy) over the full weight simplex
         — population-based, mutation + crossover, no gradient required
         — far more robust than grid search on flat surfaces
         — also optimises the signal threshold jointly with weights
      2. Falls back to grid search if scipy unavailable

    Optimisation target: macro-F1 on the validation set.
    Macro-F1 is better than accuracy here because UP/DOWN/NEUTRAL classes are
    imbalanced — accuracy rewards always predicting NEUTRAL.
    """
    from scipy.optimize import differential_evolution
    from sklearn.metrics import f1_score

    p1, p2, p3 = _stack_probas(p1, p2, p3)
    y_true = np.array(y_true[-len(p1):])

    def neg_f1(params):
        # params = [w0, w1, w2_raw, threshold_raw]
        # Normalise weights to sum to 1
        w0, w1, w2_raw = params[:3]
        total = w0 + w1 + w2_raw + 1e-9
        w = [w0 / total, w1 / total, w2_raw / total]
        threshold = params[3]

        blended   = w[0]*p1 + w[1]*p2 + w[2]*p3
        up_prob   = blended[:, 2]
        down_prob = blended[:, 0]
        best_prob = np.maximum(up_prob, down_prob)

        # Percentile threshold
        candidates = best_prob[best_prob > 0.34]
        if len(candidates) >= 5:
            pct_threshold = max(float(np.percentile(candidates, (1 - 0.15) * 100)), 0.34)
        else:
            pct_threshold = threshold   # fall back to explicit threshold

        signals = np.ones(len(blended), dtype=int)   # default NEUTRAL=1
        signals[up_prob   >= pct_threshold] = 2
        signals[down_prob >= pct_threshold] = 0
        conflict = (up_prob >= pct_threshold) & (down_prob >= pct_threshold)
        signals[conflict] = 1

        # Penalise if model fires on almost nothing (< 3% of bars)
        active_pct = (signals != 1).mean()
        if active_pct < 0.03:
            return 1.0  # penalty

        f1 = f1_score(y_true, signals, average="macro", zero_division=0)
        return -f1   # minimise negative F1

    # Bounds: [w0 ∈ (0,1), w1 ∈ (0,1), w2_raw ∈ (0,1), threshold ∈ (0.33, 0.55)]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.33, 0.55)]

    try:
        result = differential_evolution(
            neg_f1,
            bounds,
            seed=42,
            maxiter=100,
            popsize=12,
            tol=1e-5,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=True,          # final L-BFGS-B polish
            workers=1,            # keep deterministic
        )
        w0, w1, w2_raw = result.x[:3]
        total  = w0 + w1 + w2_raw + 1e-9
        best_w = [w0/total, w1/total, w2_raw/total]
        best_f1 = -result.fun

        logger.success(
            f"DE-optimised weights: m1={best_w[0]:.3f} m2={best_w[1]:.3f} "
            f"m3={best_w[2]:.3f} | macro-F1={best_f1:.4f}"
        )
        return best_w

    except Exception as e:
        logger.warning(f"Differential evolution failed ({e}), falling back to grid search")
        return _grid_search_weights(p1, p2, p3, y_true)


def _grid_search_weights(p1, p2, p3, y_true):
    """Fallback grid search (original implementation)."""
    from sklearn.metrics import f1_score

    def score(w):
        blended = w[0]*p1 + w[1]*p2 + w[2]*p3
        preds   = blended.argmax(axis=1)
        return f1_score(y_true, preds, average="macro", zero_division=0)

    best_score = -1
    best_w     = [1/3, 1/3, 1/3]
    step       = 0.05
    steps      = np.arange(0, 1 + step, step)
    for w0 in steps:
        for w1 in steps:
            w2 = 1.0 - w0 - w1
            if w2 < 0 or w2 > 1:
                continue
            s = score([w0, w1, w2])
            if s > best_score:
                best_score = s
                best_w     = [w0, w1, w2]

    total  = sum(best_w)
    best_w = [w / total for w in best_w]
    logger.success(
        f"Grid-search weights: m1={best_w[0]:.3f} m2={best_w[1]:.3f} "
        f"m3={best_w[2]:.3f} | macro-F1={best_score:.4f}"
    )
    return best_w


def compute_signal_threshold(proba: np.ndarray, top_pct: float = 0.15,
                             floor: float = 0.34) -> float:
    """
    Compute the percentile-based signal threshold from a probability array.

    Call this on the VALIDATION set and pass the result as `threshold` to
    generate_signals on the TEST set (use_percentile=False).  This eliminates
    the hindsight bias introduced when the threshold is computed from the same
    bars being evaluated.

    Returns a fixed float threshold, not a per-bar adaptive value.
    """
    up_prob   = proba[:, 2]
    down_prob = proba[:, 0]
    best_prob = np.maximum(up_prob, down_prob)
    candidates = best_prob[best_prob > floor]
    if len(candidates) >= 5:
        return float(max(np.percentile(candidates, (1 - top_pct) * 100), floor))
    return 0.40   # fallback when val set has very few confident bars


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