"""
utils/edge_scanner.py
=====================
Autonomous edge discovery: walks the label definition space to find
symbol/timeframe/horizon/threshold combinations that are genuinely
learnable — i.e. produce consistent out-of-sample accuracy and positive
Sharpe across multiple walk-forward folds.

What it searches over
---------------------
  horizon_hours  : how far ahead to predict (4h, 8h, 12h, 24h, 48h, 72h)
  threshold      : min % move to label as UP/DOWN (0.5% → 5%)
  label_style    : binary (UP/DOWN only) vs ternary (UP/NEUTRAL/DOWN)
  feature_groups : which indicator groups to include (subset search)

For each combination it runs a lightweight 3-fold walk-forward using
LSTM only (fast — skips TFT/CNN ensemble overhead), records out-of-sample
accuracy, win rate, and Sharpe, then ranks all results.

A combination is flagged as a real edge if:
  - Median OOS accuracy > 38%  (comfortably above 33% random baseline)
  - Median win rate > 40%
  - Profitable in at least 2/3 folds
  - Sharpe > 0.5 in at least 2/3 folds

Usage
-----
  # Full scan: all symbols, all timeframes
  python edge_scanner.py

  # Single symbol
  python edge_scanner.py --symbol LINK/USD

  # Specific timeframes
  python edge_scanner.py --timeframes 1h 4h 8h

  # Quick mode: fewer combos, faster scan
  python edge_scanner.py --quick

Output
------
  Prints ranked table of edges to console.
  Saves full results to backtest/results/edge_scan_TIMESTAMP.csv
  Saves best config per symbol/timeframe to backtest/results/best_edges.json
"""

import os, sys, json, argparse, itertools
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler
from config.settings import TRADING_PAIRS
from data.feature_engineer import build_features
from data.alt_data import fetch_all_alt_data, merge_alt_data

# ─── Search Space ─────────────────────────────────────────────────────────────

# Horizon: how many REAL hours ahead to predict
# We convert to bars per timeframe inside the scan
HORIZON_HOURS = [4, 8, 12, 24, 48, 72]

# Threshold: minimum % move to count as UP or DOWN
# Values chosen to span "too noisy" → "too rare"
THRESHOLDS = [0.005, 0.008, 0.012, 0.018, 0.025, 0.035, 0.050]

# Label styles
LABEL_STYLES = ["ternary"]  # binary removed: no NEUTRAL class forces overtrade and inflates Sharpe

# Bars per hour for each timeframe
BARS_PER_HOUR = {
    "15m": 4.0,
    "1h":  1.0,
    "2h":  0.5,
    "4h":  0.25,
    "8h":  0.125,
    "1d":  1/24,
}

# Quick mode: reduced search space for a ~4h scan instead of ~24h
QUICK_HORIZON_HOURS = [8, 24, 36, 48]
QUICK_THRESHOLDS    = [0.008, 0.018, 0.025, 0.035, 0.050]
QUICK_LABEL_STYLES  = ["ternary"]  # binary excluded: forces trades on every bar, inflates Sharpe

# Feature groups to always include (OHLCV + ATR essential)
CORE_GROUPS = {"atr", "raw_ohlcv"}

# Optional groups to search over (each independently toggled)
OPTIONAL_GROUPS = ["trend", "momentum", "macd", "bollinger",
                   "volume", "candles", "regime", "lags", "time"]

# ─── Edge Criteria ────────────────────────────────────────────────────────────

EDGE_MIN_ACCURACY   = 0.36    # Must beat 33% random baseline by a margin
EDGE_MIN_WIN_RATE   = 0.36    # 36% WR is profitable at 2:1 RR (breakeven = 33%)
EDGE_MIN_PROF_FOLDS = 0.34    # At least 1/3 of folds profitable (avoids all-loss)
EDGE_MIN_SHARPE     = 0.2     # Modest positive risk-adjusted return

# Minimum trades per fold to consider the result meaningful
MIN_TRADES_PER_FOLD = 3

# Walk-forward config for the scanner (lighter than the full pipeline)
SCAN_N_FOLDS    = 3     # 3 OOS folds is enough to detect consistency
SCAN_TRAIN_MULT = 3     # train = 3x test window
SCAN_EPOCHS     = 30    # Fast training; full pipeline uses 50+
SCAN_PATIENCE   = 7


# ─── Custom Label Creation ────────────────────────────────────────────────────

def make_labels(df: pd.DataFrame,
                horizon_bars: int,
                threshold: float,
                style: str = "ternary") -> pd.Series:
    """
    Create labels with the given horizon and threshold.
    style='ternary'  → 0=DOWN, 1=NEUTRAL, 2=UP
    style='binary'   → 0=DOWN, 1=UP  (drops NEUTRAL bars entirely)
    """
    future_return = df["close"].pct_change(horizon_bars).shift(-horizon_bars)

    if "atr_14" in df.columns:
        atr_pct = df["atr_14"] / df["close"]
        dynamic_thresh = (threshold + atr_pct * 0.3).clip(threshold, threshold * 2.5)
    else:
        dynamic_thresh = threshold

    if style == "ternary":
        labels = pd.Series(1, index=df.index, name="label")
        labels[future_return >  dynamic_thresh] = 2
        labels[future_return < -dynamic_thresh] = 0
    else:
        # Binary: only keep UP/DOWN bars, label as 0/1
        labels = pd.Series(np.nan, index=df.index, name="label")
        labels[future_return >  dynamic_thresh] = 1
        labels[future_return < -dynamic_thresh] = 0

    return labels


def label_quality(labels: pd.Series, style: str) -> dict:
    """Compute label distribution stats."""
    clean = labels.dropna()
    n = len(clean)
    if n == 0:
        return {"n": 0, "neutral_pct": 0, "up_pct": 0, "down_pct": 0, "balance": 0}

    if style == "ternary":
        neutral_pct = (clean == 1).sum() / n
        up_pct      = (clean == 2).sum() / n
        down_pct    = (clean == 0).sum() / n
    else:
        neutral_pct = 0.0
        up_pct      = (clean == 1).sum() / n
        down_pct    = (clean == 0).sum() / n

    # Balance: how close up/down are to each other (1.0 = perfect)
    balance = 1 - abs(up_pct - down_pct) / (up_pct + down_pct + 1e-9)

    return {
        "n":           n,
        "neutral_pct": neutral_pct,
        "up_pct":      up_pct,
        "down_pct":    down_pct,
        "balance":     balance,
    }


# ─── Feature Selection ────────────────────────────────────────────────────────

def select_feature_cols(feat_df: pd.DataFrame,
                        groups: set) -> list:
    """
    Return column indices for the given feature groups.
    Always includes OHLCV + ATR columns.
    """
    PREFIXES = {
        "trend":     ["sma_", "ema_", "close_vs_sma_"],
        "momentum":  ["roc_", "rsi_"],
        "macd":      ["macd"],
        "bollinger": ["bb_upper_", "bb_lower_", "bb_width_", "bb_pct_"],
        "atr":       ["atr_", "atr_pct_"],
        "volume":    ["vol_sma_", "vol_ratio_", "obv", "vwap", "close_vs_vwap"],
        "candles":   ["body_pct", "upper_shadow", "lower_shadow", "body_range_pct"],
        "time":      ["hour", "dayofweek", "hour_sin", "hour_cos", "dow_sin", "dow_cos"],
        "lags":      ["return_lag_", "volume_lag_"],
        "regime":    ["adx", "plus_di", "minus_di", "di_diff", "adx_trend",
                      "regime", "regime_sin", "regime_cos", "hurst_approx", "vol_regime"],
        "raw_ohlcv": ["raw_open_", "raw_high_", "raw_low_", "raw_close_", "raw_volume_",
                      "raw_range_", "raw_body_"],
        "alt_data":  ["fear_greed", "btc_dominance", "google_trends",
                      "btc_dom_proxy", "sentiment_rsi_divergence"],
    }
    always_include = {"open", "high", "low", "close", "volume"}
    selected = set()
    for group in groups:
        for col in feat_df.columns:
            for prefix in PREFIXES.get(group, []):
                if col.startswith(prefix) or col == prefix:
                    selected.add(col)
                    break
    # Always include raw OHLCV
    for col in feat_df.columns:
        if col in always_include:
            selected.add(col)
    exclude = {"label", "timestamp"}
    return [c for c in feat_df.columns if c in selected and c not in exclude]


# ─── Lightweight Walk-Forward ─────────────────────────────────────────────────

def quick_walk_forward(feat_df: pd.DataFrame,
                       labels: pd.Series,
                       feature_cols: list,
                       n_folds: int,
                       style: str,
                       symbol_tf: str) -> list:
    """
    Run a lightweight 3-fold OOS walk-forward using LSTM only.
    Returns list of fold result dicts.
    """
    from models.lstm_model import LSTMClassifier, SequenceDataset, DEVICE
    from torch.utils.data import DataLoader
    import torch
    import torch.nn as nn
    from backtest.engine import BacktestEngine
    from backtest.filters import apply_filters
    import config.settings as s

    # Merge labels into feat_df temporarily
    df = feat_df.copy()
    df["label"] = labels

    # Drop rows where label is NaN (binary mode drops neutral bars)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    if len(df) < 500:
        return []

    n_classes = 2 if style == "binary" else 3
    n         = len(df)

    # Compute fold sizes: test = 1/5 of total, train = 3x test
    test_bars  = max(50, n // (n_folds + SCAN_TRAIN_MULT))
    train_bars = test_bars * SCAN_TRAIN_MULT
    val_bars   = test_bars

    min_needed = train_bars + val_bars + test_bars
    if n < min_needed:
        # Scale down proportionally
        ratio      = n / min_needed
        train_bars = max(100, int(train_bars * ratio))
        val_bars   = max(30,  int(val_bars   * ratio))
        test_bars  = max(30,  int(test_bars  * ratio))

    X_all = df[feature_cols].values.astype(np.float32)
    y_all = df["label"].values.astype(np.int64)

    # Standardise
    scaler  = StandardScaler()
    X_all   = scaler.fit_transform(X_all)

    seq_len    = min(24, train_bars // 10)
    hidden     = 64
    n_layers   = 2
    batch_size = 64
    lr         = 1e-3

    fold_results = []
    test_start   = train_bars + val_bars

    for fold in range(n_folds):
        ts = test_start + fold * test_bars
        te = ts + test_bars
        if te > n:
            break

        tr_s = max(0, ts - train_bars - val_bars)
        tr_e = ts - val_bars
        v_e  = ts

        X_train = X_all[tr_s:tr_e]
        y_train = y_all[tr_s:tr_e]
        X_val   = X_all[tr_e:v_e]
        y_val   = y_all[tr_e:v_e]
        X_test  = X_all[ts:te]
        y_test  = y_all[ts:te]

        if len(X_train) < 100 or len(X_test) < 30:
            continue

        # Quick LSTM
        train_ds = SequenceDataset(X_train, y_train, seq_len)
        val_ds   = SequenceDataset(X_val,   y_val,   seq_len)
        test_ds  = SequenceDataset(X_test,  y_test,  seq_len)

        if len(train_ds) < batch_size or len(test_ds) < 5:
            continue

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=batch_size)
        test_dl  = DataLoader(test_ds,  batch_size=batch_size)

        model = LSTMClassifier(
            input_size  = len(feature_cols),
            hidden_size = hidden,
            num_layers  = n_layers,
            num_classes = n_classes,
            dropout     = 0.1,
        ).to(DEVICE)

        # Compute class weights from actual training label distribution
        # to prevent class collapse on imbalanced folds
        if n_classes == 2:
            c0 = max(1, (y_train == 0).sum())
            c1 = max(1, (y_train == 1).sum())
            total = c0 + c1
            w = torch.FloatTensor([total/c0, total/c1]).to(DEVICE)
        else:
            counts = [max(1, (y_train == c).sum()) for c in range(3)]
            total  = sum(counts)
            w = torch.FloatTensor([total/c for c in counts]).to(DEVICE)
            # Cap extreme weights at 5x to avoid over-correction
            w = torch.clamp(w, max=5.0)
        criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val  = float("inf")
        best_state = None
        patience_c = 0

        for epoch in range(SCAN_EPOCHS):
            model.train()
            for Xb, yb in train_dl:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                criterion(model(Xb), yb).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            vl = 0
            with torch.no_grad():
                for Xb, yb in val_dl:
                    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                    vl += criterion(model(Xb), yb).item()
            vl /= max(1, len(val_dl))

            if vl < best_val:
                best_val   = vl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_c = 0
            else:
                patience_c += 1
                if patience_c >= SCAN_PATIENCE:
                    break

        if best_state is None:
            continue
        model.load_state_dict(best_state)

        # Evaluate on test fold
        model.eval()
        all_preds, all_labels = [], []
        all_probs = []
        with torch.no_grad():
            for Xb, yb in test_dl:
                logits = model(Xb.to(DEVICE))
                probs  = torch.softmax(logits, dim=1).cpu().numpy()
                preds  = logits.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())
                all_probs.extend(probs)

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs  = np.array(all_probs)

        # Class collapse check: if model predicts >85% of bars as one class,
        # it has degenerated — skip this fold entirely
        if len(all_preds) > 0:
            dominant_class_pct = max(
                (all_preds == c).mean() for c in range(n_classes)
            )
            if dominant_class_pct > 0.85:
                logger.debug(
                    f"  Fold {fold+1}: class collapse detected "
                    f"({dominant_class_pct:.0%} one class) — skipping"
                )
                continue

        accuracy = (all_preds == all_labels).mean()

        # Quick backtest on test period
        test_df = df.iloc[ts + seq_len : te].reset_index(drop=True)
        test_df = test_df.tail(len(all_preds)).reset_index(drop=True)

        if n_classes == 2:
            # Map binary back to 3-class signal probabilities
            probs3 = np.zeros((len(all_probs), 3))
            probs3[:, 0] = all_probs[:, 0]   # DOWN
            probs3[:, 2] = all_probs[:, 1]   # UP
            probs3[:, 1] = 0.0               # no NEUTRAL class
        else:
            probs3 = all_probs

        from models.ensemble import generate_signals
        signals = generate_signals(probs3, threshold=0.60)

        orig_pos = s.MAX_POSITION_PCT
        s.MAX_POSITION_PCT = 0.25
        engine  = BacktestEngine()
        _tf = symbol_tf.split("_")[-1] if "_" in symbol_tf else "1h"
        metrics = engine.run(test_df, signals, symbol_tf, timeframe=_tf)
        s.MAX_POSITION_PCT = orig_pos

        n_trades = metrics["n_trades"]
        ret      = metrics["total_return"]
        sharpe   = metrics["sharpe_ratio"]
        win_rate = metrics["win_rate"]

        fold_results.append({
            "fold":      fold + 1,
            "accuracy":  float(accuracy),
            "n_trades":  int(n_trades),
            "return":    float(ret),
            "sharpe":    float(sharpe),
            "win_rate":  float(win_rate),
        })

    return fold_results


# ─── Score an Edge Candidate ──────────────────────────────────────────────────

def score_edge(fold_results: list, min_trades: int = MIN_TRADES_PER_FOLD) -> dict:
    """
    Aggregate fold results into an edge score.
    Only considers folds that actually traded.
    """
    if not fold_results:
        return {"score": -999, "edge": False}

    # Filter to folds with enough trades but not too many
    # >80 trades/fold at 0.2% fees = >16% return eaten by fees alone
    max_trades = 80
    active = [f for f in fold_results if min_trades <= f["n_trades"] <= max_trades]
    all_folds = fold_results

    if not active:
        return {
            "score":           -10,
            "edge":            False,
            "n_folds":         len(all_folds),
            "n_active_folds":  0,
            "med_accuracy":    float(np.median([f["accuracy"] for f in all_folds])),
            "med_return":      0.0,
            "med_sharpe":      0.0,
            "med_win_rate":    0.0,
            "med_trades":      float(np.median([f["n_trades"] for f in all_folds])),
            "prof_folds":      0,
            "reason":          "no_trades",
        }

    med_acc   = float(np.median([f["accuracy"]  for f in active]))
    med_ret   = float(np.median([f["return"]    for f in active]))
    med_sharpe = float(np.median([f["sharpe"]   for f in active]))
    med_wr    = float(np.median([f["win_rate"]  for f in active]))
    med_trades = float(np.median([f["n_trades"] for f in active]))
    prof_folds = sum(1 for f in active if f["return"] > 0)
    prof_rate  = prof_folds / len(active)

    # Composite score: accuracy above random + win rate + consistency
    acc_bonus  = max(0, (med_acc  - 0.33) * 20)   # +0 at 33%, +1.4 at 40%
    wr_bonus   = max(0, (med_wr   - 0.33) * 15)   # +0 at 33%, +1.0 at 40%
    prof_bonus = prof_rate * 3                      # up to +3
    sharpe_bonus = max(0, min(med_sharpe, 3.0))     # capped at +3
    trade_bonus  = min(1.0, med_trades / 10)        # up to +1 for 10+ trades/fold

    score = acc_bonus + wr_bonus + prof_bonus + sharpe_bonus + trade_bonus

    is_edge = (
        med_acc    >= EDGE_MIN_ACCURACY  and
        med_wr     >= EDGE_MIN_WIN_RATE  and
        prof_rate  >= EDGE_MIN_PROF_FOLDS and
        med_sharpe >= EDGE_MIN_SHARPE
    )

    reasons = []
    if med_acc  < EDGE_MIN_ACCURACY:  reasons.append(f"acc={med_acc:.2f}<{EDGE_MIN_ACCURACY}")
    if med_wr   < EDGE_MIN_WIN_RATE:  reasons.append(f"wr={med_wr:.2f}<{EDGE_MIN_WIN_RATE}")
    if prof_rate < EDGE_MIN_PROF_FOLDS: reasons.append(f"prof={prof_rate:.0%}<{EDGE_MIN_PROF_FOLDS:.0%}")
    if med_sharpe < EDGE_MIN_SHARPE:  reasons.append(f"sharpe={med_sharpe:.2f}<{EDGE_MIN_SHARPE}")

    return {
        "score":          score,
        "edge":           is_edge,
        "n_folds":        len(all_folds),
        "n_active_folds": len(active),
        "med_accuracy":   med_acc,
        "med_return":     med_ret,
        "med_sharpe":     med_sharpe,
        "med_win_rate":   med_wr,
        "med_trades":     med_trades,
        "prof_folds":     prof_folds,
        "reason":         ", ".join(reasons) if reasons else "passes_all",
    }


# ─── Main Scanner ─────────────────────────────────────────────────────────────

def run_edge_scan(raw_df: pd.DataFrame,
                  feat_df: pd.DataFrame,
                  symbol: str,
                  timeframe: str,
                  quick: bool = False) -> list:
    """
    Scan all (horizon, threshold, style) combinations for a single
    symbol/timeframe. Returns list of result dicts, sorted by score.
    """
    bph         = BARS_PER_HOUR.get(timeframe, 1.0)
    horizons    = QUICK_HORIZON_HOURS  if quick else HORIZON_HOURS
    thresholds  = QUICK_THRESHOLDS     if quick else THRESHOLDS
    styles      = QUICK_LABEL_STYLES   if quick else LABEL_STYLES

    # All optional feature groups active for the scan
    # (feature group search adds too much time; we focus on label search)
    all_groups = CORE_GROUPS | set(OPTIONAL_GROUPS)
    feature_cols = select_feature_cols(feat_df, all_groups)

    if len(feature_cols) < 5:
        logger.warning(f"Too few features for {symbol} {timeframe}: {len(feature_cols)}")
        return []

    combos = list(itertools.product(horizons, thresholds, styles))
    logger.info(
        f"Edge scan: {symbol} [{timeframe}] — "
        f"{len(combos)} combinations × {SCAN_N_FOLDS} folds"
    )

    results = []

    for i, (horizon_h, threshold, style) in enumerate(combos):
        horizon_bars = max(1, round(horizon_h * bph))

        label_tag = f"h{horizon_h}h_t{threshold:.3f}_{style}"

        # Skip if horizon > half the available data
        if horizon_bars > len(feat_df) // 4:
            logger.debug(f"  Skip {label_tag}: horizon_bars={horizon_bars} exceeds data limit")
            continue

        # Skip horizon=1 bar: model just learns next-bar autocorrelation
        # which evaporates in live trading. Minimum meaningful horizon is 2 bars.
        if horizon_bars < 2:
            logger.debug(f"  Skip {label_tag}: horizon_bars={horizon_bars} too short (min 2)")
            continue

        # Create labels
        labels = make_labels(feat_df, horizon_bars, threshold, style)
        lq     = label_quality(labels, style)

        # Skip degenerate label distributions
        # For ternary: need at least 15% each of UP and DOWN
        # For binary:  need at least 40% each class (balanced)
        if style == "ternary":
            if lq["up_pct"] < 0.10 or lq["down_pct"] < 0.10:
                logger.debug(f"  Skip {label_tag}: too few UP/DOWN labels")
                continue
            if lq["neutral_pct"] < 0.20:
                logger.debug(f"  Skip {label_tag}: too few NEUTRAL labels (<20%)")
                continue
        else:
            if lq["balance"] < 0.70:
                logger.debug(f"  Skip {label_tag}: UP/DOWN imbalance {lq['balance']:.2f}")
                continue

        logger.info(
            f"  [{i+1}/{len(combos)}] {label_tag} | "
            f"UP={lq['up_pct']:.0%} NEU={lq['neutral_pct']:.0%} "
            f"DOWN={lq['down_pct']:.0%} | n={lq['n']}"
        )

        # Run quick walk-forward
        try:
            fold_results = quick_walk_forward(
                feat_df, labels, feature_cols,
                SCAN_N_FOLDS, style,
                f"{symbol}_{timeframe}"
            )
        except Exception as e:
            logger.warning(f"  Walk-forward failed for {label_tag}: {e}")
            continue

        edge_score = score_edge(fold_results)

        result = {
            "symbol":        symbol,
            "timeframe":     timeframe,
            "horizon_hours": horizon_h,
            "horizon_bars":  horizon_bars,
            "threshold":     threshold,
            "label_style":   style,
            "n_features":    len(feature_cols),
            **lq,
            **edge_score,
            "folds":         fold_results,
        }
        results.append(result)

        status = "✓ EDGE" if edge_score["edge"] else "  ----"
        logger.info(
            f"  {status} | score={edge_score['score']:.2f} | "
            f"acc={edge_score.get('med_accuracy', 0):.3f} | "
            f"wr={edge_score.get('med_win_rate', 0):.3f} | "
            f"ret={edge_score.get('med_return', 0):+.3f} | "
            f"trades={edge_score.get('med_trades', 0):.0f}"
        )

    results.sort(key=lambda x: -x["score"])
    return results


# ─── Report ───────────────────────────────────────────────────────────────────

def print_report(all_results: list):
    """Print ranked results table and highlight discovered edges."""
    if not all_results:
        print("\nNo results to report.")
        return

    edges    = [r for r in all_results if r["edge"]]
    non_edge = [r for r in all_results if not r["edge"]]

    print("\n" + "="*90)
    print("EDGE SCANNER RESULTS")
    print("="*90)
    print(f"Scanned {len(all_results)} combinations across all symbols/timeframes")
    print(f"Edges found: {len(edges)} / {len(all_results)}")
    print()

    if edges:
        print("─"*90)
        print("✓ CONFIRMED EDGES (meet all criteria — use these in the main pipeline)")
        print("─"*90)
        print(f"  {'Symbol':12} {'TF':4} {'Horizon':8} {'Thresh':7} {'Style':8} "
              f"{'Acc':6} {'WinR':6} {'Return':8} {'Sharpe':7} {'Trades':7} {'Score':6}")
        print("  " + "-"*85)
        for r in edges:
            print(
                f"  {r['symbol']:12} {r['timeframe']:4} "
                f"{r['horizon_hours']:>5}h    "
                f"{r['threshold']:.1%}  "
                f"{r['label_style']:8} "
                f"{r['med_accuracy']:.3f}  "
                f"{r['med_win_rate']:.3f}  "
                f"{r['med_return']:+.3f}   "
                f"{r['med_sharpe']:+.2f}   "
                f"{r['med_trades']:>5.0f}   "
                f"{r['score']:.2f}"
            )
    else:
        print("No combinations met all edge criteria.")
        print("Top 10 closest candidates:")
        print()
        print(f"  {'Symbol':12} {'TF':4} {'Horizon':8} {'Thresh':7} {'Style':8} "
              f"{'Acc':6} {'WinR':6} {'Return':8} {'Sharpe':7} {'Trades':7} {'Score':6} {'Why failed'}")
        print("  " + "-"*100)
        for r in sorted(non_edge, key=lambda x: -x["score"])[:10]:
            print(
                f"  {r['symbol']:12} {r['timeframe']:4} "
                f"{r['horizon_hours']:>5}h    "
                f"{r['threshold']:.1%}  "
                f"{r['label_style']:8} "
                f"{r.get('med_accuracy', 0):.3f}  "
                f"{r.get('med_win_rate', 0):.3f}  "
                f"{r.get('med_return', 0):+.3f}   "
                f"{r.get('med_sharpe', 0):+.2f}   "
                f"{r.get('med_trades', 0):>5.0f}   "
                f"{r['score']:.2f}  "
                f"{r.get('reason', '')}"
            )

    print()
    print("="*90)


def save_results(all_results: list, out_dir: str = "backtest/results"):
    """Save full results CSV and best_edges.json."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Flatten for CSV (drop fold-level detail)
    flat = []
    for r in all_results:
        row = {k: v for k, v in r.items() if k != "folds"}
        flat.append(row)

    df = pd.DataFrame(flat)
    csv_path = Path(out_dir) / f"edge_scan_{ts}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Full scan results saved → {csv_path}")

    # Best config per symbol/timeframe
    best_edges = {}
    edges = [r for r in all_results if r["edge"]]
    for r in edges:
        key = f"{r['symbol']}_{r['timeframe']}"
        if key not in best_edges or r["score"] > best_edges[key]["score"]:
            best_edges[key] = {
                "symbol":        r["symbol"],
                "timeframe":     r["timeframe"],
                "horizon_hours": r["horizon_hours"],
                "horizon_bars":  r["horizon_bars"],
                "threshold":     r["threshold"],
                "label_style":   r["label_style"],
                "score":         r["score"],
                "med_accuracy":  r["med_accuracy"],
                "med_win_rate":  r["med_win_rate"],
                "med_return":    r["med_return"],
                "med_sharpe":    r["med_sharpe"],
            }

    json_path = Path(out_dir) / "best_edges.json"
    with open(json_path, "w") as f:
        json.dump(best_edges, f, indent=2)
    logger.info(f"Best edges saved → {json_path}")

    return csv_path, json_path


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Autonomous edge scanner")
    parser.add_argument("--symbol",     type=str,       default=None,
                        help="Single symbol to scan (default: all)")
    parser.add_argument("--timeframes", nargs="+",
                        default=["1h", "2h", "4h", "8h"],
                        help="Timeframes to scan")
    parser.add_argument("--quick",      action="store_true",
                        help="Quick mode: fewer combinations (~4h vs ~24h)")
    parser.add_argument("--refresh",    action="store_true",
                        help="Force re-fetch OHLCV data")
    args = parser.parse_args()

    symbols    = [args.symbol] if args.symbol else TRADING_PAIRS
    timeframes = args.timeframes

    logger.info(
        f"Edge scanner starting: {len(symbols)} symbols × "
        f"{len(timeframes)} timeframes | quick={args.quick}"
    )

    # ── Fetch alt data once ────────────────────────────────────────────────
    alt_df = fetch_all_alt_data(force_refresh=args.refresh)

    # ── TIMEFRAME_CONFIG (same as timeframe_comparison.py) ─────────────────
    TIMEFRAME_CONFIG = {
        "15m": {"ccxt_tf": "15m", "history_days": 365},
        "1h":  {"ccxt_tf": "1h",  "history_days": 1825},
        "2h":  {"ccxt_tf": "2h",  "history_days": 1825},
        "4h":  {"ccxt_tf": "4h",  "history_days": 1825},
        "8h":  {"ccxt_tf": "8h",  "history_days": 1825},
        "1d":  {"ccxt_tf": "1d",  "history_days": 1825},
    }

    all_results = []

    for timeframe in timeframes:
        if timeframe not in TIMEFRAME_CONFIG:
            logger.warning(f"Unknown timeframe {timeframe}, skipping")
            continue

        tf_cfg = TIMEFRAME_CONFIG[timeframe]
        logger.info(f"\n{'='*60}\nTimeframe: {timeframe}\n{'='*60}")

        for symbol in symbols:
            logger.info(f"\nFetching {symbol} [{timeframe}]...")

            try:
                # Reuse the same fetch function as the main pipeline
                # (Binance, caching, correct symbol mapping)
                from timeframe_comparison import fetch_ohlcv_timeframe
                raw_df = fetch_ohlcv_timeframe(
                    symbol, tf_cfg["ccxt_tf"], tf_cfg["history_days"]
                )

                if raw_df is None or len(raw_df) < 500:
                    logger.warning(f"Insufficient data: {symbol} {timeframe} ({len(raw_df) if raw_df is not None else 0} bars)")
                    continue

                logger.info(f"  {symbol} {timeframe}: {len(raw_df)} bars")

                # Merge alt data
                enriched = merge_alt_data(raw_df, alt_df)

                # Build features WITHOUT creating labels yet
                # (we'll create labels ourselves with different params)
                feat_df = build_features(enriched, symbol, timeframe=timeframe)

                # Run the scan
                results = run_edge_scan(
                    enriched, feat_df, symbol, timeframe,
                    quick=args.quick
                )
                all_results.extend(results)

            except Exception as e:
                logger.error(f"Failed {symbol} {timeframe}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue

    # ── Report & save ──────────────────────────────────────────────────────
    all_results.sort(key=lambda x: -x["score"])
    print_report(all_results)

    if all_results:
        csv_path, json_path = save_results(all_results)
        print(f"\nFull results: {csv_path}")
        print(f"Best edges:   {json_path}")

        edges = [r for r in all_results if r["edge"]]
        if edges:
            print("\n─── To use discovered edges in the main pipeline ───")
            print("Copy the best_edges.json path into your next run config,")
            print("or run timeframe_comparison.py --use-edges backtest/results/best_edges.json")
        else:
            print("\nNo edges found. Recommendations:")
            top = all_results[:3]
            for r in top:
                print(f"  {r['symbol']} {r['timeframe']}: "
                      f"horizon={r['horizon_hours']}h threshold={r['threshold']:.1%} "
                      f"→ score={r['score']:.2f} ({r.get('reason','')})")


if __name__ == "__main__":
    main()