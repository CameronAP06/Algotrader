"""
src/trader.py
─────────────
Core paper trading logic.

Each run (every 4h):
  1. Fetch latest DOGE/USD 4h candles from Kraken
  2. Engineer features (same pipeline as training)
  3. Load frozen LSTM, run inference
  4. Check if signal fires on the latest bar
  5. If open position, check if it should be closed
  6. Log everything to trades.csv
"""

import os
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import ccxt
import joblib

from datetime import datetime, timezone
from pathlib import Path
from loguru import logger

from src.features import build_features, get_feature_columns, fit_scaler, apply_scaler

# ── Config ────────────────────────────────────────────────────────────────────

SYMBOL        = "DOGE/USD"
TIMEFRAME     = "4h"
HISTORY_DAYS  = 60          # enough for feature engineering (200 bars needed)
SEQ_LEN       = 24
TOP_PCT       = 0.15
STOP_LOSS     = 0.05        # 5% stop loss on paper positions
TAKE_PROFIT   = 0.10        # 10% take profit
FEE_RATE      = 0.001       # 0.1% per side (Kraken taker)
INITIAL_CAPITAL = 1000.0    # paper money starting balance

DATA_DIR      = Path("data")
MODEL_DIR     = Path("models")
TRADES_CSV    = DATA_DIR / "trades.csv"
STATE_FILE    = DATA_DIR / "state.json"

TRADES_HEADER = [
    "timestamp", "action", "price", "signal", "confidence",
    "up_prob", "down_prob", "position_pnl_pct", "portfolio_value",
    "hold_bars", "reason"
]

# ── LSTM Model Definition ─────────────────────────────────────────────────────
# Must match the architecture used during training exactly

class MIOpenSafeLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))
        self.eps    = eps

    def forward(self, x):
        mean  = x.mean(dim=-1, keepdim=True)
        var   = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=0.0)
        self.norm    = MIOpenSafeLayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.norm(out[:, -1, :])
        out    = self.dropout(out)
        return self.fc(out)


# ── Data Fetching ─────────────────────────────────────────────────────────────

def fetch_candles() -> pd.DataFrame:
    """Fetch recent DOGE/USD 4h candles from Kraken."""
    exchange = ccxt.kraken({"enableRateLimit": True})
    from datetime import timedelta
    since_ms = int((datetime.utcnow() - timedelta(days=HISTORY_DAYS)).timestamp() * 1000)

    all_candles = []
    while True:
        candles = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since_ms, limit=720)
        if not candles:
            break
        all_candles.extend(candles)
        last_ts = candles[-1][0]
        # Stop when we reach near-current time
        if last_ts >= int((datetime.utcnow() - timedelta(hours=5)).timestamp() * 1000):
            break
        since_ms = last_ts + 1

    if not all_candles:
        raise RuntimeError(f"No candles returned for {SYMBOL}")

    df = pd.DataFrame(all_candles,
                      columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Fetched {len(df)} candles "
                f"({df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()})")
    return df


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_model() -> tuple:
    """Load frozen LSTM + scaler + feature info."""
    device = torch.device("cpu")  # Cloud inference on CPU

    info   = joblib.load(MODEL_DIR / "lstm_info.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")

    model = LSTMClassifier(
        input_size  = info["input_size"],
        hidden_size = info["hidden_size"],
        num_layers  = info["num_layers"],
        num_classes = 3,
        dropout     = info["dropout"],
    ).to(device)

    model.load_state_dict(
        torch.load(MODEL_DIR / "lstm_doge.pt", map_location=device)
    )
    model.eval()
    logger.info(f"Loaded LSTM: input={info['input_size']} hidden={info['hidden_size']}")
    return model, scaler, device


# ── Inference ─────────────────────────────────────────────────────────────────

def get_latest_signal(model, scaler, device, df: pd.DataFrame) -> dict:
    """
    Run inference on the latest bar only.
    Returns dict with signal, confidence, up_prob, down_prob.
    """
    feat_df   = build_features(df, timeframe=TIMEFRAME)
    feat_cols = get_feature_columns(feat_df)
    X = feat_df[feat_cols].values.astype(np.float32)
    X = apply_scaler(scaler, X)

    if len(X) < SEQ_LEN:
        return {"signal": "HOLD", "confidence": 0.0, "up_prob": 0.0, "down_prob": 0.0}

    # Only run inference on last bar — no need to score entire history
    seq = torch.FloatTensor(X[-SEQ_LEN:]).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(seq)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    down_p, neutral_p, up_p = float(probs[0]), float(probs[1]), float(probs[2])
    best_p = max(up_p, down_p)

    # Use same top-15% percentile logic as training, approximated for single bar:
    # We need a reference distribution — score last 200 bars to find percentile threshold
    all_probs = []
    n = min(len(X), 300)
    for i in range(SEQ_LEN, n + 1):
        s = torch.FloatTensor(X[i-SEQ_LEN:i]).unsqueeze(0).to(device)
        with torch.no_grad():
            p = torch.softmax(model(s), dim=1).cpu().numpy()[0]
        all_probs.append(max(p[0], p[2]))

    candidates = [p for p in all_probs if p > 0.34]
    if len(candidates) >= 5:
        threshold = max(float(np.percentile(candidates, (1 - TOP_PCT) * 100)), 0.34)
    else:
        threshold = 0.40

    if up_p >= threshold:
        signal = "BUY"
    elif down_p >= threshold:
        signal = "SELL"
    else:
        signal = "HOLD"

    latest_ts = feat_df.index[-1] if isinstance(feat_df.index, pd.DatetimeIndex) \
                else feat_df["timestamp"].iloc[-1] if "timestamp" in feat_df.columns \
                else "unknown"

    logger.info(f"Signal: {signal} | up={up_p:.3f} down={down_p:.3f} "
                f"neutral={neutral_p:.3f} | threshold={threshold:.3f} | bar={latest_ts}")

    return {
        "signal":     signal,
        "confidence": best_p,
        "up_prob":    up_p,
        "down_prob":  down_p,
        "threshold":  threshold,
    }


# ── State Management ──────────────────────────────────────────────────────────

def load_state() -> dict:
    import json
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "position":        None,   # None | "LONG" | "SHORT"
        "entry_price":     None,
        "entry_time":      None,
        "hold_bars":       0,
        "portfolio_value": INITIAL_CAPITAL,
        "total_trades":    0,
        "winning_trades":  0,
    }


def save_state(state: dict):
    import json
    DATA_DIR.mkdir(exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ── Trade Logging ─────────────────────────────────────────────────────────────

def log_trade(row: dict):
    DATA_DIR.mkdir(exist_ok=True)
    write_header = not TRADES_CSV.exists()
    with open(TRADES_CSV, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=TRADES_HEADER)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in TRADES_HEADER})


# ── Core Logic ────────────────────────────────────────────────────────────────

def run_paper_trade() -> dict | None:
    """
    Main function called every 4h.
    Returns a dict describing what happened (for Telegram notification),
    or None if nothing notable occurred (HOLD with no open position).
    """
    DATA_DIR.mkdir(exist_ok=True)

    # 1. Fetch data + run model
    df     = fetch_candles()
    model, scaler, device = load_model()
    sig    = get_latest_signal(model, scaler, device, df)
    state  = load_state()

    current_price = float(df["close"].iloc[-1])
    now           = datetime.now(timezone.utc).isoformat()
    signal        = sig["signal"]
    event         = None   # Will be populated if something happens

    # 2. Check existing position for exit conditions
    if state["position"] is not None:
        entry   = state["entry_price"]
        pos     = state["position"]
        bars    = state["hold_bars"] + 1
        state["hold_bars"] = bars

        if pos == "LONG":
            pnl_pct = (current_price - entry) / entry
        else:  # SHORT
            pnl_pct = (entry - current_price) / entry

        # Determine exit reason
        exit_reason = None
        if pnl_pct <= -STOP_LOSS:
            exit_reason = f"STOP_LOSS ({pnl_pct:+.1%})"
        elif pnl_pct >= TAKE_PROFIT:
            exit_reason = f"TAKE_PROFIT ({pnl_pct:+.1%})"
        elif pos == "LONG"  and signal == "SELL":
            exit_reason = f"SIGNAL_REVERSAL → SELL ({pnl_pct:+.1%})"
        elif pos == "SHORT" and signal == "BUY":
            exit_reason = f"SIGNAL_REVERSAL → BUY ({pnl_pct:+.1%})"

        if exit_reason:
            # Close position — apply fees both ways
            net_pnl   = pnl_pct - 2 * FEE_RATE
            old_val   = state["portfolio_value"]
            new_val   = old_val * (1 + net_pnl)
            state["portfolio_value"] = new_val
            state["total_trades"]   += 1
            if net_pnl > 0:
                state["winning_trades"] += 1

            log_trade({
                "timestamp":       now,
                "action":          f"CLOSE_{pos}",
                "price":           current_price,
                "signal":          signal,
                "confidence":      round(sig["confidence"], 4),
                "up_prob":         round(sig["up_prob"], 4),
                "down_prob":       round(sig["down_prob"], 4),
                "position_pnl_pct":round(net_pnl * 100, 2),
                "portfolio_value": round(new_val, 2),
                "hold_bars":       bars,
                "reason":          exit_reason,
            })

            wr = state["winning_trades"] / state["total_trades"] * 100
            event = {
                "type":       "CLOSE",
                "position":   pos,
                "entry":      entry,
                "exit":       current_price,
                "pnl_pct":    net_pnl * 100,
                "portfolio":  new_val,
                "reason":     exit_reason,
                "total_trades": state["total_trades"],
                "win_rate":   wr,
            }

            state["position"]    = None
            state["entry_price"] = None
            state["entry_time"]  = None
            state["hold_bars"]   = 0

    # 3. Open new position if signal fires and no existing position
    if state["position"] is None and signal in ("BUY", "SELL"):
        pos_type = "LONG" if signal == "BUY" else "SHORT"
        state["position"]    = pos_type
        state["entry_price"] = current_price
        state["entry_time"]  = now
        state["hold_bars"]   = 0

        log_trade({
            "timestamp":       now,
            "action":          f"OPEN_{pos_type}",
            "price":           current_price,
            "signal":          signal,
            "confidence":      round(sig["confidence"], 4),
            "up_prob":         round(sig["up_prob"], 4),
            "down_prob":       round(sig["down_prob"], 4),
            "position_pnl_pct": 0.0,
            "portfolio_value": round(state["portfolio_value"], 2),
            "hold_bars":       0,
            "reason":          f"Signal {signal} conf={sig['confidence']:.3f}",
        })

        event = {
            "type":      "OPEN",
            "position":  pos_type,
            "signal":    signal,
            "price":     current_price,
            "portfolio": state["portfolio_value"],
            "confidence":sig["confidence"],
            "up_prob":   sig["up_prob"],
            "down_prob": sig["down_prob"],
        }

    # 4. Periodic status log even on HOLD (every 6 bars = 24h)
    elif signal == "HOLD":
        log_trade({
            "timestamp":       now,
            "action":          "HOLD",
            "price":           current_price,
            "signal":          "HOLD",
            "confidence":      round(sig["confidence"], 4),
            "up_prob":         round(sig["up_prob"], 4),
            "down_prob":       round(sig["down_prob"], 4),
            "position_pnl_pct":"",
            "portfolio_value": round(state["portfolio_value"], 2),
            "hold_bars":       state.get("hold_bars", 0),
            "reason":          "No signal",
        })

    save_state(state)
    return event
