"""
src/trader.py
─────────────
Paper trading logic for multiple symbols.
Each 4h cycle runs independently for each symbol.
"""

import os, csv, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import ccxt
import joblib

from datetime import datetime, timezone, timedelta
from pathlib import Path
from loguru import logger

from src.feature_engineer import build_features, get_feature_columns
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────

SYMBOLS       = ["DOGE/USD", "LINK/USD", "AAVE/USD", "XMR/USD"]
TIMEFRAME     = "4h"
HISTORY_DAYS  = 400  # 52w features need 365+ days
SEQ_LEN       = 24
TOP_PCT       = 0.15
STOP_LOSS     = 0.05
TAKE_PROFIT   = 0.10
FEE_RATE      = 0.001
INITIAL_CAPITAL = 1000.0

DATA_DIR  = Path("data")
MODEL_DIR = Path("models")

TRADES_HEADER = [
    "timestamp", "symbol", "action", "price", "signal", "confidence",
    "up_prob", "down_prob", "position_pnl_pct", "portfolio_value",
    "hold_bars", "reason"
]


def safe_name(symbol: str) -> str:
    return symbol.replace("/", "_").lower()


# ── LSTM Architecture ─────────────────────────────────────────────────────────

class MIOpenSafeLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))
        self.eps    = eps

    def forward(self, x):
        mean  = x.mean(dim=-1, keepdim=True)
        var   = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias


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
        return self.fc(self.dropout(self.norm(out[:, -1, :])))


# ── Data ──────────────────────────────────────────────────────────────────────

def fetch_candles(symbol: str) -> pd.DataFrame:
    exchange = ccxt.kraken({"enableRateLimit": True})
    since_ms = int((datetime.utcnow() - timedelta(days=HISTORY_DAYS)).timestamp() * 1000)
    all_candles = []
    while True:
        candles = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since_ms, limit=720)
        if not candles:
            break
        all_candles.extend(candles)
        if candles[-1][0] >= int((datetime.utcnow() - timedelta(hours=5)).timestamp() * 1000):
            break
        since_ms = candles[-1][0] + 1

    if not all_candles:
        raise RuntimeError(f"No candles returned for {symbol}")

    df = pd.DataFrame(all_candles, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    logger.info(f"{symbol}: {len(df)} candles fetched")
    return df


# ── Model ─────────────────────────────────────────────────────────────────────

def load_ensemble(symbol: str) -> tuple:
    """Load all N models for a symbol. Returns (models, scaler, device, info)."""
    name   = safe_name(symbol)
    device = torch.device("cpu")
    info   = joblib.load(MODEL_DIR / f"lstm_{name}_info.pkl")
    scaler = joblib.load(MODEL_DIR / f"scaler_{name}.pkl")
    n_models = info.get("n_models", 9)

    models = []
    for i in range(n_models):
        m = LSTMClassifier(
            input_size  = info["input_size"],
            hidden_size = info["hidden_size"],
            num_layers  = info["num_layers"],
            num_classes = 3,
            dropout     = info["dropout"],
        ).to(device)
        m.load_state_dict(torch.load(MODEL_DIR / f"lstm_{name}_{i}.pt", map_location=device))
        m.eval()
        models.append(m)

    logger.info(f"Loaded {len(models)}-model ensemble for {symbol}")
    return models, scaler, device, info


def get_signal(models: list, scaler, device, info, df: pd.DataFrame) -> dict:
    """Run ensemble inference — average probabilities across all N models."""
    feat_df = build_features(df, timeframe=TIMEFRAME)

    saved_cols = info.get("feature_cols") or get_feature_columns(feat_df)
    for col in saved_cols:
        if col not in feat_df.columns:
            feat_df[col] = 0.0

    X = scaler.transform(feat_df[saved_cols].values.astype(np.float32))

    if len(X) < SEQ_LEN:
        return {"signal": "HOLD", "confidence": 0.0, "up_prob": 0.0, "down_prob": 0.0}

    # Score recent bars across all models to find percentile threshold
    n = min(len(X), 300)
    all_best = []
    for i in range(SEQ_LEN, n + 1):
        seq = torch.FloatTensor(X[i-SEQ_LEN:i]).unsqueeze(0).to(device)
        bar_probs = []
        with torch.no_grad():
            for model in models:
                p = torch.softmax(model(seq), dim=1).cpu().numpy()[0]
                bar_probs.append(p)
        avg_p = np.mean(bar_probs, axis=0)
        all_best.append(max(avg_p[0], avg_p[2]))

    candidates = [p for p in all_best if p > 0.34]
    threshold  = max(float(np.percentile(candidates, (1 - TOP_PCT) * 100)), 0.34) \
                 if len(candidates) >= 5 else 0.40

    # Latest bar — average ensemble probabilities
    seq = torch.FloatTensor(X[-SEQ_LEN:]).unsqueeze(0).to(device)
    bar_probs = []
    with torch.no_grad():
        for model in models:
            p = torch.softmax(model(seq), dim=1).cpu().numpy()[0]
            bar_probs.append(p)
    probs = np.mean(bar_probs, axis=0)

    down_p, up_p = float(probs[0]), float(probs[2])

    if up_p >= threshold:     signal = "BUY"
    elif down_p >= threshold: signal = "SELL"
    else:                     signal = "HOLD"

    logger.info(f"Signal: {signal} | up={up_p:.3f} down={down_p:.3f} "
                f"threshold={threshold:.3f} ensemble={len(models)} models")
    return {"signal": signal, "confidence": max(up_p, down_p),
            "up_prob": up_p, "down_prob": down_p}


# ── State & Logging ───────────────────────────────────────────────────────────

def load_state(symbol: str) -> dict:
    path = DATA_DIR / f"state_{safe_name(symbol)}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"position": None, "entry_price": None, "entry_time": None,
            "hold_bars": 0, "portfolio_value": INITIAL_CAPITAL,
            "total_trades": 0, "winning_trades": 0}


def save_state(symbol: str, state: dict):
    DATA_DIR.mkdir(exist_ok=True)
    with open(DATA_DIR / f"state_{safe_name(symbol)}.json", "w") as f:
        json.dump(state, f, indent=2, default=str)


def log_trade(row: dict):
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / "trades.csv"
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=TRADES_HEADER)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in TRADES_HEADER})


# ── Core ──────────────────────────────────────────────────────────────────────

def run_symbol(symbol: str) -> dict | None:
    df                          = fetch_candles(symbol)
    models, scaler, device, info = load_ensemble(symbol)
    sig                         = get_signal(models, scaler, device, info, df)
    state          = load_state(symbol)
    current_price  = float(df["close"].iloc[-1])
    now            = datetime.now(timezone.utc).isoformat()
    signal         = sig["signal"]
    event          = None

    def _log(action, pnl="", reason=""):
        log_trade({
            "timestamp": now, "symbol": symbol, "action": action,
            "price": current_price, "signal": signal,
            "confidence": round(sig["confidence"], 4),
            "up_prob": round(sig["up_prob"], 4),
            "down_prob": round(sig["down_prob"], 4),
            "position_pnl_pct": pnl,
            "portfolio_value": round(state["portfolio_value"], 2),
            "hold_bars": state.get("hold_bars", 0),
            "reason": reason,
        })

    # Check open position for exit
    if state["position"]:
        pos   = state["position"]
        entry = state["entry_price"]
        bars  = state["hold_bars"] + 1
        state["hold_bars"] = bars

        pnl_pct = (current_price - entry) / entry if pos == "LONG" \
                  else (entry - current_price) / entry

        exit_reason = None
        if pnl_pct <= -STOP_LOSS:
            exit_reason = f"STOP_LOSS ({pnl_pct:+.1%})"
        elif pnl_pct >= TAKE_PROFIT:
            exit_reason = f"TAKE_PROFIT ({pnl_pct:+.1%})"
        elif pos == "LONG"  and signal == "SELL":
            exit_reason = f"SIGNAL_REVERSAL ({pnl_pct:+.1%})"
        elif pos == "SHORT" and signal == "BUY":
            exit_reason = f"SIGNAL_REVERSAL ({pnl_pct:+.1%})"

        if exit_reason:
            net_pnl = pnl_pct - 2 * FEE_RATE
            state["portfolio_value"] *= (1 + net_pnl)
            state["total_trades"]    += 1
            if net_pnl > 0:
                state["winning_trades"] += 1

            _log(f"CLOSE_{pos}", round(net_pnl * 100, 2), exit_reason)
            event = {"type": "CLOSE", "symbol": symbol, "position": pos,
                     "entry": entry, "exit": current_price,
                     "pnl_pct": net_pnl * 100,
                     "portfolio": state["portfolio_value"],
                     "reason": exit_reason,
                     "total_trades": state["total_trades"],
                     "win_rate": state["winning_trades"] / state["total_trades"] * 100}

            state.update({"position": None, "entry_price": None,
                          "entry_time": None, "hold_bars": 0})

    # Open new position
    if state["position"] is None and signal in ("BUY", "SELL"):
        pos_type = "LONG" if signal == "BUY" else "SHORT"
        state.update({"position": pos_type, "entry_price": current_price,
                      "entry_time": now, "hold_bars": 0})
        _log(f"OPEN_{pos_type}", reason=f"{signal} conf={sig['confidence']:.3f}")
        event = {"type": "OPEN", "symbol": symbol, "position": pos_type,
                 "signal": signal, "price": current_price,
                 "portfolio": state["portfolio_value"],
                 "confidence": sig["confidence"],
                 "up_prob": sig["up_prob"], "down_prob": sig["down_prob"]}
    else:
        _log("HOLD", reason="No signal")

    save_state(symbol, state)
    return event


def run_paper_trade() -> list:
    """Run one cycle for all symbols. Returns list of events (may be empty)."""
    DATA_DIR.mkdir(exist_ok=True)
    events = []
    for symbol in SYMBOLS:
        try:
            event = run_symbol(symbol)
            if event:
                events.append(event)
        except Exception as e:
            logger.error(f"{symbol} cycle failed: {e}")
            events.append({"type": "ERROR", "symbol": symbol, "error": str(e)})
    return events