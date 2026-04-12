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

SYMBOLS       = ["DOGE/USD", "LINK/USD", "AAVE/USD"]
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

# ── Correlation cap ────────────────────────────────────────────────────────────
CORR_WINDOW    = 168   # bars used for rolling correlation (~1 week at 4h)
CORR_THRESHOLD = 0.70  # block new position if r > this with an open same-direction pos


def safe_name(symbol: str) -> str:
    return symbol.replace("/", "_").lower()


def _check_correlation_cap(symbol: str, new_position: str,
                            all_candles: dict, all_states: dict) -> bool:
    """
    Returns True if it's safe to open `new_position` on `symbol`.
    Returns False if a same-direction position is already open in a
    highly-correlated asset (|r| > CORR_THRESHOLD).

    Uses the last CORR_WINDOW bars of close-price returns for correlation.
    Skips the check gracefully if candles are missing or too short.
    """
    if symbol not in all_candles:
        return True

    try:
        new_ret = (all_candles[symbol]["close"]
                   .pct_change()
                   .tail(CORR_WINDOW)
                   .reset_index(drop=True))
    except Exception:
        return True

    for other_sym, state in all_states.items():
        if other_sym == symbol or not state.get("position"):
            continue
        other_pos = state["position"]
        # Only block if same direction (both LONG or both SHORT)
        if (new_position == "LONG" and other_pos != "LONG") or \
           (new_position == "SHORT" and other_pos != "SHORT"):
            continue
        if other_sym not in all_candles:
            continue
        try:
            other_ret = (all_candles[other_sym]["close"]
                         .pct_change()
                         .tail(CORR_WINDOW)
                         .reset_index(drop=True))
            combined  = pd.concat([new_ret, other_ret], axis=1).dropna()
            if len(combined) < 20:
                continue
            corr = combined.iloc[:, 0].corr(combined.iloc[:, 1])
            if abs(corr) > CORR_THRESHOLD:
                logger.info(f"Correlation cap: {symbol} {new_position} blocked by "
                            f"{other_sym} {other_pos} (r={corr:.2f})")
                return False
        except Exception:
            continue

    return True


# ── Model Architectures (must match train_and_backtest.py exactly) ────────────

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


class CNNClassifier(nn.Module):
    def __init__(self, input_size, num_filters, kernel_size, num_layers,
                 num_classes, dropout):
        super().__init__()
        layers = []
        in_ch  = input_size
        for i in range(num_layers):
            out_ch = num_filters * (2 ** min(i, 1))
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.conv_layers = nn.Sequential(*layers)
        self.avg_pool    = nn.AdaptiveAvgPool1d(1)
        self.max_pool    = nn.AdaptiveMaxPool1d(1)
        self.fc          = nn.Linear(in_ch * 2, num_classes)

    def forward(self, x):
        x   = self.conv_layers(x)
        avg = self.avg_pool(x).squeeze(-1)
        mx  = self.max_pool(x).squeeze(-1)
        return self.fc(torch.cat([avg, mx], dim=1))


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
    """
    Load the CatBoost + CNN + LSTM ensemble saved by train_and_backtest.py.
    File naming matches the main pipeline conventions:
      {SYMBOL_UPPER}_catboost.cbm / _cnn.pt / _lstm.pt / _scaler.pkl
    """
    upper  = symbol.replace("/", "_")
    device = torch.device("cpu")

    # Scaler
    scaler = joblib.load(MODEL_DIR / f"{upper}_scaler.pkl")

    # CatBoost
    from catboost import CatBoostClassifier
    cat_model = CatBoostClassifier()
    cat_model.load_model(str(MODEL_DIR / f"{upper}_catboost.cbm"))

    # CNN
    cnn_info = joblib.load(MODEL_DIR / f"{upper}_cnn_info.pkl")
    cnn_model = CNNClassifier(
        input_size  = cnn_info["input_size"],
        num_filters = cnn_info["num_filters"],
        kernel_size = cnn_info["kernel_size"],
        num_layers  = cnn_info["num_layers"],
        num_classes = 3,
        dropout     = 0.0,  # inference — no dropout
    ).to(device)
    cnn_model.load_state_dict(torch.load(MODEL_DIR / f"{upper}_cnn.pt", map_location=device))
    cnn_model.eval()

    # LSTM
    lstm_info = joblib.load(MODEL_DIR / f"{upper}_lstm_info.pkl")
    lstm_model = LSTMClassifier(
        input_size  = lstm_info["input_size"],
        hidden_size = lstm_info["hidden_size"],
        num_layers  = lstm_info["num_layers"],
        num_classes = 3,
        dropout     = 0.0,
    ).to(device)
    lstm_model.load_state_dict(torch.load(MODEL_DIR / f"{upper}_lstm.pt", map_location=device))
    lstm_model.eval()

    # Ensemble weights (DE-optimised)
    weights_path = MODEL_DIR / f"{upper}_ensemble_weights.pkl"
    weights = joblib.load(weights_path) if weights_path.exists() else [1/3, 1/3, 1/3]

    logger.info(f"Loaded CatBoost+CNN+LSTM ensemble for {symbol}")
    return cat_model, cnn_model, lstm_model, scaler, device, weights


def _predict_proba_batched(model, X: np.ndarray, seq_len: int,
                            model_type: str, device) -> np.ndarray:
    """
    Batched sliding-window inference for CNN or LSTM.
    Returns (n_samples, 3) probability array with leading padding rows.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    n, n_feat = X.shape
    n_windows = n - seq_len + 1
    if n_windows <= 0:
        return np.full((n, 3), [0.0, 1.0, 0.0], dtype=np.float32)

    windows = np.lib.stride_tricks.as_strided(
        X,
        shape=(n_windows, seq_len, n_feat),
        strides=(X.strides[0], X.strides[0], X.strides[1]),
    )

    batch_size = 256
    all_probs  = []
    with torch.no_grad():
        for start in range(0, n_windows, batch_size):
            batch_np = np.array(windows[start:start + batch_size])
            if model_type == "cnn":
                batch_np = batch_np.transpose(0, 2, 1)  # (B, features, seq_len)
            batch = torch.from_numpy(batch_np).to(device)
            probs = torch.softmax(model(batch), dim=1).cpu().numpy()
            all_probs.append(probs)

    pad = np.full((seq_len - 1, 3), [0.0, 1.0, 0.0], dtype=np.float32)
    return np.vstack([pad, np.vstack(all_probs)])


def get_signal(cat_model, cnn_model, lstm_model, scaler, device, weights,
               df: pd.DataFrame) -> dict:
    """
    Run CatBoost+CNN+LSTM ensemble inference on the latest bar.
    Uses batched stride-tricks inference — no sample-by-sample loop.
    """
    feat_df    = build_features(df, timeframe=TIMEFRAME)
    feat_cols  = get_feature_columns(feat_df)
    X          = scaler.transform(feat_df[feat_cols].values.astype(np.float32))

    if len(X) < SEQ_LEN:
        return {"signal": "HOLD", "confidence": 0.0, "up_prob": 0.0, "down_prob": 0.0}

    # CatBoost: flat features, no sequence
    cat_proba = cat_model.predict_proba(X)   # (n, 3)

    # CNN + LSTM: batched sliding-window over the whole array
    cnn_proba  = _predict_proba_batched(cnn_model,  X, SEQ_LEN, "cnn",  device)
    lstm_proba = _predict_proba_batched(lstm_model, X, SEQ_LEN, "lstm", device)

    # Align lengths (padding rows at start of seq models)
    n = min(len(cat_proba), len(cnn_proba), len(lstm_proba))
    blended = (weights[0] * cat_proba[-n:]
             + weights[1] * cnn_proba[-n:]
             + weights[2] * lstm_proba[-n:])

    # Use last N bars to compute percentile signal threshold
    recent    = min(n, 300)
    up_probs  = blended[-recent:, 2]
    dn_probs  = blended[-recent:, 0]
    best_prob = np.maximum(up_probs, dn_probs)
    candidates = best_prob[best_prob > 0.34]
    threshold  = (max(float(np.percentile(candidates, (1 - TOP_PCT) * 100)), 0.34)
                  if len(candidates) >= 5 else 0.40)

    # Signal from the very last bar
    up_p  = float(blended[-1, 2])
    down_p = float(blended[-1, 0])

    if   up_p   >= threshold: signal = "BUY"
    elif down_p >= threshold: signal = "SELL"
    else:                      signal = "HOLD"

    logger.info(f"Signal: {signal} | up={up_p:.3f} down={down_p:.3f} "
                f"threshold={threshold:.3f}")
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

def run_symbol(symbol: str,
               all_candles: dict = None,
               all_states: dict = None) -> dict | None:
    df    = all_candles[symbol] if all_candles and symbol in all_candles else fetch_candles(symbol)
    cat_model, cnn_model, lstm_model, scaler, device, weights = load_ensemble(symbol)
    sig = get_signal(cat_model, cnn_model, lstm_model, scaler, device, weights, df)
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
        # Correlation cap: don't open if a same-direction correlated trade is open
        _states_for_corr = all_states if all_states else {symbol: state}
        _candles_for_corr = all_candles if all_candles else {symbol: df}
        if not _check_correlation_cap(symbol, pos_type, _candles_for_corr, _states_for_corr):
            _log("HOLD", reason="Correlation cap — same-direction position open in correlated asset")
            save_state(symbol, state)
            return None
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
    """
    Run one cycle for all symbols. Returns list of events (may be empty).

    Fetches all candles upfront so the correlation cap can compare open
    positions across symbols without additional network calls.
    """
    DATA_DIR.mkdir(exist_ok=True)

    # Phase 1: fetch all candles (so correlation cap has all price series)
    all_candles: dict = {}
    for symbol in SYMBOLS:
        try:
            all_candles[symbol] = fetch_candles(symbol)
        except Exception as e:
            logger.error(f"{symbol} candle fetch failed: {e}")

    # Phase 2: load all current states
    all_states: dict = {sym: load_state(sym) for sym in SYMBOLS}

    # Phase 3: run each symbol with shared context
    events = []
    for symbol in SYMBOLS:
        if symbol not in all_candles:
            events.append({"type": "ERROR", "symbol": symbol,
                           "error": "Candle fetch failed"})
            continue
        try:
            event = run_symbol(symbol, all_candles=all_candles, all_states=all_states)
            if event:
                events.append(event)
            # Refresh state so the next symbol sees any position we just opened
            all_states[symbol] = load_state(symbol)
        except Exception as e:
            logger.error(f"{symbol} cycle failed: {e}")
            events.append({"type": "ERROR", "symbol": symbol, "error": str(e)})
    return events