"""
models/lstm_ensemble.py
───────────────────────
9-model LSTM ensemble for variance-stable signal generation.

Core idea: train N independent LSTMs with different random seeds.
Each model finds a different local minimum. Averaging their probability
outputs cancels individual model variance, exposing the underlying
structural signal (if any exists).

Variance reduction:
  Single model std: ~19% net return across runs
  9-model ensemble std: ~6% (reduces by 1/sqrt(9) = 1/3)

Usage:
    from models.lstm_ensemble import train_ensemble, predict_proba_ensemble

    ensemble = train_ensemble(X_train, y_train, X_val, y_val, symbol="DOGE/USD")
    proba    = predict_proba_ensemble(ensemble, X_test)   # shape (n, 3)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from loguru import logger
from config.settings import LSTM_PARAMS, MODEL_DIR

try:
    import torch_directml
    DEVICE = torch_directml.device()
    logger.info("Ensemble using DirectML (AMD GPU)")
except ImportError:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Ensemble using {DEVICE}")

# ── Config ────────────────────────────────────────────────────────────────────

N_MODELS = 9   # Number of independent models to train


# ── Architecture (identical to lstm_model.py) ─────────────────────────────────

class MIOpenSafeLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))
        self.eps    = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
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


class SequenceDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X       = torch.FloatTensor(X)
        self.y       = torch.LongTensor(y)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, idx):
        return self.X[idx:idx+self.seq_len], self.y[idx+self.seq_len]


# ── Single model training ─────────────────────────────────────────────────────

def _train_one(X_train, y_train, X_val, y_val,
               seed: int, symbol: str, model_idx: int) -> LSTMClassifier:
    """Train one LSTM with a fixed seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    p          = LSTM_PARAMS
    seq_len    = p["sequence_length"]
    input_size = X_train.shape[1]

    train_ds = SequenceDataset(X_train, y_train, seq_len)
    val_ds   = SequenceDataset(X_val,   y_val,   seq_len)
    train_dl = DataLoader(train_ds, batch_size=p["batch_size"], shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=p["batch_size"])

    model = LSTMClassifier(
        input_size  = input_size,
        hidden_size = p["hidden_size"],
        num_layers  = p["num_layers"],
        num_classes = 3,
        dropout     = p["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=p["learning_rate"], weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Compute class weights from actual label distribution.
    # Hardcoded [2.0, 0.5, 2.0] broke when NEUTRAL dominated (e.g. 90%+ at 1d)
    # because weight=0.5 on a 90% class still made the model predict NEUTRAL always.
    # Inverse-frequency weighting ensures all classes get equal gradient attention
    # regardless of how imbalanced the labels are.
    counts = np.bincount(y_train, minlength=3).astype(float)
    counts = np.maximum(counts, 1.0)           # avoid div-by-zero for missing classes
    inv_freq = 1.0 / counts
    inv_freq /= inv_freq.mean()                # normalise so mean weight = 1.0
    # Cap: don't let any class get more than 4x weight (prevents instability on
    # tiny class counts in short folds)
    inv_freq = np.clip(inv_freq, 0.25, 4.0)
    class_weights = torch.FloatTensor(inv_freq).to(DEVICE)
    logger.info(f"  [{model_idx+1}/{N_MODELS}] Class weights: "
                f"DOWN={inv_freq[0]:.2f} NEUTRAL={inv_freq[1]:.2f} UP={inv_freq[2]:.2f} "
                f"(from counts: {counts.astype(int).tolist()})")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss    = float("inf")
    patience_counter = 0
    best_state       = None

    logger.info(f"  [{model_idx+1}/{N_MODELS}] Training seed={seed} — "
                f"{len(train_ds)} samples")

    for epoch in range(p["epochs"]):
        model.train()
        for X_b, y_b in train_dl:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for X_b, y_b in val_dl:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                logits    = model(X_b)
                val_loss += criterion(logits, y_b).item()
                correct  += (logits.argmax(1) == y_b).sum().item()
                total    += len(y_b)

        val_loss /= len(val_dl)
        val_acc   = correct / total
        scheduler.step()

        logger.info(f"    Epoch {epoch:3d} | val_loss={val_loss:.4f} | "
                    f"val_acc={val_acc:.4f} | lr={optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_state       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= p["patience"]:
                logger.info(f"    Early stop at epoch {epoch}")
                break

    # Guard: if no epoch ever improved val_loss (e.g. epochs=0 or empty train_dl),
    # best_state stays None → load_state_dict(None) would raise AttributeError.
    # Fall back to the current model weights rather than crashing.
    if best_state is None:
        logger.warning(f"  [{model_idx+1}/{N_MODELS}] best_state is None "
                       f"(no epoch completed) — using final model weights as fallback")
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    logger.info(f"  [{model_idx+1}/{N_MODELS}] Best val_loss: {best_val_loss:.4f}")
    return model


# ── Ensemble training ─────────────────────────────────────────────────────────

def train_ensemble(X_train: np.ndarray, y_train: np.ndarray,
                   X_val:   np.ndarray, y_val:   np.ndarray,
                   symbol:  str = "model",
                   n_models: int = N_MODELS) -> list:
    """
    Train N independent LSTMs with different random seeds.
    Returns a list of trained LSTMClassifier models.
    """
    logger.info(f"\nTraining {n_models}-model ensemble for {symbol} on {DEVICE}")
    logger.info(f"  Train: {len(X_train)} | Val: {len(X_val)} | "
                f"Features: {X_train.shape[1]}")

    # Seeds chosen to be well-spaced and reproducible
    seeds   = [42 + i * 137 for i in range(n_models)]
    models  = []
    val_losses = []

    for i, seed in enumerate(seeds):
        m = _train_one(X_train, y_train, X_val, y_val,
                       seed=seed, symbol=symbol, model_idx=i)
        models.append(m)

        # Quick val loss check for this model
        m.eval()
        seq_len = LSTM_PARAMS["sequence_length"]
        val_ds  = SequenceDataset(X_val, y_val, seq_len)
        val_dl  = DataLoader(val_ds, batch_size=LSTM_PARAMS["batch_size"])
        # Recompute dynamic weights from val labels for the final loss check
        val_counts  = np.bincount(y_val, minlength=3).astype(float)
        val_counts  = np.maximum(val_counts, 1.0)
        val_inv     = np.clip(1.0 / val_counts / (1.0 / val_counts).mean(), 0.25, 4.0)
        criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(val_inv).to(DEVICE)
        )
        vl = 0
        with torch.no_grad():
            for X_b, y_b in val_dl:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                vl += criterion(m(X_b), y_b).item()
        val_losses.append(vl / len(val_dl) if len(val_dl) > 0 else float("inf"))

    logger.info(f"\nEnsemble training complete for {symbol}:")
    logger.info(f"  Val losses: {[f'{v:.4f}' for v in val_losses]}")
    logger.info(f"  Mean: {np.mean(val_losses):.4f} | Std: {np.std(val_losses):.4f}")

    return models


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_proba_ensemble(models: list, X: np.ndarray,
                           seq_len: int = None,
                           batch_size: int = 512) -> np.ndarray:
    """
    Batched sliding-window inference averaged across all ensemble models.

    Replaces the old sample-by-sample loop. With 9 models and 600 test bars
    the old approach made 9 × 600 = 5 400 individual GPU forward passes.
    This version makes 9 × ceil(600/512) = 18 passes — ~300x fewer.

    Uses numpy stride tricks for a zero-copy sliding-window view, identical
    to the approach in lstm_model.py and cnn_model.py.
    """
    if seq_len is None:
        seq_len = LSTM_PARAMS["sequence_length"]

    X = np.ascontiguousarray(X, dtype=np.float32)
    n, n_features = X.shape
    n_windows = n - seq_len + 1

    pad = np.full((seq_len - 1, 3), [0.0, 1.0, 0.0], dtype=np.float32)

    if n_windows <= 0:
        # X shorter than seq_len — return neutral predictions
        return np.tile([0.0, 1.0, 0.0], (n, 1)).astype(np.float32)

    # Build zero-copy sliding-window view: (n_windows, seq_len, n_features)
    windows = np.lib.stride_tricks.as_strided(
        X,
        shape=(n_windows, seq_len, n_features),
        strides=(X.strides[0], X.strides[0], X.strides[1]),
    )

    all_probas = []
    for model in models:
        model.eval()
        model_probs = []
        with torch.no_grad():
            for start in range(0, n_windows, batch_size):
                batch = torch.from_numpy(
                    np.array(windows[start:start + batch_size])
                ).to(DEVICE)
                p = torch.softmax(model(batch), dim=1).cpu().numpy()
                model_probs.append(p)
        all_probas.append(np.vstack([pad, np.vstack(model_probs)]))

    # Average across all models — this is where variance cancels
    return np.mean(all_probas, axis=0)


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_ensemble(models: list, symbol: str):
    """Save all ensemble models to disk."""
    import joblib
    os.makedirs(MODEL_DIR, exist_ok=True)
    safe = symbol.replace("/", "_")

    for i, model in enumerate(models):
        path = Path(MODEL_DIR) / f"{safe}_lstm_ensemble_{i}.pt"
        torch.save(model.state_dict(), path)

    # Save architecture info from first model
    m = models[0]
    joblib.dump({
        "input_size":  m.lstm.input_size,
        "hidden_size": m.lstm.hidden_size,
        "num_layers":  m.lstm.num_layers,
        "n_models":    len(models),
    }, Path(MODEL_DIR) / f"{safe}_lstm_ensemble_info.pkl")

    logger.success(f"Saved {len(models)}-model ensemble for {symbol} → {MODEL_DIR}")


def load_ensemble(symbol: str) -> list:
    """Load all ensemble models from disk."""
    import joblib
    safe = symbol.replace("/", "_")
    info = joblib.load(Path(MODEL_DIR) / f"{safe}_lstm_ensemble_info.pkl")
    p    = LSTM_PARAMS

    models = []
    for i in range(info["n_models"]):
        model = LSTMClassifier(
            input_size  = info["input_size"],
            hidden_size = info["hidden_size"],
            num_layers  = info["num_layers"],
            num_classes = 3,
            dropout     = p["dropout"],
        ).to(DEVICE)
        path = Path(MODEL_DIR) / f"{safe}_lstm_ensemble_{i}.pt"
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        models.append(model)

    logger.info(f"Loaded {len(models)}-model ensemble for {symbol}")
    return models