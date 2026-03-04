"""
models/cnn_model.py
1D Convolutional Neural Network for directional price prediction.
Treats the feature sequence like a signal — great at catching short-term
momentum patterns. Much faster to train than LSTM on CPU.
Drop-in replacement for xgb_model.py.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from loguru import logger
import joblib
from config.settings import MODEL_DIR, LSTM_PARAMS

try:
    import torch_directml
    DEVICE = torch_directml.device()
    logger.info("CNN using DirectML (AMD GPU)")
except ImportError:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"CNN using {DEVICE}")

CNN_PARAMS = {
    "sequence_length": 24,   # Look-back window in hours
    "num_filters":     64,   # Convolutional filters per layer
    "kernel_size":     3,    # Filter width
    "num_layers":      3,    # Number of conv blocks
    "dropout":         0.2,  # nn.Dropout only — no MIOpen kernel issues
    "epochs":          30,
    "batch_size":      128,
    "learning_rate":   0.001,
    "patience":        7,
}


# ─── Dataset ────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X      = torch.FloatTensor(np.array(X))  # np.array() makes a writable copy
        self.y      = torch.LongTensor(np.array(y))
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        # CNN expects (channels, length) — features are channels, time is length
        x_seq = self.X[idx : idx + self.seq_len].T   # (features, seq_len)
        label = self.y[idx + self.seq_len]
        return x_seq, label


# ─── Model ──────────────────────────────────────────────────────────────────

class CNNClassifier(nn.Module):
    def __init__(self, input_size: int, num_filters: int, kernel_size: int,
                 num_layers: int, num_classes: int, dropout: float):
        super().__init__()

        layers = []
        in_channels = input_size
        for i in range(num_layers):
            out_channels = num_filters * (2 ** min(i, 1))  # 64 -> 128 -> 128 (cap at 2x)
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
        self.avg_pool    = nn.AdaptiveAvgPool1d(1)   # Average across time
        self.max_pool    = nn.AdaptiveMaxPool1d(1)   # Peak activation across time
        # Concat avg+max doubles channels into classifier — captures both
        # mean trend strength and maximum pattern response
        self.fc          = nn.Linear(in_channels * 2, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)              # (batch, channels, seq_len)
        avg = self.avg_pool(x).squeeze(-1)   # (batch, channels)
        mx  = self.max_pool(x).squeeze(-1)   # (batch, channels)
        x   = torch.cat([avg, mx], dim=1)    # (batch, channels * 2)
        return self.fc(x)


# ─── Training ───────────────────────────────────────────────────────────────

def train(X_train: np.ndarray, y_train: np.ndarray,
          X_val:   np.ndarray, y_val:   np.ndarray,
          symbol:  str = "model") -> CNNClassifier:

    p          = CNN_PARAMS
    seq_len    = p["sequence_length"]
    input_size = X_train.shape[1]

    train_ds = SequenceDataset(X_train, y_train, seq_len)
    val_ds   = SequenceDataset(X_val,   y_val,   seq_len)
    train_dl = DataLoader(train_ds, batch_size=p["batch_size"], shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=p["batch_size"], shuffle=False, num_workers=0)

    model = CNNClassifier(
        input_size  = input_size,
        num_filters = p["num_filters"],
        kernel_size = p["kernel_size"],
        num_layers  = p["num_layers"],
        num_classes = 3,
        dropout     = p["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=p["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    class_weights = torch.FloatTensor([2.0, 0.5, 2.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss    = float("inf")
    patience_counter = 0
    best_state       = None

    logger.info(f"Training 1D CNN for {symbol} on {DEVICE} — {len(train_ds)} train samples")

    for epoch in range(p["epochs"]):
        # Train
        model.train()
        for X_b, y_b in train_dl:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validate
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

        logger.info(f"  Epoch {epoch:3d} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | lr={optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_state       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= p["patience"]:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    logger.success(f"1D CNN best val_loss: {best_val_loss:.4f}")
    return model


def save(model: CNNClassifier, symbol: str):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = Path(MODEL_DIR) / f"{symbol.replace('/','_')}_cnn.pt"
    torch.save(model.state_dict(), path)
    info_path = Path(MODEL_DIR) / f"{symbol.replace('/','_')}_cnn_info.pkl"
    joblib.dump({
        "input_size":  model.fc.in_features,
        "num_filters": CNN_PARAMS["num_filters"],
        "kernel_size": CNN_PARAMS["kernel_size"],
        "num_layers":  CNN_PARAMS["num_layers"],
    }, info_path)
    logger.info(f"Saved 1D CNN -> {path}")
    return path


def load(symbol: str) -> CNNClassifier:
    info_path = Path(MODEL_DIR) / f"{symbol.replace('/','_')}_cnn_info.pkl"
    info      = joblib.load(info_path)
    p         = CNN_PARAMS
    model     = CNNClassifier(
        input_size  = info["input_size"],
        num_filters = info["num_filters"],
        kernel_size = info["kernel_size"],
        num_layers  = info["num_layers"],
        num_classes = 3,
        dropout     = p["dropout"],
    ).to(DEVICE)
    path = Path(MODEL_DIR) / f"{symbol.replace('/','_')}_cnn.pt"
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model


def predict_proba(model: CNNClassifier, X: np.ndarray) -> np.ndarray:
    seq_len = CNN_PARAMS["sequence_length"]
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(seq_len, len(X) + 1):
            seq = torch.FloatTensor(X[i-seq_len:i].T).unsqueeze(0).to(DEVICE)
            p   = torch.softmax(model(seq), dim=1).cpu().detach().numpy()[0]
            probs.append(p)
    # Pad start with neutral probability to match input length
    pad = np.full((seq_len - 1, 3), [0.0, 1.0, 0.0])
    return np.vstack([pad, np.array(probs)])