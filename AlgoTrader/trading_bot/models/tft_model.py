"""
models/tft_model.py
Temporal Fusion Transformer (TFT) for directional price prediction.
Based on the architecture from Lim et al. (2019) — designed specifically
for multi-horizon time series forecasting with interpretable attention.

Key advantages over LSTM/CNN:
  - Multi-head attention captures long-range dependencies
  - Gating mechanisms suppress irrelevant features automatically
  - Variable selection network learns which features matter most
  - Significantly more expressive than tree-based or simple RNN models

Drop-in replacement for catboost_model.py.
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
from config.settings import MODEL_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TFT_PARAMS = {
    "sequence_length":  48,    # 48h look-back window
    "d_model":          64,    # Model dimension
    "n_heads":          4,     # Attention heads
    "n_lstm_layers":    2,     # LSTM layers in encoder/decoder
    "dropout":          0.1,
    "epochs":           40,
    "batch_size":       128,
    "learning_rate":    0.0005,
    "patience":         8,
}


# ─── Dataset ────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X       = torch.FloatTensor(np.array(X))
        self.y       = torch.LongTensor(np.array(y))
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]   # (seq_len, features)
        label = self.y[idx + self.seq_len]
        return x_seq, label


# ─── TFT Building Blocks ────────────────────────────────────────────────────

class GatedLinearUnit(nn.Module):
    """GLU — gates irrelevant signals, core TFT component."""
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.fc1     = nn.Linear(d_model, d_model)
        self.fc2     = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.dropout(self.fc1(x)) * torch.sigmoid(self.fc2(x)) + x)


class VariableSelectionNetwork(nn.Module):
    """Learns which input features are most relevant at each timestep."""
    def __init__(self, input_size: int, d_model: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.weight_net = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.Softmax(dim=-1),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        weights = self.weight_net(x)             # (batch, seq_len, input_size)
        x_weighted = x * weights                 # Feature selection
        return self.norm(self.dropout(self.input_proj(x_weighted)))


class TemporalSelfAttention(nn.Module):
    """Multi-head self-attention over the temporal dimension."""
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn    = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(self.dropout(attn_out) + x)


# ─── Full TFT Model ─────────────────────────────────────────────────────────

class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size: int, d_model: int, n_heads: int,
                 n_lstm_layers: int, num_classes: int, dropout: float):
        super().__init__()

        # 1. Variable selection — learn which features matter
        self.var_select = VariableSelectionNetwork(input_size, d_model, dropout)

        # 2. Local processing — LSTM captures local temporal patterns
        self.lstm = nn.LSTM(
            d_model, d_model, n_lstm_layers,
            batch_first=True, dropout=dropout if n_lstm_layers > 1 else 0
        )
        self.lstm_glu = GatedLinearUnit(d_model, dropout)

        # 3. Temporal self-attention — captures long-range dependencies
        self.attention   = TemporalSelfAttention(d_model, n_heads, dropout)
        self.attn_glu    = GatedLinearUnit(d_model, dropout)

        # 4. Feed-forward layer
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.ff_norm = nn.LayerNorm(d_model)

        # 5. Classification head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.var_select(x)                    # Feature selection
        lstm_out, _ = self.lstm(x)                # Local patterns
        x = self.lstm_glu(lstm_out)               # Gate LSTM output
        x = self.attention(x)                     # Long-range patterns
        x = self.attn_glu(x)                      # Gate attention output
        x = self.ff_norm(self.ff(x[:, -1, :]) + x[:, -1, :])  # Last timestep
        return self.classifier(x)


# ─── Training ───────────────────────────────────────────────────────────────

def train(X_train: np.ndarray, y_train: np.ndarray,
          X_val:   np.ndarray, y_val:   np.ndarray,
          symbol:  str = "model") -> TemporalFusionTransformer:

    p          = TFT_PARAMS
    seq_len    = p["sequence_length"]
    input_size = X_train.shape[1]

    train_ds = SequenceDataset(X_train, y_train, seq_len)
    val_ds   = SequenceDataset(X_val,   y_val,   seq_len)
    train_dl = DataLoader(train_ds, batch_size=p["batch_size"], shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=p["batch_size"], shuffle=False, num_workers=0)

    model = TemporalFusionTransformer(
        input_size   = input_size,
        d_model      = p["d_model"],
        n_heads      = p["n_heads"],
        n_lstm_layers= p["n_lstm_layers"],
        num_classes  = 3,
        dropout      = p["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=p["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p["epochs"])

    # Weight UP/DOWN 4x higher than NEUTRAL to combat class imbalance
    class_weights = torch.FloatTensor([2.0, 0.5, 2.0]).to(DEVICE)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss    = float("inf")
    patience_counter = 0
    best_state       = None

    logger.info(f"Training TFT for {symbol} on {DEVICE} — {len(train_ds)} train samples")

    for epoch in range(p["epochs"]):
        # Train
        model.train()
        train_loss = 0
        for X_b, y_b in train_dl:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

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

        if epoch % 5 == 0:
            logger.info(f"  Epoch {epoch:3d} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

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
    logger.success(f"TFT best val_loss: {best_val_loss:.4f}")
    return model


def save(model: TemporalFusionTransformer, symbol: str):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = Path(MODEL_DIR) / f"{symbol.replace('/','_')}_tft.pt"
    torch.save(model.state_dict(), path)
    info_path = Path(MODEL_DIR) / f"{symbol.replace('/','_')}_tft_info.pkl"
    joblib.dump({
        "input_size":    model.classifier.in_features,
        "d_model":       TFT_PARAMS["d_model"],
        "n_heads":       TFT_PARAMS["n_heads"],
        "n_lstm_layers": TFT_PARAMS["n_lstm_layers"],
    }, info_path)
    logger.info(f"Saved TFT -> {path}")
    return path


def load(symbol: str) -> TemporalFusionTransformer:
    info_path = Path(MODEL_DIR) / f"{symbol.replace('/','_')}_tft_info.pkl"
    info      = joblib.load(info_path)
    p         = TFT_PARAMS
    model     = TemporalFusionTransformer(
        input_size    = info["input_size"],
        d_model       = info["d_model"],
        n_heads       = info["n_heads"],
        n_lstm_layers = info["n_lstm_layers"],
        num_classes   = 3,
        dropout       = p["dropout"],
    ).to(DEVICE)
    path = Path(MODEL_DIR) / f"{symbol.replace('/','_')}_tft.pt"
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model


def predict_proba(model: TemporalFusionTransformer, X: np.ndarray) -> np.ndarray:
    seq_len = TFT_PARAMS["sequence_length"]
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(seq_len, len(X) + 1):
            seq = torch.FloatTensor(np.array(X[i-seq_len:i])).unsqueeze(0).to(DEVICE)
            p   = torch.softmax(model(seq), dim=1).cpu().detach().numpy()[0]
            probs.append(p)
    pad = np.full((seq_len - 1, 3), [0.0, 1.0, 0.0])
    return np.vstack([pad, np.array(probs)])
