"""
models/lstm_model.py
PyTorch LSTM for sequence-based price direction prediction.
Treats the last N candles as a sequence to capture temporal patterns.
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
    logger.info("LSTM using DirectML (AMD GPU)")
except ImportError:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"LSTM using {DEVICE}")

class MIOpenSafeLayerNorm(nn.Module):
    """LayerNorm avoiding MIOpen's broken gfx1100+MSVC14.39 reduction kernel."""
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))
        self.eps    = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias



# ─── Dataset ────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]
        label = self.y[idx + self.seq_len]
        return x_seq, label


# ─── Model ──────────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 num_classes: int, dropout: float):
        super().__init__()
        # nn.LSTM dropout triggers MIOpen JIT kernel — broken on gfx1100+MSVC14.39
        # dropout=0.0 in LSTM; nn.Dropout below applies correctly via Python
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.0  # MIOpen fused dropout disabled
        )
        self.norm   = MIOpenSafeLayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc     = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])   # Take last timestep
        out = self.dropout(out)
        return self.fc(out)


# ─── Training ───────────────────────────────────────────────────────────────

def train(X_train: np.ndarray, y_train: np.ndarray,
          X_val:   np.ndarray, y_val:   np.ndarray,
          symbol: str = "model") -> LSTMClassifier:

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=p["learning_rate"], weight_decay=1e-4)
    # CosineAnnealingWarmRestarts: LR decays and resets every T_0 epochs.
    # Decoupled from early stopping — no race condition with patience counter.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    # Focal-style weighting: UP/DOWN 4x NEUTRAL to counter ~73% NEUTRAL imbalance
    class_weights = torch.FloatTensor([2.0, 0.5, 2.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    logger.info(f"Training LSTM for {symbol} on {DEVICE} — {len(train_ds)} train samples")

    for epoch in range(p["epochs"]):
        # Training
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

        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for X_b, y_b in val_dl:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                logits = model(X_b)
                val_loss += criterion(logits, y_b).item()
                correct  += (logits.argmax(1) == y_b).sum().item()
                total    += len(y_b)

        val_loss /= len(val_dl)
        val_acc   = correct / total
        scheduler.step()

        logger.info(f"  Epoch {epoch:3d} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | lr={optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= p["patience"]:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    logger.success(f"LSTM best val_loss: {best_val_loss:.4f}")
    return model


def save(model: LSTMClassifier, symbol: str):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = Path(MODEL_DIR) / f"{symbol.replace('/','_')}_lstm.pt"
    torch.save(model.state_dict(), path)
    # Also save architecture info
    info_path = Path(MODEL_DIR) / f"{symbol.replace('/','_')}_lstm_info.pkl"
    import joblib
    joblib.dump({
        "input_size":  model.lstm.input_size,
        "hidden_size": model.lstm.hidden_size,
        "num_layers":  model.lstm.num_layers,
    }, info_path)
    logger.info(f"Saved LSTM -> {path}")
    return path


def load(symbol: str) -> LSTMClassifier:
    import joblib
    info_path = Path(MODEL_DIR) / f"{symbol.replace('/','_')}_lstm_info.pkl"
    info = joblib.load(info_path)
    p    = LSTM_PARAMS
    model = LSTMClassifier(
        input_size  = info["input_size"],
        hidden_size = info["hidden_size"],
        num_layers  = info["num_layers"],
        num_classes = 3,
        dropout     = p["dropout"],
    ).to(DEVICE)
    path = Path(MODEL_DIR) / f"{symbol.replace('/','_')}_lstm.pt"
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model


def predict_proba(model: LSTMClassifier, X: np.ndarray,
                  seq_len: int = None) -> np.ndarray:
    if seq_len is None:
        seq_len = LSTM_PARAMS["sequence_length"]
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(seq_len, len(X) + 1):
            seq = torch.FloatTensor(X[i-seq_len:i]).unsqueeze(0).to(DEVICE)
            p   = torch.softmax(model(seq), dim=1).cpu().detach().numpy()[0]
            probs.append(p)
    # Pad start with neutral probability
    pad = np.full((seq_len - 1, 3), [0.0, 1.0, 0.0])
    return np.vstack([pad, np.array(probs)])