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

Improvements over v1:
  - Self-attention layer on LSTM output sequence (learns which bars matter)
  - Focal loss replaces CrossEntropyLoss (focuses gradient on hard examples)
  - Mixed precision training on CUDA (1.5-2x speedup, no accuracy loss)
  - Fold-level model cache keyed on data hash (skip retraining if data unchanged)
  - predict_proba_ensemble returns per-model std for disagreement filtering

Usage:
    from models.lstm_ensemble import train_ensemble, predict_proba_ensemble

    ensemble = train_ensemble(X_train, y_train, X_val, y_val, symbol="DOGE/USD")
    proba, std = predict_proba_ensemble(ensemble, X_test, return_std=True)
"""

import os, sys, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from loguru import logger
from config.settings import LSTM_PARAMS, MODEL_DIR

try:
    import torch_directml
    DEVICE = torch_directml.device()
    logger.info("Ensemble using DirectML (AMD GPU)")
    USE_AMP = False   # AMP not supported on DirectML
except ImportError:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = (hasattr(DEVICE, "type") and DEVICE.type == "cuda")
    logger.info(f"Ensemble using {DEVICE} | AMP={'enabled' if USE_AMP else 'disabled'}")

# ── Config ────────────────────────────────────────────────────────────────────

N_MODELS  = 9   # Number of independent models to train
CACHE_DIR = Path(MODEL_DIR) / "fold_cache"   # cached trained models keyed by data hash

# Architecture version tag — bump this to invalidate all cached models when
# the architecture changes (e.g. after adding attention or changing focal gamma)
_ARCH_TAG = b"arch_v2_attention_focal_gamma2"


# ── Architecture ──────────────────────────────────────────────────────────────

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


class SelfAttention(nn.Module):
    """
    Single-head additive self-attention over the LSTM output sequence.

    Instead of only using the final hidden state, this learns a weighted
    average over all seq_len hidden states — letting the model focus on
    whichever bars are most predictive rather than trusting the last one alone.

    Complexity: O(seq_len * hidden_size) — negligible vs LSTM.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (batch, seq_len, hidden_size)
        weights = torch.softmax(self.score(lstm_out), dim=1)   # (batch, seq_len, 1)
        return (lstm_out * weights).sum(dim=1)                  # (batch, hidden_size)


class FocalLoss(nn.Module):
    """
    Focal loss with label smoothing for multi-class classification.

    Two complementary improvements over standard CrossEntropyLoss:
      - Focal weighting (gamma=2): down-weights easy examples so gradient
        budget flows to hard UP/DOWN calls rather than trivial neutrals.
      - Label smoothing (smoothing=0.05): replaces hard one-hot targets with
        soft distributions (e.g. [0.025, 0.025, 0.95] for UP). Prevents
        overconfidence, acts as additional regularisation.

    Combined with inverse-frequency class weights: weights handle label
    imbalance, focal handles sample-level hardness, smoothing handles
    model calibration.
    """
    def __init__(self, weight: torch.Tensor = None, gamma: float = 2.0,
                 smoothing: float = 0.05, num_classes: int = 3):
        super().__init__()
        self.weight      = weight
        self.gamma       = gamma
        self.smoothing   = smoothing
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cast weight to match logits dtype (fp16 under AMP, fp32 otherwise).
        # self.weight is always stored as fp32; under autocast logits are fp16,
        # so mixing them raises "expected scalar type Half but found Float".
        w = self.weight.to(logits.dtype) if self.weight is not None else None

        # Build soft targets: confidence on true class, uniform smoothing elsewhere
        confidence = 1.0 - self.smoothing
        smooth_val = self.smoothing / (self.num_classes - 1)
        soft = torch.full_like(logits, smooth_val)
        soft.scatter_(1, targets.unsqueeze(1), confidence)

        log_probs = F.log_softmax(logits, dim=1)
        ce = -(soft * log_probs).sum(dim=1)          # soft cross-entropy per sample

        # Apply class weights on the true class only
        if w is not None:
            ce = ce * w[targets]

        pt = torch.exp(-F.cross_entropy(logits, targets,
                                        weight=w, reduction="none").detach())
        return ((1 - pt) ** self.gamma * ce).mean()


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        # Inter-layer LSTM dropout is safe on CUDA but broken on DirectML
        # (MIOpen gfx1100 kernel crashes). Conditionally enable per device.
        inter_dropout = dropout if (USE_AMP and num_layers > 1) else 0.0
        self.lstm      = nn.LSTM(input_size, hidden_size, num_layers,
                                 batch_first=True, dropout=inter_dropout)
        self.attention = SelfAttention(hidden_size)
        self.norm      = MIOpenSafeLayerNorm(hidden_size)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)                        # (batch, seq_len, hidden)
        ctx    = self.attention(out)                  # (batch, hidden) — attention-weighted
        return self.fc(self.dropout(self.norm(ctx)))


class SequenceDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X       = torch.FloatTensor(X)
        self.y       = torch.LongTensor(y)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, idx):
        return self.X[idx:idx+self.seq_len], self.y[idx+self.seq_len]


# ── Model cache ───────────────────────────────────────────────────────────────

def _cache_key(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> str:
    """
    Stable MD5 hash of training data + LSTM_PARAMS + seed + architecture version.

    Sampling every Nth row for speed — enough to detect data changes without
    hashing 100k+ floats on every call.  The architecture tag ensures that
    cached models from old versions are never loaded by a new architecture.
    """
    h = hashlib.md5()
    h.update(str(X_train.shape).encode())
    h.update(str(y_train.shape).encode())
    stride = max(1, len(X_train) // 100)
    h.update(X_train[::stride].astype(np.float32).tobytes())
    h.update(y_train[::max(1, len(y_train) // 100)].tobytes())
    h.update(str(sorted(LSTM_PARAMS.items())).encode())
    h.update(str(seed).encode())
    h.update(_ARCH_TAG)
    return h.hexdigest()[:20]


def _try_load_cache(key: str, model: LSTMClassifier) -> bool:
    """
    Try to load a cached model state into `model` in-place.
    Returns True on success, False if cache miss or corrupt.
    """
    path = CACHE_DIR / f"{key}.pt"
    if not path.exists():
        return False
    try:
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        model.eval()
        return True
    except Exception as e:
        logger.warning(f"Cache load failed ({e}), will retrain")
        return False


def _save_cache(model: LSTMClassifier, key: str):
    """Save trained model weights to the fold cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CACHE_DIR / f"{key}.pt")


# ── Temperature scaling ───────────────────────────────────────────────────────

def _fit_temperature(model: LSTMClassifier,
                     X_val: np.ndarray, y_val: np.ndarray,
                     seq_len: int) -> float:
    """
    Fit a single temperature scalar T on the validation set by minimising
    the negative log-likelihood of softmax(logits / T).

    Neural networks are systematically overconfident — temperature scaling
    corrects this without retraining, using one scalar per model.  A T > 1
    spreads the probability mass (less confident); T < 1 sharpens it.
    Typical values: T ∈ [1.0, 2.5] for crypto LSTM ensembles.

    The temperature is stored as `model.temperature` and applied automatically
    in predict_proba_ensemble.  It is fold-specific (fitted on the current
    val set), so it is refitted even when the model weights are loaded from
    cache.
    """
    from scipy.optimize import minimize_scalar

    val_ds = SequenceDataset(X_val, y_val, seq_len)
    val_dl = DataLoader(val_ds, batch_size=512, shuffle=False)

    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for X_b, y_b in val_dl:
            X_b = X_b.to(DEVICE)
            all_logits.append(model(X_b).cpu())
            all_labels.append(y_b)

    if not all_logits:
        model.temperature = 1.0
        return 1.0

    logits = torch.cat(all_logits)   # (n_val, 3)
    labels = torch.cat(all_labels)   # (n_val,)

    def nll(T: float) -> float:
        scaled = logits / max(float(T), 0.1)
        return float(F.cross_entropy(scaled, labels).item())

    try:
        result = minimize_scalar(nll, bounds=(0.5, 5.0), method="bounded")
        T = float(np.clip(result.x, 0.5, 5.0))
    except Exception:
        T = 1.0

    model.temperature = T
    return T


# ── Single model training ─────────────────────────────────────────────────────

def _train_one(X_train, y_train, X_val, y_val,
               seed: int, symbol: str, model_idx: int) -> LSTMClassifier:
    """
    Train one LSTM with a fixed seed.

    Cache check: if a model with the same data hash + seed + params already
    exists on disk, load and return it immediately.  This makes repeated runs
    (e.g. after a crash) nearly instant for unchanged folds.

    On CUDA: uses automatic mixed precision (FP16 forward / FP32 gradient
    accumulation) for ~1.5-2x training speedup.
    On DirectML: AMP is disabled (not supported by the DirectML backend).
    """
    key = _cache_key(X_train, y_train, seed)

    p          = LSTM_PARAMS
    seq_len    = p["sequence_length"]
    input_size = X_train.shape[1]

    model = LSTMClassifier(
        input_size  = input_size,
        hidden_size = p["hidden_size"],
        num_layers  = p["num_layers"],
        num_classes = 3,
        dropout     = p["dropout"],
    ).to(DEVICE)

    # ── Cache HIT ─────────────────────────────────────────────────────────────
    if _try_load_cache(key, model):
        logger.info(f"  [{model_idx+1}/{N_MODELS}] Cache HIT seed={seed} "
                    f"(key={key[:8]}…) — skipping training")
        # Temperature must be refitted on current val set even for cached models
        T = _fit_temperature(model, X_val, y_val, seq_len)
        logger.info(f"  [{model_idx+1}/{N_MODELS}] Temperature T={T:.3f}")
        return model

    # ── Cache MISS → train ────────────────────────────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = SequenceDataset(X_train, y_train, seq_len)
    val_ds   = SequenceDataset(X_val,   y_val,   seq_len)
    train_dl = DataLoader(train_ds, batch_size=p["batch_size"], shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=p["batch_size"])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=p["learning_rate"], weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Inverse-frequency class weights — handles label imbalance
    counts    = np.bincount(y_train, minlength=3).astype(float)
    counts    = np.maximum(counts, 1.0)
    inv_freq  = 1.0 / counts
    inv_freq /= inv_freq.mean()
    inv_freq  = np.clip(inv_freq, 0.25, 4.0)
    class_weights = torch.FloatTensor(inv_freq).to(DEVICE)
    logger.info(f"  [{model_idx+1}/{N_MODELS}] Class weights: "
                f"DOWN={inv_freq[0]:.2f} NEUTRAL={inv_freq[1]:.2f} UP={inv_freq[2]:.2f} "
                f"(from counts: {counts.astype(int).tolist()})")

    criterion = FocalLoss(weight=class_weights, gamma=2.0)

    # AMP scaler — only active on CUDA
    amp_scaler = torch.cuda.amp.GradScaler() if USE_AMP else None

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

            if USE_AMP:
                with torch.cuda.amp.autocast():
                    loss = criterion(model(X_b), y_b)
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                loss = criterion(model(X_b), y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for X_b, y_b in val_dl:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                with (torch.cuda.amp.autocast() if USE_AMP else _null_ctx()):
                    logits = model(X_b)
                val_loss += criterion(logits, y_b).item()
                correct  += (logits.argmax(1) == y_b).sum().item()
                total    += len(y_b)

        val_loss /= max(len(val_dl), 1)
        val_acc   = correct / max(total, 1)
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

    if best_state is None:
        logger.warning(f"  [{model_idx+1}/{N_MODELS}] best_state is None "
                       f"(no epoch completed) — using final model weights as fallback")
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    logger.info(f"  [{model_idx+1}/{N_MODELS}] Best val_loss: {best_val_loss:.4f}")
    _save_cache(model, key)

    # Temperature calibration — always fitted on current val set regardless of cache
    T = _fit_temperature(model, X_val, y_val, seq_len)
    logger.info(f"  [{model_idx+1}/{N_MODELS}] Temperature T={T:.3f} "
                f"({'over-confident → spreading' if T > 1.0 else 'well-calibrated'})")
    return model


class _null_ctx:
    """No-op context manager for non-AMP code paths."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ── Ensemble training ─────────────────────────────────────────────────────────

def train_ensemble(X_train: np.ndarray, y_train: np.ndarray,
                   X_val:   np.ndarray, y_val:   np.ndarray,
                   symbol:  str = "model",
                   n_models: int = N_MODELS) -> list:
    """
    Train N independent LSTMs with different random seeds.
    Returns a list of trained LSTMClassifier models.

    On repeated runs with unchanged data, cached models are returned
    immediately — no retraining occurs.
    """
    logger.info(f"\nTraining {n_models}-model ensemble for {symbol} on {DEVICE}")
    logger.info(f"  Train: {len(X_train)} | Val: {len(X_val)} | "
                f"Features: {X_train.shape[1]}")

    seeds  = [42 + i * 137 for i in range(n_models)]
    models = []
    val_losses = []

    for i, seed in enumerate(seeds):
        m = _train_one(X_train, y_train, X_val, y_val,
                       seed=seed, symbol=symbol, model_idx=i)
        models.append(m)

        # Quick val loss check
        m.eval()
        seq_len = LSTM_PARAMS["sequence_length"]
        val_ds  = SequenceDataset(X_val, y_val, seq_len)
        val_dl  = DataLoader(val_ds, batch_size=LSTM_PARAMS["batch_size"])
        val_counts = np.bincount(y_val, minlength=3).astype(float)
        val_counts = np.maximum(val_counts, 1.0)
        val_inv    = np.clip(1.0 / val_counts / (1.0 / val_counts).mean(), 0.25, 4.0)
        crit = FocalLoss(weight=torch.FloatTensor(val_inv).to(DEVICE), gamma=2.0)
        vl = 0.0
        with torch.no_grad():
            for X_b, y_b in val_dl:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                vl += crit(m(X_b), y_b).item()
        val_losses.append(vl / max(len(val_dl), 1))

    logger.info(f"\nEnsemble training complete for {symbol}:")
    logger.info(f"  Val losses: {[f'{v:.4f}' for v in val_losses]}")
    logger.info(f"  Mean: {np.mean(val_losses):.4f} | Std: {np.std(val_losses):.4f}")

    return models


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_proba_ensemble(models: list, X: np.ndarray,
                           seq_len: int = None,
                           batch_size: int = 512,
                           return_std: bool = False):
    """
    Batched sliding-window inference averaged across all ensemble models.

    Args:
        models     : list of trained LSTMClassifier instances
        X          : (n_bars, n_features) array
        seq_len    : sequence length (defaults to LSTM_PARAMS["sequence_length"])
        batch_size : inference batch size
        return_std : if True, also return per-class std across models
                     (use for disagreement filtering — high std = noisy signal)

    Returns:
        mean_proba         : (n_bars, 3) float32 — averaged probabilities
        std_proba (optional): (n_bars, 3) float32 — std across models per class
    """
    if seq_len is None:
        seq_len = LSTM_PARAMS["sequence_length"]

    X = np.ascontiguousarray(X, dtype=np.float32)
    n, n_features = X.shape
    n_windows = n - seq_len + 1

    pad = np.full((seq_len - 1, 3), [0.0, 1.0, 0.0], dtype=np.float32)

    if n_windows <= 0:
        neutral = np.tile([0.0, 1.0, 0.0], (n, 1)).astype(np.float32)
        return (neutral, np.zeros_like(neutral)) if return_std else neutral

    # Zero-copy sliding-window view: (n_windows, seq_len, n_features)
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
                T = getattr(model, "temperature", 1.0)
                p = torch.softmax(model(batch) / T, dim=1).cpu().numpy()
                model_probs.append(p)
        all_probas.append(np.vstack([pad, np.vstack(model_probs)]))

    all_probas = np.stack(all_probas, axis=0)  # (n_models, n_bars, 3)
    mean_proba = all_probas.mean(axis=0)

    if return_std:
        return mean_proba, all_probas.std(axis=0)
    return mean_proba


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_ensemble(models: list, symbol: str):
    """Save all ensemble models to disk."""
    import joblib
    os.makedirs(MODEL_DIR, exist_ok=True)
    safe = symbol.replace("/", "_")

    for i, model in enumerate(models):
        path = Path(MODEL_DIR) / f"{safe}_lstm_ensemble_{i}.pt"
        torch.save(model.state_dict(), path)

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
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        model.eval()
        models.append(model)

    logger.info(f"Loaded {len(models)}-model ensemble for {symbol}")
    return models
