"""
Central configuration for the trading bot.
Edit these values to customise behaviour.
"""

# ── Exchange ──────────────────────────────────────────────────────────────────
EXCHANGE_ID = "kraken"
TRADING_PAIRS = [
    # Original 5
    "INJ/USD", "LINK/USD", "DOGE/USD", "OP/USD", "ARB/USD",
    # Mega caps
    "BTC/USD", "ETH/USD",
    # Large cap L1s
    "SOL/USD", "ADA/USD", "DOT/USD", "AVAX/USD",
    # Meme / retail momentum
    "SHIB/USD",
    # DeFi
    "UNI/USD", "AAVE/USD",
    # Payments / long history
    "XRP/USD", "LTC/USD",
    # Ecosystem
    "ATOM/USD", "NEAR/USD",
    # L2
    "MATIC/USD",
    # Extended scan
    "ALGO/USD", "FTM/USD", "VET/USD", "XLM/USD", "TRX/USD", "FIL/USD", "XTZ/USD",
    "GRT/USD", "CRV/USD", "SNX/USD", "COMP/USD", "MKR/USD",
    "1INCH/USD", "RUNE/USD", "DASH/USD", "ZEC/USD", "ETC/USD", "BCH/USD",
]
TIMEFRAME = "1h"
HISTORY_DAYS = 1825

# ── Feature Engineering ───────────────────────────────────────────────────────
FEATURE_WINDOW_SIZES = [7, 14, 21, 50, 200]
# Prediction horizon and label threshold are now computed per-timeframe
# in feature_engineer.py via get_label_params(timeframe).
# These fallback values are kept for any legacy code that imports them directly.
PREDICTION_HORIZON = 24   # fallback: 24 bars
LABEL_THRESHOLD    = 0.008  # fallback: 0.8%

# ── Model Training ────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42
N_JOBS = -1

LGBM_PARAMS = {
    "objective": "multiclass", "num_class": 3,
    "n_estimators": 200, "learning_rate": 0.05,
    "num_leaves": 63, "max_depth": -1,
    "min_child_samples": 50, "subsample": 0.8,
    "colsample_bytree": 0.8, "reg_alpha": 0.1,
    "reg_lambda": 0.1, "random_state": 42,
    "n_jobs": -1, "verbose": -1,
}

XGB_PARAMS = {
    "objective": "multi:softprob", "num_class": 3,
    "n_estimators": 200, "learning_rate": 0.05,
    "max_depth": 6, "min_child_weight": 5,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
    "random_state": 42, "n_jobs": -1, "verbosity": 0,
}

LSTM_PARAMS = {
    "sequence_length": 60, "hidden_size": 128,
    "num_layers": 2, "dropout": 0.2,
    "epochs": 100, "batch_size": 64,
    "learning_rate": 0.001,
    "patience": 30,   # was 25 — must be >= T_0*(1+T_mult)=30 so early stopping
                      # never fires before the first full cosine LR restart cycle
                      # completes (cycle 1=10 epochs, cycle 2=20 epochs → total 30).
                      # Stopping mid-cycle risks getting trapped in a local optimum.
}

ENSEMBLE_WEIGHTS = {"catboost": 0.40, "cnn": 0.35, "lstm": 0.25}

# ── Backtesting ───────────────────────────────────────────────────────────────
INITIAL_CAPITAL  = 10_000
TRADING_FEE      = 0.001   # Kraken taker fee (0.1% per leg = 0.2% round-trip)
SLIPPAGE         = 0.001   # Conservative 0.1% per leg for mid-cap alts (0.2% round-trip)
                            # Total round-trip friction: 0.4% — modelled per leg in engine.py
SIGNAL_THRESHOLD = 0.38   # Default threshold

# Per-symbol overrides — lower for symbols with compressed probability distributions
SIGNAL_THRESHOLDS = {
    "INJ/USD":  0.34,
    "LINK/USD": 0.35,
    "DOGE/USD": 0.35,
    "OP/USD":   0.35,
    "ARB/USD":  0.35,
    "SOL/USD":  0.36,
    "AVAX/USD": 0.36,
}
MAX_POSITION_PCT = 0.25   # 25% per trade — cap when Kelly is OFF (use_kelly=False)
                           # When Kelly is ON, KELLY_MAX_PCT (below) is the active cap.
                           # Both set to 0.25 so behaviour is consistent either way.

# ── Risk Management ───────────────────────────────────────────────────────────
# Fixed % stops — used as fallback when ATR stops are disabled or ATR is unavailable
STOP_LOSS_PCT      = 0.03
TAKE_PROFIT_PCT    = 0.06
MAX_DAILY_DRAWDOWN = 0.05
MAX_OPEN_TRADES    = 3

# ATR-based dynamic stops (recommended — scales with actual volatility)
# Stop  = entry_price ± ATR × ATR_STOP_MULT
# TP    = entry_price ± ATR × ATR_TP_MULT
# 2:1 ratio (2.4 TP / 1.2 stop) requires >33% win rate to be positive expectancy.
# engine.py imports these directly — change here to affect backtest behaviour.
USE_ATR_STOPS   = True
ATR_STOP_MULT   = 1.5   # stop = 1.5 × ATR from entry  (wider stop = fewer noise-outs)
ATR_TP_MULT     = 3.0   # TP   = 3.0 × ATR from entry  (2:1 reward:risk maintained)

# Kelly position sizing caps — imported by engine.py
# KELLY_MAX_PCT: hard cap on fraction Kelly can allocate per trade (Kelly ON path)
# Aligned to MAX_POSITION_PCT so both sizing modes cap at the same level.
KELLY_MAX_PCT    = 0.25   # was 0.40 hardcoded in engine.py

# Kelly ADX regime threshold — halve position size in choppy markets (ADX below this)
# Aligned to REGIME_ADX_THRESHOLD (14) so the regime label and Kelly penalty
# use the same boundary. engine.py imports KELLY_REGIME_ADX from here.
KELLY_REGIME_ADX = 14.0   # was 18.0; lowered to pass more signals in crypto consolidation

# Max bars to hold a position before forcing close (prevents dead-capital lockup)
# Set per timeframe: at 4h that's 5 days; at 1h that's ~2 days
MAX_HOLD_BARS    = 30     # bars; override per TF in engine if needed

# Amihud illiquidity filter threshold percentile (top X% most illiquid bars suppressed)
AMIHUD_FILTER_PCT = 0.80  # block signals in the top 20% most illiquid bars

# ── Signal Filters ────────────────────────────────────────────────────────────
# Each filter can be independently toggled.
# See backtest/filters.py for per-timeframe automatic relaxation via TF_RELAX.
#
# DIAGNOSIS: filters were removing 95.4% of signals (avg 3.8 trades/fold).
# Loosened defaults target 15-30 trades/fold at 1h, 8-20 at 4h.

USE_TREND_FILTER      = False   # SMA alignment — disabled: hurts at longer TFs
USE_VOLATILITY_FILTER = True    # Skip very low ATR bars (bottom VOLATILITY_FILTER_PCT)
USE_VOLUME_FILTER     = True    # Require minimum volume (VOLUME_FILTER_PCT × avg)
USE_FUNDING_FILTER    = False   # Funding rate extremes (Kraken perp only)
USE_REGIME_FILTER     = False   # Disables ADX Filter 5 only.
                                 # Choppiness (Filter 6) and Market Efficiency Ratio
                                 # (Filter 7) are unconditional and remain active.
                                 # Edge scanner validated signals without ADX gating.

# Volatility filter: skip bars in bottom X% of ATR distribution
# Was 0.20 (bottom 20%) — lowered to bottom 10%
VOLATILITY_FILTER_PCT = 0.10

# Volume filter: require vol >= X% of 14-bar average
# Was 0.80 (80%) — lowered to 40%
VOLUME_FILTER_PCT     = 0.40

# Regime filter: ADX must exceed this threshold to trade
# Was 25 → 18 → 14 (crypto spends ~60% of time below ADX 25; consolidation phases are tradeable)
# Longer timeframes get further automatic relaxation via TF_RELAX in filters.py
REGIME_WINDOW        = 50
REGIME_ADX_THRESHOLD = 14

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = "data/raw"
FEATURE_DIR = "data/features"
MODEL_DIR   = "models/saved"
RESULTS_DIR = "backtest/results"
LOG_DIR     = "logs"