"""
Central configuration for the trading bot.
Edit these values to customise behaviour.
"""

# ── Exchange ──────────────────────────────────────────────────────────────────
EXCHANGE_ID = "kraken"
TRADING_PAIRS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD"]
TIMEFRAME = "1h"
HISTORY_DAYS = 1825

# ── Feature Engineering ───────────────────────────────────────────────────────
FEATURE_WINDOW_SIZES = [7, 14, 21, 50, 200]
PREDICTION_HORIZON = 1
LABEL_THRESHOLD = 0.002

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
    "sequence_length": 24, "hidden_size": 64,
    "num_layers": 1, "dropout": 0.2,
    "epochs": 30, "batch_size": 128,
    "learning_rate": 0.001, "patience": 7,
}

ENSEMBLE_WEIGHTS = {"lgbm": 0.40, "xgb": 0.35, "lstm": 0.25}

# ── Backtesting ───────────────────────────────────────────────────────────────
INITIAL_CAPITAL  = 10_000
TRADING_FEE      = 0.0026
SLIPPAGE         = 0.001
SIGNAL_THRESHOLD = 0.38
MAX_POSITION_PCT = 0.95

# ── Risk Management ───────────────────────────────────────────────────────────
STOP_LOSS_PCT      = 0.03
TAKE_PROFIT_PCT    = 0.06
MAX_DAILY_DRAWDOWN = 0.05
MAX_OPEN_TRADES    = 3

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = "data/raw"
FEATURE_DIR = "data/features"
MODEL_DIR   = "models/saved"
RESULTS_DIR = "backtest/results"
LOG_DIR     = "logs"