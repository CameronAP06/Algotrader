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
    "sequence_length": 60,
    "stream_hidden":   48,    # hidden units per multi-stream branch.  With ~6 feature
                              # groups this gives 6×48=288 total merged hidden — more
                              # capacity than the old single-stream 192 while keeping
                              # each branch focused on one feature family.
    "hidden_size":     192,   # fallback for single-stream mode (not used in multi-stream)
    "num_layers":      2,
    "dropout":         0.35,
    "epochs":          200,
    "batch_size":      64,
    "learning_rate":   0.001,
    "patience":        50,
    "training_stride": 4,
    "noise_sigma":          0.02,
    "lr_plateau_patience":  10,
    "lr_plateau_factor":    0.5,
}

# ── Tree model parameters (XGBoost + CatBoost) ───────────────────────────────
# These are per-fold tabular models that complement the LSTM ensemble.
# Key differences from the legacy XGB_PARAMS above:
#   - More estimators with lower LR + early stopping on val set
#   - Stronger regularisation (higher min_child_weight / l2_leaf_reg)
#   - Conservative subsample/colsample to prevent feature memorisation
#   - Class-weight handled explicitly in training code (not via params here)

XGB_TREE_PARAMS = {
    "objective":         "multi:softprob",
    "num_class":         3,
    "n_estimators":      800,   # high ceiling — early stopping fires around 200-400
    "learning_rate":     0.02,
    "max_depth":         5,     # shallow = less overfitting on small folds
    "min_child_weight":  20,    # high = requires 20 samples to split (regularise)
    "subsample":         0.7,
    "colsample_bytree":  0.6,
    "reg_alpha":         0.5,
    "reg_lambda":        2.0,
    "random_state":      42,
    "n_jobs":            -1,
    "verbosity":         0,
    "eval_metric":       "mlogloss",
    "early_stopping_rounds": 50,
}

CATBOOST_PARAMS = {
    "loss_function":     "MultiClass",
    "iterations":        800,
    "learning_rate":     0.02,
    "depth":             6,
    "l2_leaf_reg":       5.0,
    "auto_class_weights": "Balanced",  # handles ~50% neutral imbalance natively
    "early_stopping_rounds": 50,
    "verbose":           0,
    "random_seed":       42,
    "thread_count":      -1,
}

# Blend weights for the three-model ensemble (LSTM + XGB + CatBoost).
# Starting point: equal weight — actual weights are optimised per fold on the
# validation set via ensemble.optimise_trio_weights() and these are only the
# fallback when optimisation fails or produces degenerate weights.
ENSEMBLE_WEIGHTS = {"lstm": 0.34, "xgb": 0.33, "catboost": 0.33}

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

# Kelly position sizing — all parameters in one place
KELLY_FRACTION   = 0.25   # fractional Kelly safety margin (full Kelly is dangerously volatile)
KELLY_MIN_PCT    = 0.05   # never risk less than 5% per trade (avoids dust trades)
KELLY_MAX_PCT    = 0.25   # hard cap per trade (was 0.40 hardcoded in engine.py)
KELLY_REGIME_ADX = 14.0   # halve Kelly fraction when ADX < this (choppy market penalty)
                           # was 18.0; aligned to REGIME_ADX_THRESHOLD so both use same boundary

# Kelly initial priors — used before enough trades to estimate win rate empirically.
# Applied until _recalc_kelly has seen >= 5 completed trades.
KELLY_INITIAL_WIN_RATE = 0.45  # conservative prior; empirical wr typically 42-52%
KELLY_INITIAL_PAYOFF   = 1.8   # conservative prior for payoff ratio (avg_win / avg_loss)

# Kelly confidence tiering — scale allocation by model signal certainty.
# Low  (< TIER_LOW): 50% of Kelly — near-threshold signal, stay small
# Mid  (< TIER_MID): 75% of Kelly — moderate confidence
# High (≥ TIER_MID): full Kelly — strong signal
KELLY_CONF_TIER_LOW = 0.40
KELLY_CONF_TIER_MID = 0.50

# Max bars to hold a position before forcing close (prevents dead-capital lockup)
# Set per timeframe: at 4h that's 5 days; at 1h that's ~2 days
MAX_HOLD_BARS    = 30     # bars; override per TF in engine if needed

# Amihud illiquidity filter threshold percentile (top X% most illiquid bars suppressed)
AMIHUD_FILTER_PCT = 0.85  # block top 15% most illiquid bars (was 0.80 / top 20% — too aggressive)

# ── Signal Filters ────────────────────────────────────────────────────────────
# Each filter can be independently toggled.
# See backtest/filters.py for per-timeframe automatic relaxation via TF_RELAX.
#
# DIAGNOSIS: filters were removing 95.4% of signals (avg 3.8 trades/fold).
# Loosened defaults target 15-30 trades/fold at 1h, 8-20 at 4h.

USE_TREND_FILTER      = False   # DIAGNOSTIC: off to test raw signal quality across symbol set
                                # SMA200 at 4h = ~33 days: meaningful long-term regime gate.
                                # Prevents BUY entries in sustained downtrends and SELL entries in uptrends.
USE_VOLATILITY_FILTER = True    # Skip very low ATR bars (bottom VOLATILITY_FILTER_PCT)
USE_VOLUME_FILTER     = True    # Require minimum volume (VOLUME_FILTER_PCT × avg)
USE_FUNDING_FILTER    = False   # Funding rate extremes (Kraken perp only)
USE_REGIME_FILTER     = False   # ADX momentum gate (Filter 5) — disabled: Kelly already penalises low-ADX bars
USE_CHOPPINESS_FILTER = False   # Choppiness Index gate (Filter 6) — killed 293 signals (53% of all kills); disabled
USE_EFFICIENCY_FILTER = False   # Market Efficiency Ratio gate (Filter 7) — redundant with choppiness; disabled
USE_AMIHUD_FILTER     = True    # Amihud illiquidity gate (Filter 8) — blocks top 20% most illiquid bars

# Trend filter SMA window — which sma_N column to use as the trend baseline.
# 200 bars at 4h = ~33 days (meaningful long-term trend indicator).
# 50 bars at 4h  = ~8 days  (short-term trend — noisier but reacts faster).
TREND_FILTER_SMA = 200   # was 50 — SMA50 proved WORSE quality: reacts faster so it sits
                          # HIGHER during post-crash periods → blocks even more BUY signals.
                          # SMA200 (33 days at 4h) is more selective but higher quality:
                          # only signals in clear uptrends pass, giving Sharpe 1.5+ on XLM/ALGO.

# Volatility filter: skip bars in bottom X% of ATR distribution
# Was 0.20 (bottom 20%) — lowered to bottom 10%
VOLATILITY_FILTER_PCT = 0.10

# Volume filter: require vol >= X% of 14-bar average
# Was 0.80 (80%) → 0.40 → 0.30 — 0.40 was still cutting too many valid bars at 4h+
VOLUME_FILTER_PCT     = 0.30

# Regime filter: ADX must exceed this threshold to trade
# Was 25 → 18 → 14 (crypto spends ~60% of time below ADX 25; consolidation phases are tradeable)
# Longer timeframes get further automatic relaxation via TF_RELAX in filters.py
REGIME_WINDOW        = 50
REGIME_ADX_THRESHOLD = 14

# Ensemble disagreement filter — suppress signals where the 9 models strongly disagree.
# Max per-class std across all 9 models. High std = genuine ambiguity = not worth trading.
# 0.15 means: if any class has std > 0.15 across the ensemble, force HOLD.
ENSEMBLE_DISAGREE_THRESHOLD = 0.15

# Ensemble temperature gate — exclude models whose temperature calibration exceeded this
# value from the ensemble average.  A T=10 model outputs near-uniform [0.33,0.33,0.33]
# probabilities that suppress confident predictions from other models AND inflate per-class
# std, causing the disagreement filter to kill most signals (e.g. fold 1 BTC: 13/14 killed).
# Models exceeding this threshold are logged and skipped during inference.
ENSEMBLE_MAX_TEMPERATURE = 5.0

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = "data/raw"
FEATURE_DIR = "data/features"
MODEL_DIR   = "models/saved"
RESULTS_DIR = "backtest/results"
LOG_DIR     = "logs"