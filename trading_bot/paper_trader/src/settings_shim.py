"""
src/settings_shim.py
Provides the constants that feature_engineer.py imports from config.settings,
without requiring the full AlgoTrader config package.
"""

FEATURE_WINDOW_SIZES  = [5, 10, 20, 50]
PREDICTION_HORIZON    = 6
LABEL_THRESHOLD       = 0.035
TIMEFRAME             = "4h"
DATA_DIR              = "data"
FEATURE_DIR           = "data/features"
USE_REGIME_FILTER     = False
REGIME_WINDOW         = 50
REGIME_ADX_THRESHOLD  = 25.0
