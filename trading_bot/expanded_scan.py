"""
expanded_scan.py
────────────────
Full ensemble scan across crypto, forex, and multiple timeframes.

Covers:
  - Crypto:   37 pairs (existing list)
  - Forex:    EUR/USD, GBP/USD, JPY/USD, CHF/USD, AUD/USD, CAD/USD
  - Timeframes: 4h, 8h, 1d

Runs a 9-model ensemble per symbol/timeframe combination.
Saves results incrementally — safe to interrupt and resume.

Usage:
    python expanded_scan.py                          # all symbols, all timeframes
    python expanded_scan.py --timeframes 4h 8h       # skip 1d
    python expanded_scan.py --asset-class forex      # forex only
    python expanded_scan.py --asset-class crypto     # crypto only
    python expanded_scan.py --resume                 # skip already-completed rows
"""

import os, sys, argparse, csv, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import set_start_method
from datetime import datetime
from pathlib import Path

_here   = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
sys.path.insert(0, _here)
sys.path.insert(0, _parent)

import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler

try:
    from timeframe_comparison import fetch_ohlcv_timeframe
except ImportError as _e:
    raise ImportError(
        f"Cannot import fetch_ohlcv_timeframe from timeframe_comparison: {_e}\n"
        f"If timeframe_comparison.py fails due to a missing 'data.alt_data' module, "
        f"add a try/except around its alt_data import or ensure alt_data.py is present."
    ) from _e
from data.feature_engineer import build_features, get_feature_columns
from backtest.engine import BacktestEngine
from backtest.filters import apply_filters
from models.ensemble import generate_signals
# NOTE: train_ensemble and predict_proba_ensemble are imported LOCALLY inside
# run_fold() — not here. This is intentional: lstm_ensemble.py resolves its
# DEVICE (cuda:N) at module import time. If imported here (main process), DEVICE
# is set before _run_on_gpu can set CUDA_VISIBLE_DEVICES in the worker, so all
# workers would see all GPUs and pile onto cuda:0. Local import inside run_fold
# ensures each worker sets CUDA_VISIBLE_DEVICES first, THEN imports the module.

# ── Symbol Lists ──────────────────────────────────────────────────────────────
# Static fallback list — used when --no-autodiscover is passed or CSV dir not found
# Auto-discovery replaces this with the full Kraken universe at runtime

# ── Previously scanned (original 35 crypto + 3 forex) ────────────────────────
PAIRS_ORIGINAL = [
    "BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "ADA/USD", "AVAX/USD",
    "DOT/USD", "POL/USD", "LTC/USD", "BCH/USD", "LINK/USD",
    "DOGE/USD", "ETC/USD", "XLM/USD", "ATOM/USD", "ALGO/USD",
    "NEAR/USD", "S/USD", "XTZ/USD", "EOS/USD",
    "UNI/USD", "AAVE/USD", "MKR/USD", "COMP/USD", "SNX/USD",
    "CRV/USD", "GRT/USD", "INJ/USD", "ARB/USD", "OP/USD",
    "RUNE/USD", "FIL/USD", "XMR/USD", "ZEC/USD", "DASH/USD",
    "EUR/USD", "GBP/USD", "AUD/USD",
]

# ── Tier 1 — full 2yr 4h history (4300+ bars), new pairs only ────────────────
PAIRS_TIER1 = [
    "1INCH/USD", "ACH/USD", "AKT/USD", "ANKR/USD", "APE/USD", "API3/USD",
    "APT/USD", "ASTR/USD", "ATLAS/USD", "AXS/USD", "BAT/USD", "BLUR/USD",
    "BLZ/USD", "BONK/USD", "BSX/USD", "BTT/USD", "CFG/USD", "CHZ/USD",
    "COTI/USD", "CVX/USD", "DENT/USD", "DYDX/USD", "EGLD/USD", "ENJ/USD",
    "ENS/USD", "EWT/USD", "FET/USD", "FLOW/USD", "FLR/USD", "FXS/USD",
    "GALA/USD", "GARI/USD", "GLMR/USD", "GNO/USD", "ICP/USD", "IMX/USD",
    "JASMY/USD", "JTO/USD", "JUNO/USD", "KAVA/USD", "KEY/USD", "KSM/USD",
    "LCX/USD", "LDO/USD", "LMWR/USD", "LPT/USD", "LRC/USD", "MANA/USD",
    "MINA/USD", "MLN/USD", "MOVR/USD", "MXC/USD", "NANO/USD", "OCEAN/USD",
    "OMG/USD", "OXT/USD", "PEPE/USD", "PHA/USD", "POLIS/USD", "PYTH/USD",
    "QNT/USD", "RAY/USD", "REN/USD", "SAND/USD", "SBR/USD", "SC/USD",
    "SCRT/USD", "SEI/USD", "SGB/USD", "SHIB/USD", "SPELL/USD", "SRM/USD",
    "STORJ/USD", "STX/USD", "SUI/USD", "SUPER/USD", "SUSHI/USD", "SYN/USD",
    "TIA/USD", "TRU/USD", "TRX/USD", "WAXL/USD", "XCN/USD", "ZRX/USD",
]

# ── Tier 2 — strong history (3000–4299 bars), new pairs only ─────────────────
PAIRS_TIER2 = [
    "ACA/USD", "ADX/USD", "AEVO/USD", "AIR/USD", "ALCX/USD", "ALICE/USD",
    "ALPHA/USD", "ALT/USD", "AR/USD", "ARKM/USD", "ARPA/USD", "AUDIO/USD",
    "BADGER/USD", "BAL/USD", "BAND/USD", "BEAM/USD", "BICO/USD", "BIGTIME/USD",
    "BIT/USD", "BNB/USD", "BNC/USD", "BNT/USD", "BOBA/USD", "BOND/USD",
    "C98/USD", "CELO/USD", "CELR/USD", "CHR/USD", "CLOUD/USD", "CPOOL/USD",
    "CQT/USD", "CTSI/USD", "CVC/USD", "CXT/USD", "DRIFT/USD", "DYM/USD",
    "EDGE/USD", "ENA/USD", "ETHFI/USD", "ETHW/USD", "EUL/USD", "FARM/USD",
    "FHE/USD", "FIDA/USD", "FIS/USD", "FLOKI/USD", "FORTH/USD", "GAL/USD",
    "GHST/USD", "GMT/USD", "GMX/USD", "GST/USD", "GTC/USD", "GUN/USD",
    "HDX/USD", "HFT/USD", "HNT/USD", "HONEY/USD", "ICX/USD", "IDEX/USD",
    "JUP/USD", "KAR/USD", "KERNEL/USD", "KILT/USD", "KIN/USD", "KINT/USD",
    "KNC/USD", "KP3R/USD", "LSK/USD", "MASK/USD", "MNGO/USD", "MNT/USD",
    "MULTI/USD", "MV/USD", "NMR/USD", "NODL/USD", "NOS/USD", "NYM/USD",
    "OGN/USD", "ONDO/USD", "ORCA/USD", "OSMO/USD", "OXY/USD", "PENDLE/USD",
    "PERP/USD", "POLS/USD", "POND/USD", "POWR/USD", "PRIME/USD", "PROMPT/USD",
    "PSTAKE/USD", "QTUM/USD", "RAD/USD", "RARE/USD", "RARI/USD", "RBC/USD",
    "RENDER/USD", "REP/USD", "REZ/USD", "RLC/USD", "RPL/USD", "SAFE/USD",
    "SAGA/USD", "SAMO/USD", "SDN/USD", "STEP/USD", "STG/USD", "STRD/USD",
    "STRK/USD", "SUN/USD", "T/USD", "TAO/USD", "TEER/USD", "TLM/USD",
    "TNSR/USD", "TOKE/USD", "TRAC/USD", "TURBO/USD", "TVK/USD", "UMA/USD",
    "UNFI/USD", "W/USD", "WCT/USD", "WIF/USD", "WOO/USD", "XRT/USD",
    "YFI/USD", "YGG/USD", "ZETA/USD", "ZEUS/USD", "ZK/USD", "ZRO/USD",
]

# ── Tier 3 — shorter history (2000–2999 bars) — skip by default ──────────────
PAIRS_TIER3 = [
    "AGLD/USD", "ALCH/USD", "APU/USD", "ARC/USD", "ATH/USD", "AUCTION/USD",
    "COW/USD", "CRO/USD", "CSM/USD", "DBR/USD", "DRV/USD", "EIGEN/USD",
    "FWOG/USD", "GFI/USD", "GIGA/USD", "INTR/USD", "JST/USD", "KAS/USD",
    "KEEP/USD", "KMNO/USD", "L3/USD", "LIT/USD", "MC/USD", "MEME/USD",
    "METIS/USD", "MEW/USD", "MOG/USD", "MOODENG/USD", "MORPHO/USD", "NEIRO/USD",
    "NTRN/USD", "PDA/USD", "PENGU/USD", "PNUT/USD", "PONKE/USD", "POPCAT/USD",
    "PORTAL/USD", "PRCL/USD", "PUFFER/USD", "REQ/USD", "ROOK/USD", "RSR/USD",
    "SKY/USD", "SPX/USD", "SSV/USD", "SWELL/USD", "SYRUP/USD", "TON/USD",
    "VANRY/USD", "ZEX/USD",
]

FOREX_PAIRS_FALLBACK = ["EUR/USD", "GBP/USD", "AUD/USD"]

# ── Always exclude — stablecoins, wrapped tokens, pegged/dead/meme assets ─────
EXCLUDE_PAIRS = {
    # Stablecoins
    "USDT/USD", "USDC/USD", "DAI/USD", "BUSD/USD", "TUSD/USD",
    "USDP/USD", "GUSD/USD", "FRAX/USD", "LUSD/USD", "SUSD/USD",
    "PYUSD/USD", "USDE/USD", "FDUSD/USD", "USDD/USD", "USDG/USD",
    "USDUC/USD", "RLUSD/USD", "EURT/USD",
    # Wrapped / synthetic
    "WBTC/USD", "WETH/USD", "STETH/USD", "CBETH/USD", "RETH/USD",
    "TBTC/USD", "MSOL/USD", "PAXG/USD",
    # Dead / collapsed projects
    "LUNA/USD", "LUNA2/USD", "UST/USD", "MIR/USD",
    # Rebranded (already scanning under new name)
    "MATIC/USD", "FTM/USD", "REPV2/USD",
    # Index / perp tokens (not real spot assets)
    "XBTPY/USD", "XRPRL/USD", "ETHPY/USD",
    # Meme / political coins
    "TRUMP/USD", "MELANIA/USD", "FARTCOIN/USD", "TITCOIN/USD",
    "TREMP/USD", "BODEN/USD", "GRIFFAIN/USD", "LOCKIN/USD",
    "GOAT/USD", "ZEREBRO/USD", "MOON/USD", "BRICK/USD",
    "BABY/USD", "WEN/USD", "WIN/USD",
    # Not found on Kraken API (CSV data exists but API rejects the symbol name)
    # These trigger 3x retry storms and waste ~90s each — skip entirely
    "ETHW/USD", "GAL/USD", "KAR/USD", "KILT/USD", "KINT/USD",
    "NODL/USD", "OXY/USD", "PSTAKE/USD", "TVK/USD",
    "ZETA/USD", "ZEUS/USD",
}

TIMEFRAMES = ["1h", "4h", "1d"]

# Minimum bars per timeframe (need enough for seq_len + train/val/test split)
# Note: there is NO maximum — more history is always better. Symbols with more
# bars than these minimums are always accepted.
MIN_BARS = {
    "30m": 4000,
    "1h":  3500,   # was 5000 — this was rejecting symbols with 4465-4999 bars
    "4h":  2000,
    "12h": 700,
    "1d":  500,
}

# Minimum trades to consider a result valid — per timeframe.
# These are per-FOLD minimums (aggregated mean across folds).
# Raised: 3 trades per fold at 4h is pure noise — not enough to distinguish
# signal from luck. Higher bar means fewer false positives in the shortlist.
MIN_TRADES_TF = {
    "30m": 10,
    "1h":  8,    # was 6
    "4h":  5,    # was 3 — 3 is statistical noise
    "12h": 3,    # was 2
    "1d":  3,    # was 2
}

# History to fetch per timeframe (used by API fallback only — CSV loader uses HISTORY_CAPS)
# Kept in sync with HISTORY_CAPS in kraken_history.py
HISTORY_DAYS = {
    "30m": 365,
    "1h":  730,    # was 365; raised for 6-fold WF
    "4h":  1095,   # was 730
    "12h": 1095,   # was 730
    "1d":  1825,   # was 1095
}

TRAIN_RATIO  = 0.60
VAL_RATIO    = 0.20
TOP_PCT      = 0.15
N_MODELS     = 9
MIN_TRADES   = 5          # global fallback — MIN_TRADES_TF takes precedence
N_WF_FOLDS   = 6         # was 3; 6 folds = more rigorous out-of-sample validation

_HDR = (
    f"{'Symbol':<12} {'TF':>4} {'Flds':>5} {'Trades':>7} "
    f"{'WR':>6} {'Ann':>7} {'±Std':>6} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>7} "
    f"{'Kelly%':>7} {'Score':>6} {'Status'}"
)
_SEP = "─" * 115


# ── Core ──────────────────────────────────────────────────────────────────────

def asset_class(symbol: str) -> str:
    forex_bases = {"EUR", "GBP", "AUD", "CAD", "CHF", "JPY", "NZD"}
    return "forex" if symbol.split("/")[0] in forex_bases else "crypto"


def _fmt_date(ts_series) -> str:
    """Format first/last timestamp of a series for logging."""
    try:
        t0 = pd.to_datetime(ts_series.iloc[0])
        t1 = pd.to_datetime(ts_series.iloc[-1])
        return f"{t0.strftime('%Y-%m-%d')}→{t1.strftime('%Y-%m-%d')}"
    except Exception:
        return "?"


def run_fold(symbol, timeframe, X_scaled, y, feat_df, train_idx, val_idx, test_idx,
             purge_bars: int = 0):
    """
    Train on one fold and return backtest metrics for the test window.

    purge_bars: strip this many bars from the END of the training set before
    training.  Prevents label leakage: labels at bar i use future prices up to
    bar i+horizon, so the last `horizon` training labels overlap with the val
    set.  Purging removes these contaminated samples.

    Disagreement filter: after ensemble inference, any bar where the 9 models
    strongly disagree (per-class std > DISAGREE_THRESHOLD) is forced to HOLD.
    High disagreement = genuinely ambiguous bar = not worth trading.
    """
    DISAGREE_THRESHOLD = 0.12  # max per-class std before suppressing signal

    from config.settings import LSTM_PARAMS
    from models.lstm_ensemble import train_ensemble, predict_proba_ensemble

    t_start, t_end = train_idx
    # Purge: trim label-leaking tail from training window
    t_end_purged = max(t_start + LSTM_PARAMS["sequence_length"] * 3,
                       t_end - purge_bars)

    X_train = X_scaled[t_start:t_end_purged]
    y_train = y[t_start:t_end_purged]
    X_val   = X_scaled[val_idx[0]:val_idx[1]]
    y_val   = y[val_idx[0]:val_idx[1]]
    X_test  = X_scaled[test_idx[0]:test_idx[1]]
    feat_test = feat_df.iloc[test_idx[0]:test_idx[1]].reset_index(drop=True)

    if len(X_train) < LSTM_PARAMS["sequence_length"] * 3:
        return None

    from models.ensemble import compute_signal_threshold

    models         = train_ensemble(X_train, y_train, X_val, y_val,
                                    symbol=f"{symbol}_{timeframe}", n_models=N_MODELS)
    proba, proba_std = predict_proba_ensemble(models, X_test, return_std=True)

    if proba is None or len(proba) == 0 or proba.shape[1] != 3:
        logger.warning(f"  Fold skipped — predict_proba_ensemble returned empty array "
                       f"(X_test={len(X_test)} rows)")
        return None

    # Fix hindsight bias: compute threshold on val set, apply frozen to test set.
    # Old approach computed the 85th-percentile threshold from the test bars
    # themselves — the backtest knew which bars would be "most confident" in
    # advance.  Now we use the val set to calibrate the threshold, then apply
    # it blindly to the test set, matching what live deployment actually does.
    val_proba = predict_proba_ensemble(models, X_val)
    val_threshold = compute_signal_threshold(val_proba, top_pct=TOP_PCT)
    logger.info(f"  Val-calibrated threshold: {val_threshold:.4f}")

    signals = generate_signals(proba, use_percentile=False, threshold=val_threshold)

    # Ensemble disagreement filter — suppress signals where models disagree
    sig_arr = signals["signal"].copy()
    if proba_std is not None and len(proba_std) == len(sig_arr):
        # Max std across all three classes for each bar
        max_std = proba_std.max(axis=1)
        suppressed = 0
        for i in range(len(sig_arr)):
            if sig_arr[i] != "HOLD" and max_std[i] > DISAGREE_THRESHOLD:
                sig_arr[i] = "HOLD"
                suppressed += 1
        if suppressed:
            logger.info(f"  Disagreement filter: suppressed {suppressed} signals "
                        f"(std > {DISAGREE_THRESHOLD})")
        signals["signal"] = sig_arr
    filtered = apply_filters(feat_test, signals, timeframe=timeframe)
    m        = BacktestEngine().run(feat_test, filtered, symbol=symbol, timeframe=timeframe)

    date_range = _fmt_date(feat_test["timestamp"]) if "timestamp" in feat_test.columns else "?"
    return {
        "net":            float(m.get("total_return",   0.0)),
        "ann":            float(m.get("ann_return",     0.0)),
        "wr":             float(m.get("win_rate",       0.0)),
        "sharpe":         float(m.get("sharpe_ratio",   0.0)),
        "sortino":        float(m.get("sortino_ratio",  0.0)),
        "calmar":         float(m.get("calmar_ratio",   0.0)),
        "dd":             float(m.get("max_drawdown",   0.0)),
        "profit_factor":  float(m.get("profit_factor",  0.0)),
        "recovery_factor":float(m.get("recovery_factor",0.0)),
        "avg_win":        float(m.get("avg_win",        0.0)),
        "avg_loss":       float(m.get("avg_loss",       0.0)),
        "payoff_ratio":   float(m.get("payoff_ratio",   0.0)),
        "max_consec_losses": int(m.get("max_consec_losses", 0)),
        "trend_wr":       float(m.get("trend_wr",       0.0)),
        "range_wr":       float(m.get("range_wr",       0.0)),
        "n_trend_trades": int(m.get("n_trend_trades",   0)),
        "n_range_trades": int(m.get("n_range_trades",   0)),
        "n_trades":       int(m.get("n_trades",         0)),
        "date_range":     date_range,
    }


def run_symbol(symbol: str, timeframe: str) -> dict | None:
    t0 = time.time()
    try:
        days = HISTORY_DAYS.get(timeframe, 730)
        raw  = fetch_ohlcv_timeframe(symbol, timeframe, history_days=days)

        min_b = MIN_BARS.get(timeframe, 1000)
        if raw is None or len(raw) < min_b:
            logger.warning(f"{symbol} {timeframe}: {len(raw) if raw is not None else 0} bars — skipping")
            return None

        # Fetch BTC data for cross-symbol features (skip for BTC itself)
        btc_raw = None
        if symbol not in ("BTC/USD", "BTC/USDT"):
            try:
                btc_raw = fetch_ohlcv_timeframe("BTC/USD", timeframe,
                                                 history_days=days)
            except Exception as _btc_err:
                logger.debug(f"BTC fetch skipped for {symbol}: {_btc_err}")

        feat_df   = build_features(raw, symbol=symbol, timeframe=timeframe,
                                   btc_df=btc_raw)
        feat_cols = get_feature_columns(feat_df)

        from config.settings import LSTM_PARAMS
        seq_len    = LSTM_PARAMS["sequence_length"]
        min_usable = seq_len * 4
        if len(feat_df) < min_usable:
            logger.warning(f"{symbol} {timeframe}: only {len(feat_df)} usable rows — skipping")
            return None

        X = feat_df[feat_cols].values.astype(np.float32)
        y = feat_df["label"].values.astype(int)
        n = len(X)

        # ── Walk-forward CV — N_WF_FOLDS folds ───────────────────────────────
        # Fold geometry is computed dynamically from N_WF_FOLDS so that
        # changing the constant from 3 → 6 automatically adjusts all windows.
        #
        # Design:
        #   - Reserve the last (N_WF_FOLDS * fold_pct) of the data as the
        #     combined test zone — each fold tests a different slice of it.
        #   - Each test window = fold_pct of total data
        #   - Val window = val_pct of total data (fixed size)
        #   - Train window grows (anchored at 0) fold by fold
        #
        # With N_WF_FOLDS=6, fold_pct=0.07, val_pct=0.13:
        #   Total test coverage = 6 × 7% = 42%
        #   First fold trains on 0→45%, tests 58→65%
        #   Last  fold trains on 0→73%, tests 87→94%
        #   Final 6% is left as a holdout buffer
        #
        # With N_WF_FOLDS=3, fold_pct=0.09, val_pct=0.16:
        #   (legacy behaviour, unchanged)

        fold_pct  = round(0.27 / N_WF_FOLDS, 3)   # total test zone / n_folds
        val_pct   = round(0.13 + 0.03 / N_WF_FOLDS, 3)  # val shrinks slightly with more folds
        fold_size = max(int(n * fold_pct), 30)     # at least 30 bars per fold
        val_size  = max(int(n * val_pct),  20)

        # First fold's train_end: enough to have val+test after it
        # Start training from index 0 and end at a point that leaves room for
        # val + all N_WF_FOLDS test windows after it.
        min_train = max(int(n * 0.35), fold_size * 3)
        train_end0 = n - val_size - N_WF_FOLDS * fold_size

        if train_end0 < min_train:
            logger.warning(
                f"{symbol} {timeframe}: insufficient data for {N_WF_FOLDS} folds "
                f"({n} bars, need ~{min_train + val_size + N_WF_FOLDS * fold_size}). "
                f"Reducing to {max(1, (n - min_train - val_size) // fold_size)} folds."
            )
            # Recompute with as many folds as the data can support
            max_folds = max(1, (n - min_train - val_size) // fold_size)
            effective_folds = min(N_WF_FOLDS, max_folds)
            train_end0 = n - val_size - effective_folds * fold_size
        else:
            effective_folds = N_WF_FOLDS

        # Purge gap = label horizon for this timeframe.
        # Strips the contaminated tail from training so labels near the
        # train/val boundary don't leak future-price information into training.
        from data.feature_engineer import get_label_params as _glp
        _horizon, _ = _glp(timeframe)

        fold_results = []
        for fold in range(effective_folds):
            train_end = train_end0 + fold * fold_size
            val_end   = train_end  + val_size
            test_end  = val_end    + fold_size
            if test_end > n:
                break

            # Fit scaler on the most recent ~1 year of training data only.
            # Fitting on all history (e.g. 2020–2024) bakes in old volatility
            # regimes — the scaler's mean/std are dominated by conditions that
            # no longer hold.  Using only recent data means the normalisation
            # reflects the current distributional environment.
            _scaler_window = {"1h": 8760, "4h": 2190, "8h": 1095,
                              "12h": 730, "1d": 365}.get(timeframe, 8760)
            _scaler_start  = max(0, train_end - _scaler_window)
            scaler   = StandardScaler()
            scaler.fit(X[_scaler_start:train_end])   # fit on recent year only
            X_scaled = X.copy()
            X_scaled[:train_end]        = scaler.transform(X[:train_end])
            X_scaled[train_end:val_end] = scaler.transform(X[train_end:val_end])
            X_scaled[val_end:test_end]  = scaler.transform(X[val_end:test_end])

            fr = run_fold(symbol, timeframe, X_scaled, y, feat_df,
                          train_idx=(0, train_end),
                          val_idx  =(train_end, val_end),
                          test_idx =(val_end, test_end),
                          purge_bars=_horizon)
            if fr:
                logger.info(f"  Fold {fold+1}/{effective_folds} [{fr['date_range']}]: "
                            f"net={fr['net']:+.1%} ann={fr['ann']:+.1%} "
                            f"WR={fr['wr']:.1%} Sh={fr['sharpe']:.2f} trades={fr['n_trades']}")
                fold_results.append(fr)

        if not fold_results:
            logger.warning(f"{symbol} {timeframe}: all folds failed")
            return None

        # ── Aggregate across folds ────────────────────────────────────────────
        def _mean(key):  return float(np.mean([f[key] for f in fold_results]))
        def _std(key):   return float(np.std( [f[key] for f in fold_results]))
        def _imean(key): return int(round(np.mean([f[key] for f in fold_results])))

        nets        = [f["net"] for f in fold_results]
        date_ranges = [f["date_range"] for f in fold_results]

        net      = _mean("net")
        ann      = _mean("ann")
        ann_std  = _std("ann")
        wr       = _mean("wr")
        sharpe   = _mean("sharpe")
        sortino  = _mean("sortino")
        calmar   = _mean("calmar")
        dd       = _mean("dd")
        pf       = _mean("profit_factor")
        rf       = _mean("recovery_factor")
        payoff_r = _mean("payoff_ratio")
        avg_win  = _mean("avg_win")
        avg_loss = _mean("avg_loss")
        max_cl   = max(f["max_consec_losses"] for f in fold_results)
        trend_wr = _mean("trend_wr")
        range_wr = _mean("range_wr")
        n_trend  = _imean("n_trend_trades")
        n_range  = _imean("n_range_trades")
        n_trades = _imean("n_trades")
        n_folds_pos = sum(1 for x in nets if x > 0)

        # ── Wilcoxon signed-rank test for fold-return significance ────────────
        # Tests whether the distribution of fold returns is significantly > 0.
        # With only 6 folds, t-test assumptions don't hold (non-normal, small n).
        # Wilcoxon is non-parametric and valid here.  p < 0.10 = 90% confidence
        # the median fold return is genuinely positive.
        try:
            from scipy.stats import wilcoxon as _wilcoxon
            if len(nets) >= 4 and any(x != nets[0] for x in nets):  # need variance
                _, _p_val = _wilcoxon(nets, alternative="greater", zero_method="wilcox")
            else:
                _p_val = 1.0   # too few folds or all identical (degenerate)
        except Exception:
            _p_val = 1.0
        p_significant = _p_val < 0.10

        # ── Quality score (composite, 0-100) ──────────────────────────────────
        consistency  = n_folds_pos / max(len(fold_results), 1)
        ann_norm     = float(np.clip(ann / 0.40, -1, 1)) * 0.5 + 0.5
        sharpe_norm  = float(np.clip(sharpe / 2.5, -1, 1)) * 0.5 + 0.5
        dd_norm      = float(np.clip(1 + dd / 0.30, 0, 1))
        wr_norm      = float(np.clip(wr / 0.60, 0, 1))
        trade_norm   = float(np.clip(np.log1p(n_trades) / np.log1p(50), 0, 1))
        quality      = round(
            ann_norm   * 25 +
            sharpe_norm* 25 +
            consistency* 20 +
            dd_norm    * 15 +
            wr_norm    * 10 +
            trade_norm *  5, 1)

        # ── Kelly position size recommendation ────────────────────────────────
        # Quarter-Kelly from measured win-rate and payoff ratio
        b            = max(payoff_r, 0.01)
        full_kelly   = max(0.0, (wr * b - (1 - wr)) / b)
        kelly_pct    = round(float(np.clip(full_kelly * 0.25, 0.03, 0.40)) * 100, 1)

        # ── Regime attribution ────────────────────────────────────────────────
        if n_trend >= 3 and n_range >= 3:
            if trend_wr > 0.52 and range_wr < 0.48:
                regime_edge = "trend-following"
            elif range_wr > 0.52 and trend_wr < 0.48:
                regime_edge = "mean-reversion"
            elif trend_wr > 0.50 and range_wr > 0.50:
                regime_edge = "regime-robust"
            else:
                regime_edge = "mixed"
        elif n_trend >= 3:
            regime_edge = f"trend-only({trend_wr:.0%})"
        elif n_range >= 3:
            regime_edge = f"range-only({range_wr:.0%})"
        else:
            regime_edge = "insufficient"

        # ── Status label ──────────────────────────────────────────────────────
        min_trades = MIN_TRADES_TF.get(timeframe, MIN_TRADES)
        if n_trades < min_trades:
            status = "TOO_FEW_TRADES"
        elif quality >= 65 and n_folds_pos == len(fold_results) and p_significant:
            status = "** STRONG"   # great metrics AND statistically significant
        elif quality >= 65 and n_folds_pos == len(fold_results):
            status = "* GOOD"      # great metrics but p >= 0.10 (not enough folds)
        elif quality >= 52 and n_folds_pos >= len(fold_results) - 1:
            status = "* GOOD"
        elif ann > 0.0 and wr >= 0.35:
            status = "+ MARGINAL"
        elif ann > 0.0:
            status = "~ WEAK+"
        else:
            status = "- LOSS"

        # Consistency flag — if any fold was strongly negative, downgrade
        if any(x < -0.05 for x in nets) and status in ("** STRONG", "* GOOD"):
            status = "~ INCONSISTENT"

        elapsed = time.time() - t0
        label_counts = {int(k): int(v)
                        for k, v in zip(*np.unique(y, return_counts=True))}

        return {
            "symbol":           symbol,
            "timeframe":        timeframe,
            "asset_class":      asset_class(symbol),
            "n_models":         N_MODELS,
            "n_bars":           n,
            "n_folds":          len(fold_results),
            "folds_positive":   n_folds_pos,
            "fold_dates":       " | ".join(date_ranges),
            "label_down":       label_counts.get(0, 0),
            "label_neutral":    label_counts.get(1, 0),
            "label_up":         label_counts.get(2, 0),
            "n_trades":         n_trades,
            "win_rate":         round(wr, 4),
            "net_return":       round(net, 4),
            "ann_return":       round(ann, 4),
            "ann_std":          round(ann_std, 4),
            "sharpe":           round(sharpe, 3),
            "sortino":          round(sortino, 3),
            "calmar":           round(calmar, 3),
            "max_drawdown":     round(dd, 4),
            "profit_factor":    round(pf, 3),
            "recovery_factor":  round(rf, 3),
            "payoff_ratio":     round(payoff_r, 3),
            "avg_win":          round(avg_win, 4),
            "avg_loss":         round(avg_loss, 4),
            "max_consec_losses":max_cl,
            "trend_wr":         round(trend_wr, 4),
            "range_wr":         round(range_wr, 4),
            "n_trend_trades":   n_trend,
            "n_range_trades":   n_range,
            "regime_edge":      regime_edge,
            "quality_score":    quality,
            "wilcox_p":         round(_p_val, 4),
            "kelly_pct":        kelly_pct,
            "status":           status,
            "elapsed_s":        round(elapsed, 1),
        }

    except Exception as e:
        logger.error(f"{symbol} {timeframe} failed: {e}")
        import traceback; traceback.print_exc()
        return None


def print_row(r: dict):
    folds_str = f"{r.get('folds_positive', '?')}/{r.get('n_folds', '?')}"
    score     = r.get('quality_score', 0)
    kelly     = r.get('kelly_pct', 0)
    tag       = " ★★" if score >= 65 else (" ★" if score >= 52 else "")
    print(
        f"  {r['symbol']:<12} {r['timeframe']:>4}  {folds_str:>5}  "
        f"{r['n_trades']:>6}  {r['win_rate']:>5.1%}  "
        f"{r['ann_return']:>+6.1%}  ±{r.get('ann_std', 0):>4.1%}  "
        f"{r['sharpe']:>6.2f}  {r.get('sortino', 0):>7.2f}  "
        f"{r['max_drawdown']:>6.1%}  "
        f"{kelly:>5.1f}%  {score:>5.1f}  {r['status']}{tag}"
    )


# Canonical column order — must match result dict keys from run_symbol().
# Defined here so the CSV header is stable whether we write one row or many.
_FIELDNAMES = [
    "symbol", "timeframe", "asset_class", "n_models", "n_bars",
    "n_folds", "folds_positive", "fold_dates",
    "label_down", "label_neutral", "label_up",
    "n_trades", "win_rate", "net_return", "ann_return", "ann_std",
    "sharpe", "sortino", "calmar", "max_drawdown",
    "profit_factor", "recovery_factor", "payoff_ratio",
    "avg_win", "avg_loss", "max_consec_losses",
    "trend_wr", "range_wr", "n_trend_trades", "n_range_trades",
    "regime_edge", "quality_score", "wilcox_p",
    "kelly_pct", "status", "elapsed_s",
]


def append_result_row(path: Path, row: dict) -> None:
    """
    Append one result row to the CSV immediately after it completes.
    Writes the header iff the file is new or empty — safe to call row-by-row.
    All writes happen in the main process so no cross-process lock is needed.
    """
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDNAMES, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def save_csv(results, path):
    """Rewrite the full CSV — used at the end for a clean sorted snapshot."""
    if not results:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)


def load_completed(path: Path) -> set:
    """Load already-completed symbol/timeframe pairs for resume support."""
    if not path.exists():
        return set()
    completed = set()
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            completed.add((row["symbol"], row["timeframe"]))
    return completed


def print_summary(results: list):
    valid  = [r for r in results
              if isinstance(r.get("n_trades"), (int, float))
              and r["n_trades"] >= MIN_TRADES_TF.get(r.get("timeframe", "4h"), MIN_TRADES)]
    if not valid:
        print("\nNo valid results.")
        return

    # Rank by quality score (falls back to ann_return for old-format CSVs)
    ranked     = sorted(valid,
                        key=lambda r: float(r.get("quality_score", r.get("ann_return", 0))),
                        reverse=True)
    profitable = [r for r in ranked if float(r.get("ann_return", 0)) > 0]

    sep = "─" * 115
    print(f"\n{'='*115}")
    print(f"  EXPANDED ENSEMBLE SCAN — FINAL SUMMARY  (Walk-forward CV: {N_WF_FOLDS} folds)")
    print(f"{'='*115}")
    print(f"  {_HDR}")
    print(f"  {sep}")
    for r in ranked[:30]:
        print_row(r)
    print(f"  {sep}")

    print(f"\n  Profitable: {len(profitable)}/{len(ranked)} symbol/timeframe combos")

    # Best per asset class by quality score
    for cls in ["crypto", "forex"]:
        cls_results = [r for r in profitable if r.get("asset_class") == cls]
        if cls_results:
            best = max(cls_results, key=lambda r: float(r.get("quality_score", 0)))
            print(f"  Best {cls}: {best['symbol']} {best['timeframe']} — "
                  f"Score={best.get('quality_score',0):.1f} | "
                  f"{best['ann_return']:+.1%} ann ±{best.get('ann_std',0):.1%} | "
                  f"{best['win_rate']:.1%} WR | {best['sharpe']:.2f} Sh | "
                  f"{best.get('sortino',0):.2f} So | "
                  f"Kelly={best.get('kelly_pct',0):.1f}%")

    # Shortlist — quality score threshold + all folds positive
    shortlist = [r for r in ranked
                 if float(r.get("quality_score", 0)) >= 52
                 and float(r.get("ann_return", 0)) > 0
                 and int(r.get("folds_positive", 0)) >= int(r.get("n_folds", 1))]

    print(f"\n  SHORTLIST for live deployment ({len(shortlist)} combos — quality≥52, all folds positive):")
    print(f"  {'Symbol':<12} {'TF':>4} {'Score':>6} {'Ann%':>7} {'Sharpe':>7} {'Sortino':>8} "
          f"{'MaxDD':>7} {'WR':>6} {'PF':>6} {'Kelly%':>7} {'RegimeEdge'}")
    print(f"  {'─'*113}")
    for r in shortlist:
        print(
            f"  {r['symbol']:<12} {r['timeframe']:>4}  "
            f"{float(r.get('quality_score',0)):>5.1f}  "
            f"{float(r['ann_return']):>+6.1%}  "
            f"{float(r['sharpe']):>6.2f}  "
            f"{float(r.get('sortino',0)):>7.2f}  "
            f"{float(r['max_drawdown']):>6.1%}  "
            f"{float(r['win_rate']):>5.1%}  "
            f"{float(r.get('profit_factor',0)):>5.2f}  "
            f"{float(r.get('kelly_pct',0)):>5.1f}%  "
            f"{r.get('regime_edge','?')}"
        )

    # Watchlist — quality >= 40 but not shortlisted
    watchlist = [r for r in ranked
                 if float(r.get("quality_score", 0)) >= 40
                 and float(r.get("ann_return", 0)) > 0
                 and r not in shortlist
                 and int(r.get("folds_positive", 0)) >= max(1, int(r.get("n_folds", 1)) - 1)]
    if watchlist:
        print(f"\n  WATCHLIST (quality 40-52, promising but needs more evidence):")
        for r in watchlist:
            print(f"    {r['symbol']:<12} {r['timeframe']:>4} | "
                  f"Score={float(r.get('quality_score',0)):.1f} | "
                  f"{float(r['ann_return']):+.1%} ann ±{float(r.get('ann_std',0)):.1%} | "
                  f"{int(r.get('folds_positive',0))}/{int(r.get('n_folds',0))} folds | "
                  f"{r.get('regime_edge','?')} | "
                  f"fold dates: {r.get('fold_dates','')}")

    # Regime summary — surface regime-robust pairs separately
    regime_robust = [r for r in shortlist if r.get("regime_edge") == "regime-robust"]
    if regime_robust:
        print(f"\n  REGIME-ROBUST (edge holds in both trending AND ranging markets):")
        for r in regime_robust:
            print(f"    {r['symbol']:<12} {r['timeframe']:>4} | "
                  f"TrendWR={float(r.get('trend_wr',0)):.1%}({int(r.get('n_trend_trades',0))}t) | "
                  f"RangeWR={float(r.get('range_wr',0)):.1%}({int(r.get('n_range_trades',0))}t) | "
                  f"Kelly={float(r.get('kelly_pct',0)):.1f}%")

    # Multi-TF winners
    tf_groups = {}
    for r in profitable:
        tf_groups.setdefault(r["symbol"], []).append(r)
    multi_tf = {s: rs for s, rs in tf_groups.items() if len(rs) > 1}
    if multi_tf:
        print(f"\n  MULTI-TIMEFRAME WINNERS (profitable across >1 TF):")
        for sym, rs in sorted(multi_tf.items()):
            tfs = " | ".join(
                f"{r['timeframe']} {float(r['ann_return']):+.1%} "
                f"(Sh={float(r['sharpe']):.2f} Score={float(r.get('quality_score',0)):.0f} "
                f"{int(r.get('folds_positive',0))}/{int(r.get('n_folds',0))}flds)"
                for r in sorted(rs, key=lambda x: x["timeframe"])
            )
            print(f"    {sym:<12}  {tfs}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def _run_single(gpu_id: int, symbol: str, timeframe: str) -> dict | None:
    """
    One (symbol, timeframe) combo per future. Sets CUDA_VISIBLE_DEVICES before
    any torch import so each task lands on the right GPU.

    Submitting one future per combo (instead of one per worker partition) means
    as_completed() fires after every result — the main process can write to CSV
    and update the dashboard immediately, giving live progress throughout the run.
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        return run_symbol(symbol, timeframe)
    except Exception as e:
        logger.error(f"GPU {gpu_id} | {symbol} {timeframe} crashed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Full universe ensemble scan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tier definitions (4h bar counts):
  original  : 38 symbols previously scanned
  1         : 84 new pairs with full 2yr history (4300+ bars)
  2         : 132 new pairs with strong history (3000-4299 bars)
  3         : 50 new pairs with shorter history (2000-2999 bars)
  all       : original + tier1 + tier2 + tier3 (full universe, ~260 symbols)

Examples:
  python expanded_scan.py --tier original 1 2 --gpus 4 --workers 8
  python expanded_scan.py --tier original 1 2 --gpus 8 --workers 16
  python expanded_scan.py --tier 1
  python expanded_scan.py --tier all --gpus 8 --workers 16
  python expanded_scan.py --tier 1 2 --timeframes 4h 1h
  python expanded_scan.py --symbols SNX/USD EOS/USD --timeframes 4h
        """
    )
    parser.add_argument("--timeframes",  nargs="+", default=TIMEFRAMES)
    parser.add_argument("--symbols",     nargs="+", default=None,
                        help="Explicit symbol list — overrides --tier")
    parser.add_argument("--tier",        nargs="+",
                        choices=["original", "1", "2", "3", "all"],
                        default=["original", "1", "2"],
                        help="Which tier(s) to scan. Default: original+1+2. Use 'all' for everything.")
    parser.add_argument("--asset-class", choices=["all", "crypto", "forex"],
                        default="all")
    parser.add_argument("--resume",      action="store_true",
                        help="Skip symbol/TF combos already in output CSV")
    parser.add_argument("--workers",     type=int, default=None,
                        help="Number of parallel worker processes. Each gets its own "
                             "CUDA_VISIBLE_DEVICES assignment (round-robin across --gpus). "
                             "On AMD/DirectML, CUDA_VISIBLE_DEVICES has no effect — use "
                             "--workers 1 to avoid GPU memory contention. "
                             "Default: same as --gpus (1 worker per GPU).")
    parser.add_argument("--gpus",        type=int, default=1,
                        help="Number of CUDA GPUs available (default: 1). Used for "
                             "CUDA_VISIBLE_DEVICES round-robin assignment across workers. "
                             "On AMD/DirectML ignore this — set --workers 1 instead.")
    args = parser.parse_args()
    if args.workers is None:
        args.workers = args.gpus  # 1 worker per GPU by default

    timeframes = args.timeframes

    # ── Symbol selection ──────────────────────────────────────────────────────
    if args.symbols:
        symbols = args.symbols
        tier_label = "explicit"
        logger.info(f"Using explicit symbol list: {len(symbols)} symbols")

    else:
        tiers = set(args.tier)
        # "all" expands to every tier
        if "all" in tiers:
            tiers = {"original", "1", "2", "3"}

        pool = []
        if "original" in tiers:
            pool += PAIRS_ORIGINAL
        if "1" in tiers:
            pool += PAIRS_TIER1
        if "2" in tiers:
            pool += PAIRS_TIER2
        if "3" in tiers:
            pool += PAIRS_TIER3

        # Deduplicate preserving order, apply exclude list
        seen = set()
        crypto_symbols = []
        for s in pool:
            if s not in seen and s not in EXCLUDE_PAIRS:
                seen.add(s)
                crypto_symbols.append(s)

        if args.asset_class == "forex":
            symbols = FOREX_PAIRS_FALLBACK
        elif args.asset_class == "crypto":
            symbols = crypto_symbols
        else:
            # Add forex (original already contains them; avoid dupes)
            forex_new = [f for f in FOREX_PAIRS_FALLBACK if f not in seen]
            symbols = crypto_symbols + forex_new

        tier_label = "+".join(sorted(tiers))
        logger.info(f"Tiers [{tier_label}]: {len(symbols)} symbols selected")

    total = len(symbols) * len(timeframes)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir  = Path("backtest/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"expanded_scan_{ts}.csv"

    # Resume support
    completed = set()
    if args.resume:
        existing = sorted(out_dir.glob("expanded_scan_*.csv"), reverse=True)
        if existing:
            csv_path  = existing[0]
            completed = load_completed(csv_path)
            logger.info(f"Resuming from {csv_path} — {len(completed)} combos already done")

    discovery_mode = tier_label
    print(f"\n{'='*85}")
    print(f"  EXPANDED ENSEMBLE SCAN — FULL UNIVERSE")
    print(f"  Discovery: {discovery_mode} | Symbols: {len(symbols)} | Timeframes: {timeframes}")
    print(f"  Total combos: {total} | Models per combo: {N_MODELS} | WF folds: {N_WF_FOLDS}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Output:  {csv_path}")
    print(f"{'='*85}")
    print(f"  {_HDR}")
    print("  " + _SEP)

    t_scan_start = time.time()
    results, failed, skipped = [], [], 0

    # Load existing results if resuming
    if completed and csv_path.exists():
        with open(csv_path, encoding="utf-8-sig") as f:
            results = list(csv.DictReader(f))
            # Coerce numeric fields back from strings
            int_fields   = ["n_bars","n_models","n_folds","folds_positive",
                            "n_trades","n_trend_trades","n_range_trades",
                            "label_down","label_neutral","label_up","max_consec_losses"]
            float_fields = ["win_rate","net_return","ann_return","ann_std",
                            "sharpe","sortino","calmar","max_drawdown",
                            "profit_factor","recovery_factor","payoff_ratio",
                            "avg_win","avg_loss","trend_wr","range_wr",
                            "quality_score","wilcox_p","kelly_pct","elapsed_s"]
            for r in results:
                for k in int_fields:
                    if k in r and r[k] != "":
                        r[k] = int(float(r[k]))
                for k in float_fields:
                    if k in r and r[k] != "":
                        r[k] = float(r[k])

    # Build work queue — filter already-completed combos
    work = []
    for symbol in symbols:
        for timeframe in timeframes:
            if (symbol, timeframe) in completed:
                skipped += 1
            else:
                work.append((symbol, timeframe))

    total_work = len(work)
    n_workers  = min(args.workers, total_work) if total_work > 0 else 1
    logger.info(
        f"Running {total_work} combos with {n_workers} parallel worker(s) "
        f"across {args.gpus} GPU(s)"
    )

    # Assign each combo a GPU round-robin — on DirectML this has no effect.
    combo_gpu = {(sym, tf): i % args.gpus for i, (sym, tf) in enumerate(work)}

    done_count = 0

    # One future per combo — as_completed() fires after every single result so
    # the CSV and dashboard update in real time, not in one burst at the end.
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_run_single, combo_gpu[(sym, tf)], sym, tf): (sym, tf)
            for sym, tf in work
        }

        for future in as_completed(futures):
            sym, tf = futures[future]
            done_count += 1
            try:
                r = future.result()
            except Exception as e:
                logger.error(f"{sym} {tf} crashed: {e}")
                failed.append(f"{sym}_{tf}")
                continue

            if r is None:
                failed.append(f"{sym}_{tf}")
                logger.warning(f"[{done_count}/{total_work}] {sym} {tf} — no result")
                continue

            results.append(r)
            print_row(r)
            append_result_row(csv_path, r)   # write immediately — dashboard sees it now
            logger.info(f"[{done_count}/{total_work}] {sym} {tf} done")

    print("  " + _SEP)
    print_summary(results)

    if skipped:
        print(f"  Skipped (already complete): {skipped}")
    if failed:
        print(f"  Failed/skipped ({len(failed)}): {', '.join(failed[:10])}"
              + ("..." if len(failed) > 10 else ""))

    # Final rewrite — sorts by quality_score desc so the CSV opens cleanly
    # in the dashboard. Rows were already written incrementally during the run.
    results_valid = [r for r in results if isinstance(r, dict)]
    results_valid.sort(key=lambda r: float(r.get("quality_score", 0)), reverse=True)
    save_csv(results_valid, csv_path)
    print(f"  Results saved -> {csv_path}\n")

    # ── Telegram completion notification ─────────────────────────────────────
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent / "paper_trader"))
        from src.notifier import send as _tg_send
        elapsed_h = (time.time() - t_scan_start) / 3600
        strong = sum(1 for r in results if "STRONG" in str(r.get("status", "")))
        good   = sum(1 for r in results if str(r.get("status", "")) == "* GOOD")
        _tg_send(
            f"✅ <b>Scan Complete</b> ({elapsed_h:.1f}h)\n"
            f"Combos: {total} | Failed: {len(failed)} | Skipped: {skipped}\n"
            f"<b>Strong: {strong}</b> | Good: {good}\n"
            f"Output: {csv_path.name}"
        )
    except Exception:
        pass   # Telegram is optional


if __name__ == "__main__":
    set_start_method("spawn", force=True)
    main()