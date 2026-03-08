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
from datetime import datetime
from pathlib import Path

_here   = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
sys.path.insert(0, _here)
sys.path.insert(0, _parent)

import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler

from timeframe_comparison import fetch_ohlcv_timeframe, fetch_ohlcv_12h
from data.feature_engineer import build_features, get_feature_columns
from backtest.engine import BacktestEngine
from backtest.filters import apply_filters
from models.ensemble import generate_signals
from models.lstm_ensemble import train_ensemble, predict_proba_ensemble

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

TIMEFRAMES = ["30m", "1h", "4h", "12h", "1d"]

# Minimum bars per timeframe (need enough for seq_len + train/val/test split)
MIN_BARS = {
    "30m": 6000,
    "1h":  5000,
    "4h":  3000,
    "12h": 1000,
    "1d":   500,
}

# Minimum trades to consider a result valid — per timeframe
# Higher-frequency TFs have more opportunity so we require more trades
MIN_TRADES_TF = {
    "30m": 15,
    "1h":  15,
    "4h":   8,
    "12h":  5,
    "1d":   5,
}

# History to fetch per timeframe (used by API fallback only — CSV loader uses HISTORY_CAPS)
HISTORY_DAYS = {
    "30m": 180,
    "1h":  365,
    "4h":  730,
    "12h": 730,
    "1d":  1095,
}

TRAIN_RATIO  = 0.60
VAL_RATIO    = 0.20
TOP_PCT      = 0.15
N_MODELS     = 9
MIN_TRADES   = 5          # global fallback — MIN_TRADES_TF takes precedence
N_WF_FOLDS   = 3         # walk-forward CV folds

_HDR = (
    f"{'Symbol':<12} {'TF':>4} {'Flds':>5} {'Trades':>7} "
    f"{'WR':>6} {'Net':>7} {'Ann':>7} {'±Std':>6} {'Sharpe':>7} {'Calmar':>7} {'MaxDD':>7} {'Status'}"
)
_SEP = "─" * 105


# ── Core ──────────────────────────────────────────────────────────────────────

def asset_class(symbol: str) -> str:
    forex_bases = {"EUR", "GBP", "AUD", "CAD", "CHF", "JPY", "NZD"}
    return "forex" if symbol.split("/")[0] in forex_bases else "crypto"


def _fmt_date(ts_series) -> str:
    """Format first/last timestamp of a series for logging."""
    try:
        return f"{ts_series.iloc[0].strftime('%Y-%m-%d')}→{ts_series.iloc[-1].strftime('%Y-%m-%d')}"
    except Exception:
        return "?"


def run_fold(symbol, timeframe, X_scaled, y, feat_df, train_idx, val_idx, test_idx):
    """Train on one fold and return backtest metrics for the test window."""
    from config.settings import LSTM_PARAMS
    X_train = X_scaled[train_idx[0]:train_idx[1]]
    y_train = y[train_idx[0]:train_idx[1]]
    X_val   = X_scaled[val_idx[0]:val_idx[1]]
    y_val   = y[val_idx[0]:val_idx[1]]
    X_test  = X_scaled[test_idx[0]:test_idx[1]]
    feat_test = feat_df.iloc[test_idx[0]:test_idx[1]].reset_index(drop=True)

    if len(X_train) < LSTM_PARAMS["sequence_length"] * 3:
        return None

    models   = train_ensemble(X_train, y_train, X_val, y_val,
                               symbol=f"{symbol}_{timeframe}", n_models=N_MODELS)
    proba    = predict_proba_ensemble(models, X_test)

    # Guard: if proba is empty or too short to produce signals, skip fold
    if proba is None or len(proba) == 0 or proba.shape[1] != 3:
        logger.warning(f"  Fold skipped — predict_proba_ensemble returned empty array "
                       f"(X_test={len(X_test)} rows)")
        return None

    signals  = generate_signals(proba, use_percentile=True, top_pct=TOP_PCT)
    filtered = apply_filters(feat_test, signals, timeframe=timeframe)
    m        = BacktestEngine().run(feat_test, filtered, symbol=symbol, timeframe=timeframe)

    date_range = _fmt_date(feat_test["timestamp"]) if "timestamp" in feat_test.columns else "?"
    return {
        "net":        float(m.get("total_return", 0.0)),
        "ann":        float(m.get("ann_return",   0.0)),
        "wr":         float(m.get("win_rate",      0.0)),
        "sharpe":     float(m.get("sharpe_ratio",  0.0)),
        "calmar":     float(m.get("calmar_ratio",  0.0)),
        "dd":         float(m.get("max_drawdown",  0.0)),
        "n_trades":   int(m.get("n_trades",        0)),
        "date_range": date_range,
    }


def run_symbol(symbol: str, timeframe: str) -> dict | None:
    t0 = time.time()
    try:
        days = HISTORY_DAYS.get(timeframe, 730)
        if timeframe == "12h":
            raw = fetch_ohlcv_12h(symbol, history_days=days)
        else:
            raw  = fetch_ohlcv_timeframe(symbol, timeframe, history_days=days)

        min_b = MIN_BARS.get(timeframe, 1000)
        if raw is None or len(raw) < min_b:
            logger.warning(f"{symbol} {timeframe}: {len(raw) if raw is not None else 0} bars — skipping")
            return None

        feat_df   = build_features(raw, timeframe=timeframe)
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
        # Each fold: train on older data, validate on the next slice, test on
        # a fresh slice that was never seen during training.
        # Fold windows stagger forward so the test periods span different regimes.
        #
        # Example with 3 folds over n bars:
        #   Fold 0: train[0:0.48n] val[0.48:0.64n] test[0.64:0.73n]
        #   Fold 1: train[0:0.57n] val[0.57:0.73n] test[0.73:0.82n]
        #   Fold 2: train[0:0.66n] val[0.66:0.82n] test[0.82:n]
        #
        # The train window grows (anchored start) while test steps forward,
        # mimicking how a deployed model would be retrained over time.

        fold_size  = int(n * 0.09)    # each fold's test window = ~9% of data
        val_size   = int(n * 0.16)
        train_end0 = int(n * 0.48)    # first fold train end

        fold_results = []
        for fold in range(N_WF_FOLDS):
            train_end = train_end0 + fold * fold_size
            val_end   = train_end  + val_size
            test_end  = val_end    + fold_size
            if test_end > n:
                break

            # Fit scaler on this fold's train only
            scaler   = StandardScaler()
            X_scaled = X.copy()
            X_scaled[:train_end]        = scaler.fit_transform(X[:train_end])
            X_scaled[train_end:val_end] = scaler.transform(X[train_end:val_end])
            X_scaled[val_end:test_end]  = scaler.transform(X[val_end:test_end])

            fr = run_fold(symbol, timeframe, X_scaled, y, feat_df,
                          train_idx=(0, train_end),
                          val_idx  =(train_end, val_end),
                          test_idx =(val_end, test_end))
            if fr:
                logger.info(f"  Fold {fold+1}/{N_WF_FOLDS} [{fr['date_range']}]: "
                            f"net={fr['net']:+.1%} ann={fr['ann']:+.1%} "
                            f"WR={fr['wr']:.1%} Sh={fr['sharpe']:.2f} trades={fr['n_trades']}")
                fold_results.append(fr)

        if not fold_results:
            logger.warning(f"{symbol} {timeframe}: all folds failed")
            return None

        # ── Aggregate across folds ────────────────────────────────────────────
        nets      = [f["net"]    for f in fold_results]
        anns      = [f["ann"]    for f in fold_results]
        wrs       = [f["wr"]     for f in fold_results]
        sharpes   = [f["sharpe"] for f in fold_results]
        calmars   = [f["calmar"] for f in fold_results]
        dds       = [f["dd"]     for f in fold_results]
        all_trades= [f["n_trades"] for f in fold_results]
        date_ranges = [f["date_range"] for f in fold_results]

        net      = float(np.mean(nets))
        ann      = float(np.mean(anns))
        ann_std  = float(np.std(anns))
        wr       = float(np.mean(wrs))
        sharpe   = float(np.mean(sharpes))
        calmar   = float(np.mean(calmars))
        dd       = float(np.mean(dds))
        n_trades = int(np.mean(all_trades))
        n_folds_pos = sum(1 for x in nets if x > 0)

        min_trades = MIN_TRADES_TF.get(timeframe, MIN_TRADES)

        if n_trades < min_trades:
            status = "TOO_FEW_TRADES"
        elif ann > 0.20 and wr >= 0.42 and n_folds_pos == len(fold_results):
            status = "** STRONG"
        elif ann > 0.10 and wr >= 0.38 and n_folds_pos >= len(fold_results) - 1:
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
            "symbol":        symbol,
            "timeframe":     timeframe,
            "asset_class":   asset_class(symbol),
            "n_models":      N_MODELS,
            "n_bars":        n,
            "n_folds":       len(fold_results),
            "folds_positive": n_folds_pos,
            "fold_dates":    " | ".join(date_ranges),
            "label_down":    label_counts.get(0, 0),
            "label_neutral": label_counts.get(1, 0),
            "label_up":      label_counts.get(2, 0),
            "n_trades":      n_trades,
            "win_rate":      round(wr, 4),
            "net_return":    round(net, 4),
            "ann_return":    round(ann, 4),
            "ann_std":       round(ann_std, 4),
            "sharpe":        round(sharpe, 3),
            "calmar":        round(calmar, 3),
            "max_drawdown":  round(dd, 4),
            "status":        status,
            "elapsed_s":     round(elapsed, 1),
        }

    except Exception as e:
        logger.error(f"{symbol} {timeframe} failed: {e}")
        import traceback; traceback.print_exc()
        return None


def print_row(r: dict):
    folds_str = f"{r.get('folds_positive', '?')}/{r.get('n_folds', '?')}"
    print(
        f"  {r['symbol']:<12} {r['timeframe']:>4}  {folds_str:>5}  "
        f"{r['n_trades']:>6}  {r['win_rate']:>5.1%}  "
        f"{r['net_return']:>+6.1%}  {r['ann_return']:>+6.1%}  ±{r.get('ann_std', 0):>4.1%}  "
        f"{r['sharpe']:>6.2f}  {r.get('calmar', 0):>6.2f}  "
        f"{r['max_drawdown']:>6.1%}  {r['status']}"
    )


def save_csv(results, path):
    if not results:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
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
              and r["n_trades"] >= MIN_TRADES_TF.get(r.get("timeframe","4h"), MIN_TRADES)]
    if not valid:
        print("\nNo valid results.")
        return

    ranked     = sorted(valid, key=lambda r: r["ann_return"], reverse=True)
    profitable = [r for r in ranked if r["ann_return"] > 0]

    print(f"\n{'='*105}")
    print(f"  EXPANDED ENSEMBLE SCAN — FINAL SUMMARY  (Walk-forward CV: {N_WF_FOLDS} folds)")
    print(f"{'='*105}")
    print(f"  {_HDR}")
    print("  " + _SEP)
    for r in ranked[:30]:
        print_row(r)
    print("  " + _SEP)

    print(f"\n  Profitable: {len(profitable)}/{len(ranked)} symbol/timeframe combos")

    # Best per asset class by Calmar (better than Sharpe for deployment decisions)
    for cls in ["crypto", "forex"]:
        cls_results = [r for r in profitable if r.get("asset_class") == cls]
        if cls_results:
            best = max(cls_results, key=lambda r: r.get("calmar", 0))
            print(f"  Best {cls}: {best['symbol']} {best['timeframe']} — "
                  f"{best['ann_return']:+.1%} ann | ±{best.get('ann_std',0):.1%} std | "
                  f"{best['win_rate']:.1%} WR | {best['sharpe']:.2f} Sh | {best.get('calmar',0):.2f} Calmar")

    # Shortlist — now requires consistency across folds (folds_positive = n_folds)
    shortlist = [r for r in ranked
                 if r["ann_return"] > 0.10
                 and r["win_rate"] >= 0.38
                 and r.get("folds_positive", 0) >= r.get("n_folds", 1)]
    print(f"\n  SHORTLIST for OOT validation ({len(shortlist)} combos — all folds positive):")
    for r in shortlist:
        print(f"    {r['symbol']:<12} {r['timeframe']:>4} | "
              f"{r['ann_return']:+.1%} ann ±{r.get('ann_std',0):.1%} | "
              f"{r['win_rate']:.1%} WR | {r['sharpe']:.2f} Sh | "
              f"{r.get('calmar',0):.2f} Calmar | "
              f"{r.get('folds_positive',0)}/{r.get('n_folds',0)} folds pos")

    # Watchlist — promising but inconsistent
    watchlist = [r for r in ranked
                 if r["ann_return"] > 0.08
                 and r not in shortlist
                 and r.get("folds_positive", 0) >= max(1, r.get("n_folds", 1) - 1)]
    if watchlist:
        print(f"\n  WATCHLIST (promising but inconsistent across folds):")
        for r in watchlist:
            print(f"    {r['symbol']:<12} {r['timeframe']:>4} | "
                  f"{r['ann_return']:+.1%} ann ±{r.get('ann_std',0):.1%} | "
                  f"{r.get('folds_positive',0)}/{r.get('n_folds',0)} folds pos | "
                  f"fold dates: {r.get('fold_dates','')}")

    # Multi-TF winners
    tf_groups = {}
    for r in profitable:
        tf_groups.setdefault(r["symbol"], []).append(r)
    multi_tf = {s: rs for s, rs in tf_groups.items() if len(rs) > 1}
    if multi_tf:
        print(f"\n  MULTI-TIMEFRAME WINNERS (profitable across >1 TF):")
        for sym, rs in sorted(multi_tf.items()):
            tfs = " | ".join(
                f"{r['timeframe']} {r['ann_return']:+.1%} ({r['sharpe']:.2f}Sh "
                f"{r.get('folds_positive',0)}/{r.get('n_folds',0)}folds)"
                for r in sorted(rs, key=lambda x: x["timeframe"])
            )
            print(f"    {sym:<12}  {tfs}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

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
  python expanded_scan.py --tier 1
  python expanded_scan.py --tier all
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
    args = parser.parse_args()

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

    results, failed, skipped = [], [], 0

    # Load existing results if resuming
    if completed and csv_path.exists():
        with open(csv_path, encoding="utf-8-sig") as f:
            results = list(csv.DictReader(f))
            # Convert numeric fields back
            for r in results:
                for k in ["n_bars","n_train","n_val","n_test","n_trades",
                          "n_buy","n_sell","n_models",
                          "label_down","label_neutral","label_up"]:
                    if k in r:
                        r[k] = int(r[k])
                for k in ["win_rate","net_return","sharpe","max_drawdown",
                          "mean_conf","elapsed_s"]:
                    if k in r:
                        r[k] = float(r[k])

    idx = 0
    for symbol in symbols:
        for timeframe in timeframes:
            idx += 1

            if (symbol, timeframe) in completed:
                skipped += 1
                continue

            logger.info(f"[{idx}/{total}] {symbol} {timeframe}")
            r = run_symbol(symbol, timeframe)

            if r is None:
                failed.append(f"{symbol} {timeframe}")
                continue

            results.append(r)
            print_row(r)
            save_csv(results, csv_path)

    print("  " + _SEP)
    print_summary(results)

    if skipped:
        print(f"  Skipped (already complete): {skipped}")
    if failed:
        print(f"  Failed/skipped ({len(failed)}): {', '.join(failed[:10])}"
              + ("..." if len(failed) > 10 else ""))
    print(f"  Results saved -> {csv_path}\n")


if __name__ == "__main__":
    main()