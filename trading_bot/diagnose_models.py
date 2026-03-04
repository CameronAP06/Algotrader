"""
diagnose_models.py
──────────────────
Trains each model (TFT, CNN, LSTM, LightGBM) independently on a single
symbol/timeframe and prints a side-by-side diagnostic report so you can
see exactly what each model is outputting before committing to a full run.

Usage:
    python diagnose_models.py --symbol DOGE/USD --timeframe 4h
    python diagnose_models.py --symbol LINK/USD --timeframe 2h
    python diagnose_models.py  # defaults to DOGE/USD 4h
"""
import os, sys, argparse

# Support running from either the repo root or the trading_bot subdirectory
_here   = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
sys.path.insert(0, _here)
sys.path.insert(0, _parent)

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler

try:
    from data.feature_engineer import build_features, get_feature_columns
except ModuleNotFoundError:
    from feature_engineer import build_features, get_feature_columns
from backtest.engine import BacktestEngine
from backtest.filters import apply_filters
from models.ensemble import generate_signals, weighted_ensemble

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
TEST_RATIO  = 0.20
FEE         = 0.001
TOP_PCT     = 0.15   # percentile signal threshold

# ── Helpers ───────────────────────────────────────────────────────────────────

def split_data(X, y, feat_df):
    n = len(X)
    t1 = int(n * TRAIN_RATIO)
    t2 = int(n * (TRAIN_RATIO + VAL_RATIO))
    return (
        X[:t1],  y[:t1],
        X[t1:t2], y[t1:t2],
        X[t2:],  y[t2:],
        feat_df.iloc[t2:].reset_index(drop=True),
    )


def evaluate_model(name, proba, y_test, feat_df_test, timeframe):
    """Run backtest on a model's probability output and return metrics."""
    signals  = generate_signals(proba, use_percentile=True, top_pct=TOP_PCT)
    filtered = apply_filters(feat_df_test, signals, timeframe=timeframe)
    engine   = BacktestEngine()
    m        = engine.run(feat_df_test, filtered, timeframe=timeframe)

    n_trades = m.get("n_trades", 0)
    ret      = m.get("total_return", 0.0)
    wr       = m.get("win_rate", 0.0)
    sharpe   = m.get("sharpe_ratio", 0.0)
    dd       = m.get("max_drawdown", 0.0)
    net      = ret - n_trades * FEE
    acc      = (proba.argmax(axis=1) == y_test).mean()

    # Signal breakdown
    up_p   = proba[:, 2]
    dn_p   = proba[:, 0]
    best_p = np.maximum(up_p, dn_p)
    thresh = float(np.percentile(best_p[best_p > 0.34], (1 - TOP_PCT) * 100)) \
             if (best_p > 0.34).sum() > 5 else 0.40

    sig_arr = np.array(signals["signal"])
    n_buy  = (sig_arr == "BUY").sum()
    n_sell = (sig_arr == "SELL").sum()
    n_hold = (sig_arr == "HOLD").sum()

    # Probability distribution
    pred_classes = proba.argmax(axis=1)
    n_pred_down    = (pred_classes == 0).sum()
    n_pred_neutral = (pred_classes == 1).sum()
    n_pred_up      = (pred_classes == 2).sum()

    return {
        "name":       name,
        "acc":        acc,
        "threshold":  thresh,
        "n_buy":      n_buy,
        "n_sell":     n_sell,
        "n_hold":     n_hold,
        "pred_down":  n_pred_down,
        "pred_neut":  n_pred_neutral,
        "pred_up":    n_pred_up,
        "max_prob":   float(best_p.max()),
        "mean_prob":  float(best_p.mean()),
        "n_trades":   n_trades,
        "win_rate":   wr,
        "gross_ret":  ret,
        "net_ret":    net,
        "sharpe":     sharpe,
        "max_dd":     dd,
    }


def print_report(results, symbol, timeframe, n_train, n_val, n_test, label_dist):
    bar = "=" * 90
    print(f"\n{bar}")
    print(f"  MODEL DIAGNOSTIC — {symbol} [{timeframe}]")
    print(f"  Train: {n_train} bars | Val: {n_val} bars | Test: {n_test} bars")
    print(f"  Label dist — DOWN={label_dist.get(0,0)} NEUTRAL={label_dist.get(1,0)} UP={label_dist.get(2,0)}")
    print(bar)

    # Header
    print(f"\n{'Model':<12} {'Acc':>6} {'Thresh':>7} {'BUY':>5} {'SELL':>5} "
          f"{'Trades':>7} {'WR':>6} {'Net':>7} {'Sharpe':>7} {'MaxDD':>7}")
    print("-" * 80)

    for r in results:
        flag = ""
        if r["n_trades"] == 0:       flag = "  ← NO SIGNALS"
        elif r["win_rate"] < 0.30:   flag = "  ← POOR WR"
        elif r["net_ret"] > 0.05:    flag = "  ★ STRONG"
        elif r["net_ret"] > 0:       flag = "  ✓"

        print(f"  {r['name']:<10} {r['acc']:>6.3f}  {r['threshold']:>6.3f}  "
              f"{r['n_buy']:>4}  {r['n_sell']:>4}  {r['n_trades']:>6}  "
              f"{r['win_rate']:>5.1%}  {r['net_ret']:>+6.1%}  "
              f"{r['sharpe']:>6.2f}  {r['max_dd']:>6.1%}{flag}")

    print()
    print("PROBABILITY DISTRIBUTIONS (what the model thinks on test bars):")
    print(f"  {'Model':<12} {'↓DOWN':>7} {'=NEUT':>7} {'↑UP':>7} {'MaxConf':>8} {'MeanConf':>9}")
    print("  " + "-" * 50)
    for r in results:
        n = r["pred_down"] + r["pred_neut"] + r["pred_up"]
        print(f"  {r['name']:<12} {r['pred_down']:>5} ({r['pred_down']/n:.0%})  "
              f"{r['pred_neut']:>5} ({r['pred_neut']/n:.0%})  "
              f"{r['pred_up']:>5} ({r['pred_up']/n:.0%})  "
              f"{r['max_prob']:>7.3f}   {r['mean_prob']:>8.3f}")

    print()
    print("INTERPRETATION GUIDE:")
    print("  Acc ~33%       = random (3-class baseline)")
    print("  Acc >45%       = model is learning something")
    print("  MeanConf ~0.33 = flat/uninformative probabilities")
    print("  MeanConf >0.45 = model has some conviction")
    print("  WR >33%        = profitable at 2:1 RR (breakeven)")
    print("  WR >45%        = genuinely useful edge")
    print(f"\n{'='*90}\n")


def save_results(results: list, symbol: str, timeframe: str):
    """Save diagnostic results to CSV and JSON in backtest/results/."""
    import csv, json, os
    from datetime import datetime
    from pathlib import Path

    out_dir = Path("backtest/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    combo = f"{symbol.replace('/','_')}_{timeframe}"
    base  = out_dir / f"diag_{combo}_{ts}"

    # CSV
    csv_path = base.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        fields = list(results[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    # JSON — custom encoder to handle numpy int64/float32 from model outputs
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):  return int(obj)
            if isinstance(obj, (np.floating,)):  return float(obj)
            if isinstance(obj, np.ndarray):      return obj.tolist()
            return super().default(obj)

    json_path = base.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump({"symbol": symbol, "timeframe": timeframe,
                   "generated": datetime.now().isoformat(),
                   "results": results}, f, indent=2, cls=NumpyEncoder)

    logger.success(f"Diagnostic saved → {csv_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",    default="DOGE/USD")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--models",    nargs="+",
                        default=["lgbm", "lstm", "cnn", "tft"],
                        help="Which models to run (lgbm lstm cnn tft)")
    parser.add_argument("--refresh",   action="store_true")
    args = parser.parse_args()

    symbol    = args.symbol
    timeframe = args.timeframe
    logger.info(f"Diagnosing {symbol} [{timeframe}] with models: {args.models}")

    # ── Fetch & build features ────────────────────────────────────────────────
    from timeframe_comparison import fetch_ohlcv_timeframe, TIMEFRAME_CONFIG, merge_alt_data
    from pathlib import Path
    tf_config = TIMEFRAME_CONFIG[timeframe]

    # Handle --refresh by deleting cache file first
    if args.refresh:
        cache = Path(f"data/raw/{symbol.replace('/','_')}_{tf_config['ccxt_tf']}.csv")
        if cache.exists():
            cache.unlink()
            logger.info(f"Cache cleared for {symbol} {timeframe}")

    raw_df = fetch_ohlcv_timeframe(symbol, tf_config["ccxt_tf"], tf_config["history_days"])
    if raw_df.empty:
        logger.error(f"No data for {symbol} {timeframe}")
        return

    alt_df   = {}  # skip alt data for speed
    feat_df  = build_features(raw_df, symbol, timeframe)
    feat_cols = get_feature_columns(feat_df)

    X = feat_df[feat_cols].values
    y = feat_df["label"].values

    scaler = StandardScaler()
    X_train, y_train, X_val, y_val, X_test, y_test, feat_df_test = split_data(X, y, feat_df)
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    label_dist = {int(k): int(v) for k, v in
                  zip(*np.unique(y_test, return_counts=True))}

    logger.info(f"Data split — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    results  = []
    all_proba = {}

    # ── LightGBM ──────────────────────────────────────────────────────────────
    if "lgbm" in args.models:
        logger.info("\n" + "─"*40 + " LightGBM " + "─"*40)
        from models import lgbm_model
        model = lgbm_model.train(X_train, y_train, X_val, y_val, symbol)
        proba = lgbm_model.predict_proba(model, X_test)
        all_proba["lgbm"] = proba

        # Print top 20 feature importances
        imp = lgbm_model.feature_importance(model, feat_cols)
        top20 = list(imp.items())[:20]
        logger.info("Top 20 features: " + ", ".join(f"{k}({v})" for k,v in top20))

        results.append(evaluate_model("LightGBM", proba, y_test, feat_df_test, timeframe))

    # ── LSTM ──────────────────────────────────────────────────────────────────
    if "lstm" in args.models:
        logger.info("\n" + "─"*40 + " LSTM " + "─"*40)
        from models import lstm_model
        model = lstm_model.train(X_train, y_train, X_val, y_val, symbol)
        proba = lstm_model.predict_proba(model, X_test)
        all_proba["lstm"] = proba
        results.append(evaluate_model("LSTM", proba, y_test, feat_df_test, timeframe))

    # ── CNN ───────────────────────────────────────────────────────────────────
    if "cnn" in args.models:
        logger.info("\n" + "─"*40 + " CNN " + "─"*40)
        from models import cnn_model
        model = cnn_model.train(X_train, y_train, X_val, y_val, symbol)
        proba = cnn_model.predict_proba(model, X_test)
        all_proba["cnn"] = proba
        results.append(evaluate_model("CNN", proba, y_test, feat_df_test, timeframe))

    # ── TFT ───────────────────────────────────────────────────────────────────
    if "tft" in args.models:
        logger.info("\n" + "─"*40 + " TFT " + "─"*40)
        from models import tft_model
        model = tft_model.train(X_train, y_train, X_val, y_val, symbol)
        proba = tft_model.predict_proba(model, X_test)
        all_proba["tft"] = proba
        results.append(evaluate_model("TFT", proba, y_test, feat_df_test, timeframe))

    # ── Ensemble of whatever ran ───────────────────────────────────────────────
    if len(all_proba) > 1:
        logger.info("\n" + "─"*40 + " Ensemble " + "─"*40)
        probas = list(all_proba.values())
        # Equal weights
        blended = np.mean(probas, axis=0)
        results.append(evaluate_model("Ensemble", blended, y_test, feat_df_test, timeframe))

    # ── Print report & save ───────────────────────────────────────────────────
    print_report(results, symbol, timeframe,
                 len(X_train), len(X_val), len(X_test), label_dist)
    save_results(results, symbol, timeframe)


if __name__ == "__main__":
    main()