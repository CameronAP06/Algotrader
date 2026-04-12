"""
dashboard.py — Interactive Trading Bot Dashboard
─────────────────────────────────────────────────
Run with:
    streamlit run dashboard.py

Tabs:
  1. Scan Results  — load any scan CSV, filter/sort/chart results
  2. Live Monitor  — auto-refresh the newest scan CSV as it grows
  3. Paper Trader  — portfolio equity, open positions, trade log
  4. Model Cache   — inspect the fold cache to see what's cached vs stale

Dependencies (pip install if missing):
    streamlit plotly pandas
"""

import os, sys, json, time
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

RESULTS_DIR     = _ROOT / "backtest" / "results"
PAPER_DATA_DIR  = _ROOT / "paper_trader" / "data"
MODEL_CACHE_DIR = _ROOT / "models" / "saved" / "fold_cache"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e; border-radius: 8px; padding: 12px 16px;
        border-left: 4px solid #7c3aed;
    }
    .strong  { color: #22c55e; font-weight: bold; }
    .good    { color: #86efac; }
    .marginal{ color: #fbbf24; }
    .loss    { color: #ef4444; }
    .hold    { color: #94a3b8; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

STATUS_COLOR = {
    "** STRONG":     "#22c55e",
    "* GOOD":        "#86efac",
    "+ MARGINAL":    "#fbbf24",
    "~ WEAK+":       "#f97316",
    "~ INCONSISTENT":"#f97316",
    "- LOSS":        "#ef4444",
    "TOO_FEW_TRADES":"#94a3b8",
}

def _color_status(val: str) -> str:
    c = STATUS_COLOR.get(val, "#94a3b8")
    return f"color: {c}; font-weight: bold"

def _pct(v, decimals=1):
    try:
        return f"{float(v)*100:.{decimals}f}%"
    except Exception:
        return str(v)

def _f(v, decimals=2):
    try:
        return f"{float(v):.{decimals}f}"
    except Exception:
        return str(v)

def load_scan_csv(path: Path) -> pd.DataFrame:
    """Load a scan results CSV and coerce numeric columns."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    num_cols = [
        "n_bars", "n_trades", "win_rate", "net_return", "ann_return",
        "ann_std", "sharpe", "sortino", "calmar", "max_drawdown",
        "profit_factor", "payoff_ratio", "quality_score", "kelly_pct",
        "folds_positive", "n_folds", "elapsed_s",
        "trend_wr", "range_wr", "n_trend_trades", "n_range_trades",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def find_scan_files() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("expanded_scan_*.csv"), reverse=True)

def find_newest_scan() -> Path | None:
    files = find_scan_files()
    return files[0] if files else None

def load_trades() -> pd.DataFrame:
    path = PAPER_DATA_DIR / "trades.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in ["price", "confidence", "up_prob", "down_prob",
              "position_pnl_pct", "portfolio_value"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_all_states() -> dict:
    states = {}
    if not PAPER_DATA_DIR.exists():
        return states
    for f in PAPER_DATA_DIR.glob("state_*.json"):
        try:
            with open(f) as fh:
                data = json.load(fh)
            sym = f.stem.replace("state_", "").upper()
            # Convert back to slash format
            if "_usd" in sym.lower():
                sym = sym.replace("_USD", "/USD").replace("_usd", "/USD")
            states[sym] = data
        except Exception:
            pass
    return states


# ── Tab 1: Scan Results ───────────────────────────────────────────────────────

def tab_scan_results():
    st.header("Scan Results Explorer")

    files = find_scan_files()
    if not files:
        st.warning(f"No scan CSV files found in {RESULTS_DIR}")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        chosen = st.selectbox(
            "Scan file",
            options=files,
            format_func=lambda p: f"{p.name}  ({p.stat().st_size / 1024:.0f} KB)",
        )
    with col2:
        if st.button("🔄 Refresh file list"):
            st.rerun()

    df = load_scan_csv(chosen)
    if df.empty:
        st.error("File loaded but contains no rows.")
        return

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.subheader("Summary")
    valid = df[df["n_trades"].fillna(0) >= 5]
    profitable = valid[valid["ann_return"].fillna(0) > 0]
    strong = valid[valid["status"].str.contains("STRONG", na=False)]
    good   = valid[valid["status"].str.contains("GOOD",   na=False)]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total combos",   len(df))
    c2.metric("Profitable",     len(profitable))
    c3.metric("★★ Strong",     len(strong))
    c4.metric("★ Good",        len(good))
    c5.metric("Avg Sharpe",    f"{valid['sharpe'].mean():.2f}" if len(valid) else "—")

    # ── Filters ───────────────────────────────────────────────────────────────
    st.subheader("Filters")
    fc1, fc2, fc3, fc4, fc5 = st.columns(5)
    with fc1:
        tfs = ["All"] + sorted(df["timeframe"].dropna().unique().tolist())
        tf_filter = st.selectbox("Timeframe", tfs)
    with fc2:
        statuses = ["All"] + sorted(df["status"].dropna().unique().tolist())
        status_filter = st.selectbox("Status", statuses)
    with fc3:
        min_sharpe = st.slider("Min Sharpe", -2.0, 5.0, 0.0, 0.1)
    with fc4:
        min_trades = st.slider("Min trades/fold", 0, 30, 3)
    with fc5:
        min_score = st.slider("Min quality score", 0, 100, 0)

    fdf = df.copy()
    if tf_filter != "All":
        fdf = fdf[fdf["timeframe"] == tf_filter]
    if status_filter != "All":
        fdf = fdf[fdf["status"] == status_filter]
    fdf = fdf[fdf["sharpe"].fillna(-99) >= min_sharpe]
    fdf = fdf[fdf["n_trades"].fillna(0) >= min_trades]
    fdf = fdf[fdf["quality_score"].fillna(0) >= min_score]

    st.caption(f"Showing {len(fdf)} / {len(df)} combos")

    # ── Results table ─────────────────────────────────────────────────────────
    display_cols = [
        "symbol", "timeframe", "status", "quality_score",
        "ann_return", "sharpe", "sortino", "max_drawdown",
        "win_rate", "n_trades", "folds_positive", "n_folds",
        "profit_factor", "kelly_pct", "regime_edge",
    ]
    show_cols = [c for c in display_cols if c in fdf.columns]
    tbl = fdf[show_cols].sort_values("quality_score", ascending=False).reset_index(drop=True)

    # Format for display
    fmt = tbl.copy()
    for c in ["ann_return", "max_drawdown", "win_rate"]:
        if c in fmt.columns:
            fmt[c] = fmt[c].apply(lambda v: _pct(v) if pd.notna(v) else "—")
    for c in ["sharpe", "sortino", "profit_factor"]:
        if c in fmt.columns:
            fmt[c] = fmt[c].apply(lambda v: _f(v) if pd.notna(v) else "—")

    st.dataframe(
        fmt.style.applymap(_color_status, subset=["status"]),
        use_container_width=True,
        height=420,
    )

    # ── Charts ────────────────────────────────────────────────────────────────
    st.subheader("Charts")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        scatter_df = fdf.dropna(subset=["ann_return", "sharpe", "quality_score", "status"])
        if not scatter_df.empty:
            fig = px.scatter(
                scatter_df,
                x="sharpe", y="ann_return",
                size="quality_score",
                color="status",
                hover_name="symbol",
                hover_data=["timeframe", "n_trades", "max_drawdown", "win_rate"],
                color_discrete_map={k: v for k, v in STATUS_COLOR.items()},
                title="Sharpe vs Annualised Return",
                labels={"ann_return": "Ann Return", "sharpe": "Sharpe Ratio"},
                size_max=30,
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.update_layout(height=400, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for scatter plot.")

    with chart_col2:
        # Quality score bar chart — top 20
        top = fdf.nlargest(20, "quality_score")[["symbol", "timeframe", "quality_score", "status"]]
        if not top.empty:
            top["label"] = top["symbol"] + " " + top["timeframe"]
            top["color"] = top["status"].map(STATUS_COLOR).fillna("#94a3b8")
            fig2 = px.bar(
                top, x="quality_score", y="label",
                orientation="h",
                color="status",
                color_discrete_map=STATUS_COLOR,
                title="Top 20 by Quality Score",
                labels={"quality_score": "Score", "label": ""},
            )
            fig2.update_layout(height=400, template="plotly_dark",
                               yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig2, use_container_width=True)

    # ── Heatmap ───────────────────────────────────────────────────────────────
    st.subheader("Quality Score Heatmap — Symbol × Timeframe")
    heat_df = fdf.pivot_table(
        index="symbol", columns="timeframe",
        values="quality_score", aggfunc="mean"
    )
    if not heat_df.empty and len(heat_df) <= 80:
        fig3 = px.imshow(
            heat_df, color_continuous_scale="RdYlGn",
            text_auto=".0f",
            title="Quality Score by Symbol × Timeframe",
            zmin=0, zmax=100,
            aspect="auto",
        )
        fig3.update_layout(height=max(300, len(heat_df) * 18),
                           template="plotly_dark")
        st.plotly_chart(fig3, use_container_width=True)
    elif len(heat_df) > 80:
        st.info(f"Heatmap skipped ({len(heat_df)} symbols — apply filters to narrow down)")

    # ── Symbol drill-down ─────────────────────────────────────────────────────
    st.subheader("Symbol Detail")
    symbols_avail = sorted(fdf["symbol"].dropna().unique().tolist())
    if symbols_avail:
        sel_sym = st.selectbox("Select symbol", symbols_avail)
        sym_rows = fdf[fdf["symbol"] == sel_sym]
        for _, row in sym_rows.iterrows():
            with st.expander(f"{row['symbol']} — {row['timeframe']}  |  "
                             f"Score={row.get('quality_score', '?'):.1f}  "
                             f"Status={row.get('status', '?')}"):
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Ann Return",    _pct(row.get("ann_return", 0)))
                d2.metric("Sharpe",        _f(row.get("sharpe", 0)))
                d3.metric("Win Rate",      _pct(row.get("win_rate", 0)))
                d4.metric("Max Drawdown",  _pct(row.get("max_drawdown", 0)))
                d5, d6, d7, d8 = st.columns(4)
                d5.metric("Profit Factor", _f(row.get("profit_factor", 0)))
                d6.metric("Kelly %",       f"{row.get('kelly_pct', 0):.1f}%")
                d7.metric("Trades/fold",   str(row.get("n_trades", "?")))
                d8.metric("Regime Edge",   str(row.get("regime_edge", "?")))
                if "fold_dates" in row and pd.notna(row["fold_dates"]):
                    st.caption(f"Fold dates: {row['fold_dates']}")


# ── Tab 2: Live Monitor ───────────────────────────────────────────────────────

def tab_live_monitor():
    st.header("Live Scan Monitor")

    newest = find_newest_scan()
    if not newest:
        st.warning(f"No scan file found in {RESULTS_DIR}")
        return

    # Auto-refresh toggle
    col1, col2 = st.columns([2, 1])
    with col1:
        st.caption(f"Watching: **{newest.name}**  "
                   f"({newest.stat().st_size / 1024:.1f} KB)")
    with col2:
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=True)

    df = load_scan_csv(newest)
    if df.empty:
        st.info("Scan file exists but has no results yet. Waiting...")
        if auto_refresh:
            time.sleep(15)
            st.rerun()
        return

    # ── Progress ──────────────────────────────────────────────────────────────
    done = len(df)
    st.metric("Combos completed", done)

    # ── Live results ──────────────────────────────────────────────────────────
    st.subheader("Latest Results (newest first)")

    display = df.copy().iloc[::-1].reset_index(drop=True)
    show = [c for c in ["symbol", "timeframe", "status", "quality_score",
                         "ann_return", "sharpe", "max_drawdown", "win_rate",
                         "n_trades", "folds_positive", "n_folds"]
            if c in display.columns]
    st.dataframe(
        display[show].head(50).style.applymap(_color_status, subset=["status"]),
        use_container_width=True,
        height=360,
    )

    # ── Running totals ────────────────────────────────────────────────────────
    valid = df[df["n_trades"].fillna(0) >= 5]
    profitable = valid[valid["ann_return"].fillna(0) > 0]
    strong = valid[valid["status"].str.contains("STRONG", na=False)]

    st.subheader("Running Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Profitable", f"{len(profitable)} / {len(valid)}")
    m2.metric("Strong",     len(strong))
    m3.metric("Avg Sharpe (valid)", f"{valid['sharpe'].mean():.2f}" if len(valid) else "—")
    m4.metric("Best score", f"{df['quality_score'].max():.1f}" if "quality_score" in df else "—")

    # Live scatter
    scatter_df = valid.dropna(subset=["ann_return", "sharpe"])
    if not scatter_df.empty:
        fig = px.scatter(
            scatter_df, x="sharpe", y="ann_return",
            color="status", hover_name="symbol",
            hover_data=["timeframe", "n_trades"],
            color_discrete_map=STATUS_COLOR,
            title="All Results So Far",
            template="plotly_dark", height=350,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.4)
        st.plotly_chart(fig, use_container_width=True)

    if auto_refresh:
        time.sleep(30)
        st.rerun()


# ── Tab 3: Paper Trader ───────────────────────────────────────────────────────

def tab_paper_trader():
    st.header("Paper Trader")

    # ── Open positions ────────────────────────────────────────────────────────
    states = load_all_states()
    trades = load_trades()

    st.subheader("Open Positions")
    if not states:
        st.info("No state files found — paper trader may not have run yet.")
    else:
        pos_rows = []
        for sym, s in states.items():
            pos_rows.append({
                "Symbol":     sym,
                "Position":   s.get("position") or "—",
                "Entry":      f"${s['entry_price']:,.4f}" if s.get("entry_price") else "—",
                "Hold bars":  s.get("hold_bars", 0),
                "Portfolio":  f"${s.get('portfolio_value', 0):,.2f}",
                "Trades":     s.get("total_trades", 0),
                "Win rate":   f"{s['winning_trades']/s['total_trades']*100:.1f}%"
                              if s.get("total_trades") else "—",
            })
        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True)

    # ── Portfolio equity curve ─────────────────────────────────────────────────
    if not trades.empty and "portfolio_value" in trades.columns:
        st.subheader("Portfolio Equity Curve")

        # Aggregate: one portfolio_value row per timestamp (sum across symbols)
        if "symbol" in trades.columns:
            # Use per-symbol portfolio values — show each as a line
            syms = trades["symbol"].dropna().unique().tolist()
            fig = go.Figure()
            for sym in syms:
                sym_t = trades[trades["symbol"] == sym].dropna(subset=["timestamp"])
                if sym_t.empty:
                    continue
                sym_t = sym_t.sort_values("timestamp")
                fig.add_trace(go.Scatter(
                    x=sym_t["timestamp"], y=sym_t["portfolio_value"],
                    mode="lines", name=sym,
                ))
            fig.update_layout(
                title="Portfolio Value per Symbol",
                xaxis_title="Time", yaxis_title="Value ($)",
                template="plotly_dark", height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Trade history ─────────────────────────────────────────────────────────
    st.subheader("Trade History")
    if trades.empty:
        st.info("No trades logged yet.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            sym_filter = ["All"] + sorted(trades["symbol"].dropna().unique().tolist())
            sel = st.selectbox("Filter by symbol", sym_filter)
        with col2:
            action_filter = ["All"] + sorted(trades["action"].dropna().unique().tolist())
            act = st.selectbox("Filter by action", action_filter)

        tdf = trades.copy()
        if sel != "All":
            tdf = tdf[tdf["symbol"] == sel]
        if act != "All":
            tdf = tdf[tdf["action"] == act]

        tdf = tdf.sort_values("timestamp", ascending=False).reset_index(drop=True)
        st.dataframe(tdf.head(200), use_container_width=True, height=360)

        # PnL distribution
        closed = tdf[tdf["action"].str.startswith("CLOSE", na=False)].copy()
        if not closed.empty:
            st.subheader("Closed Trade PnL Distribution")
            fig2 = px.histogram(
                closed, x="position_pnl_pct", nbins=30,
                color="symbol" if "symbol" in closed.columns else None,
                title="PnL % per closed trade",
                template="plotly_dark", height=280,
            )
            fig2.add_vline(x=0, line_dash="dash", line_color="white")
            st.plotly_chart(fig2, use_container_width=True)


# ── Tab 4: Model Cache ────────────────────────────────────────────────────────

def tab_model_cache():
    st.header("Model Cache Inspector")

    if not MODEL_CACHE_DIR.exists():
        st.info(f"Cache directory does not exist yet: {MODEL_CACHE_DIR}\n\n"
                "It will be created on the first training run.")
        return

    cache_files = sorted(MODEL_CACHE_DIR.glob("*.pt"))
    if not cache_files:
        st.info("Cache directory exists but is empty — no models cached yet.")
        return

    total_mb = sum(f.stat().st_size for f in cache_files) / 1024 / 1024
    st.metric("Cached models", len(cache_files))
    st.metric("Total cache size", f"{total_mb:.1f} MB")

    rows = []
    for f in cache_files:
        stat = f.stat()
        rows.append({
            "Key (first 8 chars)": f.stem[:8] + "…",
            "Full key":            f.stem,
            "Size (KB)":           f"{stat.st_size / 1024:.1f}",
            "Cached at":           datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.caption(
        "Each entry is one trained LSTM (1 of 9 per fold per symbol/timeframe combo). "
        "Keys are MD5 hashes of training data + hyperparams + seed. "
        "If data changes or hyperparams change, the old file is ignored and a new one is written."
    )

    if st.button("🗑️ Clear entire cache", type="secondary"):
        for f in cache_files:
            f.unlink()
        st.success(f"Deleted {len(cache_files)} cached models.")
        st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.sidebar.title("📈 Trading Bot")
    st.sidebar.markdown("---")

    # Sidebar scan summary
    newest = find_newest_scan()
    if newest:
        try:
            df_s = load_scan_csv(newest)
            n_strong = df_s["status"].str.contains("STRONG", na=False).sum()
            n_total  = len(df_s)
            st.sidebar.metric("Latest scan combos", n_total)
            st.sidebar.metric("Strong opportunities", int(n_strong))
            st.sidebar.caption(f"File: {newest.name}")
        except Exception:
            pass

    # Paper trader sidebar
    states = load_all_states()
    open_positions = sum(1 for s in states.values() if s.get("position"))
    if open_positions:
        st.sidebar.markdown(f"**Open positions:** {open_positions}")

    st.sidebar.markdown("---")
    st.sidebar.caption("Refresh any tab manually with the button, "
                       "or enable auto-refresh in the Live Monitor tab.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Scan Results",
        "🔴 Live Monitor",
        "💼 Paper Trader",
        "🗄️ Model Cache",
    ])

    with tab1:
        tab_scan_results()
    with tab2:
        tab_live_monitor()
    with tab3:
        tab_paper_trader()
    with tab4:
        tab_model_cache()


if __name__ == "__main__":
    main()
