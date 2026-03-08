"""
run_discovery.py
────────────────
Run from trading_bot directory:
    python run_discovery.py
"""
import os, sys, glob, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Find the OHLC_Kraken directory ────────────────────────────────────────────
candidates = [
    os.environ.get("KRAKEN_HISTORY_DIR", ""),
    "data/OHLC_Kraken",
    os.path.join(os.path.dirname(__file__), "data", "OHLC_Kraken"),
]
history_dir = None
for path in candidates:
    if path and os.path.isdir(path):
        csvs = glob.glob(os.path.join(path, "*.csv"))
        if csvs:
            history_dir = path
            print(f"Found OHLC_Kraken at: {path} ({len(csvs)} CSV files)")
            break

if not history_dir:
    print("ERROR: Could not find OHLC_Kraken directory.")
    print("Set KRAKEN_HISTORY_DIR environment variable or place data in trading_bot/data/OHLC_Kraken")
    sys.exit(1)

# ── Timeframe → minutes mapping ───────────────────────────────────────────────
TF_MINUTES = {"30m": 30, "1h": 60, "4h": 240, "12h": 720, "1d": 1440}

# Minimum bars per timeframe to be worth scanning
MIN_BARS = {"30m": 5000, "1h": 4000, "4h": 2000, "12h": 700, "1d": 365}

# Known legacy Kraken name → standard symbol
LEGACY = {
    "XBTUSD": "BTC/USD", "ETHUSD": "ETH/USD", "XRPUSD": "XRP/USD",
    "LTCUSD": "LTC/USD", "XMRUSD": "XMR/USD", "ZECUSD": "ZEC/USD",
    "ETCUSD": "ETC/USD", "XDGUSD": "DOGE/USD", "DASHUSD": "DASH/USD",
    "EURUSD": "EUR/USD", "GBPUSD": "GBP/USD", "AUDUSD": "AUD/USD",
}

# Stablecoins / wrapped tokens to skip
EXCLUDE = {
    "USDT", "USDC", "DAI", "BUSD", "TUSD", "USDP", "GUSD",
    "FRAX", "LUSD", "WBTC", "WETH", "STETH", "CBETH", "RETH",
    "PYUSD", "USDE", "FDUSD",
}

# ── Discover ──────────────────────────────────────────────────────────────────
results = {}

for tf, minutes in TF_MINUTES.items():
    min_bars = MIN_BARS[tf]
    pattern  = os.path.join(history_dir, f"*USD_{minutes}.csv")
    files    = glob.glob(pattern)
    pairs    = []

    for fpath in sorted(files):
        fname      = os.path.basename(fpath)
        kraken_name = fname.replace(f"_{minutes}.csv", "")

        # Convert to standard symbol
        if kraken_name in LEGACY:
            symbol = LEGACY[kraken_name]
        elif kraken_name.endswith("USD"):
            base = kraken_name[:-3]
            # Handle XBT prefix just in case
            if base == "XBT": base = "BTC"
            if base == "XDG": base = "DOGE"
            symbol = f"{base}/USD"
        else:
            continue

        base = symbol.split("/")[0]
        if base in EXCLUDE:
            continue

        # Count lines (fast bar count without full parse)
        try:
            with open(fpath, "rb") as f:
                n_bars = sum(1 for _ in f)
        except Exception:
            continue

        if n_bars >= min_bars:
            pairs.append((symbol, n_bars))

    results[tf] = sorted(pairs)
    print(f"  {tf:>4}: {len(pairs)} pairs (min {min_bars} bars)")

# ── Save results ──────────────────────────────────────────────────────────────
out_path = "discovery_results.txt"
with open(out_path, "w") as f:
    for tf, pairs in results.items():
        f.write(f"\n=== {tf} ({len(pairs)} pairs, min {MIN_BARS[tf]} bars) ===\n")
        for symbol, n_bars in pairs:
            f.write(f"  {symbol:<16} {n_bars:>8} bars\n")

print(f"\nFull list saved to {out_path}")