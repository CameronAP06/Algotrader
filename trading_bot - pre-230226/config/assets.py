"""
config/assets.py
────────────────
Define the universe of assets the bot will trade.
Edit these lists to expand or narrow your coverage.
"""

# ── Stocks ────────────────────────────────────────────────────────────────────
STOCK_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",   # Mega-cap tech
    "META", "TSLA", "JPM", "V", "UNH",           # Diversification
    "SPY", "QQQ",                                 # ETFs for regime signals
]

# ── Crypto ────────────────────────────────────────────────────────────────────
# Symbols must match CCXT/Binance format: BASE/QUOTE
CRYPTO_UNIVERSE = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
]

# ── Forex ─────────────────────────────────────────────────────────────────────
# OANDA instrument format: BASE_QUOTE
FOREX_UNIVERSE = [
    "EUR_USD",
    "GBP_USD",
    "USD_JPY",
    "AUD_USD",
    "USD_CAD",
]

# ── Reference / regime assets (not traded, used as features) ──────────────────
REFERENCE_ASSETS = {
    "vix":  "^VIX",      # Equity fear gauge
    "dxy":  "DX-Y.NYB",  # US Dollar Index
    "gold": "GC=F",      # Gold futures
    "oil":  "CL=F",      # Crude oil futures
}

# ── Market type lookup ────────────────────────────────────────────────────────
def get_market_type(symbol: str) -> str:
    """Return 'stock', 'crypto', or 'forex' for a given symbol."""
    if "/" in symbol:
        return "crypto"
    if "_" in symbol:
        return "forex"
    return "stock"
