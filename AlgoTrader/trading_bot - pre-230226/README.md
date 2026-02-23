# 🤖 ML Ensemble Trading Bot — Phase 1

Algorithmic trading bot using **LightGBM + XGBoost + LSTM ensemble** 
to predict crypto price direction on **Kraken** (UK-friendly, no fees for data).

## Architecture

```
trading_bot/
├── config/settings.py        ← All tunable parameters
├── data/
│   ├── kraken_fetcher.py     ← Pulls OHLCV from Kraken (no API key needed)
│   └── feature_engineer.py  ← 80+ technical features + labels
├── models/
│   ├── lgbm_model.py         ← LightGBM classifier
│   ├── xgb_model.py          ← XGBoost classifier
│   ├── lstm_model.py         ← PyTorch LSTM (sequence model)
│   └── ensemble.py           ← Blends all three + weight optimisation
├── utils/splitter.py         ← Chronological train/val/test split
├── backtest/
│   ├── engine.py             ← Event-driven backtester (fees + slippage)
│   └── plot_results.py       ← Equity curves + metrics charts
└── train_and_backtest.py     ← MAIN SCRIPT
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline (fetches data, trains models, backtests)
python train_and_backtest.py

# 3. Force refresh data
python train_and_backtest.py --refresh

# 4. Single pair only
python train_and_backtest.py --symbol BTC/USD
```

## What It Does

1. **Fetches** 2 years of hourly OHLCV from Kraken (free, no key)
2. **Engineers** 80+ features: moving averages, RSI, MACD, Bollinger Bands, 
   ATR, volume ratios, OBV, VWAP, candle patterns, time cyclicals, lagged returns
3. **Labels** each bar as UP / NEUTRAL / DOWN based on next-bar return
4. **Trains** LightGBM, XGBoost, and LSTM on chronological splits (no leakage)
5. **Optimises** ensemble blend weights on the validation set
6. **Backtests** with 0.26% fees, 0.1% slippage, 3% stop-loss, 6% take-profit
7. **Reports** Sharpe ratio, max drawdown, win rate, profit factor

## Key Config (config/settings.py)

| Parameter | Default | Description |
|---|---|---|
| `TIMEFRAME` | `1h` | Candle size |
| `SIGNAL_THRESHOLD` | `0.55` | Min confidence to trade |
| `STOP_LOSS_PCT` | `3%` | Per-trade stop loss |
| `TAKE_PROFIT_PCT` | `6%` | Per-trade take profit |
| `TRADING_FEE` | `0.26%` | Kraken maker fee |

## Next Steps (Phases 2–4)

- **Phase 2**: Connect to Kraken live API for paper trading
- **Phase 3**: Kelly Criterion position sizing + kill switches
- **Phase 4**: Docker + VPS deployment + Telegram alerts
