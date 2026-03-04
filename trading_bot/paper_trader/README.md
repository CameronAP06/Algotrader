# DOGE/USD Paper Trader

LSTM-based paper trading bot for DOGE/USD 4h signals.
Runs on Railway (free tier). Sends Telegram alerts on every signal.

## Structure

```
paper_trader/
├── main.py              ← entry point, runs every 4h candle close
├── freeze_model.py      ← run locally to train + freeze the model
├── src/
│   ├── trader.py        ← signal generation + paper execution logic
│   ├── features.py      ← feature engineering (mirrors training pipeline)
│   └── notifier.py      ← Telegram alerts
├── models/              ← frozen model files (committed to repo)
│   ├── lstm_doge.pt
│   ├── lstm_info.pkl
│   └── scaler.pkl
├── data/                ← paper trade log (gitignored, persisted on Railway)
│   ├── trades.csv
│   └── state.json
├── requirements.txt
├── Dockerfile
└── railway.toml
```

---

## Step 1 — Train and freeze the model (local, one-time)

Run this from your main AlgoTrader project directory:

```bash
python paper_trader/freeze_model.py
```

This trains the LSTM on 80% of DOGE/USD history, then saves:
- `models/lstm_doge.pt`   — frozen weights
- `models/lstm_info.pkl`  — architecture config
- `models/scaler.pkl`     — fitted StandardScaler

Commit these three files to the repo.

---

## Step 2 — Set up Telegram bot (one-time, 2 minutes)

1. Open Telegram → search **@BotFather** → `/newbot`
2. Follow prompts → copy the **token** (looks like `123456:ABC-DEF...`)
3. Send any message to your new bot
4. Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
5. Copy the `"id"` value from the `"chat"` object — that's your **chat_id**

---

## Step 3 — Deploy to Railway

1. Push this folder as a GitHub repo (or subfolder)
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Select the repo → Railway auto-detects the Dockerfile
4. Go to **Variables** tab and add:
   ```
   TELEGRAM_TOKEN   = your_bot_token_here
   TELEGRAM_CHAT_ID = your_chat_id_here
   ```
5. Deploy — the bot starts immediately and sends a startup message

Railway free tier gives $5/month credit. This service uses ~$1-2/month.

---

## What it does every 4 hours

1. Fetches latest DOGE/USD 4h candles from Kraken (no API key needed)
2. Engineers features using the same pipeline as training
3. Loads frozen LSTM, runs inference on the latest bar
4. If BUY or SELL signal: opens paper position, sends Telegram alert
5. Checks open positions for stop loss (5%) / take profit (10%) / reversal
6. Logs everything to `data/trades.csv`

---

## Monitoring

**Telegram** — alerts on every trade open/close.

**trades.csv** columns:
```
timestamp, action, price, signal, confidence, up_prob, down_prob,
position_pnl_pct, portfolio_value, hold_bars, reason
```
Actions: `OPEN_LONG`, `OPEN_SHORT`, `CLOSE_LONG`, `CLOSE_SHORT`, `HOLD`

**state.json** — current position, portfolio value, trade count.

---

## Updating the model (variance fix)

When the improved (ensemble/fixed) model is ready:

1. Run `freeze_model.py` again locally with the new model code
2. Commit the new `models/` files
3. Push to GitHub → Railway auto-redeploys
4. Bot sends startup message confirming new model loaded

The trades.csv history is preserved across redeploys (Railway persistent volume).

---

## Paper trade config

| Parameter      | Value    |
|---------------|----------|
| Symbol        | DOGE/USD |
| Timeframe     | 4h       |
| Starting capital | $1,000 |
| Stop loss     | 5%       |
| Take profit   | 10%      |
| Fee model     | 0.1% per side |
| Signal filter | Top 15% confidence |
