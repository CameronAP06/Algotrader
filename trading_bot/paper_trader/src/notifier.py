"""
src/notifier.py
───────────────
Telegram notifications for paper trade events.
Uses the Bot API directly — no SDK needed, just HTTP.

Setup (one time):
  1. Message @BotFather on Telegram → /newbot → follow prompts → get TOKEN
  2. Message your new bot once (so it can find your chat_id)
  3. Visit https://api.telegram.org/bot<TOKEN>/getUpdates → copy chat_id
  4. Set env vars: TELEGRAM_TOKEN and TELEGRAM_CHAT_ID
"""

import os
import requests
from loguru import logger

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

ENABLED = bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)


def send(message: str):
    """Send a Telegram message. Fails silently if not configured."""
    if not ENABLED:
        logger.info(f"[Telegram disabled] {message}")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, json={
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       message,
            "parse_mode": "HTML",
        }, timeout=10)
        if not resp.ok:
            logger.warning(f"Telegram send failed: {resp.status_code} {resp.text}")
        else:
            logger.info("Telegram notification sent")
    except Exception as e:
        logger.warning(f"Telegram error: {e}")


def format_event(event: dict) -> str:
    """Format a trade event into a readable Telegram message."""
    if event is None:
        return None

    symbol = "DOGE/USD"
    etype  = event.get("type")

    if etype == "OPEN":
        pos  = event["position"]
        icon = "🟢" if pos == "LONG" else "🔴"
        return (
            f"{icon} <b>PAPER TRADE — {pos} OPENED</b>\n"
            f"Symbol:     {symbol}\n"
            f"Price:      ${event['price']:,.5f}\n"
            f"Signal:     {event['signal']}\n"
            f"Confidence: {event['confidence']:.3f}\n"
            f"  UP prob:  {event['up_prob']:.3f}\n"
            f"  DOWN prob:{event['down_prob']:.3f}\n"
            f"Portfolio:  ${event['portfolio']:,.2f}"
        )

    elif etype == "CLOSE":
        pos    = event["position"]
        pnl    = event["pnl_pct"]
        icon   = "✅" if pnl > 0 else "❌"
        return (
            f"{icon} <b>PAPER TRADE — {pos} CLOSED</b>\n"
            f"Symbol:     {symbol}\n"
            f"Entry:      ${event['entry']:,.5f}\n"
            f"Exit:       ${event['exit']:,.5f}\n"
            f"PnL:        {pnl:+.2f}%\n"
            f"Reason:     {event['reason']}\n"
            f"Portfolio:  ${event['portfolio']:,.2f}\n"
            f"Record:     {event['total_trades']} trades | "
            f"{event['win_rate']:.1f}% WR"
        )

    return None


def notify_startup():
    send(
        "🤖 <b>DOGE Paper Trader — STARTED</b>\n"
        "Monitoring DOGE/USD 4h signals.\n"
        "Will alert on every BUY/SELL signal and position close."
    )


def notify_error(error: str):
    send(f"⚠️ <b>Paper Trader Error</b>\n{error}")
