"""
src/notifier.py — Telegram alerts
"""

import os, requests
from loguru import logger

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ENABLED = bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)


def send(message: str):
    if not ENABLED:
        logger.info(f"[Telegram disabled] {message}")
        return
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=10)
        if not resp.ok:
            logger.warning(f"Telegram failed: {resp.status_code}")
    except Exception as e:
        logger.warning(f"Telegram error: {e}")


def format_event(event: dict) -> str | None:
    if not event:
        return None

    etype  = event.get("type")
    symbol = event.get("symbol", "?")

    if etype == "ERROR":
        return f"⚠️ <b>{symbol} Error</b>\n{event.get('error','unknown')}"

    if etype == "OPEN":
        pos  = event["position"]
        icon = "🟢" if pos == "LONG" else "🔴"
        return (
            f"{icon} <b>{symbol} — {pos} OPENED</b>\n"
            f"Price:      ${event['price']:,.5f}\n"
            f"Signal:     {event['signal']}\n"
            f"Confidence: {event['confidence']:.3f} "
            f"(up={event['up_prob']:.3f} down={event['down_prob']:.3f})\n"
            f"Portfolio:  ${event['portfolio']:,.2f}"
        )

    if etype == "CLOSE":
        pnl  = event["pnl_pct"]
        icon = "✅" if pnl > 0 else "❌"
        return (
            f"{icon} <b>{symbol} — {event['position']} CLOSED</b>\n"
            f"Entry:     ${event['entry']:,.5f}\n"
            f"Exit:      ${event['exit']:,.5f}\n"
            f"PnL:       {pnl:+.2f}%\n"
            f"Reason:    {event['reason']}\n"
            f"Portfolio: ${event['portfolio']:,.2f}\n"
            f"Record:    {event['total_trades']} trades | {event['win_rate']:.1f}% WR"
        )

    return None


def notify_startup():
    send(
        "🤖 <b>Paper Trader — STARTED</b>\n"
        "Symbols: DOGE/USD + LINK/USD\n"
        "Timeframe: 4h | Capital: $1,000 each\n"
        "Alerts on every signal and position close."
    )


def notify_error(error: str):
    send(f"⚠️ <b>Paper Trader Error</b>\n{error[:500]}")
