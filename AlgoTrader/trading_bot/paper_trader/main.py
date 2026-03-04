"""
main.py
───────
Entry point for the Railway deployment.

Runs immediately on startup, then sleeps and repeats every 4 hours,
aligned to the 4h candle close times (00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC).
"""

import time
import traceback
from datetime import datetime, timezone, timedelta
from loguru import logger

from src.trader   import run_paper_trade
from src.notifier import send, format_event, notify_startup, notify_error


INTERVAL_HOURS = 4


def next_candle_close() -> datetime:
    """Return the next 4h candle close time in UTC."""
    now   = datetime.now(timezone.utc)
    hour  = (now.hour // INTERVAL_HOURS + 1) * INTERVAL_HOURS
    day   = now.date()
    if hour >= 24:
        hour -= 24
        day  += timedelta(days=1)
    return datetime(day.year, day.month, day.day, hour, 1, 0, tzinfo=timezone.utc)
    # +1 minute after candle close to ensure the bar is finalised


def run_cycle():
    """Execute one paper trading cycle."""
    logger.info(f"─── Paper trade cycle @ {datetime.now(timezone.utc).isoformat()} ───")
    try:
        event = run_paper_trade()
        if event is not None:
            msg = format_event(event)
            if msg:
                send(msg)
    except Exception as e:
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}"
        logger.error(err)
        notify_error(err)


def main():
    logger.info("DOGE/USD Paper Trader starting up...")
    notify_startup()

    # Run immediately on startup
    run_cycle()

    # Then run at every 4h candle close
    while True:
        target = next_candle_close()
        now    = datetime.now(timezone.utc)
        sleep_secs = max(0, (target - now).total_seconds())
        logger.info(f"Next cycle at {target.isoformat()} "
                    f"(sleeping {sleep_secs/3600:.1f}h)")
        time.sleep(sleep_secs)
        run_cycle()


if __name__ == "__main__":
    main()
