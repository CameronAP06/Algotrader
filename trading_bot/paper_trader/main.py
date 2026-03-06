"""
main.py — runs every 4h candle close
"""

import time, traceback
from datetime import datetime, timezone, timedelta
from loguru import logger

from src.trader   import run_paper_trade
from src.notifier import send, format_event, notify_startup, notify_error

INTERVAL_HOURS = 4

def next_candle_close() -> datetime:
    now  = datetime.now(timezone.utc)
    hour = (now.hour // INTERVAL_HOURS + 1) * INTERVAL_HOURS
    day  = now.date()
    if hour >= 24:
        hour -= 24
        day  += timedelta(days=1)
    return datetime(day.year, day.month, day.day, hour, 1, 0, tzinfo=timezone.utc)

def run_cycle():
    logger.info(f"─── Cycle @ {datetime.now(timezone.utc).isoformat()} ───")
    try:
        events = run_paper_trade()
        for event in events:
            msg = format_event(event)
            if msg:
                send(msg)
    except Exception as e:
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}"
        logger.error(err)
        notify_error(err)

def main():
    logger.info("Paper Trader starting — DOGE/USD + LINK/USD + AAVE/USD + XMR/USD 4h")
    notify_startup()
    run_cycle()
    while True:
        target     = next_candle_close()
        sleep_secs = max(0, (target - datetime.now(timezone.utc)).total_seconds())
        logger.info(f"Next cycle at {target.isoformat()} (sleeping {sleep_secs/3600:.1f}h)")
        time.sleep(sleep_secs)
        run_cycle()

if __name__ == "__main__":
    main()