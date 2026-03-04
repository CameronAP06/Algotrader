"""
backtest/engine.py
Realistic event-driven backtester with fees, slippage, stop-loss, and take-profit.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from config.settings import (
    INITIAL_CAPITAL, TRADING_FEE, SLIPPAGE,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT, MAX_POSITION_PCT,
    RESULTS_DIR,
    USE_ATR_STOPS, ATR_STOP_MULT, ATR_TP_MULT,
)


class BacktestEngine:
    """
    Simple long/short backtester.
    Supports stop-loss, take-profit, fees, and slippage.
    """

    def __init__(self, initial_capital=INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.reset()

    def reset(self):
        self.cash        = self.initial_capital
        self.position    = 0.0   # Units held (positive = long, negative = short)
        self.entry_price = None
        self.stop_price  = None  # Dynamic stop level
        self.tp_price    = None  # Dynamic take-profit level
        self.side        = None  # "LONG" or "SHORT"
        self.margin_held = 0.0
        self.trades      = []
        self.equity_curve = []

    def _apply_slippage(self, price, side):
        if side == "BUY":
            return price * (1 + SLIPPAGE)
        return price * (1 - SLIPPAGE)

    def _apply_fee(self, trade_value):
        return trade_value * TRADING_FEE

    def run(self, df: pd.DataFrame, signals: dict, symbol: str = "") -> dict:
        """
        Run backtest over a DataFrame with signals dict from ensemble.generate_signals().
        df must contain 'close', 'high', 'low' columns aligned to signals.
        Supports both long and short positions.
        """
        self.reset()
        signal_arr  = signals["signal"]
        confidence  = signals["confidence"]

        n = min(len(df), len(signal_arr))
        prices = df["close"].values[-n:]
        highs  = df["high"].values[-n:]
        lows   = df["low"].values[-n:]
        # ATR for dynamic stop/TP sizing
        atrs   = df["atr_14"].values[-n:] if "atr_14" in df.columns else np.full(n, np.nan)

        for i in range(n):
            price = prices[i]
            if self.position >= 0:
                equity = self.cash + self.position * price
            else:
                # Short: cash includes sale proceeds; subtract current buyback cost
                buyback_cost = abs(self.position) * price
                equity = self.cash - buyback_cost
            self.equity_curve.append(equity)

            # ── Check stop-loss / take-profit for LONG ────────────────────
            if self.position > 0 and self.entry_price:
                if lows[i] <= self.stop_price:
                    self._close_position(self.stop_price, i, "STOP_LOSS")
                    continue
                if highs[i] >= self.tp_price:
                    self._close_position(self.tp_price, i, "TAKE_PROFIT")
                    continue

            # ── Check stop-loss / take-profit for SHORT ───────────────────
            elif self.position < 0 and self.entry_price:
                if highs[i] >= self.stop_price:
                    self._close_position(self.stop_price, i, "STOP_LOSS")
                    continue
                if lows[i] <= self.tp_price:
                    self._close_position(self.tp_price, i, "TAKE_PROFIT")
                    continue

            sig = signal_arr[i]

            # ── Entry / flip logic ────────────────────────────────────────
            if sig == "BUY" and self.position <= 0:
                if self.position < 0:
                    self._close_position(price, i, "SIGNAL_FLIP")
                self._open_long(price, i, atrs[i])

            elif sig == "SELL" and self.position >= 0:
                if self.position > 0:
                    self._close_position(price, i, "SIGNAL_FLIP")
                self._open_short(price, i, atrs[i])

            elif sig == "HOLD":
                pass  # Hold existing position

        # Close any open position at end
        if self.position != 0:
            self._close_position(prices[-1], n-1, "END_OF_PERIOD")

        return self._compute_metrics(symbol)

    def _open_long(self, price, idx, atr=None):
        exec_price       = self._apply_slippage(price, "BUY")
        trade_value      = self.cash * MAX_POSITION_PCT
        fee              = self._apply_fee(trade_value)
        self.position    = (trade_value - fee) / exec_price
        self.cash        -= trade_value
        self.entry_price = exec_price
        self.side        = "LONG"
        if USE_ATR_STOPS and atr is not None and atr > 0:
            self.stop_price = exec_price - ATR_STOP_MULT * atr
            self.tp_price   = exec_price + ATR_TP_MULT   * atr
        else:
            self.stop_price = exec_price * (1 - STOP_LOSS_PCT)
            self.tp_price   = exec_price * (1 + TAKE_PROFIT_PCT)

    def _open_short(self, price, idx, atr=None):
        exec_price        = self._apply_slippage(price, "SELL")
        margin            = self.cash * MAX_POSITION_PCT
        units             = margin / exec_price
        fee               = self._apply_fee(units * exec_price)
        # Sell units short: receive proceeds into cash, deduct fee
        self.position     = -units
        self.cash        += (units * exec_price) - fee  # proceeds received
        self.margin_held  = margin                       # track capital at risk
        self.entry_price  = exec_price
        self.side         = "SHORT"
        if USE_ATR_STOPS and atr is not None and atr > 0:
            self.stop_price = exec_price + ATR_STOP_MULT * atr
            self.tp_price   = exec_price - ATR_TP_MULT   * atr
        else:
            self.stop_price = exec_price * (1 + STOP_LOSS_PCT)
            self.tp_price   = exec_price * (1 - TAKE_PROFIT_PCT)

    def _close_position(self, price, idx, reason):
        if self.position > 0:
            # Closing a long
            exec_price = self._apply_slippage(price, "SELL")
            proceeds   = self.position * exec_price
            fee        = self._apply_fee(proceeds)
            pnl        = proceeds - fee - (self.position * self.entry_price)
            self.cash += proceeds - fee
        else:
            # Closing a short — buy back units to cover
            exec_price   = self._apply_slippage(price, "BUY")
            units        = abs(self.position)
            cost_to_buy  = units * exec_price
            fee          = self._apply_fee(cost_to_buy)
            entry_value  = units * self.entry_price
            pnl          = entry_value - cost_to_buy - fee
            # Deduct buyback cost + fee from cash (proceeds were already added on open)
            self.cash   -= cost_to_buy + fee

        self.trades.append({
            "idx":         idx,
            "entry_price": self.entry_price,
            "exit_price":  exec_price,
            "pnl":         pnl,
            "reason":      reason,
            "side":        getattr(self, "side", "LONG"),
        })
        self.position    = 0.0
        self.entry_price = None
        self.stop_price  = None
        self.tp_price    = None
        self.side        = None

    def _compute_metrics(self, symbol) -> dict:
        equity = np.array(self.equity_curve)
        returns = pd.Series(equity).pct_change().dropna()

        total_return = (equity[-1] - equity[0]) / equity[0]
        n_trades     = len(self.trades)
        winning      = [t for t in self.trades if t["pnl"] > 0]
        win_rate     = len(winning) / n_trades if n_trades else 0
        n_longs      = len([t for t in self.trades if t.get("side") == "LONG"])
        n_shorts     = len([t for t in self.trades if t.get("side") == "SHORT"])

        # Sharpe ratio (annualised, hourly bars)
        sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(8760)

        # Max drawdown
        roll_max  = pd.Series(equity).cummax()
        drawdown  = (pd.Series(equity) - roll_max) / roll_max
        max_dd    = drawdown.min()

        # Profit factor
        gross_profit = sum(t["pnl"] for t in self.trades if t["pnl"] > 0)
        gross_loss   = abs(sum(t["pnl"] for t in self.trades if t["pnl"] < 0))
        profit_factor = gross_profit / (gross_loss + 1e-9)

        metrics = {
            "symbol":        symbol,
            "final_equity":  equity[-1],
            "total_return":  total_return,
            "n_trades":      n_trades,
            "n_longs":       n_longs,
            "n_shorts":      n_shorts,
            "win_rate":      win_rate,
            "sharpe_ratio":  sharpe,
            "max_drawdown":  max_dd,
            "profit_factor": profit_factor,
        }

        logger.info(
            f"{symbol} Results: Return={total_return:.2%} | "
            f"Sharpe={sharpe:.2f} | MaxDD={max_dd:.2%} | "
            f"WinRate={win_rate:.2%} | Trades={n_trades} "
            f"(L:{n_longs} S:{n_shorts})"
        )
        return metrics

    def save_results(self, metrics, symbol):
        import json
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = Path(RESULTS_DIR) / f"{symbol.replace('/','_')}_backtest.json"
        with open(path, "w") as f:
            json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v
                       for k, v in metrics.items()}, f, indent=2)
        # Save equity curve
        eq_path = Path(RESULTS_DIR) / f"{symbol.replace('/','_')}_equity.csv"
        pd.DataFrame({"equity": self.equity_curve}).to_csv(eq_path, index=False)
        logger.info(f"Results saved -> {path}")
