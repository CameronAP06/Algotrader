"""
backtest/engine.py
Realistic event-driven backtester.

Upgrades vs previous version:
  - Kelly Criterion position sizing  — sizes each trade by estimated edge strength
  - Fractional Kelly (0.25x)         — standard safety margin against estimation error
  - ATR-based dynamic stop-loss      — stops scale with realised volatility, not fixed %
  - ATR-based dynamic take-profit    — 3x ATR TP (positive expectancy by construction)
  - Regime-aware Kelly fraction      — halved in ranging/choppy markets (ADX < threshold)
  - Richer metrics                   — Sortino, avg trade duration, consecutive losses,
    recovery factor, confidence vs outcome, regime breakdown of wins/losses
  - Per-trade confidence tracking    — surfaces whether high-conf signals are more profitable
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
    ATR_STOP_MULT, ATR_TP_MULT,       # ATR stop/TP multipliers — editable in settings
    KELLY_MAX_PCT, KELLY_REGIME_ADX,   # Kelly caps — editable in settings
)

# ── Kelly constants ────────────────────────────────────────────────────────────
KELLY_FRACTION   = 0.25    # quarter-Kelly — standard safety margin
KELLY_MIN_PCT    = 0.05    # never risk less than 5% (avoids dust trades)
# KELLY_MAX_PCT and KELLY_REGIME_ADX imported from settings above

# ATR-based stop/TP multipliers — imported from settings above
ATR_WINDOW    = 14


class BacktestEngine:
    """
    Event-driven backtester with Kelly position sizing and ATR-dynamic stops.
    """

    _BARS_PER_YEAR = {
        "15m":  35040, "30m": 17520, "1h": 8760,
        "4h":   2190,  "12h": 730,   "1d": 365,
    }

    def __init__(self, initial_capital=INITIAL_CAPITAL,
                 use_kelly: bool = True,
                 use_atr_stops: bool = True):
        self.initial_capital = initial_capital
        self.use_kelly       = use_kelly
        self.use_atr_stops   = use_atr_stops
        self.reset()

    def reset(self):
        self.cash          = self.initial_capital
        self.position      = 0.0
        self.entry_price   = None
        self.stop_price    = None
        self.tp_price      = None
        self.side          = None
        self.margin_held   = 0.0
        self.trades        = []
        self.equity_curve  = []
        self._timeframe    = "4h"

    # ── Kelly sizing ───────────────────────────────────────────────────────────

    def _kelly_fraction(self, confidence: float, win_rate_est: float,
                        payoff_ratio: float, adx: float = 100.0) -> float:
        p = float(np.clip(confidence, 0.34, 0.95))
        b = max(payoff_ratio, 0.5)
        q = 1.0 - p
        kelly = (p * b - q) / b
        if adx < KELLY_REGIME_ADX:
            kelly *= 0.5   # halve in choppy markets
        kelly *= KELLY_FRACTION
        return float(np.clip(kelly, KELLY_MIN_PCT, KELLY_MAX_PCT))

    def _compute_atr(self, highs, lows, closes, idx) -> float:
        start = max(0, idx - ATR_WINDOW)
        h = highs[start:idx + 1]
        l = lows[start:idx + 1]
        c = closes[start:idx + 1]
        if len(h) < 2:
            return closes[idx] * 0.02
        prev_c = c[:-1]
        trs = np.maximum(h[1:] - l[1:],
               np.maximum(np.abs(h[1:] - prev_c), np.abs(l[1:] - prev_c)))
        return float(np.mean(trs))

    # ── Position management ────────────────────────────────────────────────────

    def _apply_slippage(self, price, side):
        return price * (1 + SLIPPAGE) if side == "BUY" else price * (1 - SLIPPAGE)

    def _apply_fee(self, trade_value):
        return trade_value * TRADING_FEE

    def _open_long(self, price, idx, confidence, win_rate_est, payoff_ratio, adx, atr):
        exec_price = self._apply_slippage(price, "BUY")
        frac = (self._kelly_fraction(confidence, win_rate_est, payoff_ratio, adx)
                if self.use_kelly else MAX_POSITION_PCT)
        trade_value   = self.cash * frac
        fee           = self._apply_fee(trade_value)
        self.position = (trade_value - fee) / exec_price
        self.cash    -= trade_value
        self.entry_price = exec_price
        self.side        = "LONG"
        if self.use_atr_stops and atr > 0:
            self.stop_price = exec_price - ATR_STOP_MULT * atr
            self.tp_price   = exec_price + ATR_TP_MULT   * atr
        else:
            self.stop_price = exec_price * (1 - STOP_LOSS_PCT)
            self.tp_price   = exec_price * (1 + TAKE_PROFIT_PCT)
        self._entry_confidence = confidence
        self._entry_atr        = atr
        self._entry_adx        = adx

    def _open_short(self, price, idx, confidence, win_rate_est, payoff_ratio, adx, atr):
        exec_price = self._apply_slippage(price, "SELL")
        frac = (self._kelly_fraction(confidence, win_rate_est, payoff_ratio, adx)
                if self.use_kelly else MAX_POSITION_PCT)
        margin        = self.cash * frac
        units         = margin / exec_price
        fee           = self._apply_fee(units * exec_price)
        self.position = -units
        self.cash    += (units * exec_price) - fee
        self.margin_held = margin
        self.entry_price = exec_price
        self.side        = "SHORT"
        if self.use_atr_stops and atr > 0:
            self.stop_price = exec_price + ATR_STOP_MULT * atr
            self.tp_price   = exec_price - ATR_TP_MULT   * atr
        else:
            self.stop_price = exec_price * (1 + STOP_LOSS_PCT)
            self.tp_price   = exec_price * (1 - TAKE_PROFIT_PCT)
        self._entry_confidence = confidence
        self._entry_atr        = atr
        self._entry_adx        = adx

    def _close_position(self, price, idx, reason):
        if self.position > 0:
            exec_price = self._apply_slippage(price, "SELL")
            proceeds   = self.position * exec_price
            fee        = self._apply_fee(proceeds)
            pnl        = proceeds - fee - (self.position * self.entry_price)
            self.cash += proceeds - fee
        else:
            exec_price   = self._apply_slippage(price, "BUY")
            units        = abs(self.position)
            cost_to_buy  = units * exec_price
            fee          = self._apply_fee(cost_to_buy)
            entry_value  = units * self.entry_price
            pnl          = entry_value - cost_to_buy - fee
            self.cash   -= cost_to_buy + fee
        self.trades.append({
            "idx":        idx,
            "entry_price": self.entry_price,
            "exit_price":  exec_price,
            "pnl":         pnl,
            "reason":      reason,
            "side":        self.side or "LONG",
            "confidence":  getattr(self, "_entry_confidence", 0.5),
            "atr":         getattr(self, "_entry_atr", 0.0),
            "adx":         getattr(self, "_entry_adx", 0.0),
        })
        self.position = 0.0; self.entry_price = None
        self.stop_price = None; self.tp_price = None; self.side = None

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame, signals: dict, symbol: str = "",
            timeframe: str = "4h") -> dict:
        self._timeframe = timeframe
        self.reset()

        signal_arr = signals["signal"]
        confidence = signals["confidence"]
        n          = min(len(df), len(signal_arr))
        prices     = df["close"].values[-n:]
        highs      = df["high"].values[-n:]
        lows       = df["low"].values[-n:]
        adx_arr    = df["adx"].values[-n:] if "adx" in df.columns else np.full(n, 100.0)

        win_rate_est = 0.50
        payoff_ratio = 2.0
        wins, losses = [], []

        for i in range(n):
            price = prices[i]
            adx   = float(adx_arr[i])
            atr   = self._compute_atr(highs, lows, prices, i)

            equity = (self.cash + self.position * price if self.position >= 0
                      else self.cash - abs(self.position) * price)
            self.equity_curve.append(equity)

            # Stop / TP checks
            if self.position > 0 and self.stop_price is not None:
                if lows[i] <= self.stop_price:
                    self._close_position(self.stop_price, i, "STOP_LOSS")
                    t = self.trades[-1]; losses.append(abs(t["pnl"]))
                    win_rate_est, payoff_ratio = _recalc_kelly(wins, losses); continue
                if self.tp_price is not None and highs[i] >= self.tp_price:
                    self._close_position(self.tp_price, i, "TAKE_PROFIT")
                    t = self.trades[-1]; wins.append(abs(t["pnl"]))
                    win_rate_est, payoff_ratio = _recalc_kelly(wins, losses); continue

            elif self.position < 0 and self.stop_price is not None:
                if highs[i] >= self.stop_price:
                    self._close_position(self.stop_price, i, "STOP_LOSS")
                    t = self.trades[-1]; losses.append(abs(t["pnl"]))
                    win_rate_est, payoff_ratio = _recalc_kelly(wins, losses); continue
                if self.tp_price is not None and lows[i] <= self.tp_price:
                    self._close_position(self.tp_price, i, "TAKE_PROFIT")
                    t = self.trades[-1]; wins.append(abs(t["pnl"]))
                    win_rate_est, payoff_ratio = _recalc_kelly(wins, losses); continue

            sig  = signal_arr[i]
            conf = float(confidence[i])

            if sig == "BUY" and self.position <= 0:
                if self.position < 0:
                    self._close_position(price, i, "SIGNAL_FLIP")
                    t = self.trades[-1]
                    (wins if t["pnl"] > 0 else losses).append(abs(t["pnl"]))
                    win_rate_est, payoff_ratio = _recalc_kelly(wins, losses)
                self._open_long(price, i, conf, win_rate_est, payoff_ratio, adx, atr)

            elif sig == "SELL" and self.position >= 0:
                if self.position > 0:
                    self._close_position(price, i, "SIGNAL_FLIP")
                    t = self.trades[-1]
                    (wins if t["pnl"] > 0 else losses).append(abs(t["pnl"]))
                    win_rate_est, payoff_ratio = _recalc_kelly(wins, losses)
                self._open_short(price, i, conf, win_rate_est, payoff_ratio, adx, atr)

        if self.position != 0:
            self._close_position(prices[-1], n - 1, "END_OF_PERIOD")

        return self._compute_metrics(symbol)

    # ── Metrics ────────────────────────────────────────────────────────────────

    def _compute_metrics(self, symbol: str) -> dict:
        equity  = np.array(self.equity_curve)
        if len(equity) == 0:
            equity = np.array([self.initial_capital, self.initial_capital])
        returns = pd.Series(equity).pct_change().dropna()

        total_return = (equity[-1] - equity[0]) / equity[0] if equity[0] != 0 else 0.0
        n_trades = len(self.trades)
        winning  = [t for t in self.trades if t["pnl"] > 0]
        losing   = [t for t in self.trades if t["pnl"] <= 0]
        win_rate = len(winning) / n_trades if n_trades else 0
        n_longs  = sum(1 for t in self.trades if t.get("side") == "LONG")
        n_shorts = sum(1 for t in self.trades if t.get("side") == "SHORT")

        bars_per_year = self._BARS_PER_YEAR.get(self._timeframe, 2190)
        sharpe  = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(bars_per_year)
        dn      = returns[returns < 0]
        sortino = (returns.mean() / (dn.std() + 1e-9)) * np.sqrt(bars_per_year) if len(dn) > 1 else 0.0

        roll_max = pd.Series(equity).cummax()
        drawdown = (pd.Series(equity) - roll_max) / roll_max
        max_dd   = float(drawdown.min())

        # Max drawdown duration (consecutive bars in drawdown)
        dd_len = max_dd_len = 0
        for v in (drawdown < 0):
            dd_len = dd_len + 1 if v else 0
            max_dd_len = max(max_dd_len, dd_len)

        gross_profit  = sum(t["pnl"] for t in self.trades if t["pnl"] > 0)
        gross_loss    = abs(sum(t["pnl"] for t in self.trades if t["pnl"] < 0))
        profit_factor = gross_profit / (gross_loss + 1e-9)

        recovery_factor = abs(total_return) / (abs(max_dd) + 1e-9) if max_dd < 0 else float("inf")

        hours_per_bar = {"15m": 0.25, "30m": 0.5, "1h": 1.0, "2h": 2.0,
                         "4h": 4.0, "8h": 8.0, "12h": 12.0, "1d": 24.0, "1w": 168.0}
        n_bars     = len(equity)
        test_years = max(n_bars * hours_per_bar.get(self._timeframe, 4.0) / 8760, 1/365)
        ann_ret    = (1 + total_return) ** (1 / test_years) - 1
        calmar     = ann_ret / (abs(max_dd) + 1e-9)

        # Consecutive losses
        max_consec = consec = 0
        for t in self.trades:
            consec = consec + 1 if t["pnl"] <= 0 else 0
            max_consec = max(max_consec, consec)

        avg_pnl  = float(np.mean([t["pnl"] for t in self.trades])) if n_trades else 0
        avg_win  = float(np.mean([t["pnl"] for t in winning]))      if winning  else 0
        avg_loss = float(np.mean([t["pnl"] for t in losing]))       if losing   else 0
        payoff_r = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
        expectancy = avg_pnl / self.initial_capital

        # Exit reason counts
        exit_reasons = {}
        for t in self.trades:
            r = t.get("reason", "UNKNOWN")
            exit_reasons[r] = exit_reasons.get(r, 0) + 1

        # High-confidence subset
        hc = [t for t in self.trades if t.get("confidence", 0) > 0.45]
        high_conf_wr = sum(1 for t in hc if t["pnl"] > 0) / len(hc) if hc else 0

        # Regime breakdown
        trend_t = [t for t in self.trades if t.get("adx", 0) >= KELLY_REGIME_ADX]
        range_t = [t for t in self.trades if t.get("adx", 0) <  KELLY_REGIME_ADX]
        trend_wr = sum(1 for t in trend_t if t["pnl"] > 0) / len(trend_t) if trend_t else 0
        range_wr = sum(1 for t in range_t if t["pnl"] > 0) / len(range_t) if range_t else 0

        metrics = {
            "symbol":           symbol,
            "final_equity":     float(equity[-1]),
            "total_return":     total_return,
            "ann_return":       ann_ret,
            "n_trades":         n_trades,
            "n_longs":          n_longs,
            "n_shorts":         n_shorts,
            "win_rate":         win_rate,
            "sharpe_ratio":     float(sharpe),
            "sortino_ratio":    float(sortino),
            "calmar_ratio":     calmar,
            "max_drawdown":     max_dd,
            "max_dd_duration":  max_dd_len,
            "profit_factor":    float(profit_factor),
            "recovery_factor":  float(recovery_factor),
            "avg_trade_pnl":    float(avg_pnl),
            "avg_win":          float(avg_win),
            "avg_loss":         float(avg_loss),
            "payoff_ratio":     float(payoff_r),
            "expectancy":       float(expectancy),
            "max_consec_losses": max_consec,
            "high_conf_wr":     float(high_conf_wr),
            "trend_wr":         float(trend_wr),
            "range_wr":         float(range_wr),
            "n_trend_trades":   len(trend_t),
            "n_range_trades":   len(range_t),
            "exit_reasons":     exit_reasons,
        }

        logger.info(
            f"{symbol} | Ann={ann_ret:.2%} Sh={sharpe:.2f} So={sortino:.2f} "
            f"Cal={calmar:.2f} DD={max_dd:.2%}({max_dd_len}b) "
            f"WR={win_rate:.2%} PF={profit_factor:.2f} RF={recovery_factor:.2f} "
            f"Trades={n_trades}(L{n_longs}/S{n_shorts}) "
            f"TrendWR={trend_wr:.2%}({len(trend_t)}) "
            f"RangeWR={range_wr:.2%}({len(range_t)}) "
            f"Exits={exit_reasons}"
        )
        return metrics

    def save_results(self, metrics, symbol):
        import json
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = Path(RESULTS_DIR) / f"{symbol.replace('/','_')}_backtest.json"
        serialisable = {
            k: (float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in metrics.items()
        }
        with open(path, "w") as f:
            json.dump(serialisable, f, indent=2)
        pd.DataFrame({"equity": self.equity_curve}).to_csv(
            Path(RESULTS_DIR) / f"{symbol.replace('/','_')}_equity.csv", index=False)
        logger.info(f"Results saved -> {path}")


# ── Standalone Kelly helpers ────────────────────────────────────────────────────

def _recalc_kelly(wins: list, losses: list):
    """Recompute (win_rate, payoff_ratio) from running trade history."""
    n = len(wins) + len(losses)
    if n < 5:
        return 0.50, 2.0
    wr       = len(wins) / n
    avg_win  = float(np.mean(wins))  if wins   else 1.0
    avg_loss = float(np.mean(losses)) if losses else 0.5
    pr       = avg_win / max(avg_loss, 1e-9)
    return float(np.clip(wr, 0.25, 0.80)), float(np.clip(pr, 0.3, 10.0))


def kelly_position_size(win_rate: float, payoff_ratio: float,
                        capital: float,
                        fraction: float = KELLY_FRACTION) -> float:
    """
    Standalone Kelly position size calculator.
    Returns the dollar amount to allocate to the next trade.
    """
    b      = max(payoff_ratio, 0.01)
    kelly_f = max(0.0, (win_rate * b - (1 - win_rate)) / b) * fraction
    kelly_f = float(np.clip(kelly_f, KELLY_MIN_PCT, KELLY_MAX_PCT))
    return capital * kelly_f