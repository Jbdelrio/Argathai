"""
s2_maker_mr.py — Maker-Only Mean Reversion
Logic: Bollinger Band mean reversion on 15-min candles
- Enter at lower/upper band via limit orders (maker rebate)
- RSI filter: only enter when momentum aligns
- Exit at BB mid (target) or ATR stop
- Maker rebate offsets fees materially
"""
import logging
import numpy as np
import pandas as pd
from typing import Optional
from itertools import product

try:
    from .base import BaseStrategy, Signal
except ImportError:
    from strategies.base import BaseStrategy, Signal

log = logging.getLogger(__name__)


class MakerMRStrategy(BaseStrategy):

    NAME = "s2_maker_mr"

    DEFAULT_PARAMS = {
        "bb_period": 20,
        "bb_std": 2.0,
        "rsi_period": 14,
        "rsi_oversold": 35,
        "rsi_overbought": 65,
        "atr_period": 14,
        "stop_atr_mult": 2.0,
        "position_pct": 0.15,
        "min_bb_width_pct": 0.002,   # Min BB width to avoid flat markets
    }

    PARAM_GRID = [
        {
            "bb_period": bp,
            "bb_std": bs,
            "rsi_period": rp,
            "rsi_oversold": ro,
            "rsi_overbought": rob,
            "atr_period": 14,
            "stop_atr_mult": sa,
            "position_pct": 0.15,
            "min_bb_width_pct": 0.002,
        }
        for bp  in [15, 20, 30]
        for bs  in [1.5, 2.0, 2.5]
        for rp  in [10, 14]
        for ro  in [30, 35, 40]
        for rob in [60, 65, 70]
        for sa  in [1.5, 2.0, 2.5]
        if ro < rob
    ]

    def backtest(self,
                 candles: pd.DataFrame,
                 params: Optional[dict] = None) -> list[float]:
        """
        Vectorised BB mean reversion backtest.
        Entry: close crosses below lower BB (long) or above upper BB (short)
        Exit: close reaches BB mid OR ATR stop hit
        """
        p = {**self.params, **(params or {})}
        if candles.empty or len(candles) < p["bb_period"] + p["atr_period"]:
            return []

        closes = candles["close"].values.astype(float)
        highs  = candles["high"].values.astype(float)
        lows   = candles["low"].values.astype(float)
        n = len(closes)

        capital   = 500.0
        pos_size  = capital * p["position_pct"]

        # Indicators
        bb_upper, bb_mid, bb_lower = self._bollinger(closes, p["bb_period"], p["bb_std"])
        rsi  = self._rsi(closes, p["rsi_period"])
        atr  = self._atr(highs, lows, closes, p["atr_period"])

        # BB width filter
        bb_width = (bb_upper - bb_lower) / np.where(bb_mid > 0, bb_mid, 1)

        trades  = []
        in_trade = False
        side     = "long"
        entry_p  = 0.0
        stop_p   = 0.0
        tp_p     = 0.0

        warmup = max(p["bb_period"], p["rsi_period"], p["atr_period"]) + 2

        for i in range(warmup, n):
            if not in_trade:
                # Skip flat markets
                if bb_width[i] < p["min_bb_width_pct"]:
                    continue

                # Long entry: close < lower BB AND RSI oversold
                if (closes[i] < bb_lower[i] and rsi[i] < p["rsi_oversold"]):
                    in_trade = True
                    side     = "long"
                    entry_p  = closes[i]
                    stop_p   = entry_p - p["stop_atr_mult"] * atr[i]
                    tp_p     = bb_mid[i]

                # Short entry: close > upper BB AND RSI overbought
                elif (closes[i] > bb_upper[i] and rsi[i] > p["rsi_overbought"]):
                    in_trade = True
                    side     = "short"
                    entry_p  = closes[i]
                    stop_p   = entry_p + p["stop_atr_mult"] * atr[i]
                    tp_p     = bb_mid[i]

            else:
                if side == "long":
                    price_pnl_pct = (closes[i] - entry_p) / entry_p
                    hit_stop = closes[i] <= stop_p or lows[i] <= stop_p
                    hit_tp   = closes[i] >= tp_p or highs[i] >= tp_p
                else:
                    price_pnl_pct = (entry_p - closes[i]) / entry_p
                    hit_stop = closes[i] >= stop_p or highs[i] >= stop_p
                    hit_tp   = closes[i] <= tp_p or lows[i] <= tp_p

                if hit_stop or hit_tp:
                    if hit_stop and not hit_tp:
                        # Use actual stop price for realistic slippage
                        if side == "long":
                            exit_p = min(closes[i], stop_p)
                        else:
                            exit_p = max(closes[i], stop_p)
                        if side == "long":
                            price_pnl_pct = (exit_p - entry_p) / entry_p
                        else:
                            price_pnl_pct = (entry_p - exit_p) / entry_p
                    # Maker entry rebate: -0.3bp, taker exit: +2.5bp
                    # Net fee: 2.5 - 0.3 + 2*0.8 = 3.8bp per side (entry maker, exit taker)
                    gross_pnl = price_pnl_pct * pos_size
                    fee = pos_size * 3.8 / 10_000
                    trades.append(gross_pnl - fee)
                    in_trade = False

        return trades

    def signal(self,
               candles: pd.DataFrame,
               funding_df: Optional[pd.DataFrame],
               params: Optional[dict],
               equity: float,
               open_positions: list[dict]) -> list[Signal]:
        """Generate live signals from current candle data."""
        p = {**self.params, **(params or {})}
        signals = []

        if candles.empty or len(candles) < p["bb_period"] + 10:
            return signals

        closes = candles["close"].values.astype(float)
        highs  = candles["high"].values.astype(float)
        lows   = candles["low"].values.astype(float)

        bb_upper, bb_mid, bb_lower = self._bollinger(closes, p["bb_period"], p["bb_std"])
        rsi = self._rsi(closes, p["rsi_period"])
        atr = self._atr(highs, lows, closes, p["atr_period"])

        i = -1  # last bar
        close = closes[i]
        bb_width = (bb_upper[i] - bb_lower[i]) / max(bb_mid[i], 1)

        if bb_width < p["min_bb_width_pct"]:
            return signals

        existing_mr = [p for p in open_positions if p.get("strategy") == self.NAME]
        if existing_mr:
            return signals

        pos_size = equity * p["position_pct"]
        if pos_size < 10:
            return signals

        coin = candles.get("coin", "BTC") if hasattr(candles, "get") else "BTC"

        if close < bb_lower[i] and rsi[i] < p["rsi_oversold"]:
            stop_price = close - p["stop_atr_mult"] * atr[i]
            signals.append(Signal(
                symbol=coin,
                side="long",
                size_usd=pos_size,
                entry_type="maker",
                stop_price=stop_price,
                tp_price=bb_mid[i],
                confidence=min((p["rsi_oversold"] - rsi[i]) / p["rsi_oversold"], 1.0),
                meta={"strategy": self.NAME, "rsi": rsi[i], "bb_pos": "below_lower"}
            ))

        elif close > bb_upper[i] and rsi[i] > p["rsi_overbought"]:
            stop_price = close + p["stop_atr_mult"] * atr[i]
            signals.append(Signal(
                symbol=coin,
                side="short",
                size_usd=pos_size,
                entry_type="maker",
                stop_price=stop_price,
                tp_price=bb_mid[i],
                confidence=min((rsi[i] - p["rsi_overbought"]) / (100 - p["rsi_overbought"]), 1.0),
                meta={"strategy": self.NAME, "rsi": rsi[i], "bb_pos": "above_upper"}
            ))

        return signals


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rng = np.random.default_rng(42)
    n = 3000
    ts = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    # Mean-reverting price
    close = 100 + np.cumsum(rng.normal(0, 0.2, n) - 0.01 * (100 + np.cumsum(rng.normal(0, 0.2, n)) - 100))
    close = np.clip(close, 50, 200)
    candles = pd.DataFrame({
        "ts": ts,
        "open": close,
        "high": close + abs(rng.normal(0, 0.3, n)),
        "low":  close - abs(rng.normal(0, 0.3, n)),
        "close": close,
        "volume": rng.uniform(100, 1000, n),
    })

    strat = MakerMRStrategy()
    trades = strat.backtest(candles)
    print(f"Trades: {len(trades)}")
    if trades:
        pnl = np.array(trades)
        print(f"Win rate: {(pnl > 0).mean()*100:.1f}%")
        print(f"Total P&L: ${pnl.sum():.2f}")
        print(f"Avg trade: ${pnl.mean():.3f}")
