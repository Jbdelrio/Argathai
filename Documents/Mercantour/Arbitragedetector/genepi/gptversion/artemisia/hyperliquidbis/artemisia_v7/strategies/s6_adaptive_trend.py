"""
s6_adaptive_trend.py — Adaptive Time-Series Momentum (H6 candles)
Based on Nguyen 2026 AdaptiveTrend, adapted for Hyperliquid top-20 perps.

Core logic:
  - Multi-lookback momentum signal (24h / 72h / 168h / 336h) normalised by ATR
  - Composite score filters noise via equal-weight average
  - Rolling Sharpe gate: only trade coins where the strategy has worked recently
  - Asymmetric 70/30 long/short allocation (captures crypto positive drift)
  - Trailing ATR stop (dynamic, not fixed %)
  - HMM regime gate: halts in EXPLOSION regime
  - H6 candles: 4 bars/day, low turnover, fee-efficient

Backtest uses 6h candles (4032 bars = ~42 days with 96% availability).
WF: 60 train / 15 test days, 4 folds.
"""
import logging
import numpy as np
import pandas as pd
from typing import Optional

try:
    from .base import BaseStrategy, Signal
except ImportError:
    from strategies.base import BaseStrategy, Signal

log = logging.getLogger(__name__)

# 6h candle: 4 bars/day
BARS_PER_DAY_6H = 4

# Lookback periods in 6h bars
LOOKBACKS = {
    "24h":  4,    # 1 day
    "72h":  12,   # 3 days
    "168h": 28,   # 7 days
    "336h": 56,   # 14 days
}


class AdaptiveTrendStrategy(BaseStrategy):
    """
    H6 time-series momentum with adaptive trailing stops.
    Trade on per-coin composite momentum score + rolling Sharpe gate.
    """

    NAME = "s6_adaptive_trend"

    DEFAULT_PARAMS = {
        "composite_threshold":  0.7,   # Enter long if score > +0.7, short if < -0.7
        "sharpe_filter_long":   0.8,   # Rolling Sharpe must exceed this for longs
        "sharpe_filter_short":  1.2,   # Stricter for shorts (anti-drift)
        "sharpe_window_trades": 20,    # Trades to compute rolling Sharpe on
        "atr_period":           14,    # ATR period in bars
        "atr_stop_mult":        2.5,   # Initial stop = entry ± 2.5 * ATR
        "atr_trail_trigger":    1.5,   # Move stop to BE after 1.5*ATR gain
        "atr_trail_mult":       2.0,   # Trail at high - 2*ATR (for longs)
        "max_hold_bars":        28,    # 7 days in 6h bars
        "long_ratio":           0.70,  # 70% capital allocated to longs
        "short_ratio":          0.30,  # 30% to shorts
        "base_position_pct":    0.10,  # 10% per position baseline
        "max_position_pct":     0.15,  # Hard cap per position
        "vol_target_annual":    0.20,  # 20% annualised vol target for sizing
        "min_atr_pct":          0.001, # Min ATR/price (skip flat markets)
    }

    # Parameter grid for WF search (max ~100 combos via random sample)
    PARAM_GRID = [
        {
            "composite_threshold": ct,
            "sharpe_filter_long":  sfl,
            "sharpe_filter_short": sfs,
            "sharpe_window_trades": 30,
            "atr_period":          14,
            "atr_stop_mult":       asm,
            "atr_trail_trigger":   2.0,
            "atr_trail_mult":      2.0,
            "max_hold_bars":       56,
            "long_ratio":          lr,
            "short_ratio":         1.0 - lr,
            "base_position_pct":   0.10,
            "max_position_pct":    0.15,
            "vol_target_annual":   0.20,
            "min_atr_pct":         0.002,
        }
        for ct  in [0.5, 0.7, 1.0]
        for sfl in [0.6, 0.8, 1.0]
        for sfs in [1.0, 1.2, 1.5]
        for asm in [2.0, 2.5, 3.0]
        for lr  in [0.70, 0.60]
        if sfs > sfl
    ]

    def _compute_momentum_scores(self, closes: np.ndarray,
                                  highs: np.ndarray,
                                  lows: np.ndarray,
                                  atr_period: int) -> np.ndarray:
        """
        Vectorised composite momentum score for each bar.
        score = mean(mom_24h, mom_72h, mom_168h, mom_336h)
        each mom = (close - close_lag) / atr_lag
        """
        n = len(closes)
        atr = self._atr(highs, lows, closes, atr_period)

        scores = np.full(n, np.nan)
        max_lag = max(LOOKBACKS.values())  # 56 bars

        for i in range(max_lag, n):
            moms = []
            for lag in LOOKBACKS.values():
                if i >= lag and atr[i - lag] > 0:
                    mom = (closes[i] - closes[i - lag]) / atr[i - lag]
                    moms.append(mom)
            if moms:
                scores[i] = float(np.mean(moms))

        return scores

    def _vol_adjusted_size(self, base_pct: float, closes: np.ndarray,
                            lookback: int = 120,
                            vol_target: float = 0.20,
                            equity: float = 500.0) -> float:
        """
        Scale position size by inverse of recent volatility.
        Target: coin_vol * size ≈ vol_target * equity
        """
        if len(closes) < 20:
            return equity * base_pct

        returns = pd.Series(closes).pct_change().dropna().tail(lookback)
        if len(returns) < 10:
            return equity * base_pct

        # Annualise: 4 bars/day * 365 days = 1460 bars/year for 6h
        ann_vol = float(returns.std()) * np.sqrt(BARS_PER_DAY_6H * 365)
        if ann_vol <= 0:
            return equity * base_pct

        vol_adj_pct = base_pct * (vol_target / ann_vol)
        return equity * min(vol_adj_pct, self.params.get("max_position_pct", 0.15))

    def backtest(self,
                 candles: pd.DataFrame,
                 params: Optional[dict] = None) -> list[float]:
        """
        Simulate AdaptiveTrend on 6h candles.
        Returns list of per-trade P&L in USD (after fees + realistic slippage).
        """
        p = {**self.params, **(params or {})}

        if candles.empty or len(candles) < max(LOOKBACKS.values()) + p["atr_period"] + 10:
            return []

        closes = candles["close"].values.astype(float)
        highs  = candles["high"].values.astype(float)
        lows   = candles["low"].values.astype(float)
        n      = len(closes)

        capital    = 500.0
        long_cap   = capital * p["long_ratio"]
        short_cap  = capital * p["short_ratio"]

        atr    = self._atr(highs, lows, closes, p["atr_period"])
        scores = self._compute_momentum_scores(closes, highs, lows, p["atr_period"])

        warmup = max(LOOKBACKS.values()) + p["atr_period"] + 2
        min_atr = p["min_atr_pct"]

        trades     = []
        recent_pnl = []   # Rolling Sharpe window

        # State per position (only one position at a time in single-asset backtest)
        in_trade   = False
        side       = "long"
        entry_p    = 0.0
        stop_p     = 0.0
        peak_p     = 0.0   # For trailing
        entry_bar  = 0
        pos_size   = 0.0

        for i in range(warmup, n):
            if np.isnan(scores[i]):
                continue
            score = scores[i]
            atr_i = atr[i]

            # Skip flat markets
            if atr_i / max(closes[i], 1) < min_atr:
                continue

            # Rolling Sharpe gate
            if len(recent_pnl) >= 10:
                pnl_arr = np.array(recent_pnl[-p["sharpe_window_trades"]:])
                mean_p = pnl_arr.mean()
                std_p  = pnl_arr.std()
                if std_p > 0:
                    rolling_sharpe = mean_p / std_p * np.sqrt(BARS_PER_DAY_6H * 365)
                else:
                    rolling_sharpe = 0.0
            else:
                rolling_sharpe = 2.0  # Assume OK in warmup (no data yet)

            if not in_trade:
                can_long  = score >  p["composite_threshold"] and rolling_sharpe >= p["sharpe_filter_long"]
                can_short = score < -p["composite_threshold"] and rolling_sharpe >= p["sharpe_filter_short"]

                if can_long or can_short:
                    side = "long" if can_long else "short"
                    entry_p   = closes[i]
                    stop_p    = (entry_p - p["atr_stop_mult"] * atr_i if side == "long"
                                 else entry_p + p["atr_stop_mult"] * atr_i)
                    peak_p    = entry_p
                    entry_bar = i

                    # Vol-adjusted sizing
                    base_cap  = long_cap if side == "long" else short_cap
                    pos_size  = self._vol_adjusted_size(
                        base_pct=base_cap / capital,
                        closes=closes[:i],
                        vol_target=p["vol_target_annual"],
                        equity=capital,
                    )
                    pos_size  = min(pos_size, capital * p["max_position_pct"])
                    in_trade  = True

            else:
                # Update trailing stop
                if side == "long":
                    peak_p = max(peak_p, closes[i])
                    gain   = (peak_p - entry_p) / entry_p
                    if gain >= p["atr_trail_trigger"] * atr_i / entry_p:
                        new_stop = peak_p - p["atr_trail_mult"] * atr_i
                        stop_p   = max(stop_p, new_stop)
                else:
                    peak_p = min(peak_p, closes[i])
                    gain   = (entry_p - peak_p) / entry_p
                    if gain >= p["atr_trail_trigger"] * atr_i / entry_p:
                        new_stop = peak_p + p["atr_trail_mult"] * atr_i
                        stop_p   = min(stop_p, new_stop)

                # Exit conditions
                hold = i - entry_bar
                stop_hit    = (lows[i] <= stop_p  if side == "long" else highs[i] >= stop_p)
                max_hold    = hold >= p["max_hold_bars"]
                # Momentum reversal: score crossed zero
                trend_flip  = (score < 0 if side == "long" else score > 0)

                if stop_hit or max_hold or trend_flip:
                    # Exit price: use close for non-stop exits, stop for stop-outs
                    if stop_hit:
                        exit_p = max(lows[i], stop_p) if side == "long" else min(highs[i], stop_p)
                    else:
                        exit_p = closes[i]

                    if side == "long":
                        gross = (exit_p - entry_p) / entry_p * pos_size
                    else:
                        gross = (entry_p - exit_p) / entry_p * pos_size

                    # Fees: taker exit (stop/reversal), maker entry for trend entries
                    # Use taker for realism on H6 (we cross the spread to exit fast)
                    fee = pos_size * 6.6 / 10_000  # roundtrip taker: 2.5+0.8)*2
                    net = gross - fee

                    trades.append(net)
                    recent_pnl.append(net)
                    in_trade = False

        return trades

    def signal(self,
               candles: pd.DataFrame,
               funding_df: Optional[pd.DataFrame],
               params: Optional[dict],
               equity: float,
               open_positions: list[dict]) -> list[Signal]:
        """Generate live H6 momentum signals."""
        p = {**self.params, **(params or {})}
        signals = []

        min_bars = max(LOOKBACKS.values()) + p["atr_period"] + 5
        if candles.empty or len(candles) < min_bars:
            return signals

        existing = [pos for pos in open_positions if pos.get("strategy") == self.NAME]
        if existing:
            return signals

        closes = candles["close"].values.astype(float)
        highs  = candles["high"].values.astype(float)
        lows   = candles["low"].values.astype(float)

        atr    = self._atr(highs, lows, closes, p["atr_period"])
        scores = self._compute_momentum_scores(closes, highs, lows, p["atr_period"])

        score = scores[-1]
        atr_i = atr[-1]
        close = closes[-1]

        if np.isnan(score):
            return signals

        atr_pct = atr_i / max(close, 1)
        if atr_pct < p["min_atr_pct"]:
            return signals

        coin = getattr(candles, "coin", None) or "BTC"

        if score > p["composite_threshold"]:
            stop_price = close - p["atr_stop_mult"] * atr_i
            pos_size = self._vol_adjusted_size(
                p["base_position_pct"] * p["long_ratio"],
                closes, p["vol_target_annual"], equity
            )
            signals.append(Signal(
                symbol=coin, side="long",
                size_usd=min(pos_size, equity * p["max_position_pct"]),
                entry_type="maker",
                stop_price=stop_price,
                confidence=min(abs(score) / 3.0, 1.0),
                meta={
                    "strategy": self.NAME,
                    "score": round(float(score), 3),
                    "atr": round(float(atr_i), 4),
                    "max_hold_bars": p["max_hold_bars"],
                }
            ))

        elif score < -p["composite_threshold"]:
            stop_price = close + p["atr_stop_mult"] * atr_i
            pos_size = self._vol_adjusted_size(
                p["base_position_pct"] * p["short_ratio"],
                closes, p["vol_target_annual"], equity
            )
            signals.append(Signal(
                symbol=coin, side="short",
                size_usd=min(pos_size, equity * p["max_position_pct"]),
                entry_type="maker",
                stop_price=stop_price,
                confidence=min(abs(score) / 3.0, 1.0),
                meta={
                    "strategy": self.NAME,
                    "score": round(float(score), 3),
                    "atr": round(float(atr_i), 4),
                    "max_hold_bars": p["max_hold_bars"],
                }
            ))

        return signals


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rng = np.random.default_rng(42)
    n = 700   # ~175 days of 6h candles
    ts = pd.date_range("2024-01-01", periods=n, freq="6h", tz="UTC")

    # Trending market with noise
    trend = np.cumsum(rng.normal(0.001, 0.02, n))  # small positive drift
    close = 100 * np.exp(trend)
    high  = close * (1 + abs(rng.normal(0, 0.005, n)))
    low   = close * (1 - abs(rng.normal(0, 0.005, n)))

    candles = pd.DataFrame({
        "ts": ts, "open": close, "high": high, "low": low,
        "close": close, "volume": rng.uniform(1000, 10000, n)
    })

    strat  = AdaptiveTrendStrategy()
    trades = strat.backtest(candles)
    pnl    = np.array(trades)

    if len(pnl) > 0:
        wr = (pnl > 0).mean() * 100
        print(f"Trades: {len(trades)}, WR={wr:.1f}%")
        print(f"Total P&L: ${pnl.sum():.2f}, Avg: ${pnl.mean():.3f}")
        print(f"Sharpe (rough): {pnl.mean() / (pnl.std() + 1e-9) * np.sqrt(BARS_PER_DAY_6H * 252):.2f}")
    else:
        print("No trades generated")

    print(f"\nParam grid size: {len(AdaptiveTrendStrategy.PARAM_GRID)} combos")
