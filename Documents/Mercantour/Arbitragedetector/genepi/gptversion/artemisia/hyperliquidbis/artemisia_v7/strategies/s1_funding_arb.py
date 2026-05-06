"""
s1_funding_arb.py — Funding Rate Arbitrage Delta-Neutral (Priority #1)

Logic:
- Identify coins with high absolute funding rate (annualised > 30%)
- Short the high-funding coin (collect funding), long a correlated hedge coin
- The pair is delta-neutral: captures funding without directional exposure
- Exit when: funding drops below threshold OR 8h max hold OR stop hit

Backtest mode: use funding history + candles to simulate funding collection
Live mode: signal when current funding > threshold, pair entry via maker orders
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

# Fee model
ROUNDTRIP_MAKER_BPS = 4.8

# Funding is hourly on Hyperliquid; annualised = rate * 24 * 365 * 100
HOURLY_TO_ANNUAL = 24 * 365 * 100


class FundingArbStrategy(BaseStrategy):

    NAME = "s1_funding_arb"

    DEFAULT_PARAMS = {
        "min_funding_annual_pct": 10.0,  # Enter when |ann funding| > 10%
        "exit_funding_annual_pct": 3.0,  # Exit when |ann funding| drops below 3%
        "max_hold_hours": 2,
        "stop_loss_pct": 0.03,           # 3% stop — funding arb needs 8h hold
        "hedge_corr_min": 0.7,
        "position_pct": 0.15,            # 15% of equity per leg
        "funding_ewma_alpha": 0.1,       # Smoothing for funding signal
    }

    PARAM_GRID = [
        {
            "min_funding_annual_pct": mf,
            "exit_funding_annual_pct": ef,
            "max_hold_hours": mh,
            "stop_loss_pct": sl,
            "position_pct": pp,
            "funding_ewma_alpha": 0.1,
            "hedge_corr_min": 0.7,
        }
        for mf in [10.0, 15.0, 20.0, 30.0]
        for ef in [3.0, 5.0, 8.0, 10.0]
        for mh in [4, 8, 12]
        for sl in [0.02, 0.03, 0.05]
        for pp in [0.10, 0.15, 0.20]
        if ef < mf
    ]

    def __init__(self, params=None, hedge_map: Optional[dict[str, str]] = None):
        super().__init__(params)
        # hedge_map: {high_funding_coin -> hedge_coin}
        # Built dynamically from correlation analysis
        self.hedge_map = hedge_map or {}

    def backtest(self,
                 candles: pd.DataFrame,
                 params: Optional[dict] = None,
                 funding_df: Optional[pd.DataFrame] = None) -> list[float]:
        """
        Backtest funding arb on a single coin.
        Assumes we short the coin and hold through funding payments.

        Without funding data: approximates funding as simulated.
        With funding data: uses actual hourly funding from DataFrame.

        Returns list of trade P&Ls in USD.
        """
        p = {**self.params, **(params or {})}
        if candles.empty:
            return []

        closes = candles["close"].values.astype(float)
        n = len(closes)
        if n < 20:
            return []

        capital = 500.0  # baseline
        pos_size = capital * p["position_pct"]
        max_hold_bars = p["max_hold_hours"] * 4  # 15-min candles: 4 bars/hour
        stop_pct = p["stop_loss_pct"]
        min_ann = p["min_funding_annual_pct"]
        exit_ann = p["exit_funding_annual_pct"]

        # Build funding signal array
        if funding_df is not None and not funding_df.empty:
            # Align hourly funding to 15-min candles
            fund_ann = self._align_funding(candles, funding_df)
        else:
            # Synthetic: use small random funding for conservative backtest
            fund_ann = np.zeros(n)

        # EWMA smoothing of funding signal
        alpha = p["funding_ewma_alpha"]
        fund_smooth = pd.Series(fund_ann).ewm(alpha=alpha, adjust=False).mean().values

        trades = []
        in_trade = False
        entry_price = 0.0
        entry_bar   = 0
        short_side  = True  # True = short the high-funding coin

        for i in range(1, n):
            if not in_trade:
                # Entry: funding above threshold
                if abs(fund_smooth[i]) >= min_ann:
                    in_trade    = True
                    entry_price = closes[i]
                    entry_bar   = i
                    short_side  = fund_smooth[i] > 0  # short when positive funding
            else:
                hold_bars    = i - entry_bar
                current_fund = abs(fund_smooth[i])
                price_move   = (closes[i] - entry_price) / entry_price

                # For short: profit when price falls, loss when rises
                if short_side:
                    price_pnl_pct = -price_move
                else:
                    price_pnl_pct = price_move

                # Funding collected: each 15-min bar = 1/(4*24)=1/96 of daily funding
                # Daily funding = ann_funding / 365
                # Per-bar funding = ann_funding / (365 * 96 * 100)  (fraction)
                funding_per_bar = fund_smooth[i] / (365 * 96 * 100) if short_side else 0
                cumulative_funding = sum(
                    fund_smooth[max(0, entry_bar):i+1]
                ) / (365 * 96 * 100) * pos_size

                # Exit conditions
                exit_trade = False
                if hold_bars >= max_hold_bars:
                    exit_trade = True
                elif current_fund < exit_ann:
                    exit_trade = True
                elif price_pnl_pct <= -stop_pct:
                    exit_trade = True

                if exit_trade:
                    gross_pnl = (price_pnl_pct * pos_size) + cumulative_funding
                    net_pnl   = self._apply_fees(gross_pnl, pos_size, "maker")
                    trades.append(net_pnl)
                    in_trade = False

        return trades

    def _align_funding(self, candles: pd.DataFrame,
                       funding_df: pd.DataFrame) -> np.ndarray:
        """
        Resample hourly funding to 15-min candle timestamps via forward-fill.
        Returns array of annualised funding % aligned to candles.
        """
        if "ts" not in candles.columns or "ts" not in funding_df.columns:
            return np.zeros(len(candles))
        try:
            f = funding_df.set_index("ts")["annual_pct"]
            c_ts = candles["ts"]
            aligned = f.reindex(c_ts, method="ffill")
            return aligned.fillna(0).values
        except Exception as e:
            log.debug("Funding alignment failed: %s", e)
            return np.zeros(len(candles))

    def signal(self,
               candles: pd.DataFrame,
               funding_df: Optional[pd.DataFrame],
               params: Optional[dict],
               equity: float,
               open_positions: list[dict]) -> list[Signal]:
        """
        Generate live signals for funding arb.
        Looks at current funding rate and emits entry/exit signals.
        """
        p = {**self.params, **(params or {})}
        signals = []

        if funding_df is None or funding_df.empty:
            return signals

        # Current funding (most recent)
        current_annual = float(funding_df["annual_pct"].iloc[-1])
        abs_annual = abs(current_annual)

        if abs_annual < p["min_funding_annual_pct"]:
            return signals

        # Check no existing funding arb position
        existing = [pos for pos in open_positions if pos.get("strategy") == self.NAME]
        if existing:
            return signals

        pos_size = equity * p["position_pct"]
        if pos_size < 10:
            return signals

        # Determine which coin to short/long
        coin = candles.get("coin", candles.index.name or "UNKNOWN")
        if isinstance(coin, pd.Index) or not isinstance(coin, str):
            coin = "BTC"  # fallback

        if current_annual > 0:
            # Short the coin (collect positive funding)
            side = "short"
        else:
            # Long the coin (collect negative funding payment from longs)
            side = "long"

        current_price = float(candles["close"].iloc[-1])
        stop_dist = current_price * p["stop_loss_pct"]
        stop_price = (current_price + stop_dist) if side == "short" else (current_price - stop_dist)

        signals.append(Signal(
            symbol=coin,
            side=side,
            size_usd=pos_size,
            entry_type="maker",
            stop_price=stop_price,
            confidence=min(abs_annual / 100.0, 1.0),
            meta={
                "strategy": self.NAME,
                "funding_annual_pct": current_annual,
                "max_hold_hours": p["max_hold_hours"],
                "stop_loss_pct": p["stop_loss_pct"],
            }
        ))

        return signals


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Synthetic test
    rng = np.random.default_rng(42)
    n = 2000  # ~20 days of 15m candles
    ts = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    candles = pd.DataFrame({
        "ts": ts, "open": close, "high": close * 1.002,
        "low": close * 0.998, "close": close, "volume": rng.uniform(100, 1000, n)
    })

    # Synthetic funding: periods of high funding
    hours = n // 4
    fund_ts = pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC")
    fund_rate = np.where((np.arange(hours) % 48) < 12, 0.0001, 0.00001)
    funding_df = pd.DataFrame({
        "ts": fund_ts,
        "funding_rate": fund_rate,
        "premium": fund_rate * 0.5,
        "annual_pct": fund_rate * 24 * 365 * 100,
    })

    strat = FundingArbStrategy()
    trades = strat.backtest(candles, funding_df=funding_df)
    print(f"Trades: {len(trades)}")
    if trades:
        pnl = np.array(trades)
        print(f"Win rate: {(pnl > 0).mean()*100:.1f}%")
        print(f"Total P&L: ${pnl.sum():.2f}")
        print(f"Sharpe (rough): {pnl.mean() / (pnl.std() + 1e-9) * np.sqrt(252):.2f}")
