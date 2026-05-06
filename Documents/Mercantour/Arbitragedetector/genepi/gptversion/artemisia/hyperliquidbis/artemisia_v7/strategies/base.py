"""
base.py — Abstract Strategy base class for Artemisia v7
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Optional


class Signal:
    """A trading signal emitted by a strategy."""
    __slots__ = ("symbol", "side", "size_usd", "entry_type",
                 "stop_price", "tp_price", "confidence", "meta")

    def __init__(self,
                 symbol: str,
                 side: str,          # "long" | "short" | "close"
                 size_usd: float,
                 entry_type: str = "maker",  # "maker" | "taker"
                 stop_price: Optional[float] = None,
                 tp_price: Optional[float] = None,
                 confidence: float = 1.0,
                 meta: Optional[dict] = None):
        self.symbol    = symbol
        self.side      = side
        self.size_usd  = size_usd
        self.entry_type = entry_type
        self.stop_price = stop_price
        self.tp_price   = tp_price
        self.confidence = confidence
        self.meta       = meta or {}

    def __repr__(self):
        return (f"Signal({self.symbol} {self.side} ${self.size_usd:.0f} "
                f"stop={self.stop_price} tp={self.tp_price})")


class BaseStrategy(ABC):
    """
    Abstract base for all v7 strategies.
    Each strategy must implement:
      - NAME: str
      - PARAM_GRID: list[dict]
      - DEFAULT_PARAMS: dict
      - backtest(candles, params) -> list[float]  (per-trade P&L in USD)
      - signal(candles, funding_df, params, equity, open_positions) -> list[Signal]
    """

    NAME: str = "base"
    PARAM_GRID: list[dict] = []
    DEFAULT_PARAMS: dict = {}

    def __init__(self, params: Optional[dict] = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}

    @abstractmethod
    def backtest(self,
                 candles: pd.DataFrame,
                 params: Optional[dict] = None) -> list[float]:
        """
        Run backtest on candle slice. Returns list of per-trade P&L in USD.
        Used by walk_forward.py.

        Args:
            candles: DataFrame [ts, open, high, low, close, volume]
            params: Override parameters (uses self.params if None)

        Returns:
            List of trade P&Ls in USD (positive = win, negative = loss)
        """
        ...

    @abstractmethod
    def signal(self,
               candles: pd.DataFrame,
               funding_df: Optional[pd.DataFrame],
               params: Optional[dict],
               equity: float,
               open_positions: list[dict]) -> list[Signal]:
        """
        Generate live trading signals from current market data.

        Args:
            candles: Recent candles (enough for indicator warmup)
            funding_df: Recent funding history (may be None)
            params: Strategy params
            equity: Current account equity in USD
            open_positions: List of currently open positions

        Returns:
            List of Signal objects (may be empty)
        """
        ...

    def _apply_fees(self,
                    gross_pnl: float,
                    position_size_usd: float,
                    order_type: str = "maker") -> float:
        """Subtract realistic round-trip fees from gross P&L."""
        if order_type == "maker":
            roundtrip_bps = 4.8  # -0.3 rebate * 2 + 0.8 slip * 2 + 0.3 taker exit * 2
        else:
            roundtrip_bps = 6.6  # 2.5 taker * 2 + 0.8 slip * 2
        fee = position_size_usd * roundtrip_bps / 10_000
        return gross_pnl - fee

    def _atr(self, highs: np.ndarray, lows: np.ndarray,
             closes: np.ndarray, period: int = 14) -> np.ndarray:
        """ATR via EWM."""
        tr = np.maximum(highs - lows,
               np.maximum(abs(highs - np.roll(closes, 1)),
                          abs(lows  - np.roll(closes, 1))))
        tr[0] = highs[0] - lows[0]
        return pd.Series(tr).ewm(span=period, adjust=False).mean().values

    def _ema(self, values: np.ndarray, period: int) -> np.ndarray:
        return pd.Series(values).ewm(span=period, adjust=False).mean().values

    def _rsi(self, closes: np.ndarray, period: int = 14) -> np.ndarray:
        s = pd.Series(closes)
        delta = s.diff()
        gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
        loss = (-delta).clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
        rs = gain / loss.replace(0, np.nan)
        return (100 - 100 / (1 + rs)).fillna(50).values

    def _bollinger(self, closes: np.ndarray, period: int = 20,
                   n_std: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (upper, mid, lower) Bollinger Bands."""
        s = pd.Series(closes)
        mid   = s.rolling(period, min_periods=1).mean().values
        std   = s.rolling(period, min_periods=1).std(ddof=1).fillna(0).values
        upper = mid + n_std * std
        lower = mid - n_std * std
        return upper, mid, lower
