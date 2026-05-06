"""
s7_aggressive_maker_scalping.py — Aggressive market-making on Hyperliquid L2.

All logic here is STATELESS: given a book snapshot and parameters, return
decisions. State (open positions, pending orders) lives in the executor.

Edge:
  WR_breakeven = 67.4% (see brief §4.2)
  Target WR: 70-74% for +$1-$4/day on $500 at 8x lever
"""
import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------

@dataclass
class QuoteDecision:
    """Output of compute_quotes(): where to place the two limit orders."""
    symbol: str
    buy_price: float
    sell_price: float
    size_units: float       # in coin units
    notional_usd: float     # buy_price * size_units  (approximate)
    fair_value: float
    skew: float
    spread_bps: float
    imbalance: float


@dataclass
class TPStopDecision:
    """Output of compute_tp_stop(): where to set TP and stop after a fill."""
    tp_price: float
    stop_price: float
    max_hold_until: float   # unix timestamp


# ---------------------------------------------------------------------------
# ATR helper (1m bars)
# ---------------------------------------------------------------------------

def _compute_atr(highs: list[float], lows: list[float],
                 closes: list[float], period: int = 14) -> float:
    """True-range ATR over the last `period` bars."""
    if len(closes) < period + 1:
        # Fallback: use 0.5× typical spread as proxy
        return 0.0

    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)

    if not trs:
        return 0.0
    # Simple MA of last `period` TRs
    return float(np.mean(trs[-period:]))


# ---------------------------------------------------------------------------
# Core S7 logic
# ---------------------------------------------------------------------------

class S7AggressiveMakerScalping:
    """
    Stateless computation module.
    The engine feeds it a book snapshot; it returns quote / TP-stop decisions.
    """

    # ---- Default parameters (override via config_s7.json) ----
    DEFAULT_PARAMS = {
        # Coin filter
        "spread_bps_min":      4.0,    # only quote if spread >= 4 bps
        "spread_bps_max":     15.0,    # don't quote if spread > 15 bps (toxic)
        "volume_5m_min_usd":  100_000, # min $ volume in last 5 min
        "atr_ratio_max":       0.0008, # max ATR/mid (vol guard)

        # Quote placement
        "notional_pct":        0.04,   # 4% of equity per quote ($20 on $500)
        "leverage":            8.0,    # aggressive leverage
        "skew_factor":         0.30,   # 30% of spread as max skew magnitude
        "fair_vwap_weight":    0.40,   # 40% VWAP, 60% mid in fair_value
        "imbalance_levels":    5,      # top-N levels for imbalance computation

        # TP & stop
        "tp_spread_capture":   0.60,   # capture 60% of spread as TP
        "stop_atr_mult":       3.0,    # stop = entry ± 3 × ATR_1m
        "max_hold_s":          60,     # max 60s before market-close

        # Blacklist
        "blacklist_stops":     3,      # 3 consecutive stops → blacklist
        "blacklist_window_s":  300,    # within 5 min
        "blacklist_duration_s":1800,   # blacklist for 30 min
    }

    def __init__(self, params: Optional[dict] = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}

    # ------------------------------------------------------------------
    # 1. Tradability filter
    # ------------------------------------------------------------------

    def is_tradable(
        self,
        symbol: str,
        best_bid: float,
        best_ask: float,
        volume_5m_usd: float,
        atr_1m: float,
        blacklist: dict[str, float],
    ) -> tuple[bool, str]:
        """
        Returns (True, "") if we should quote this coin right now.
        Returns (False, reason) if not tradable.
        """
        p = self.params

        # Blacklist check
        if symbol in blacklist and time.time() < blacklist[symbol]:
            remaining = blacklist[symbol] - time.time()
            return False, f"blacklisted ({remaining:.0f}s)"

        if best_bid <= 0 or best_ask <= 0:
            return False, "no book"

        mid = (best_bid + best_ask) / 2
        if mid <= 0:
            return False, "zero mid"

        spread_bps = (best_ask - best_bid) / mid * 10_000
        if spread_bps < p["spread_bps_min"]:
            return False, f"spread {spread_bps:.1f}bps < {p['spread_bps_min']}"
        if spread_bps > p["spread_bps_max"]:
            return False, f"spread {spread_bps:.1f}bps > {p['spread_bps_max']}"

        if volume_5m_usd < p["volume_5m_min_usd"]:
            return False, f"vol {volume_5m_usd/1e3:.0f}k < {p['volume_5m_min_usd']/1e3:.0f}k"

        if atr_1m > 0 and (atr_1m / mid) > p["atr_ratio_max"]:
            return False, f"ATR/mid {atr_1m/mid*1e4:.1f} > {p['atr_ratio_max']*1e4:.1f} (vol too high)"

        return True, ""

    # ------------------------------------------------------------------
    # 2. Fair value
    # ------------------------------------------------------------------

    def compute_fair_value(
        self,
        best_bid: float,
        best_ask: float,
        trades_30s: list,         # list of Trade objects with .price and .size
    ) -> float:
        """
        Weighted average of mid price (60%) and VWAP_30s (40%).
        Falls back to mid if no recent trades.
        """
        mid = (best_bid + best_ask) / 2
        if not trades_30s:
            return mid

        total_value = sum(t.price * t.size for t in trades_30s)
        total_size  = sum(t.size for t in trades_30s)
        if total_size <= 0:
            return mid

        vwap = total_value / total_size
        w    = self.params["fair_vwap_weight"]
        return (1 - w) * mid + w * vwap

    # ------------------------------------------------------------------
    # 3. Imbalance skew
    # ------------------------------------------------------------------

    def compute_skew(
        self,
        best_bid: float,
        best_ask: float,
        imbalance: float,        # from OrderBook.imbalance()
    ) -> float:
        """
        Skew our quotes toward the imbalance direction.
        imbalance = +1 → strong buy pressure → both prices skewed up
        skew capped at 30% of raw spread
        """
        spread = best_ask - best_bid
        return imbalance * spread * self.params["skew_factor"]

    # ------------------------------------------------------------------
    # 4. Quote decision
    # ------------------------------------------------------------------

    def compute_quotes(
        self,
        symbol: str,
        best_bid: float,
        best_ask: float,
        imbalance: float,
        trades_30s: list,
        equity: float,
    ) -> Optional[QuoteDecision]:
        """
        Compute the two limit order prices and sizes.
        Returns None if prices would cross the book (POST_ONLY would reject).
        """
        p = self.params

        fair_value = self.compute_fair_value(best_bid, best_ask, trades_30s)
        skew       = self.compute_skew(best_bid, best_ask, imbalance)
        spread     = best_ask - best_bid

        buy_price  = fair_value - spread / 2 + skew
        sell_price = fair_value + spread / 2 + skew

        # POST_ONLY guard: our BUY must be strictly below best_ask
        #                  our SELL must be strictly above best_bid
        if buy_price  >= best_ask:
            buy_price  = best_ask - best_ask * 0.00005   # pull back 0.5 bps
        if sell_price <= best_bid:
            sell_price = best_bid + best_bid * 0.00005

        # Both quotes must still be on their correct side
        if buy_price >= sell_price:
            return None   # degenerate book, skip

        notional_usd  = equity * p["notional_pct"] * p["leverage"]
        size_units    = notional_usd / max(fair_value, 1e-9)

        mid = (best_bid + best_ask) / 2
        spread_bps = (best_ask - best_bid) / max(mid, 1e-9) * 10_000

        return QuoteDecision(
            symbol=symbol,
            buy_price=round(buy_price, 8),
            sell_price=round(sell_price, 8),
            size_units=size_units,
            notional_usd=notional_usd,
            fair_value=fair_value,
            skew=skew,
            spread_bps=spread_bps,
            imbalance=imbalance,
        )

    # ------------------------------------------------------------------
    # 5. TP & stop after fill
    # ------------------------------------------------------------------

    def compute_tp_stop(
        self,
        entry_price: float,
        filled_side: str,     # "BUY" or "SELL"
        atr_1m: float,
        fair_value: float,
    ) -> TPStopDecision:
        """
        Compute TP (maker limit) and stop (market trigger) after a fill.
        """
        p = self.params

        # TP: capture 60% of spread from the fair value
        spread_est = atr_1m * 0.5 if atr_1m > 0 else fair_value * 0.0006

        if filled_side == "BUY":
            tp_price   = entry_price + spread_est * p["tp_spread_capture"]
            stop_price = entry_price - p["stop_atr_mult"] * atr_1m if atr_1m > 0 \
                         else entry_price * (1 - 0.003)
        else:  # SELL
            tp_price   = entry_price - spread_est * p["tp_spread_capture"]
            stop_price = entry_price + p["stop_atr_mult"] * atr_1m if atr_1m > 0 \
                         else entry_price * (1 + 0.003)

        max_hold_until = time.time() + p["max_hold_s"]

        return TPStopDecision(
            tp_price=round(tp_price, 8),
            stop_price=round(stop_price, 8),
            max_hold_until=max_hold_until,
        )

    # ------------------------------------------------------------------
    # 6. ATR helper (wraps module-level fn)
    # ------------------------------------------------------------------

    def atr_from_candles(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        period: int = 14,
    ) -> float:
        return _compute_atr(highs, lows, closes, period)

    # ------------------------------------------------------------------
    # 7. Blacklist update
    # ------------------------------------------------------------------

    def check_blacklist(
        self,
        symbol: str,
        blacklist: dict[str, float],
        recent_stops: dict[str, list[float]],
    ) -> dict[str, float]:
        """
        Add `symbol` to a new stop record. If >= blacklist_stops in the
        window → set blacklist[symbol].
        Returns updated blacklist dict.
        """
        p = self.params
        now = time.time()

        if symbol not in recent_stops:
            recent_stops[symbol] = []

        recent_stops[symbol].append(now)
        # Prune old stops outside the window
        recent_stops[symbol] = [
            t for t in recent_stops[symbol]
            if now - t < p["blacklist_window_s"]
        ]

        if len(recent_stops[symbol]) >= p["blacklist_stops"]:
            until = now + p["blacklist_duration_s"]
            blacklist[symbol] = until
            log.warning(
                "BLACKLIST %s: %d stops in %ds → paused %.0fs",
                symbol, len(recent_stops[symbol]),
                p["blacklist_window_s"], p["blacklist_duration_s"],
            )
            recent_stops[symbol].clear()

        return blacklist
