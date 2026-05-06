"""
high_freq_executor.py — Paper/live order execution for S7.

PAPER mode (default):
  Simulates POST_ONLY fills: a BUY at price P fills when the book's ASK
  drops to <= P. A SELL at P fills when the BID rises to >= P.
  Market closes (stops / max-hold) execute at current mid.

LIVE mode (future):
  Would call Hyperliquid REST/WS API. Stub only — not implemented yet.
  Do not flip paper=False until API keys + reconciliation are tested.

Thread-safety: all state mutations go through self._lock.
"""
import csv
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PendingOrder:
    order_id: str
    symbol: str
    side: str          # "BUY" or "SELL"
    price: float
    size_units: float
    notional_usd: float
    placed_at: float
    quote_pair_id: str  # ties BUY and SELL of the same quote cycle


@dataclass
class OpenPosition:
    pos_id: str
    symbol: str
    side: str           # "BUY" (long) or "SELL" (short)
    size_units: float
    notional_usd: float
    entry_price: float
    tp_price: float
    stop_price: float
    max_hold_until: float
    entry_ts: float
    quote_fair: float   # fair value at entry time


@dataclass
class Fill:
    fill_id: str
    symbol: str
    side: str
    price: float
    size_units: float
    notional_usd: float
    ts: float
    order_id: str
    is_maker: bool = True


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class HighFreqExecutor:
    """
    Manages the full lifecycle of S7 quotes:
      1. place_quotes()   → record two pending limit orders
      2. check_fills()    → detect if book crossed our prices (paper mode)
      3. on_fill()        → cancel opposite leg, set TP+stop, open position
      4. check_exits()    → close positions hitting stop/TP/max_hold
      5. cancel_quotes()  → cancel pending orders for a symbol
    """

    MAKER_REBATE_BPS  = 0.3   # +0.3 bps per maker fill
    TAKER_FEE_BPS     = 3.0   # −3.0 bps for market (stop) close

    def __init__(
        self,
        paper: bool = True,
        trade_log_path: str = "logs/fills_s7.csv",
        on_position_open:  Optional[Callable] = None,
        on_position_close: Optional[Callable] = None,
    ):
        self.paper = paper
        self.trade_log_path = trade_log_path
        self.on_position_open  = on_position_open
        self.on_position_close = on_position_close

        # State
        self._pending: dict[str, PendingOrder] = {}   # order_id → PendingOrder
        self._positions: dict[str, OpenPosition] = {} # pos_id → OpenPosition
        self._quote_pairs: dict[str, list[str]] = {}  # quote_pair_id → [buy_id, sell_id]

        self._lock = threading.Lock()
        self._fills: list[Fill] = []

        Path(trade_log_path).parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def open_positions(self) -> list[OpenPosition]:
        with self._lock:
            return list(self._positions.values())

    @property
    def pending_orders(self) -> list[PendingOrder]:
        with self._lock:
            return list(self._pending.values())

    def symbols_with_pending(self) -> set[str]:
        with self._lock:
            return {o.symbol for o in self._pending.values()}

    def symbols_with_positions(self) -> set[str]:
        with self._lock:
            return {p.symbol for p in self._positions.values()}

    # ------------------------------------------------------------------
    # 1. Place quotes
    # ------------------------------------------------------------------

    def place_quotes(self, quote) -> Optional[str]:
        """
        Place a BUY + SELL maker order pair.
        Returns quote_pair_id or None if skipped.
        """
        if not self.paper:
            raise NotImplementedError("Live execution not implemented yet")

        with self._lock:
            # Only one pending quote per symbol allowed
            existing = [o for o in self._pending.values() if o.symbol == quote.symbol]
            if existing:
                return None

        pair_id = str(uuid.uuid4())[:8]
        buy_id  = f"b_{pair_id}"
        sell_id = f"s_{pair_id}"
        now = time.time()

        buy = PendingOrder(
            order_id=buy_id, symbol=quote.symbol,
            side="BUY", price=quote.buy_price,
            size_units=quote.size_units, notional_usd=quote.notional_usd,
            placed_at=now, quote_pair_id=pair_id,
        )
        sell = PendingOrder(
            order_id=sell_id, symbol=quote.symbol,
            side="SELL", price=quote.sell_price,
            size_units=quote.size_units, notional_usd=quote.notional_usd,
            placed_at=now, quote_pair_id=pair_id,
        )

        with self._lock:
            self._pending[buy_id]  = buy
            self._pending[sell_id] = sell
            self._quote_pairs[pair_id] = [buy_id, sell_id]

        log.debug("[QUOTE] %s pair=%s buy=%.6f sell=%.6f size=%.4f notional=$%.1f",
                  quote.symbol, pair_id, quote.buy_price, quote.sell_price,
                  quote.size_units, quote.notional_usd)
        return pair_id

    # ------------------------------------------------------------------
    # 2. Check fills (called on every orderbook tick in paper mode)
    # ------------------------------------------------------------------

    def check_fills(
        self,
        symbol: str,
        best_bid: float,
        best_ask: float,
        mid: float,
        tp_stop_callback: Callable,   # fn(fill) → TPStopDecision
    ) -> list[Fill]:
        """
        In paper mode: check if any pending orders for `symbol` are crossed.
        A BUY fills when best_ask <= buy_price (maker fill).
        A SELL fills when best_bid >= sell_price (maker fill).
        """
        new_fills = []

        with self._lock:
            orders = [o for o in self._pending.values() if o.symbol == symbol]

        for order in orders:
            filled = False
            fill_price = order.price

            if order.side == "BUY" and best_ask <= order.price:
                filled = True
                fill_price = min(order.price, best_ask)   # realistic fill

            elif order.side == "SELL" and best_bid >= order.price:
                filled = True
                fill_price = max(order.price, best_bid)

            if filled:
                fill = Fill(
                    fill_id=str(uuid.uuid4())[:8],
                    symbol=symbol,
                    side=order.side,
                    price=fill_price,
                    size_units=order.size_units,
                    notional_usd=order.notional_usd,
                    ts=time.time(),
                    order_id=order.order_id,
                    is_maker=True,
                )
                new_fills.append(fill)
                self._on_fill(fill, order, tp_stop_callback)

        return new_fills

    # ------------------------------------------------------------------
    # 3. Check exits (stops / TP / max-hold) — called every 500ms
    # ------------------------------------------------------------------

    def check_exits(
        self,
        mids: dict[str, float],
        best_bids: dict[str, float],
        best_asks: dict[str, float],
    ) -> list[tuple[OpenPosition, float, str]]:
        """
        Returns list of (position, exit_price, reason) for positions that
        should be closed. Caller is responsible for booking PnL.
        """
        now = time.time()
        to_close = []

        with self._lock:
            positions = list(self._positions.values())

        for pos in positions:
            mid = mids.get(pos.symbol)
            if mid is None:
                continue

            bid = best_bids.get(pos.symbol, mid)
            ask = best_asks.get(pos.symbol, mid)

            reason = None
            exit_price = mid

            # Max hold
            if now >= pos.max_hold_until:
                reason = "max_hold"
                exit_price = mid

            # Stop loss (market close)
            elif pos.side == "BUY" and mid <= pos.stop_price:
                reason = "stop_loss"
                exit_price = bid   # simulated market sell at bid

            elif pos.side == "SELL" and mid >= pos.stop_price:
                reason = "stop_loss"
                exit_price = ask   # simulated market buy at ask

            # Take profit (maker — check crossed)
            elif pos.side == "BUY" and ask <= pos.tp_price:
                reason = "take_profit"
                exit_price = pos.tp_price

            elif pos.side == "SELL" and bid >= pos.tp_price:
                reason = "take_profit"
                exit_price = pos.tp_price

            if reason:
                to_close.append((pos, exit_price, reason))

        return to_close

    def close_position(self, pos: OpenPosition, exit_price: float, reason: str) -> float:
        """
        Book the P&L and remove from open positions. Returns net PnL in USD.
        """
        if pos.side == "BUY":
            gross = (exit_price - pos.entry_price) / pos.entry_price * pos.notional_usd
        else:
            gross = (pos.entry_price - exit_price) / pos.entry_price * pos.notional_usd

        is_maker_exit = reason == "take_profit"
        exit_fee_bps  = -self.MAKER_REBATE_BPS if is_maker_exit else self.TAKER_FEE_BPS
        entry_rebate  = -self.MAKER_REBATE_BPS  # always maker entry

        fee = pos.notional_usd * (exit_fee_bps - entry_rebate) / 10_000
        net = gross - fee

        hold_s = time.time() - pos.entry_ts

        log.info(
            "[%s] %s %s size=$%.1f entry=%.6f exit=%.6f | "
            "gross=$%.4f fee=$%.4f net=$%.4f hold=%.0fs reason=%s",
            "PAPER" if self.paper else "LIVE",
            pos.symbol, pos.side, pos.notional_usd,
            pos.entry_price, exit_price,
            gross, fee, net, hold_s, reason,
        )

        with self._lock:
            self._positions.pop(pos.pos_id, None)
            self._fills.append(Fill(
                fill_id=str(uuid.uuid4())[:8],
                symbol=pos.symbol, side="CLOSE_" + pos.side,
                price=exit_price, size_units=pos.size_units,
                notional_usd=pos.notional_usd, ts=time.time(),
                order_id="", is_maker=is_maker_exit,
            ))

        self._log_trade(pos, exit_price, gross, fee, net, reason, hold_s)

        if self.on_position_close:
            self.on_position_close(pos, net, reason)

        return net

    def cancel_quotes(self, symbol: str):
        """Cancel all pending orders for a symbol (e.g. before requoting)."""
        with self._lock:
            to_remove = [oid for oid, o in self._pending.items() if o.symbol == symbol]
            for oid in to_remove:
                self._pending.pop(oid)

    def cancel_all(self):
        """Emergency: cancel everything."""
        with self._lock:
            self._pending.clear()

    def close_all_market(self, mids: dict[str, float], reason: str = "emergency") -> float:
        """Force-close all open positions at mid. Returns total net PnL."""
        with self._lock:
            positions = list(self._positions.values())
        self.cancel_all()
        total = 0.0
        for pos in positions:
            mid = mids.get(pos.symbol, pos.entry_price)
            total += self.close_position(pos, mid, reason)
        return total

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_fill(
        self,
        fill: Fill,
        order: PendingOrder,
        tp_stop_callback: Callable,
    ):
        """Called when a pending order is filled. Opens position, sets TP/stop."""
        pair_id = order.quote_pair_id

        with self._lock:
            # Cancel the opposite leg
            sibling_ids = self._quote_pairs.get(pair_id, [])
            for oid in sibling_ids:
                if oid != order.order_id and oid in self._pending:
                    self._pending.pop(oid)
                    log.debug("Cancelled sibling order %s", oid)

            # Remove filled order
            self._pending.pop(order.order_id, None)
            self._quote_pairs.pop(pair_id, None)

        # Get TP + stop from strategy
        tp_stop = tp_stop_callback(fill)

        pos = OpenPosition(
            pos_id=fill.fill_id,
            symbol=fill.symbol,
            side=fill.side,
            size_units=fill.size_units,
            notional_usd=fill.notional_usd,
            entry_price=fill.price,
            tp_price=tp_stop.tp_price,
            stop_price=tp_stop.stop_price,
            max_hold_until=tp_stop.max_hold_until,
            entry_ts=fill.ts,
            quote_fair=fill.price,
        )

        with self._lock:
            self._positions[pos.pos_id] = pos
            self._fills.append(fill)

        hold_budget = tp_stop.max_hold_until - time.time()
        log.info(
            "[FILL] %s %s @ %.6f | tp=%.6f stop=%.6f max_hold=%.0fs",
            fill.symbol, fill.side, fill.price,
            tp_stop.tp_price, tp_stop.stop_price, hold_budget,
        )

        if self.on_position_open:
            self.on_position_open(pos)

    def _log_trade(self, pos: OpenPosition, exit_price: float,
                   gross: float, fee: float, net: float,
                   reason: str, hold_s: float):
        try:
            write_header = not Path(self.trade_log_path).exists()
            with open(self.trade_log_path, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow([
                        "ts", "symbol", "side", "notional_usd",
                        "entry", "exit", "gross", "fee", "net",
                        "hold_s", "reason",
                    ])
                w.writerow([
                    time.strftime("%Y-%m-%dT%H:%M:%S"),
                    pos.symbol, pos.side, round(pos.notional_usd, 2),
                    round(pos.entry_price, 8), round(exit_price, 8),
                    round(gross, 6), round(fee, 6), round(net, 6),
                    round(hold_s, 1), reason,
                ])
        except Exception as e:
            log.error("Trade log write failed: %s", e)
