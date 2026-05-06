"""
orderbook_manager.py — Live L2 orderbook via Hyperliquid WebSocket.

Maintains a real-time L2 book + rolling 30s trade tape for up to 16 symbols.
Thread-safe reads via property access; WebSocket runs on a dedicated asyncio loop
in a background thread so the rest of the engine stays synchronous.

Reconnects automatically with exponential back-off (1s → 64s).
"""
import asyncio
import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

try:
    import websockets
except ImportError:
    raise ImportError("pip install websockets>=11")

log = logging.getLogger(__name__)

HL_WS_URL = "wss://api.hyperliquid.xyz/ws"
PING_INTERVAL = 20      # keepalive (seconds)
MAX_TRADES_AGE = 60     # keep trades up to 60s


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Level:
    price: float
    size: float
    n_orders: int = 1


@dataclass
class OrderBook:
    coin: str
    bids: list = field(default_factory=list)   # list[Level], best first
    asks: list = field(default_factory=list)   # list[Level], best first
    ts: float = 0.0
    seq: int = 0

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> Optional[float]:
        b, a = self.best_bid, self.best_ask
        return (b + a) / 2 if b and a else None

    @property
    def spread_bps(self) -> Optional[float]:
        m = self.mid
        b, a = self.best_bid, self.best_ask
        if m and b and a and m > 0:
            return (a - b) / m * 10_000
        return None

    def bid_depth(self, levels: int = 5) -> float:
        return sum(lv.size for lv in self.bids[:levels])

    def ask_depth(self, levels: int = 5) -> float:
        return sum(lv.size for lv in self.asks[:levels])

    def imbalance(self, levels: int = 5) -> float:
        """Order book imbalance [-1, +1]. +1 = all bids, -1 = all asks."""
        bd = self.bid_depth(levels)
        ad = self.ask_depth(levels)
        total = bd + ad
        if total == 0:
            return 0.0
        return (bd - ad) / total


@dataclass
class Trade:
    price: float
    size: float
    side: str    # "B" (buy aggressor) or "A" (sell aggressor)
    ts: float


# ---------------------------------------------------------------------------
# OrderbookManager
# ---------------------------------------------------------------------------

class OrderbookManager:
    """
    Manages live L2 orderbooks for a list of coins via Hyperliquid WebSocket.

    Usage:
        obm = OrderbookManager(["BTC", "ETH", "SOL"])
        obm.start()
        ...
        book = obm.get_book("BTC")   # always safe, may be empty if not yet received
        trades = obm.get_trades("BTC", seconds=30)
        obm.stop()
    """

    def __init__(self, symbols: list[str]):
        self.symbols = [s.upper() for s in symbols]
        self._books: dict[str, OrderBook] = {s: OrderBook(coin=s) for s in self.symbols}
        self._trades: dict[str, deque] = {s: deque() for s in self.symbols}
        self._lock = threading.Lock()

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Connection health
        self.connected = False
        self.reconnections = 0
        self.last_message_ts: float = 0.0

    # ------------------------------------------------------------------
    # Public API (thread-safe)
    # ------------------------------------------------------------------

    def start(self):
        """Start WebSocket thread."""
        self._running = True
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="ws-orderbook"
        )
        self._thread.start()
        log.info("OrderbookManager started for %s", self.symbols)

    def stop(self):
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

    def get_book(self, symbol: str) -> OrderBook:
        with self._lock:
            return self._books.get(symbol.upper(), OrderBook(coin=symbol))

    def get_trades(self, symbol: str, seconds: float = 30) -> list[Trade]:
        cutoff = time.time() - seconds
        with self._lock:
            dq = self._trades.get(symbol.upper(), deque())
            return [t for t in dq if t.ts >= cutoff]

    def is_stale(self, symbol: str, max_age_s: float = 5.0) -> bool:
        with self._lock:
            book = self._books.get(symbol.upper())
            if not book or book.ts == 0:
                return True
            return (time.time() - book.ts) > max_age_s

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._ws_loop())

    async def _ws_loop(self):
        backoff = 1
        while self._running:
            try:
                async with websockets.connect(
                    HL_WS_URL,
                    ping_interval=PING_INTERVAL,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self.connected = True
                    self.reconnections += 1 if backoff > 1 else 0
                    backoff = 1
                    log.info("WebSocket connected (reconnections=%d)", self.reconnections)

                    # Subscribe to l2Book + trades for every symbol
                    for sym in self.symbols:
                        await ws.send(json.dumps({
                            "method": "subscribe",
                            "subscription": {"type": "l2Book", "coin": sym}
                        }))
                        await ws.send(json.dumps({
                            "method": "subscribe",
                            "subscription": {"type": "trades", "coin": sym}
                        }))

                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            self._handle_message(json.loads(raw))
                        except Exception as e:
                            log.debug("Message parse error: %s", e)

            except Exception as e:
                self.connected = False
                log.warning("WebSocket error (%s), retrying in %ds", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 64)

    def _handle_message(self, msg: dict):
        self.last_message_ts = time.time()
        channel = msg.get("channel", "")
        data = msg.get("data", {})

        if channel == "l2Book":
            self._update_book(data)
        elif channel == "trades":
            self._update_trades(data)

    def _update_book(self, data: dict):
        coin = data.get("coin", "")
        if coin not in self._books:
            return

        levels_raw = data.get("levels", [[], []])
        ts = data.get("time", time.time() * 1000) / 1000

        def parse(raw_list) -> list[Level]:
            out = []
            for lv in raw_list:
                try:
                    out.append(Level(
                        price=float(lv["px"]),
                        size=float(lv["sz"]),
                        n_orders=int(lv.get("n", 1)),
                    ))
                except (KeyError, ValueError):
                    pass
            return out

        bids = parse(levels_raw[0] if len(levels_raw) > 0 else [])
        asks = parse(levels_raw[1] if len(levels_raw) > 1 else [])

        # HL sends bids in descending order, asks in ascending — verify/enforce
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)

        with self._lock:
            book = self._books[coin]
            book.bids = bids
            book.asks = asks
            book.ts = ts
            book.seq += 1

    def _update_trades(self, data):
        # data is a list of trade dicts
        if not isinstance(data, list):
            data = [data]
        now = time.time()

        with self._lock:
            for t in data:
                coin = t.get("coin", "")
                if coin not in self._trades:
                    continue
                try:
                    trade = Trade(
                        price=float(t["px"]),
                        size=float(t["sz"]),
                        side=t.get("side", "B"),
                        ts=float(t.get("time", now * 1000)) / 1000,
                    )
                    self._trades[coin].append(trade)
                    # Prune old trades
                    cutoff = now - MAX_TRADES_AGE
                    while self._trades[coin] and self._trades[coin][0].ts < cutoff:
                        self._trades[coin].popleft()
                except (KeyError, ValueError):
                    pass


# ---------------------------------------------------------------------------
# Minimal test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    obm = OrderbookManager(["BTC", "ETH"])
    obm.start()
    try:
        for _ in range(20):
            time.sleep(1)
            btc = obm.get_book("BTC")
            trades = obm.get_trades("BTC", seconds=10)
            if btc.mid:
                print(f"BTC mid={btc.mid:.1f}  spread={btc.spread_bps:.2f}bps  "
                      f"imb={btc.imbalance():.3f}  trades_10s={len(trades)}")
    finally:
        obm.stop()
