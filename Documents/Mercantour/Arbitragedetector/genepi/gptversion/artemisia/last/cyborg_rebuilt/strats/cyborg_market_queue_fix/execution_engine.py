"""
execution_engine.py – Real & paper execution layer for CYBORG / Coin5min

Modes:
    "paper"  → orders are logged but never sent to Polymarket
    "live"   → orders are signed and posted via py-clob-client

Safety:
    - max_order_usd hard cap (default $5)
    - kill-switch flag
    - every live call wrapped in try/except with alert
    - cancel_all on shutdown / panic
    - order journal persisted to SQLite

Requires:
    pip install py-clob-client python-dotenv

Environment (see .env.polymarket.example):
    POLYMARKET_PRIVATE_KEY
    POLYMARKET_FUNDER           (proxy wallet address shown on polymarket.com profile)
    POLYMARKET_SIGNATURE_TYPE   (0=EOA, 1=email/Magic, 2=browser wallet proxy)
    POLYMARKET_CHAIN_ID         (137 = Polygon mainnet)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("cyborg.execution")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLOB_HOST = "https://clob.polymarket.com"
DEFAULT_CHAIN_ID = 137
ORDER_DB = "orders_live.db"

# Hard safety cap – never exceeded even if strategy asks for more
ABSOLUTE_MAX_ORDER_USD = 50.0


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ExecMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class OrderState(str, Enum):
    PENDING = "PENDING"        # created locally, not yet sent
    POSTING = "POSTING"        # sent to CLOB, awaiting ack
    OPEN = "OPEN"              # resting on book
    PARTIAL = "PARTIAL"        # partially filled
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"
    EXPIRED = "EXPIRED"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class OrderRecord:
    client_order_id: str
    clob_order_id: Optional[str] = None
    symbol: str = ""
    token_id: str = ""
    side: str = "BUY"          # BUY or SELL
    price: float = 0.0
    size: float = 0.0          # in USD for BUY, in shares for SELL
    order_type: str = "GTC"    # GTC, GTD, FOK
    state: str = OrderState.PENDING
    fill_price: Optional[float] = None
    fill_size: Optional[float] = None
    created_ts: float = 0.0
    updated_ts: float = 0.0
    tags: Dict[str, str] = field(default_factory=dict)
    error_msg: str = ""
    raw_response: str = ""
    mode: str = "paper"

    def to_dict(self) -> Dict:
        return {
            "client_order_id": self.client_order_id,
            "clob_order_id": self.clob_order_id,
            "symbol": self.symbol,
            "token_id": self.token_id,
            "side": self.side,
            "price": self.price,
            "size": self.size,
            "order_type": self.order_type,
            "state": self.state,
            "fill_price": self.fill_price,
            "fill_size": self.fill_size,
            "created_ts": self.created_ts,
            "updated_ts": self.updated_ts,
            "tags": self.tags,
            "error_msg": self.error_msg,
            "mode": self.mode,
        }


@dataclass
class Alert:
    level: str          # INFO, WARN, ERROR, CRIT
    code: str           # ORDER_PLACED, FILL, PARTIAL, REJECTED, ERROR, CANCEL, SAFETY
    message: str
    order_id: str = ""
    symbol: str = ""
    sound_hint: str = ""  # fill, partial, error, cancel
    ts: float = 0.0


# ---------------------------------------------------------------------------
# Order journal (SQLite)
# ---------------------------------------------------------------------------

def _init_order_db(db_path: str = ORDER_DB):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS order_journal (
            client_order_id TEXT PRIMARY KEY,
            clob_order_id   TEXT,
            symbol          TEXT,
            token_id        TEXT,
            side            TEXT,
            price           REAL,
            size            REAL,
            order_type      TEXT,
            state           TEXT,
            fill_price      REAL,
            fill_size       REAL,
            created_ts      REAL,
            updated_ts      REAL,
            tags            TEXT,
            error_msg       TEXT,
            raw_response    TEXT,
            mode            TEXT
        )
    """)
    conn.commit()
    return conn


def _persist_order(conn: sqlite3.Connection, o: OrderRecord):
    conn.execute("""
        INSERT OR REPLACE INTO order_journal
        (client_order_id, clob_order_id, symbol, token_id, side, price, size,
         order_type, state, fill_price, fill_size, created_ts, updated_ts,
         tags, error_msg, raw_response, mode)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        o.client_order_id, o.clob_order_id, o.symbol, o.token_id,
        o.side, o.price, o.size, o.order_type, o.state,
        o.fill_price, o.fill_size, o.created_ts, o.updated_ts,
        json.dumps(o.tags), o.error_msg, o.raw_response, o.mode,
    ))
    conn.commit()


# ---------------------------------------------------------------------------
# Execution Engine
# ---------------------------------------------------------------------------

class ExecutionEngine:
    """
    Unified execution interface.

    In PAPER mode  → orders are logged locally, never touch Polymarket.
    In LIVE mode   → orders are signed + posted via py-clob-client.
    """

    def __init__(
        self,
        mode: ExecMode = ExecMode.PAPER,
        private_key: str = "",
        funder: str = "",
        signature_type: int = 2,
        chain_id: int = DEFAULT_CHAIN_ID,
        max_order_usd: float = 5.0,
        db_path: str = ORDER_DB,
    ):
        self.mode = ExecMode(mode)
        self._private_key = private_key
        self._funder = funder
        self._signature_type = signature_type
        self._chain_id = chain_id
        self.max_order_usd = min(float(max_order_usd), ABSOLUTE_MAX_ORDER_USD)

        self._lock = threading.RLock()
        self._client = None          # ClobClient (lazy)
        self._connected = False
        self._killed = False         # emergency kill switch

        self._orders: Dict[str, OrderRecord] = {}
        self._alerts: List[Alert] = []

        self._db = _init_order_db(db_path)

        logger.info("ExecutionEngine created | mode=%s | max_order=$%.2f", self.mode, self.max_order_usd)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Initialise the CLOB client and derive API creds.  Paper mode always succeeds."""
        if self.mode == ExecMode.PAPER:
            self._connected = True
            self._emit_alert("INFO", "CONNECT", "Paper execution engine ready", sound_hint="")
            return True

        try:
            from py_clob_client.client import ClobClient
            self._client = ClobClient(
                CLOB_HOST,
                key=self._private_key,
                chain_id=self._chain_id,
                signature_type=self._signature_type,
                funder=self._funder,
            )
            # derive or reuse API credentials
            creds = self._client.create_or_derive_api_creds()
            self._client.set_api_creds(creds)

            # quick health-check
            ok = self._client.get_ok()
            if ok:
                self._connected = True
                self._emit_alert("INFO", "CONNECT", "Live CLOB connection OK", sound_hint="fill")
                logger.info("CLOB connection established | funder=%s", self._funder)
                return True
            else:
                self._emit_alert("ERROR", "CONNECT", "CLOB health-check failed", sound_hint="error")
                return False
        except Exception as exc:
            self._emit_alert("CRIT", "CONNECT", f"Connection failed: {exc}", sound_hint="error")
            logger.exception("CLOB connect error")
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected and not self._killed

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def kill(self):
        """Emergency: cancel all open orders and block further trading."""
        self._killed = True
        self._emit_alert("CRIT", "SAFETY", "KILL SWITCH activated – all orders cancelled", sound_hint="error")
        self.cancel_all_orders()

    def unkill(self):
        """Re-enable trading after review."""
        self._killed = False
        self._emit_alert("INFO", "SAFETY", "Kill switch released", sound_hint="fill")

    # ------------------------------------------------------------------
    # Place limit order  (main entry point for data_manager)
    # ------------------------------------------------------------------

    def place_limit_order(
        self,
        symbol: str,
        token_id: str,
        side: str,           # "BUY" or "SELL"
        price: float,
        size: float,         # USD amount for BUY
        order_type: str = "GTC",
        tags: Optional[Dict[str, str]] = None,
    ) -> OrderRecord:
        """
        Place a limit order.

        Returns an OrderRecord immediately.
        In LIVE mode the order is signed + posted to CLOB.
        In PAPER mode it is recorded locally as FILLED at the requested price.
        """
        cid = f"CYB-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}"
        now = time.time()

        order = OrderRecord(
            client_order_id=cid,
            symbol=symbol,
            token_id=token_id,
            side=side.upper(),
            price=round(float(price), 4),
            size=round(float(size), 4),
            order_type=order_type,
            state=OrderState.PENDING,
            created_ts=now,
            updated_ts=now,
            tags=tags or {},
            mode=self.mode.value,
        )

        # ---- safety checks ----
        if self._killed:
            order.state = OrderState.REJECTED
            order.error_msg = "Kill switch active"
            self._save(order)
            self._emit_alert("WARN", "REJECTED", f"Order blocked (kill switch) {symbol} {side}", symbol=symbol, order_id=cid, sound_hint="error")
            return order

        if float(size) > self.max_order_usd:
            order.state = OrderState.REJECTED
            order.error_msg = f"Size ${size:.2f} exceeds max ${self.max_order_usd:.2f}"
            self._save(order)
            self._emit_alert("WARN", "REJECTED", order.error_msg, symbol=symbol, order_id=cid, sound_hint="error")
            return order

        if float(price) <= 0 or float(price) >= 1.0:
            order.state = OrderState.REJECTED
            order.error_msg = f"Price {price} out of valid range (0, 1)"
            self._save(order)
            self._emit_alert("WARN", "REJECTED", order.error_msg, symbol=symbol, order_id=cid, sound_hint="error")
            return order

        if not token_id:
            order.state = OrderState.REJECTED
            order.error_msg = "Missing token_id"
            self._save(order)
            return order

        # ---- paper mode ----
        if self.mode == ExecMode.PAPER:
            order.state = OrderState.FILLED
            order.fill_price = order.price
            order.fill_size = order.size
            order.updated_ts = time.time()
            self._save(order)
            self._emit_alert("INFO", "FILL", f"[PAPER] {side} {symbol} {size:.2f}$ @ {price:.4f}", symbol=symbol, order_id=cid, sound_hint="fill")
            return order

        # ---- live mode ----
        if not self._connected or self._client is None:
            order.state = OrderState.ERROR
            order.error_msg = "Not connected to CLOB"
            self._save(order)
            self._emit_alert("ERROR", "ERROR", "Cannot place order: not connected", symbol=symbol, order_id=cid, sound_hint="error")
            return order

        order.state = OrderState.POSTING
        self._save(order)

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY, SELL

            clob_side = BUY if side.upper() == "BUY" else SELL

            # Polymarket sizes: for BUY, size = number of shares (size_usd / price)
            shares = round(size / price, 2) if price > 0 else 0.0

            args = OrderArgs(
                token_id=token_id,
                price=round(price, 2),    # Polymarket prices have 2 decimal precision
                size=round(shares, 2),
                side=clob_side,
            )

            ot_map = {
                "GTC": OrderType.GTC,
                "FOK": OrderType.FOK,
            }
            clob_order_type = ot_map.get(order_type, OrderType.GTC)

            signed = self._client.create_order(args)
            resp = self._client.post_order(signed, clob_order_type)

            order.raw_response = json.dumps(resp) if isinstance(resp, dict) else str(resp)

            if isinstance(resp, dict) and resp.get("success"):
                order.clob_order_id = resp.get("orderID", "")
                order.state = OrderState.OPEN
                order.updated_ts = time.time()
                self._save(order)
                self._emit_alert(
                    "INFO", "ORDER_PLACED",
                    f"[LIVE] {side} {symbol} {shares:.2f} shares @ {price:.4f} → {order.clob_order_id[:12]}…",
                    symbol=symbol, order_id=cid, sound_hint="fill",
                )
            else:
                err = resp.get("errorMsg", "unknown") if isinstance(resp, dict) else str(resp)
                order.state = OrderState.REJECTED
                order.error_msg = err
                order.updated_ts = time.time()
                self._save(order)
                self._emit_alert("ERROR", "REJECTED", f"Order rejected: {err}", symbol=symbol, order_id=cid, sound_hint="error")

        except Exception as exc:
            order.state = OrderState.ERROR
            order.error_msg = str(exc)
            order.updated_ts = time.time()
            self._save(order)
            self._emit_alert("CRIT", "ERROR", f"Order exception: {exc}", symbol=symbol, order_id=cid, sound_hint="error")
            logger.exception("place_limit_order error")

        return order

    # ------------------------------------------------------------------
    # Refresh order state from CLOB
    # ------------------------------------------------------------------

    def refresh_order(self, client_order_id: str) -> Optional[OrderRecord]:
        """Query CLOB for latest state of an order."""
        with self._lock:
            order = self._orders.get(client_order_id)
        if not order or not order.clob_order_id:
            return order
        if self.mode == ExecMode.PAPER:
            return order

        try:
            resp = self._client.get_order(order.clob_order_id)
            if isinstance(resp, dict):
                clob_state = str(resp.get("status", "")).upper()
                state_map = {
                    "LIVE": OrderState.OPEN,
                    "OPEN": OrderState.OPEN,
                    "MATCHED": OrderState.FILLED,
                    "FILLED": OrderState.FILLED,
                    "CANCELLED": OrderState.CANCELLED,
                    "DELAYED": OrderState.POSTING,
                }
                new_state = state_map.get(clob_state, order.state)

                # detect partial fills
                size_matched = float(resp.get("size_matched", 0) or 0)
                original_size = float(resp.get("original_size", order.size) or order.size)
                if size_matched > 0 and new_state == OrderState.OPEN:
                    new_state = OrderState.PARTIAL

                if new_state != order.state:
                    old_state = order.state
                    order.state = new_state
                    order.updated_ts = time.time()
                    if size_matched > 0:
                        order.fill_size = size_matched
                        avg_price = float(resp.get("associate_trades", [{}])[0].get("price", order.price)) if resp.get("associate_trades") else order.price
                        order.fill_price = avg_price
                    self._save(order)

                    if new_state == OrderState.FILLED:
                        self._emit_alert("INFO", "FILL", f"Order FILLED {order.symbol} {order.side}", symbol=order.symbol, order_id=client_order_id, sound_hint="fill")
                    elif new_state == OrderState.PARTIAL:
                        self._emit_alert("WARN", "PARTIAL", f"Partial fill {order.symbol} {size_matched}/{original_size}", symbol=order.symbol, order_id=client_order_id, sound_hint="partial")
                    elif new_state == OrderState.CANCELLED:
                        self._emit_alert("INFO", "CANCEL", f"Order cancelled {order.symbol}", symbol=order.symbol, order_id=client_order_id, sound_hint="cancel")

        except Exception as exc:
            logger.warning("refresh_order error for %s: %s", client_order_id, exc)

        return order

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    def cancel_order(self, client_order_id: str) -> bool:
        with self._lock:
            order = self._orders.get(client_order_id)
        if not order:
            return False
        if self.mode == ExecMode.PAPER:
            order.state = OrderState.CANCELLED
            order.updated_ts = time.time()
            self._save(order)
            return True

        if not order.clob_order_id:
            order.state = OrderState.CANCELLED
            order.updated_ts = time.time()
            self._save(order)
            return True

        try:
            self._client.cancel(order_id=order.clob_order_id)
            order.state = OrderState.CANCELLED
            order.updated_ts = time.time()
            self._save(order)
            self._emit_alert("INFO", "CANCEL", f"Cancelled {order.symbol} {order.side}", symbol=order.symbol, order_id=client_order_id, sound_hint="cancel")
            return True
        except Exception as exc:
            logger.warning("cancel_order error: %s", exc)
            self._emit_alert("ERROR", "ERROR", f"Cancel failed: {exc}", symbol=order.symbol, order_id=client_order_id, sound_hint="error")
            return False

    def cancel_all_orders(self):
        """Cancel every open/pending order."""
        if self.mode == ExecMode.LIVE and self._client:
            try:
                self._client.cancel_all()
                logger.info("cancel_all sent to CLOB")
            except Exception as exc:
                logger.warning("cancel_all error: %s", exc)

        with self._lock:
            for o in self._orders.values():
                if o.state in (OrderState.OPEN, OrderState.PENDING, OrderState.POSTING, OrderState.PARTIAL):
                    o.state = OrderState.CANCELLED
                    o.updated_ts = time.time()
                    _persist_order(self._db, o)
        self._emit_alert("INFO", "CANCEL", "All orders cancelled", sound_hint="cancel")

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def list_orders(self, states: Optional[List[str]] = None) -> List[Dict]:
        """Return orders, optionally filtered by state."""
        with self._lock:
            orders = list(self._orders.values())
        if states:
            states_set = set(s.upper() for s in states)
            orders = [o for o in orders if o.state in states_set]
        return [o.to_dict() for o in orders]

    def get_order(self, client_order_id: str) -> Optional[Dict]:
        with self._lock:
            o = self._orders.get(client_order_id)
        return o.to_dict() if o else None

    def get_alerts(self, limit: int = 50) -> List[Dict]:
        with self._lock:
            return [
                {
                    "level": a.level,
                    "code": a.code,
                    "message": a.message,
                    "order_id": a.order_id,
                    "symbol": a.symbol,
                    "sound_hint": a.sound_hint,
                    "ts": a.ts,
                    "time": datetime.fromtimestamp(a.ts).strftime("%H:%M:%S") if a.ts else "",
                }
                for a in self._alerts[:limit]
            ]

    def get_balance(self) -> Optional[float]:
        """Fetch USDC balance from CLOB (live mode only)."""
        if self.mode == ExecMode.PAPER or not self._client:
            return None
        try:
            from py_clob_client.clob_types import AssetType
            bal = self._client.get_balance_allowance(asset_type=AssetType.COLLATERAL)
            if isinstance(bal, dict):
                # balance is in wei (6 decimals for USDC)
                raw = float(bal.get("balance", 0))
                return raw / 1e6
            return None
        except Exception as exc:
            logger.warning("get_balance error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save(self, order: OrderRecord):
        with self._lock:
            self._orders[order.client_order_id] = order
            _persist_order(self._db, order)

    def _emit_alert(self, level: str, code: str, message: str, symbol: str = "", order_id: str = "", sound_hint: str = ""):
        a = Alert(
            level=level,
            code=code,
            message=message,
            order_id=order_id,
            symbol=symbol,
            sound_hint=sound_hint,
            ts=time.time(),
        )
        with self._lock:
            self._alerts.insert(0, a)
            self._alerts = self._alerts[:200]
        logger.info("[%s] %s: %s", level, code, message)


# ---------------------------------------------------------------------------
# Factory: build from .env
# ---------------------------------------------------------------------------

def build_engine_from_env(
    mode: str = "paper",
    max_order_usd: float = 5.0,
    env_path: str = ".env.polymarket",
) -> ExecutionEngine:
    """
    Convenience builder that reads credentials from a .env file or environment.

    Usage:
        engine = build_engine_from_env(mode="paper")   # safe default
        engine = build_engine_from_env(mode="live", max_order_usd=3.0)
    """
    # try loading dotenv if available
    try:
        from dotenv import load_dotenv
        if os.path.exists(env_path):
            load_dotenv(env_path)
            logger.info("Loaded env from %s", env_path)
    except ImportError:
        pass

    private_key = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
    funder = os.environ.get("POLYMARKET_FUNDER", "")
    sig_type = int(os.environ.get("POLYMARKET_SIGNATURE_TYPE", "2"))
    chain_id = int(os.environ.get("POLYMARKET_CHAIN_ID", "137"))

    engine = ExecutionEngine(
        mode=ExecMode(mode),
        private_key=private_key,
        funder=funder,
        signature_type=sig_type,
        chain_id=chain_id,
        max_order_usd=max_order_usd,
    )
    return engine
