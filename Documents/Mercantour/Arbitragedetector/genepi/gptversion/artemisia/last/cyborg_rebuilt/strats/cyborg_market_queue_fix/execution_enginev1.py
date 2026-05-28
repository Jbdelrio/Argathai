from __future__ import annotations

import os
import time
import uuid
import threading
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class EngineMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class OrderState(str, Enum):
    NEW = "NEW"
    POSTING = "POSTING"
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"


class AlertLevel(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    CRIT = "CRIT"


@dataclass
class ExecutionAlert:
    ts: float
    level: AlertLevel
    code: str
    message: str
    order_id: Optional[str] = None
    symbol: Optional[str] = None
    sound_hint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["level"] = self.level.value
        return d


@dataclass
class OrderRecord:
    client_order_id: str
    symbol: str
    token_id: str
    side: str
    price: float
    size: float
    mode: EngineMode
    state: OrderState = OrderState.NEW
    exchange_order_id: Optional[str] = None
    filled_size: float = 0.0
    avg_fill_price: Optional[float] = None
    remaining_size: Optional[float] = None
    posted_ts: Optional[float] = None
    updated_ts: float = field(default_factory=time.time)
    last_error: str = ""
    tags: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["mode"] = self.mode.value
        d["state"] = self.state.value
        return d


class BaseExecutionEngine:
    """
    Common order state / alert manager.
    This file is safe to use without credentials.
    LIVE mode requires `py-clob-client` and valid Polymarket credentials.
    """

    def __init__(self, mode: EngineMode):
        self.mode = mode
        self._lock = threading.RLock()
        self._orders: Dict[str, OrderRecord] = {}
        self._alerts: List[ExecutionAlert] = []

    def _emit(self, level: AlertLevel, code: str, message: str, *,
              order_id: Optional[str] = None,
              symbol: Optional[str] = None,
              sound_hint: Optional[str] = None) -> None:
        with self._lock:
            self._alerts.insert(
                0,
                ExecutionAlert(
                    ts=time.time(),
                    level=level,
                    code=code,
                    message=message,
                    order_id=order_id,
                    symbol=symbol,
                    sound_hint=sound_hint,
                ),
            )
            self._alerts = self._alerts[:500]

    def get_alerts(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [a.to_dict() for a in self._alerts]

    def clear_alerts(self) -> None:
        with self._lock:
            self._alerts = []

    def list_orders(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [o.to_dict() for o in self._orders.values()]

    def get_order(self, client_order_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            o = self._orders.get(client_order_id)
            return o.to_dict() if o else None

    def _new_client_id(self) -> str:
        return f"cx-{uuid.uuid4().hex[:16]}"

    def connect(self) -> Dict[str, Any]:
        raise NotImplementedError

    def get_balance(self) -> Dict[str, Any]:
        raise NotImplementedError

    def place_limit_order(self, *,
                          symbol: str,
                          token_id: str,
                          side: str,
                          price: float,
                          size: float,
                          tags: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raise NotImplementedError

    def cancel_order(self, client_order_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    def refresh_order(self, client_order_id: str) -> Dict[str, Any]:
        raise NotImplementedError


class PaperMirrorExecutionEngine(BaseExecutionEngine):
    """
    Useful bridge for GUI integration before LIVE.
    Keeps real order states / alerts, but does not touch the exchange.
    """

    def __init__(self):
        super().__init__(EngineMode.PAPER)

    def connect(self) -> Dict[str, Any]:
        self._emit(AlertLevel.INFO, "PAPER_CONNECTED", "Paper execution engine connected.")
        return {"ok": True, "mode": self.mode.value}

    def get_balance(self) -> Dict[str, Any]:
        return {"ok": True, "mode": self.mode.value, "currency": "USDC", "available": None}

    def place_limit_order(self, *, symbol: str, token_id: str, side: str, price: float, size: float,
                          tags: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cid = self._new_client_id()
        order = OrderRecord(
            client_order_id=cid,
            symbol=symbol,
            token_id=token_id,
            side=side,
            price=float(price),
            size=float(size),
            mode=self.mode,
            state=OrderState.OPEN,
            posted_ts=time.time(),
            remaining_size=float(size),
            tags=tags or {},
        )
        with self._lock:
            self._orders[cid] = order
        self._emit(AlertLevel.INFO, "PAPER_ORDER_POSTED", f"Paper order posted: {side} {size} @ {price}", order_id=cid, symbol=symbol)
        return {"ok": True, "order": order.to_dict()}

    def cancel_order(self, client_order_id: str) -> Dict[str, Any]:
        with self._lock:
            order = self._orders.get(client_order_id)
            if not order:
                return {"ok": False, "error": "unknown_order"}
            order.state = OrderState.CANCELLED
            order.updated_ts = time.time()
        self._emit(AlertLevel.WARN, "PAPER_CANCELLED", "Paper order cancelled.", order_id=client_order_id, symbol=order.symbol)
        return {"ok": True, "order": order.to_dict()}

    def refresh_order(self, client_order_id: str) -> Dict[str, Any]:
        with self._lock:
            order = self._orders.get(client_order_id)
            if not order:
                return {"ok": False, "error": "unknown_order"}
            return {"ok": True, "order": order.to_dict()}


class PolymarketLiveExecutionEngine(BaseExecutionEngine):
    """
    Real Polymarket execution layer.
    Uses py-clob-client when available.
    This class is intentionally conservative:
    - tracks posted orders
    - surfaces pending / partial / filled alerts
    - does NOT auto-scale size
    - does NOT assume immediate fills
    """

    def __init__(self,
                 *,
                 host: str = "https://clob.polymarket.com",
                 private_key: Optional[str] = None,
                 chain_id: int = 137,
                 signature_type: int = 0,
                 funder: Optional[str] = None):
        super().__init__(EngineMode.LIVE)
        self.host = host
        self.private_key = private_key or os.getenv("POLYMARKET_PRIVATE_KEY", "")
        self.chain_id = int(os.getenv("POLYMARKET_CHAIN_ID", chain_id))
        self.signature_type = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", signature_type))
        self.funder = funder or os.getenv("POLYMARKET_FUNDER", "")
        self._client = None

    def connect(self) -> Dict[str, Any]:
        if not self.private_key:
            return {"ok": False, "error": "missing_private_key"}

        try:
            from py_clob_client.client import ClobClient
        except Exception as exc:
            return {"ok": False, "error": f"py_clob_client_import_failed: {exc}"}

        try:
            self._client = ClobClient(
                host=self.host,
                key=self.private_key,
                chain_id=self.chain_id,
                signature_type=self.signature_type,
                funder=self.funder or None,
            )
            self._emit(AlertLevel.INFO, "LIVE_CONNECTED", "Live Polymarket engine connected.")
            return {"ok": True, "mode": self.mode.value, "host": self.host}
        except Exception as exc:
            self._emit(AlertLevel.CRIT, "LIVE_CONNECT_ERROR", f"Live connect failed: {exc}", sound_hint="error")
            return {"ok": False, "error": str(exc)}

    def _require_client(self):
        if self._client is None:
            result = self.connect()
            if not result.get("ok"):
                raise RuntimeError(result.get("error", "live_client_not_connected"))

    def get_balance(self) -> Dict[str, Any]:
        self._require_client()
        try:
            # Keep generic because client versions vary.
            if hasattr(self._client, "get_balance_allowance"):
                bal = self._client.get_balance_allowance()
                return {"ok": True, "raw": bal}
            if hasattr(self._client, "get_balance"):
                bal = self._client.get_balance()
                return {"ok": True, "raw": bal}
            return {"ok": False, "error": "balance_method_not_found"}
        except Exception as exc:
            self._emit(AlertLevel.WARN, "BALANCE_ERROR", f"Balance fetch failed: {exc}")
            return {"ok": False, "error": str(exc)}

    def place_limit_order(self, *,
                          symbol: str,
                          token_id: str,
                          side: str,
                          price: float,
                          size: float,
                          tags: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._require_client()
        cid = self._new_client_id()
        order = OrderRecord(
            client_order_id=cid,
            symbol=symbol,
            token_id=token_id,
            side=side,
            price=float(price),
            size=float(size),
            mode=self.mode,
            state=OrderState.POSTING,
            posted_ts=time.time(),
            remaining_size=float(size),
            tags=tags or {},
        )
        with self._lock:
            self._orders[cid] = order

        try:
            # NOTE:
            # py-clob-client versions differ. We keep a permissive payload.
            payload = {
                "token_id": token_id,
                "side": side.upper(),
                "price": float(price),
                "size": float(size),
                "client_order_id": cid,
            }

            if hasattr(self._client, "create_order"):
                resp = self._client.create_order(payload)
            elif hasattr(self._client, "post_order"):
                resp = self._client.post_order(payload)
            else:
                raise RuntimeError("order_create_method_not_found")

            exchange_id = None
            if isinstance(resp, dict):
                exchange_id = (
                    resp.get("orderID")
                    or resp.get("order_id")
                    or resp.get("id")
                )

            with self._lock:
                order.exchange_order_id = exchange_id
                order.state = OrderState.OPEN
                order.updated_ts = time.time()

            self._emit(
                AlertLevel.INFO,
                "ORDER_POSTED",
                f"Live order posted: {side} {size} @ {price}",
                order_id=cid,
                symbol=symbol,
            )
            return {"ok": True, "order": order.to_dict(), "raw": resp}
        except Exception as exc:
            with self._lock:
                order.state = OrderState.ERROR
                order.last_error = str(exc)
                order.updated_ts = time.time()
            self._emit(AlertLevel.CRIT, "ORDER_POST_ERROR", f"Order post failed: {exc}", order_id=cid, symbol=symbol, sound_hint="error")
            return {"ok": False, "error": str(exc), "order": order.to_dict()}

    def cancel_order(self, client_order_id: str) -> Dict[str, Any]:
        self._require_client()
        with self._lock:
            order = self._orders.get(client_order_id)
            if not order:
                return {"ok": False, "error": "unknown_order"}
            ex_id = order.exchange_order_id or client_order_id
            order.state = OrderState.CANCEL_PENDING
            order.updated_ts = time.time()

        try:
            if hasattr(self._client, "cancel"):
                resp = self._client.cancel(ex_id)
            elif hasattr(self._client, "cancel_order"):
                resp = self._client.cancel_order(ex_id)
            else:
                raise RuntimeError("cancel_method_not_found")

            with self._lock:
                order.state = OrderState.CANCELLED
                order.updated_ts = time.time()

            self._emit(AlertLevel.WARN, "ORDER_CANCELLED", "Order cancelled.", order_id=client_order_id, symbol=order.symbol)
            return {"ok": True, "order": order.to_dict(), "raw": resp}
        except Exception as exc:
            with self._lock:
                order.state = OrderState.ERROR
                order.last_error = str(exc)
                order.updated_ts = time.time()
            self._emit(AlertLevel.CRIT, "ORDER_CANCEL_ERROR", f"Cancel failed: {exc}", order_id=client_order_id, symbol=order.symbol, sound_hint="error")
            return {"ok": False, "error": str(exc), "order": order.to_dict()}

    def refresh_order(self, client_order_id: str) -> Dict[str, Any]:
        self._require_client()
        with self._lock:
            order = self._orders.get(client_order_id)
            if not order:
                return {"ok": False, "error": "unknown_order"}
            ex_id = order.exchange_order_id or client_order_id

        try:
            if hasattr(self._client, "get_order"):
                resp = self._client.get_order(ex_id)
            elif hasattr(self._client, "get_orders"):
                resp = self._client.get_orders()
            else:
                raise RuntimeError("order_status_method_not_found")

            new_state = None
            filled = order.filled_size
            avg_price = order.avg_fill_price
            remaining = order.remaining_size

            if isinstance(resp, dict):
                status_raw = str(resp.get("status", "")).upper()
                if status_raw in {"OPEN", "LIVE"}:
                    new_state = OrderState.OPEN
                elif status_raw in {"PARTIAL", "PARTIALLY_FILLED"}:
                    new_state = OrderState.PARTIAL
                elif status_raw in {"FILLED", "MATCHED"}:
                    new_state = OrderState.FILLED
                elif status_raw in {"CANCELLED", "CANCELED"}:
                    new_state = OrderState.CANCELLED
                elif status_raw in {"REJECTED"}:
                    new_state = OrderState.REJECTED

                if resp.get("filled_size") is not None:
                    filled = float(resp.get("filled_size"))
                if resp.get("avg_fill_price") is not None:
                    avg_price = float(resp.get("avg_fill_price"))
                if resp.get("remaining_size") is not None:
                    remaining = float(resp.get("remaining_size"))

            with self._lock:
                prev_state = order.state
                if new_state is not None:
                    order.state = new_state
                order.filled_size = filled
                order.avg_fill_price = avg_price
                order.remaining_size = remaining
                order.updated_ts = time.time()

            if prev_state != order.state:
                if order.state == OrderState.PARTIAL:
                    self._emit(AlertLevel.WARN, "ORDER_PARTIAL", "Order partially filled.", order_id=client_order_id, symbol=order.symbol, sound_hint="partial")
                elif order.state == OrderState.FILLED:
                    self._emit(AlertLevel.INFO, "ORDER_FILLED", "Order fully filled.", order_id=client_order_id, symbol=order.symbol, sound_hint="fill")
                elif order.state == OrderState.CANCELLED:
                    self._emit(AlertLevel.WARN, "ORDER_CANCELLED", "Order cancelled by venue.", order_id=client_order_id, symbol=order.symbol)
                elif order.state == OrderState.REJECTED:
                    self._emit(AlertLevel.CRIT, "ORDER_REJECTED", "Order rejected by venue.", order_id=client_order_id, symbol=order.symbol, sound_hint="error")

            return {"ok": True, "order": order.to_dict(), "raw": resp}
        except Exception as exc:
            with self._lock:
                order.state = OrderState.ERROR
                order.last_error = str(exc)
                order.updated_ts = time.time()
            self._emit(AlertLevel.CRIT, "ORDER_REFRESH_ERROR", f"Refresh failed: {exc}", order_id=client_order_id, symbol=order.symbol, sound_hint="error")
            return {"ok": False, "error": str(exc), "order": order.to_dict()}


def build_engine_from_env(mode: str = "paper") -> BaseExecutionEngine:
    if str(mode).lower() == "live":
        return PolymarketLiveExecutionEngine()
    return PaperMirrorExecutionEngine()
