"""
data_manager.py – runtime state + live paper trading loops for CYBORG

Goals of this rewrite:
- no fake trades or markets injected on startup
- Start really starts a live paper loop
- controls are updated explicitly from the GUI
- capital and allocations are configurable
- Coin5min has a visible warmup stage before trading
"""

from __future__ import annotations

import json
import logging
import difflib
import sqlite3
import threading
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd

from paper_trading import PaperTradingEngine

logger = logging.getLogger("cyborg.data")
DB = "cyborg.db"


# ============================================================
# Persistence helpers (backtest history table kept for compatibility)
# ============================================================

def init_db():
    conn = sqlite3.connect(DB)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS backtest_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            strategy TEXT,
            params TEXT,
            total_pnl REAL,
            n_trades INTEGER,
            win_rate REAL,
            sharpe REAL,
            max_drawdown REAL,
            roi REAL,
            created_at REAL
        )"""
    )
    conn.commit()
    conn.close()


# ============================================================
# Utility helpers
# ============================================================

def _now_str(ts: Optional[float] = None) -> str:
    return datetime.fromtimestamp(ts or time.time()).strftime("%H:%M:%S")


def _normalize_question(q: str) -> str:
    q = (q or "").lower()
    keep = []
    for ch in q:
        keep.append(ch if ch.isalnum() or ch.isspace() else " ")
    return " ".join("".join(keep).split())


def _detect_asset(text: str) -> str:
    q = (text or "").lower()
    mapping = [
        ("bitcoin", "BTC"), ("btc", "BTC"),
        ("ethereum", "ETH"), ("eth", "ETH"),
        ("solana", "SOL"), ("sol", "SOL"),
        ("xrp", "XRP"),
        ("dogecoin", "DOGE"), ("doge", "DOGE"),
    ]
    for key, asset in mapping:
        if key in q:
            return asset
    return "?"


def _extract_yes_price(market: Dict) -> float:
    prices = market.get("outcomePrices", "[]")
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except Exception:
            prices = []
    try:
        return float(prices[0]) if prices else 0.5
    except Exception:
        return 0.5


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _coinbase_granularity(interval_min: int) -> int:
    mapping = {1: 60, 5: 300, 15: 900}
    return mapping.get(int(interval_min), 300)


def _binance_interval(interval_min: int) -> str:
    mapping = {1: '1m', 3: '3m', 5: '5m', 15: '15m', 30: '30m'}
    return mapping.get(int(interval_min), '5m')


def _round_window_start(ts: float, interval_min: int) -> int:
    interval_sec = max(60, int(interval_min) * 60)
    return int(float(ts) // interval_sec * interval_sec)


def _round_window_end(ts: float, interval_min: int) -> int:
    start = _round_window_start(ts, interval_min)
    interval_sec = max(60, int(interval_min) * 60)
    return start + interval_sec


# ============================================================
# HTTP client helpers
# ============================================================

class PublicFeeds:
    def __init__(self, timeout: float = 6.0):
        self.client = httpx.Client(timeout=timeout, headers={"User-Agent": "cyborg-paper-live/1.0"})

    @staticmethod
    def _normalize_polymarket_event_order(order: str) -> str:
        mapping = {
            "volume24hr": "volume_24hr",
            "startDate": "start_date",
            "endDate": "end_date",
            "closedTime": "closed_time",
        }
        return mapping.get(str(order or "").strip(), str(order or "").strip())

    def get_polymarket_markets(self, limit: int = 100, order: str = "volume24hr", ascending: bool = False, offset: int = 0) -> List[Dict]:
        # Docs recommend /events for active market discovery, then reading nested events[].markets.
        # In practice, Gamma can return 422 for some valid-looking query combinations on /events.
        # So we progressively fall back to simpler parameter sets and sort client-side when needed.
        base_params = {
            "active": "true",
            "closed": "false",
            "limit": limit,
            "offset": offset,
        }
        normalized_order = self._normalize_polymarket_event_order(order)

        param_candidates = []
        full = dict(base_params)
        full["ascending"] = "true" if ascending else "false"
        if normalized_order:
            full["order"] = normalized_order
        param_candidates.append(full)
        param_candidates.append(dict(base_params))
        param_candidates.append({"limit": limit, "offset": offset})

        last_exc = None
        payload = None
        used_fallback = None
        for params in param_candidates:
            try:
                resp = self.client.get("https://gamma-api.polymarket.com/events", params=params)
                resp.raise_for_status()
                payload = resp.json()
                used_fallback = params
                break
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response is None or exc.response.status_code != 422:
                    raise
                continue

        if payload is None:
            if last_exc:
                raise last_exc
            raise RuntimeError("polymarket event discovery failed without HTTP response")

        events = payload if isinstance(payload, list) else payload.get("data", [])

        markets: List[Dict] = []
        for ev in events:
            if not isinstance(ev, dict):
                continue
            for m in ev.get("markets", []) or []:
                if not isinstance(m, dict):
                    continue
                mm = dict(m)
                mm.setdefault("event_id", ev.get("id"))
                mm.setdefault("event_slug", ev.get("slug"))
                mm.setdefault("event_title", ev.get("title"))
                mm["category"] = mm.get("category") or ev.get("category")
                mm["startDate"] = mm.get("startDate") or ev.get("startDate")
                mm["endDate"] = mm.get("endDate") or ev.get("endDate")
                markets.append(mm)

        if normalized_order and used_fallback is not None and "order" not in used_fallback:
            reverse = not ascending
            key_map = {
                "volume_24hr": lambda x: _safe_float(x.get("volume24hr") or x.get("volume24Hr") or x.get("volume_24hr"), 0.0),
                "volume": lambda x: _safe_float(x.get("volume"), 0.0),
                "liquidity": lambda x: _safe_float(x.get("liquidity"), 0.0),
                "start_date": lambda x: str(x.get("startDate") or ""),
                "end_date": lambda x: str(x.get("endDate") or ""),
                "closed_time": lambda x: str(x.get("closedTime") or ""),
                "competitive": lambda x: _safe_float(x.get("competitive"), 0.0),
            }
            key_fn = key_map.get(normalized_order)
            if key_fn:
                markets.sort(key=key_fn, reverse=reverse)

        return markets

    def get_polymarket_orderbook(self, token_id: str) -> Dict:
        resp = self.client.get("https://clob.polymarket.com/book", params={"token_id": token_id})
        resp.raise_for_status()
        return resp.json()

    def get_polymarket_market_by_slug(self, slug: str) -> Dict:
        resp = self.client.get(f"https://gamma-api.polymarket.com/markets/slug/{slug}")
        resp.raise_for_status()
        return resp.json()

    def get_polymarket_event_by_slug(self, slug: str) -> Dict:
        resp = self.client.get(f"https://gamma-api.polymarket.com/events/slug/{slug}")
        resp.raise_for_status()
        return resp.json()

    def get_polymarket_event_page(self, slug: str) -> str:
        resp = self.client.get(f"https://polymarket.com/event/{slug}")
        resp.raise_for_status()
        return resp.text

    def get_polymarket_public_search(self, q: str, page: int = 1, limit_per_type: int = 10, events_status: str = "active") -> Dict:
        params = {
            "q": q,
            "page": page,
            "limit_per_type": limit_per_type,
            "events_status": events_status,
            "search_tags": "false",
            "search_profiles": "false",
            "optimized": "true",
        }
        resp = self.client.get("https://gamma-api.polymarket.com/public-search", params=params)
        resp.raise_for_status()
        return resp.json()

    def get_polymarket_price_to_beat_by_slug(self, slug: str) -> Optional[float]:
        html = self.get_polymarket_event_page(slug)
        patterns = [
            r'Price\s+to\s+Beat[^$0-9]{0,80}\$?\s*([0-9][0-9,]*\.?[0-9]*)',
            r'Price\s+To\s+Beat[^$0-9]{0,80}\$?\s*([0-9][0-9,]*\.?[0-9]*)',
            r'opening\s+reference\s+price[^$0-9]{0,80}\$?\s*([0-9][0-9,]*\.?[0-9]*)',
            r'\"priceToBeat\"\s*:\s*\"?\$?([0-9][0-9,]*\.?[0-9]*)',
            r'priceToBeat\s*[=:]\s*\"?\$?([0-9][0-9,]*\.?[0-9]*)',
        ]
        for pattern in patterns:
            m = re.search(pattern, html, re.I | re.S)
            if not m:
                continue
            raw = m.group(1).replace(',', '')
            try:
                val = float(raw)
            except Exception:
                continue
            if np.isfinite(val) and val > 0:
                return float(val)
        return None

    def get_kalshi_markets(self, limit: int = 100) -> List[Dict]:
        resp = self.client.get(
            "https://demo-api.kalshi.co/trade-api/v2/markets",
            params={"limit": limit, "status": "open"},
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload.get("markets", []) if isinstance(payload, dict) else []

    def get_coinbase_ticker(self, product_id: str) -> Dict:
        resp = self.client.get(f"https://api.exchange.coinbase.com/products/{product_id}/ticker")
        resp.raise_for_status()
        return resp.json()

    def get_coinbase_candles(self, product_id: str, granularity: int) -> List[List[float]]:
        resp = self.client.get(
            f"https://api.exchange.coinbase.com/products/{product_id}/candles",
            params={"granularity": granularity},
        )
        resp.raise_for_status()
        rows = resp.json()
        rows.sort(key=lambda r: r[0])
        return rows

    def get_binance_book_ticker(self, symbol: str) -> Dict:
        resp = self.client.get("https://api.binance.com/api/v3/ticker/bookTicker", params={"symbol": symbol})
        resp.raise_for_status()
        return resp.json()

    def get_binance_klines(self, symbol: str, interval: str = '1m', limit: int = 120) -> List[List]:
        resp = self.client.get("https://api.binance.com/api/v3/klines", params={"symbol": symbol, "interval": interval, "limit": limit})
        resp.raise_for_status()
        rows = resp.json()
        rows.sort(key=lambda r: r[0])
        return rows


# ============================================================
# Strategy state models
# ============================================================

@dataclass
class StrategyRuntime:
    active: bool = False
    paused: bool = False
    status: str = "stopped"  # stopped / connecting / warming / running / paused / error
    last_action: str = "Not started"
    last_update_ts: float = 0.0
    last_error: str = ""
    started_at: Optional[float] = None


@dataclass
class BetMissState(StrategyRuntime):
    min_edge_pct: float = 2.0
    trade_size_pct: float = 2.0
    allocation_pct: float = 50.0
    opportunities: List[Dict] = field(default_factory=list)


@dataclass
class Coin5MinState(StrategyRuntime):
    min_edge_pct: float = 1.0
    interval_min: int = 5
    trade_size_pct: float = 2.0
    allocation_pct: float = 50.0
    assets: List[str] = field(default_factory=lambda: ["BTC", "ETH", "SOL"])
    warmup_required: int = 8
    warmup_started_at: Optional[float] = None
    legging_enabled: bool = True
    leg1_directional: bool = True
    max_leg_hold_sec: int = 45
    max_first_leg_price: float = 0.65
    max_sum_yes_no: float = 0.99
    max_trade_usd: float = 20.0
    trend_lookback_min: int = 20
    spot_source: str = "binance"
    signals: Dict[str, Dict] = field(default_factory=dict)
    markets: List[Dict] = field(default_factory=list)
    market_meta: Dict[str, Dict] = field(default_factory=dict)
    current_market_ids: Dict[str, str] = field(default_factory=dict)
    orderbooks: Dict[str, Dict] = field(default_factory=dict)
    spot_rows: Dict[str, Dict] = field(default_factory=dict)
    reference_spot: Dict[str, float] = field(default_factory=dict)
    reference_source: Dict[str, str] = field(default_factory=dict)
    ref_cache: Dict[str, Dict] = field(default_factory=dict)
    open_trade_ids: Dict[str, str] = field(default_factory=dict)
    open_trade_meta: Dict[str, Dict] = field(default_factory=dict)
    partial_legs: Dict[str, Dict] = field(default_factory=dict)


# ============================================================
# Thread-safe live state
# ============================================================

class LiveState:
    def __init__(self):
        self._lock = threading.RLock()
        self._started = False
        self._stop_event = threading.Event()

        self.mode = "paper_live"
        self.kill_switch = False
        self.capital = 10000.0
        self.allocations = {"betmiss": 50.0, "coin5min": 50.0}

        self.platforms: Dict[str, Dict] = {
            "polymarket": {"status": "disconnected", "latency": None, "requests": 0, "errors": 0},
            "kalshi": {"status": "disconnected", "latency": None, "requests": 0, "errors": 0},
            "coinbase": {"status": "disconnected", "latency": None, "requests": 0, "errors": 0},
            "binance": {"status": "disconnected", "latency": None, "requests": 0, "errors": 0},
        }

        self.betmiss = BetMissState()
        self.coin5min = Coin5MinState()

        self.recent_trades: List[Dict] = []
        self.alerts: List[Dict] = []
        self.live_markets: List[Dict] = []
        self.last_tick_ts: float = 0.0

        self.price_history: Dict[str, Deque[Tuple[float, float]]] = defaultdict(lambda: deque(maxlen=600))
        self.coin_market_history: Dict[str, Deque[Dict]] = defaultdict(lambda: deque(maxlen=300))
        self.paper = PaperTradingEngine(initial_capital=self.capital, db_path=":memory:")
        self.betmiss_realized_pnl = 0.0

        self.feeds = PublicFeeds(timeout=6.0)

    # ---------- state snapshots ----------
    def snap(self) -> Dict:
        with self._lock:
            metrics = self.paper.get_metrics()
            total_pnl = metrics["total_pnl"] + self.betmiss_realized_pnl
            warmup_points = {a: len(self.price_history[a]) for a in self.coin5min.assets}
            warmup_missing = {a: max(0, self.coin5min.warmup_required - warmup_points.get(a, 0)) for a in self.coin5min.assets}
            warmup_elapsed_sec = 0.0
            if self.coin5min.warmup_started_at:
                warmup_elapsed_sec = max(0.0, time.time() - self.coin5min.warmup_started_at)
            warmup_eta_sec = 2.0 if (self.coin5min.active and any(v > 0 for v in warmup_missing.values())) else 0.0
            return {
                "mode": self.mode,
                "kill_switch": self.kill_switch,
                "capital": self.capital,
                "allocations": dict(self.allocations),
                "platforms": {k: dict(v) for k, v in self.platforms.items()},
                "betmiss": {
                    **self.betmiss.__dict__,
                    "opportunities": list(self.betmiss.opportunities),
                },
                "coin5min": {
                    **self.coin5min.__dict__,
                    "signals": {k: dict(v) for k, v in self.coin5min.signals.items()},
                    "markets": list(self.coin5min.markets),
                    "orderbooks": {k: dict(v) for k, v in self.coin5min.orderbooks.items()},
                    "spot_rows": {k: dict(v) for k, v in self.coin5min.spot_rows.items()},
                    "reference_spot": dict(self.coin5min.reference_spot),
                    "reference_source": dict(self.coin5min.reference_source),
                    "market_meta": {k: dict(v) for k, v in self.coin5min.market_meta.items()},
                    "current_market_ids": dict(self.coin5min.current_market_ids),
                    "open_trade_ids": dict(self.coin5min.open_trade_ids),
                    "open_trade_meta": {k: dict(v) for k, v in self.coin5min.open_trade_meta.items()},
                    "warmup_points": warmup_points,
                    "warmup_missing": warmup_missing,
                    "warmup_elapsed_sec": round(warmup_elapsed_sec, 1),
                    "warmup_eta_sec": round(warmup_eta_sec, 1),
                    "price_history": {
                        a: [{"ts": ts, "price": price} for ts, price in list(self.price_history[a])[-180:]]
                        for a in self.coin5min.assets
                    },
                    "market_history": {
                        a: list(self.coin_market_history[a])[-180:]
                        for a in self.coin5min.assets
                    },
                },
                "portfolio": {
                    **metrics,
                    "total_pnl": total_pnl,
                    "total_value": self.capital + total_pnl,
                },
                "recent_trades": list(self.recent_trades),
                "alerts": list(self.alerts[:120]),
                "live_markets": list(self.live_markets),
                "last_tick_ts": self.last_tick_ts,
            }

    def alert(self, level: str, strat: str, msg: str):
        with self._lock:
            self.alerts.insert(0, {"time": _now_str(), "level": level, "strat": strat, "msg": msg})
            self.alerts = self.alerts[:200]

    # ---------- configuration ----------
    def reset_session(self, capital: Optional[float] = None, bm_alloc: Optional[float] = None, c5_alloc: Optional[float] = None) -> Tuple[bool, str]:
        with self._lock:
            if capital is not None and capital <= 0:
                return False, "Le capital doit être > 0."
            if bm_alloc is not None and not (0 <= bm_alloc <= 100):
                return False, "Allocation BetMiss invalide."
            if c5_alloc is not None and not (0 <= c5_alloc <= 100):
                return False, "Allocation Coin5min invalide."
            if bm_alloc is not None and c5_alloc is not None and bm_alloc + c5_alloc > 100:
                return False, "La somme des allocations doit être ≤ 100%."

            self._stop_strategies_locked(clear_runtime=False)

            if capital is not None:
                self.capital = float(capital)
            if bm_alloc is not None:
                self.allocations["betmiss"] = float(bm_alloc)
                self.betmiss.allocation_pct = float(bm_alloc)
            if c5_alloc is not None:
                self.allocations["coin5min"] = float(c5_alloc)
                self.coin5min.allocation_pct = float(c5_alloc)

            self.paper = PaperTradingEngine(initial_capital=self.capital, db_path=":memory:")
            self.betmiss_realized_pnl = 0.0
            self.recent_trades = []
            self.live_markets = []
            self.price_history = defaultdict(lambda: deque(maxlen=600))
            self.coin_market_history = defaultdict(lambda: deque(maxlen=300))
            self.coin5min.signals = {}
            self.coin5min.markets = []
            self.coin5min.orderbooks = {}
            self.coin5min.spot_rows = {}
            self.coin5min.reference_spot = {}
            self.coin5min.reference_source = {}
            self.coin5min.ref_cache = {}
            self.coin5min.market_meta = {}
            self.coin5min.current_market_ids = {}
            self.coin5min.warmup_started_at = None
            self.coin5min.open_trade_ids = {}
            self.coin5min.open_trade_meta = {}
            self.coin5min.partial_legs = {}
            self.coin_market_history = defaultdict(lambda: deque(maxlen=300))
            for p in self.platforms.values():
                p.update({"status": "disconnected", "latency": None, "requests": 0, "errors": 0})

        self.alert("INFO", "SYS", f"Session paper réinitialisée | capital=${self.capital:,.0f} | BM {self.allocations['betmiss']:.0f}% | C5 {self.allocations['coin5min']:.0f}%")
        return True, "Session paper réinitialisée."

    def update_betmiss_params(self, min_edge_pct: Optional[float] = None, trade_size_pct: Optional[float] = None):
        with self._lock:
            changed = False
            if min_edge_pct is not None and float(min_edge_pct) != self.betmiss.min_edge_pct:
                self.betmiss.min_edge_pct = float(min_edge_pct)
                changed = True
            if trade_size_pct is not None and float(trade_size_pct) != self.betmiss.trade_size_pct:
                self.betmiss.trade_size_pct = float(trade_size_pct)
                changed = True
            if changed:
                self.betmiss.last_action = f"Params mis à jour à {_now_str()}"

    def update_coin5min_params(
        self,
        min_edge_pct: Optional[float] = None,
        interval_min: Optional[int] = None,
        assets: Optional[List[str]] = None,
        trade_size_pct: Optional[float] = None,
        max_leg_hold_sec: Optional[int] = None,
        max_first_leg_price: Optional[float] = None,
        max_sum_yes_no: Optional[float] = None,
        max_trade_usd: Optional[float] = None,
        leg1_directional: Optional[bool] = None,
        spot_source: Optional[str] = None,
    ):
        with self._lock:
            restart_warmup = False
            changed = False
            if min_edge_pct is not None and float(min_edge_pct) != self.coin5min.min_edge_pct:
                self.coin5min.min_edge_pct = float(min_edge_pct)
                changed = True
            if trade_size_pct is not None and float(trade_size_pct) != self.coin5min.trade_size_pct:
                self.coin5min.trade_size_pct = float(trade_size_pct)
                changed = True
            if interval_min is not None and int(interval_min) != self.coin5min.interval_min:
                self.coin5min.interval_min = int(interval_min)
                restart_warmup = True
                changed = True
            if max_leg_hold_sec is not None and int(max_leg_hold_sec) != self.coin5min.max_leg_hold_sec:
                self.coin5min.max_leg_hold_sec = int(max_leg_hold_sec)
                changed = True
            if max_first_leg_price is not None and float(max_first_leg_price) != self.coin5min.max_first_leg_price:
                self.coin5min.max_first_leg_price = float(max_first_leg_price)
                changed = True
            if max_sum_yes_no is not None and float(max_sum_yes_no) != self.coin5min.max_sum_yes_no:
                self.coin5min.max_sum_yes_no = float(max_sum_yes_no)
                changed = True
            if max_trade_usd is not None and float(max_trade_usd) != self.coin5min.max_trade_usd:
                self.coin5min.max_trade_usd = max(1.0, float(max_trade_usd))
                changed = True
            if leg1_directional is not None and bool(leg1_directional) != self.coin5min.leg1_directional:
                self.coin5min.leg1_directional = bool(leg1_directional)
                changed = True
            if spot_source is not None and str(spot_source) != self.coin5min.spot_source:
                self.coin5min.spot_source = str(spot_source)
                restart_warmup = True
                changed = True
            if assets is not None:
                assets = list(dict.fromkeys(assets))
                if assets != self.coin5min.assets:
                    self.coin5min.assets = assets
                    restart_warmup = True
                    changed = True
            if restart_warmup:
                self.price_history = defaultdict(lambda: deque(maxlen=600))
                self.coin5min.signals = {}
                self.coin5min.spot_rows = {}
                self.coin5min.markets = []
                self.coin5min.orderbooks = {}
                self.coin5min.reference_spot = {}
                self.coin5min.reference_source = {}
                self.coin5min.ref_cache = {}
                self.coin5min.market_meta = {}
                self.coin5min.current_market_ids = {}
                self.coin5min.partial_legs = {}
                self.coin_market_history = defaultdict(lambda: deque(maxlen=300))
                if self.coin5min.active:
                    self.coin5min.status = "warming"
                    self.coin5min.warmup_started_at = time.time()
                    self.coin5min.last_action = f"Warmup relancé à {_now_str()}"
            elif changed:
                self.coin5min.last_action = f"Params mis à jour à {_now_str()}"

    def toggle_kill_switch(self) -> Tuple[bool, str]:
        with self._lock:
            self.kill_switch = not self.kill_switch
            enabled = self.kill_switch
            if enabled:
                self._stop_strategies_locked(clear_runtime=False)
                msg = "KILL SWITCH ACTIVÉ – toutes les stratégies sont stoppées."
            else:
                msg = "KILL SWITCH désactivé. Tu peux relancer les stratégies."
        self.alert("WARN" if enabled else "INFO", "SYS", msg)
        return enabled, msg

    # ---------- strategy lifecycle ----------
    def start_strategy(self, strategy: str):
        with self._lock:
            if self.kill_switch:
                return False, "Kill switch actif – Start bloqué."
            if strategy == "betmiss":
                self.betmiss.active = True
                self.betmiss.paused = False
                self.betmiss.status = "connecting"
                self.betmiss.started_at = time.time()
                self.betmiss.last_action = f"Start à {_now_str()}"
                self.betmiss.last_error = ""
                self.betmiss.opportunities = []
                self.platforms["polymarket"]["status"] = "connecting"
                self.platforms["kalshi"]["status"] = "connecting"
                msg = "BetMiss démarrée."
            elif strategy == "coin5min":
                self.coin5min.active = True
                self.coin5min.paused = False
                self.coin5min.status = "connecting"
                self.coin5min.started_at = time.time()
                self.coin5min.warmup_started_at = time.time()
                self.coin5min.last_action = f"Start à {_now_str()}"
                self.coin5min.last_error = ""
                self.coin5min.signals = {}
                self.coin5min.markets = []
                self.coin5min.orderbooks = {}
                self.coin5min.spot_rows = {}
                self.coin5min.reference_spot = {}
                self.coin5min.reference_source = {}
                self.coin5min.ref_cache = {}
                self.coin5min.market_meta = {}
                self.coin5min.current_market_ids = {}
                self.coin5min.open_trade_ids = {}
                self.coin5min.open_trade_meta = {}
                self.coin5min.partial_legs = {}
                self.price_history = defaultdict(lambda: deque(maxlen=600))
                self.coin_market_history = defaultdict(lambda: deque(maxlen=300))
                self.live_markets = []
                self.platforms["polymarket"]["status"] = "connecting"
                self.platforms["coinbase"]["status"] = "connecting"
                self.platforms["binance"]["status"] = "connecting"
                msg = "Coin5min démarrée."
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        self.alert("INFO", strategy.upper(), msg)
        return True, msg

    def pause_strategy(self, strategy: str):
        with self._lock:
            s = self.betmiss if strategy == "betmiss" else self.coin5min
            if not s.active:
                return
            s.paused = not s.paused
            s.status = "paused" if s.paused else "running"
            s.last_action = f"{'Pause' if s.paused else 'Reprise'} à {_now_str()}"
        self.alert("INFO", strategy.upper(), s.last_action)

    def stop_strategy(self, strategy: str):
        with self._lock:
            self._stop_one_locked(strategy)
        self.alert("INFO", strategy.upper(), f"{strategy} arrêtée")

    def _stop_one_locked(self, strategy: str):
        s = self.betmiss if strategy == "betmiss" else self.coin5min
        s.active = False
        s.paused = False
        s.status = "stopped"
        s.last_action = f"Stop à {_now_str()}"
        if strategy == "betmiss":
            s.opportunities = []
            self.platforms["kalshi"]["status"] = "disconnected"
        else:
            s.signals = {}
            s.markets = []
            s.orderbooks = {}
            s.spot_rows = {}
            s.reference_spot = {}
            s.reference_source = {}
            s.ref_cache = {}
            s.market_meta = {}
            s.current_market_ids = {}
            s.partial_legs = {}
            s.warmup_started_at = None
            self.live_markets = []
            self.platforms["coinbase"]["status"] = "disconnected"
            self.platforms["binance"]["status"] = "disconnected"
        self.platforms["polymarket"]["status"] = "disconnected"

    def _stop_strategies_locked(self, clear_runtime: bool = False):
        self._stop_one_locked("betmiss")
        self._stop_one_locked("coin5min")
        if clear_runtime:
            self.recent_trades = []
            self.live_markets = []
            self.coin5min.open_trade_ids = {}
            self.coin5min.open_trade_meta = {}
            self.coin5min.partial_legs = {}
            self.coin_market_history = defaultdict(lambda: deque(maxlen=300))


ST = LiveState()


# ============================================================
# Connectors parsing helpers
# ============================================================

def _parse_polymarket_markets(raw_markets: List[Dict]) -> List[Dict]:
    out = []
    for m in raw_markets:
        q = m.get("question") or m.get("event_title") or m.get("title") or ""
        slug = str(m.get("slug") or m.get("event_slug") or "")
        asset = _detect_asset(f"{q} {slug} {m.get('description') or ''}")
        token_ids = m.get("clobTokenIds", [])
        if isinstance(token_ids, str):
            try:
                token_ids = json.loads(token_ids)
            except Exception:
                token_ids = []
        if not token_ids:
            tokens = m.get("tokens") or m.get("outcomes") or []
            if isinstance(tokens, list):
                tmp = []
                for tok in tokens:
                    if not isinstance(tok, dict):
                        continue
                    tok_id = tok.get("token_id") or tok.get("id") or tok.get("tokenId")
                    if tok_id is not None:
                        tmp.append(str(tok_id))
                token_ids = tmp
        start_raw = m.get("startDate") or m.get("start_date") or m.get("startTime") or m.get("event_start_date")
        end_raw = m.get("endDate") or m.get("end_date") or m.get("endTime") or m.get("closedTime") or m.get("event_end_date")
        start_ts = None
        end_ts = None
        try:
            if start_raw:
                start_ts = pd.to_datetime(start_raw, utc=True).timestamp()
        except Exception:
            start_ts = None
        try:
            if end_raw:
                end_ts = pd.to_datetime(end_raw, utc=True).timestamp()
        except Exception:
            end_ts = None
        yes_price = _extract_yes_price(m)
        out.append(
            {
                "platform": "polymarket",
                "market_id": str(m.get("conditionId") or m.get("id") or slug or ""),
                "slug": slug,
                "question": q,
                "question_norm": _normalize_question(q),
                "asset": asset,
                "yes_price": yes_price,
                "no_price": 1 - yes_price,
                "token_yes": token_ids[0] if len(token_ids) >= 1 else None,
                "token_no": token_ids[1] if len(token_ids) >= 2 else None,
                "volume_24h": _safe_float(m.get("volume24hr") or m.get("volume24Hr") or m.get("volume_24hr") or 0),
                "liquidity": _safe_float(m.get("liquidity", 0)),
                "start_ts": start_ts,
                "end_ts": end_ts,
                "event_title": m.get("event_title") or m.get("title") or q,
                "event_slug": m.get("event_slug") or slug,
            }
        )
    return out


def _parse_kalshi_markets(raw_markets: List[Dict]) -> List[Dict]:
    out = []
    for m in raw_markets:
        q = m.get("title") or m.get("subtitle") or m.get("question") or m.get("ticker") or ""
        yes_bid = _safe_float(m.get("yes_bid") or m.get("yes_ask") or m.get("last_price") or 50, 50.0) / 100.0
        no_bid = 1 - yes_bid
        out.append(
            {
                "platform": "kalshi",
                "market_id": str(m.get("ticker", "")),
                "question": q,
                "question_norm": _normalize_question(q),
                "asset": _detect_asset(q),
                "yes_price": yes_bid,
                "no_price": no_bid,
                "volume_24h": _safe_float(m.get("volume", 0)),
                "liquidity": _safe_float(m.get("open_interest", 0)),
            }
        )
    return out


def _parse_interval_minutes(question: str) -> Optional[int]:
    q = question or ""
    m = re.search(r"(\d{1,2}):(\d{2})(AM|PM)\s*-\s*(\d{1,2}):(\d{2})(AM|PM)", q, re.I)
    if m:
        h1, mn1, ap1, h2, mn2, ap2 = m.groups()
        h1 = int(h1) % 12 + (12 if ap1.upper() == "PM" else 0)
        h2 = int(h2) % 12 + (12 if ap2.upper() == "PM" else 0)
        t1 = h1 * 60 + int(mn1)
        t2 = h2 * 60 + int(mn2)
        if t2 < t1:
            t2 += 24 * 60
        diff = t2 - t1
        if diff in (5, 10, 15, 30, 60):
            return diff
    ql = q.lower()
    if any(x in ql for x in ["15min", "15 min", "15-minute", "15 minute", "15m"]):
        return 15
    if any(x in ql for x in ["5min", "5 min", "5-minute", "5 minute", "5m"]):
        return 5
    return None


def _is_coin_window_market(market: Dict, asset: str, interval_min: int) -> bool:
    q = (market.get("question") or market.get("event_title") or "").lower()
    slug = (market.get("slug") or market.get("event_slug") or "").lower()
    text = f"{q} {slug}"
    detected_asset = market.get("asset")
    if detected_asset == "?":
        detected_asset = _detect_asset(text)
    if detected_asset != asset:
        return False
    interval = _parse_interval_minutes(text)
    if interval_min and interval and interval != interval_min:
        return False
    patterns = [
        "up or down", "updown", "above or below", "higher or lower",
        "up/down", "price above", "price below"
    ]
    return any(p in text for p in patterns)


def _time_left_sec(market: Optional[Dict]) -> Optional[float]:
    if not market:
        return None
    end_ts = market.get("end_ts")
    if not end_ts:
        return None
    return max(0.0, float(end_ts) - time.time())


def _choose_coin_markets(asset: str, markets: List[Dict], interval_min: int) -> Optional[Dict]:
    candidates = []
    now = time.time()
    for m in markets:
        if not _is_coin_window_market(m, asset, interval_min):
            continue
        m_interval = _parse_interval_minutes(f"{m.get('question','')} {m.get('slug','')} {m.get('event_title','')} {m.get('event_slug','')}")
        mm = dict(m)
        mm["interval_min"] = m_interval or interval_min
        start_ts = mm.get("start_ts")
        end_ts = mm.get("end_ts")
        if end_ts and end_ts < now - 10:
            continue

        phase = 2
        if start_ts and end_ts:
            if start_ts <= now <= end_ts:
                phase = 0
            elif now < start_ts and (start_ts - now) <= (interval_min * 60 + 90):
                phase = 1

        time_left = max(0.0, end_ts - now) if end_ts else 999999.0
        time_to_start = max(0.0, start_ts - now) if start_ts else 999999.0
        if phase == 2 and time_to_start > (2 * interval_min * 60 + 120):
            continue

        liquidity_score = mm.get("volume_24h", 0) + mm.get("liquidity", 0)
        candidates.append((phase, time_to_start, time_left, -liquidity_score, mm))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    return candidates[0][4]


def _simple_match_key(question: str) -> str:
    q = _normalize_question(question)
    for chunk in ["up or down", "will", " by ", " tomorrow", " today", " this week"]:
        q = q.replace(chunk, " ")
    return " ".join(q.split())


def _betmiss_tokens(question: str) -> set[str]:
    stop = {"will", "the", "a", "an", "be", "is", "are", "in", "on", "at", "of", "to", "for", "by", "and", "or", "up", "down", "minutes", "minute", "5", "15"}
    toks = {t for t in _normalize_question(question).split() if len(t) > 2 and t not in stop}
    return toks


def _betmiss_match_score(pm: Dict, ks: Dict) -> Tuple[float, str]:
    q1 = pm.get("question", "")
    q2 = ks.get("question", "")
    n1 = _normalize_question(q1)
    n2 = _normalize_question(q2)
    t1 = _betmiss_tokens(q1)
    t2 = _betmiss_tokens(q2)
    if not t1 or not t2:
        return 0.0, "empty_tokens"
    overlap = len(t1 & t2) / max(1, len(t1 | t2))
    nums1 = set(re.findall(r"\d+(?:\.\d+)?", n1))
    nums2 = set(re.findall(r"\d+(?:\.\d+)?", n2))
    number_bonus = 0.25 if nums1 and nums1 == nums2 else 0.0
    asset_bonus = 0.2 if pm.get("asset") != "?" and pm.get("asset") == ks.get("asset") else 0.0
    raw_ratio = difflib.SequenceMatcher(None, n1, n2).ratio()
    score = 0.55 * overlap + 0.25 * raw_ratio + number_bonus + asset_bonus
    reason = f"overlap={overlap:.2f} ratio={raw_ratio:.2f} nums={'match' if number_bonus else 'na'} asset={'match' if asset_bonus else 'na'}"
    return score, reason


# ============================================================
# Strategy logic helpers
# ============================================================

def _spot_trend_stats(history_rows, lookback_sec: int = 20 * 60) -> Dict:
    rows = list(history_rows or [])
    if not rows:
        return {"trend_bps": 0.0, "vol_bps": 0.0, "trend_dir": "flat"}
    now_ts = float(rows[-1][0]) if rows and isinstance(rows[-1], (tuple, list)) else time.time()
    cutoff = now_ts - float(lookback_sec)
    window = [(float(ts), float(px)) for ts, px in rows if float(ts) >= cutoff and np.isfinite(float(px))]
    if len(window) < 2:
        window = [(float(ts), float(px)) for ts, px in rows[-min(len(rows), 20):] if np.isfinite(float(px))]
    if len(window) < 2:
        return {"trend_bps": 0.0, "vol_bps": 0.0, "trend_dir": "flat"}
    start_px = max(window[0][1], 1e-9)
    end_px = window[-1][1]
    trend_bps = ((end_px / start_px) - 1.0) * 10000.0
    rets = []
    for i in range(1, len(window)):
        p0 = max(window[i-1][1], 1e-9)
        p1 = window[i][1]
        rets.append(((p1 / p0) - 1.0) * 10000.0)
    vol_bps = float(np.std(rets)) if rets else 0.0
    if trend_bps > 3.0:
        trend_dir = 'above'
    elif trend_bps < -3.0:
        trend_dir = 'below'
    else:
        trend_dir = 'flat'
    return {"trend_bps": float(trend_bps), "vol_bps": float(vol_bps), "trend_dir": trend_dir}


def _coin_signal_from_prices(prices: List[float], market_yes: float, market_no: float, ref_price: float, min_edge_pct: float, max_sum_yes_no: Optional[float] = None) -> Optional[Dict]:
    if not prices:
        return None
    current_price = float(prices[-1])
    yes_buy = float(market_yes)
    no_buy = float(market_no)
    total_cost = yes_buy + no_buy
    edge = 1.0 - total_cost
    direction = "above" if current_price >= ref_price else "below"
    move_bps = ((current_price / ref_price) - 1.0) * 10000 if ref_price else 0.0
    effective_cap = min(1.0 - (min_edge_pct / 100.0), float(max_sum_yes_no)) if max_sum_yes_no is not None else (1.0 - (min_edge_pct / 100.0))
    return {
        "direction": direction,
        "current_price": current_price,
        "reference_price": ref_price,
        "market_yes_price": yes_buy,
        "market_no_price": no_buy,
        "sum_yes_no": total_cost,
        "edge": edge,
        "max_sum_yes_no": effective_cap,
        "move_bps": move_bps,
        "actionable": total_cost <= effective_cap and total_cost < 1.0,
    }


def _coin_first_leg_plan(signal: Dict, time_left_sec: Optional[float], c5: Coin5MinState) -> Optional[Dict]:
    if not c5.legging_enabled:
        return None
    if time_left_sec is not None and time_left_sec < 15:
        return None

    yes_px = float(signal.get('market_yes_price', 1.0))
    no_px = float(signal.get('market_no_price', 1.0))
    if c5.leg1_directional:
        bias = signal.get('leg1_bias') or signal.get('direction')
        if bias == 'above':
            side = 'yes'
            first_price = yes_px
            hedge_price = no_px
        else:
            side = 'no'
            first_price = no_px
            hedge_price = yes_px
    else:
        if yes_px <= no_px:
            side = 'yes'
            first_price = yes_px
            hedge_price = no_px
        else:
            side = 'no'
            first_price = no_px
            hedge_price = yes_px
    if not (0 < first_price <= c5.max_first_leg_price):
        return None
    hedge_trigger = max(0.0, min(c5.max_sum_yes_no, 1.0 - (c5.min_edge_pct / 100.0)) - first_price)
    return {
        'side': side,
        'first_price': first_price,
        'hedge_side': 'no' if side == 'yes' else 'yes',
        'hedge_price_now': hedge_price,
        'hedge_trigger': hedge_trigger,
    }


def _resolve_single_leg(order: Optional[object], outcome: str) -> float:
    if order is None:
        return 0.0
    if getattr(order, 'status', None) in ('resolved_win', 'resolved_loss'):
        return float(getattr(order, 'pnl', 0.0) or 0.0)
    resolved = ST.paper.resolve_trade(order.id, outcome)
    return float(getattr(resolved, 'pnl', 0.0) or 0.0)


def _record_trade_row(order_id: str, strategy: str, asset: str, side: str, price: float, size: float, status: str, pnl: float = 0.0, confidence: Optional[float] = None, edge: Optional[float] = None, note: str = "") -> Dict:
    return {
        "id": order_id,
        "time": _now_str(),
        "timestamp": time.time(),
        "strategy": strategy,
        "asset": asset,
        "side": side.upper(),
        "price": round(price, 4),
        "size": round(size, 2),
        "pnl": round(pnl, 2),
        "status": status,
        "confidence": round(confidence, 4) if confidence is not None else None,
        "edge": round(edge, 4) if edge is not None else None,
        "latency": 0.0,
        "note": note,
    }


# ============================================================
# Main runtime loop
# ============================================================

def _poll_polymarket_markets(limit: int = 120, order: str = "volume24hr", ascending: bool = False, offset: int = 0) -> List[Dict]:
    t0 = time.time()
    try:
        raw = ST.feeds.get_polymarket_markets(limit=limit, order=order, ascending=ascending, offset=offset)
        latency = (time.time() - t0) * 1000
        with ST._lock:
            ST.platforms["polymarket"]["status"] = "connected"
            ST.platforms["polymarket"]["latency"] = round(latency, 1)
            ST.platforms["polymarket"]["requests"] += 1
        return _parse_polymarket_markets(raw)
    except Exception as exc:
        with ST._lock:
            ST.platforms["polymarket"]["status"] = "disconnected"
            ST.platforms["polymarket"]["errors"] += 1
        logger.warning("polymarket market discovery failed: %s", exc)
        return []


def _poll_polymarket_coin_markets(assets: List[str], interval_min: int) -> List[Dict]:
    wanted = set(assets)
    found: List[Dict] = []
    found_assets = set()

    # 1) Deterministic slug lookup around the current window.
    for asset in assets:
        direct = _poll_coin_market_by_slug(asset, interval_min)
        if direct:
            found.append(direct)
            found_assets.add(asset)

    # 2) Targeted public-search fallback for assets still missing.
    for asset in assets:
        if asset in found_assets:
            continue
        direct = _poll_coin_market_by_search(asset, interval_min)
        if direct:
            found.append(direct)
            found_assets.add(asset)

    if wanted.issubset(found_assets):
        return found

    # 3) Broad paginated discovery fallback.
    for offset in (0, 100, 200, 300):
        page = _poll_polymarket_markets(limit=100, order='end_date', ascending=True, offset=offset)
        for m in page:
            a = m.get('asset')
            if a in wanted and _is_coin_window_market(m, a, interval_min):
                found.append(m)
                found_assets.add(a)
        if wanted.issubset(found_assets):
            break
    unique = {}
    for m in found:
        unique[m.get('market_id') or m.get('slug') or m.get('event_slug')] = m
    return list(unique.values())


def _poll_kalshi_markets() -> List[Dict]:
    t0 = time.time()
    raw = ST.feeds.get_kalshi_markets(limit=120)
    latency = (time.time() - t0) * 1000
    with ST._lock:
        ST.platforms["kalshi"]["status"] = "connected"
        ST.platforms["kalshi"]["latency"] = round(latency, 1)
        ST.platforms["kalshi"]["requests"] += 1
    return _parse_kalshi_markets(raw)


def _fill_reference_from_history(asset: str, start_ts: Optional[float], fallback_price: float) -> float:
    if not start_ts:
        return float(fallback_price)
    hist = list(ST.price_history[asset])
    if not hist:
        return float(fallback_price)
    after = [p for ts, p in hist if float(ts) >= float(start_ts) - 1.0]
    if after:
        return float(after[0])
    before = [p for ts, p in hist if float(ts) <= float(start_ts) + 60.0]
    if before:
        return float(before[-1])
    return float(fallback_price)


def _get_true_polymarket_reference_price(mkt: Dict, asset: str, fallback_price: float) -> Tuple[float, str]:
    slug = str(mkt.get('slug') or mkt.get('event_slug') or '').strip()
    if not slug:
        return float(fallback_price), 'history_fallback'
    now = time.time()
    with ST._lock:
        cached = ST.coin5min.ref_cache.get(slug)
    if cached and (now - float(cached.get('ts', 0.0))) <= 30.0 and cached.get('ref_px'):
        return float(cached['ref_px']), str(cached.get('source') or 'polymarket_page_cache')
    try:
        ref_px = ST.feeds.get_polymarket_price_to_beat_by_slug(slug)
        if ref_px is not None and np.isfinite(ref_px) and ref_px > 0:
            with ST._lock:
                ST.coin5min.ref_cache[slug] = {'ref_px': float(ref_px), 'source': 'polymarket_page', 'ts': now}
            return float(ref_px), 'polymarket_page'
    except Exception as exc:
        logger.info('polymarket price-to-beat fetch failed for %s: %s', slug, exc)
    hist_ref = _fill_reference_from_history(asset, mkt.get('start_ts'), fallback_price)
    with ST._lock:
        ST.coin5min.ref_cache[slug] = {'ref_px': float(hist_ref), 'source': 'history_fallback', 'ts': now}
    return float(hist_ref), 'history_fallback'


def _poll_spot_ticker(asset: str, interval_min: int, source: str = 'binance') -> Dict:
    source = (source or 'binance').lower()
    history = ST.price_history[asset]
    ts = time.time()

    if source == 'binance':
        symbol = f"{asset}USDT"
        if len(history) < ST.coin5min.warmup_required:
            t0 = time.time()
            klines = ST.feeds.get_binance_klines(symbol, interval=_binance_interval(interval_min), limit=max(60, ST.coin5min.warmup_required * 3))
            latency = (time.time() - t0) * 1000
            with ST._lock:
                ST.platforms['binance']['status'] = 'connected'
                ST.platforms['binance']['latency'] = round(latency, 1)
                ST.platforms['binance']['requests'] += 1
            for row in klines[-max(20, ST.coin5min.warmup_required * 2):]:
                open_ts = float(row[0]) / 1000.0
                close_px = _safe_float(row[4], np.nan)
                if np.isfinite(close_px):
                    history.append((open_ts, float(close_px)))

        t0 = time.time()
        ticker = ST.feeds.get_binance_book_ticker(symbol)
        latency = (time.time() - t0) * 1000
        bid = _safe_float(ticker.get('bidPrice'), np.nan)
        ask = _safe_float(ticker.get('askPrice'), np.nan)
        price = float(np.nanmean([bid, ask])) if np.isfinite(bid) and np.isfinite(ask) else (_safe_float(ticker.get('price'), np.nan))
        with ST._lock:
            ST.platforms['binance']['status'] = 'connected'
            ST.platforms['binance']['latency'] = round(latency, 1)
            ST.platforms['binance']['requests'] += 1
            ST.platforms['coinbase']['status'] = 'disconnected'
        if np.isfinite(price):
            history.append((ts, float(price)))
        return {'asset': asset, 'price': float(price), 'bid': float(bid), 'ask': float(ask), 'timestamp': ts, 'source': 'binance'}

    product_id = f"{asset}-USD"
    if len(history) < ST.coin5min.warmup_required:
        granularity = _coinbase_granularity(interval_min)
        t0 = time.time()
        candles = ST.feeds.get_coinbase_candles(product_id, granularity=granularity)
        latency = (time.time() - t0) * 1000
        with ST._lock:
            ST.platforms['coinbase']['status'] = 'connected'
            ST.platforms['coinbase']['latency'] = round(latency, 1)
            ST.platforms['coinbase']['requests'] += 1
            ST.platforms['binance']['status'] = 'disconnected'
        for row in candles[-max(20, ST.coin5min.warmup_required * 2):]:
            cts, _low, _high, _open, close, _vol = row
            history.append((float(cts), float(close)))

    t0 = time.time()
    ticker = ST.feeds.get_coinbase_ticker(product_id)
    latency = (time.time() - t0) * 1000
    price = _safe_float(ticker.get('price'), np.nan)
    bid = _safe_float(ticker.get('bid'), price)
    ask = _safe_float(ticker.get('ask'), price)
    with ST._lock:
        ST.platforms['coinbase']['status'] = 'connected'
        ST.platforms['coinbase']['latency'] = round(latency, 1)
        ST.platforms['coinbase']['requests'] += 1
        ST.platforms['binance']['status'] = 'disconnected'
    if np.isfinite(price):
        history.append((ts, float(price)))
    return {'asset': asset, 'price': float(price), 'bid': float(bid), 'ask': float(ask), 'timestamp': ts, 'source': 'coinbase'}


def _candidate_coin_slugs(asset: str, interval_min: int, now_ts: Optional[float] = None) -> List[str]:
    now_ts = float(now_ts or time.time())
    base_start = _round_window_start(now_ts, interval_min)
    base_end = _round_window_end(now_ts, interval_min)
    candidates = []

    # Crypto Up/Down event slugs commonly behave like asset-updown-5m-<window_end_ts>.
    # Keep start-based variants too as fallback because Gamma payloads have not always been stable.
    explicit = []
    for ts in (base_end, base_end + interval_min * 60):
        explicit.extend([
            f"{asset.lower()}-updown-{int(interval_min)}m-{int(ts)}",
            f"{asset.lower()}-up-or-down-{int(interval_min)}m-{int(ts)}",
            f"{asset.lower()}-up-down-{int(interval_min)}m-{int(ts)}",
        ])

    for shift in (-2, -1, 0, 1, 2, 3, 4):
        start_ts = base_start + shift * interval_min * 60
        end_ts = start_ts + interval_min * 60
        for ts in (end_ts, start_ts):
            candidates.append(f"{asset.lower()}-updown-{int(interval_min)}m-{int(ts)}")
            candidates.append(f"{asset.lower()}-up-or-down-{int(interval_min)}m-{int(ts)}")
            candidates.append(f"{asset.lower()}-up-down-{int(interval_min)}m-{int(ts)}")
    return list(dict.fromkeys(explicit + candidates))



def _asset_search_terms(asset: str) -> List[str]:
    mapping = {
        'BTC': ['Bitcoin Up or Down', 'BTC Up or Down', 'Bitcoin Up or Down - 5 Minutes', 'BTC Up or Down - 5 Minutes'],
        'ETH': ['Ethereum Up or Down', 'ETH Up or Down', 'Ethereum Up or Down - 5 Minutes', 'ETH Up or Down - 5 Minutes'],
        'SOL': ['Solana Up or Down', 'SOL Up or Down'],
        'XRP': ['XRP Up or Down', 'Ripple Up or Down'],
        'DOGE': ['Dogecoin Up or Down', 'DOGE Up or Down'],
    }
    return mapping.get(asset, [f'{asset} Up or Down'])


def _parse_polymarket_event_markets(ev: Dict) -> List[Dict]:
    markets = []
    if not isinstance(ev, dict):
        return markets
    for m in ev.get('markets', []) or []:
        if not isinstance(m, dict):
            continue
        mm = dict(m)
        mm.setdefault('event_id', ev.get('id'))
        mm.setdefault('event_slug', ev.get('slug'))
        mm.setdefault('event_title', ev.get('title'))
        mm.setdefault('event_description', ev.get('description'))
        mm['startDate'] = mm.get('startDate') or ev.get('startDate')
        mm['endDate'] = mm.get('endDate') or ev.get('endDate')
        markets.append(mm)
    return _parse_polymarket_markets(markets)


def _poll_coin_market_by_slug(asset: str, interval_min: int) -> Optional[Dict]:
    last_exc = None
    for slug in _candidate_coin_slugs(asset, interval_min):
        try:
            ev = ST.feeds.get_polymarket_event_by_slug(slug)
            parsed = _parse_polymarket_event_markets(ev)
            for m in parsed:
                if _is_coin_window_market(m, asset, interval_min):
                    return m
        except Exception as exc:
            last_exc = exc
        try:
            raw = ST.feeds.get_polymarket_market_by_slug(slug)
            parsed = _parse_polymarket_markets([raw])
            for m in parsed:
                if _is_coin_window_market(m, asset, interval_min):
                    return m
        except Exception as exc:
            last_exc = exc
            continue
    if last_exc:
        logger.info("coin slug lookup miss for %s %sm: %s", asset, interval_min, last_exc)
    return None


def _poll_coin_market_by_search(asset: str, interval_min: int) -> Optional[Dict]:
    queries = []
    for term in _asset_search_terms(asset):
        queries.append(term)
        queries.append(f"{term} {interval_min} Minutes")
        queries.append(f"{term} {interval_min}m")
    seen = set()
    queries = [q for q in queries if not (q in seen or seen.add(q))]

    best = None
    best_score = None
    last_exc = None
    for q in queries:
        try:
            res = ST.feeds.get_polymarket_public_search(q=q, page=1, limit_per_type=10, events_status='active')
            events = res.get('events') or []
            for ev in events:
                for m in _parse_polymarket_event_markets(ev):
                    if not _is_coin_window_market(m, asset, interval_min):
                        continue
                    score = _choose_coin_markets(asset, [m], interval_min)
                    if score is not None:
                        mm = score
                    else:
                        mm = m
                    time_left = _time_left_sec(mm) or 999999.0
                    running = 0 if (mm.get('start_ts') and mm.get('end_ts') and mm.get('start_ts') <= time.time() <= mm.get('end_ts')) else 1
                    rank = (running, time_left, -(mm.get('volume_24h',0)+mm.get('liquidity',0)))
                    if best_score is None or rank < best_score:
                        best_score = rank
                        best = mm
            if best is not None:
                return best
        except Exception as exc:
            last_exc = exc
            continue
    if last_exc:
        logger.info("coin search lookup miss for %s %sm: %s", asset, interval_min, last_exc)
    return None


def _sync_trade_row(order_id: str, pnl: float, status: str, note: str = ""):
    with ST._lock:
        for row in ST.recent_trades:
            if row.get("id") == order_id:
                row["pnl"] = round(pnl, 2)
                row["status"] = status
                if note:
                    row["note"] = note
                row["time"] = _now_str()
                row["timestamp"] = time.time()
                break


def _run_coin5min_cycle():
    with ST._lock:
        c5 = ST.coin5min
        if not c5.active and not c5.open_trade_meta:
            return
        if ST.kill_switch and not c5.open_trade_meta:
            c5.status = "stopped"
            return
        if c5.paused and not c5.open_trade_meta:
            c5.status = "paused"
            return
        c5.status = "connecting" if c5.status == "stopped" else c5.status
        assets = list(c5.assets)
        interval_min = c5.interval_min
        min_edge = c5.min_edge_pct
        allocation_pct = c5.allocation_pct
        trade_size_pct = c5.trade_size_pct
        spot_source = c5.spot_source

    markets = _poll_polymarket_coin_markets(assets, interval_min)
    selected_markets: List[Dict] = []
    waiting_assets: List[str] = []

    with ST._lock:
        c5.markets = []
        ST.live_markets = []

    for asset in assets:
        spot = _poll_spot_ticker(asset, interval_min, source=spot_source)
        with ST._lock:
            c5.spot_rows[asset] = spot

        warm_points = len(ST.price_history[asset])
        if warm_points < c5.warmup_required:
            with ST._lock:
                c5.signals.pop(asset, None)
            continue

        mkt = _choose_coin_markets(asset, markets, interval_min)
        if not mkt:
            waiting_assets.append(asset)
            with ST._lock:
                c5.market_meta[asset] = {
                    "asset": asset,
                    "status": "waiting_market",
                    "message": f"Aucun marché Polymarket {interval_min}m trouvé pour {asset}",
                    "candidate_slugs": _candidate_coin_slugs(asset, interval_min),
                    "search_queries": [f"{asset} Up or Down", f"{asset} Up or Down {interval_min} Minutes"],
                    "spot": float(spot["price"]),
                }
                c5.signals.pop(asset, None)
            continue

        selected_markets.append(mkt)
        time_left = _time_left_sec(mkt)

        with ST._lock:
            prev_market_id = c5.current_market_ids.get(asset)
            if prev_market_id != mkt["market_id"]:
                ST.coin_market_history[asset] = deque(maxlen=300)
                c5.current_market_ids[asset] = mkt["market_id"]
                ref_px, ref_src = _get_true_polymarket_reference_price(mkt, asset, float(spot["price"]))
                c5.reference_spot[asset] = float(ref_px)
                c5.reference_source[asset] = str(ref_src)
                c5.last_action = f"Marché {asset} détecté à {_now_str()}"
            elif asset not in c5.reference_spot or c5.reference_source.get(asset) != 'polymarket_page':
                ref_px, ref_src = _get_true_polymarket_reference_price(mkt, asset, float(spot["price"]))
                c5.reference_spot[asset] = float(ref_px)
                c5.reference_source[asset] = str(ref_src)
            ref_price = c5.reference_spot.get(asset, float(spot["price"]))
            ref_source = c5.reference_source.get(asset, 'history_fallback')
            c5.market_meta[asset] = {
                "asset": asset,
                "market_id": mkt["market_id"],
                "question": mkt.get("question", ""),
                "slug": mkt.get("slug", ""),
                "interval_min": mkt.get("interval_min", interval_min),
                "time_left_sec": round(time_left, 1) if time_left is not None else None,
                "ref_px": round(ref_price, 4),
                "ref_source": ref_source,
                "resolution_source": "chainlink_stream",
                "spot": round(float(spot["price"]), 4),
                "spot_source": spot.get("source"),
                "yes_mark": round(float(mkt.get("yes_price", 0.0)), 4),
                "no_mark": round(float(mkt.get("no_price", 0.0)), 4),
                "status": "market_found",
                "window_start_ts": mkt.get("start_ts"),
                "window_end_ts": mkt.get("end_ts"),
                "is_running_window": bool(mkt.get("start_ts") and mkt.get("end_ts") and mkt.get("start_ts") <= time.time() <= mkt.get("end_ts")),
                "max_trade_usd": round(float(c5.max_trade_usd), 2),
            }

        yes_buy = float(mkt.get("yes_price", 0.5))
        no_buy = float(mkt.get("no_price", 0.5))
        yes_bid = yes_ask = no_bid = no_ask = None
        yes_bid_size = yes_ask_size = no_bid_size = no_ask_size = None
        bid_depth = ask_depth = 0.0
        try:
            if mkt.get("token_yes"):
                book_yes = ST.feeds.get_polymarket_orderbook(mkt["token_yes"])
                bids_yes = book_yes.get("bids", [])
                asks_yes = book_yes.get("asks", [])
                if bids_yes:
                    yes_bid = float(bids_yes[0].get("price", 0))
                    yes_bid_size = float(bids_yes[0].get("size", 0))
                if asks_yes:
                    yes_ask = float(asks_yes[0].get("price", 0))
                    yes_ask_size = float(asks_yes[0].get("size", 0))
                bid_depth += sum(float(x.get("size", 0)) for x in bids_yes[:8])
                ask_depth += sum(float(x.get("size", 0)) for x in asks_yes[:8])
            if mkt.get("token_no"):
                book_no = ST.feeds.get_polymarket_orderbook(mkt["token_no"])
                bids_no = book_no.get("bids", [])
                asks_no = book_no.get("asks", [])
                if bids_no:
                    no_bid = float(bids_no[0].get("price", 0))
                    no_bid_size = float(bids_no[0].get("size", 0))
                if asks_no:
                    no_ask = float(asks_no[0].get("price", 0))
                    no_ask_size = float(asks_no[0].get("size", 0))
                bid_depth += sum(float(x.get("size", 0)) for x in bids_no[:8])
                ask_depth += sum(float(x.get("size", 0)) for x in asks_no[:8])
        except Exception:
            pass

        yes_buy = yes_ask if yes_ask is not None else yes_buy
        no_buy = no_ask if no_ask is not None else no_buy
        with ST._lock:
            c5.orderbooks[asset] = {
                "yes_best_bid": yes_bid,
                "yes_best_ask": yes_ask,
                "yes_bid_size": yes_bid_size,
                "yes_ask_size": yes_ask_size,
                "no_best_bid": no_bid,
                "no_best_ask": no_ask,
                "no_bid_size": no_bid_size,
                "no_ask_size": no_ask_size,
                "bid_depth": bid_depth,
                "ask_depth": ask_depth,
            }
            ref_price = c5.reference_spot.get(asset, float(spot["price"]))

        prices = [p for _ts, p in ST.price_history[asset]]
        trend = _spot_trend_stats(ST.price_history[asset], lookback_sec=max(300, int(c5.trend_lookback_min) * 60))
        signal = _coin_signal_from_prices(prices, yes_buy, no_buy, ref_price, min_edge, c5.max_sum_yes_no)
        if signal:
            directional_bias = signal.get('direction')
            if c5.leg1_directional and abs(float(trend.get('trend_bps', 0.0))) >= 5.0 and trend.get('trend_dir') in ('above', 'below'):
                directional_bias = trend.get('trend_dir')
            signal.update({
                "trend_bps": round(float(trend.get('trend_bps', 0.0)), 2),
                "trend_vol_bps": round(float(trend.get('vol_bps', 0.0)), 2),
                "trend_dir": trend.get('trend_dir'),
                "leg1_bias": directional_bias,
                "asset": asset,
                "warmup_points": warm_points,
                "timestamp": time.time(),
                "interval_min": mkt.get("interval_min", interval_min),
                "question": mkt.get("question", ""),
                "market_id": mkt.get("market_id"),
                "time_left_sec": round(time_left, 1) if time_left is not None else None,
                "yes_mark_price": round(float(mkt.get("yes_price", 0.0)), 4),
                "no_mark_price": round(float(mkt.get("no_price", 0.0)), 4),
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "no_bid": no_bid,
                "no_ask": no_ask,
                "spot_source": spot.get("source"),
                "max_trade_usd": round(float(c5.max_trade_usd), 2),
            })
            with ST._lock:
                c5.signals[asset] = signal
                c5.market_meta[asset].update({
                    "yes_ask": round(yes_buy, 4),
                    "no_ask": round(no_buy, 4),
                    "sum_yes_no": round(signal["sum_yes_no"], 4),
                    "edge_pct": round(signal["edge"] * 100.0, 3),
                    "direction": signal.get("direction"),
                    "leg1_bias": signal.get("leg1_bias"),
                    "trend_bps": signal.get("trend_bps"),
                    "trend_dir": signal.get("trend_dir"),
                    "max_trade_usd": round(float(c5.max_trade_usd), 2),
                    "max_sum_yes_no": round(signal.get("max_sum_yes_no", c5.max_sum_yes_no), 4),
                })

        hist_row = {
            "ts": time.time(),
            "spot": round(float(spot["price"]), 6),
            "ref_px": round(float(ref_price), 6),
            "yes_mark": round(float(mkt.get("yes_price", 0.0)), 6),
            "no_mark": round(float(mkt.get("no_price", 0.0)), 6),
            "yes_bid": round(float(yes_bid), 6) if yes_bid is not None else None,
            "yes_ask": round(float(yes_buy), 6),
            "no_bid": round(float(no_bid), 6) if no_bid is not None else None,
            "no_ask": round(float(no_buy), 6),
            "sum_yes_no": round(float(yes_buy + no_buy), 6),
            "time_left_sec": round(time_left, 3) if time_left is not None else None,
            "market_id": mkt.get("market_id"),
            "question": mkt.get("question", ""),
        }
        with ST._lock:
            ST.coin_market_history[asset].append(hist_row)

        now_ts = time.time()
        with ST._lock:
            open_id = c5.open_trade_ids.get(asset)
            open_meta = c5.open_trade_meta.get(asset)
            partial_leg = c5.partial_legs.get(asset)
        if open_id and open_meta and now_ts >= open_meta["expiry_ts"]:
            outcome = "yes" if spot["price"] >= open_meta["reference_price"] else "no"
            yes_order = ST.paper.resolve_trade(open_meta["yes_order_id"], outcome)
            no_order = ST.paper.resolve_trade(open_meta["no_order_id"], outcome)
            pair_pnl = 0.0
            if yes_order:
                pair_pnl += yes_order.pnl
            if no_order:
                pair_pnl += no_order.pnl
            status = "WIN" if pair_pnl >= 0 else "LOSS"
            _sync_trade_row(
                open_id,
                pair_pnl,
                status,
                note=f"expiry | ref={open_meta['reference_price']:.2f} final={spot['price']:.2f} outcome={outcome.upper()}",
            )
            with ST._lock:
                c5.open_trade_ids.pop(asset, None)
                c5.open_trade_meta.pop(asset, None)
                c5.reference_spot[asset] = float(spot["price"])
                c5.reference_source[asset] = "post_expiry_spot"
                open_id = None
                open_meta = None

        if partial_leg and now_ts >= partial_leg["expiry_ts"]:
            outcome = "yes" if spot["price"] >= partial_leg["reference_price"] else "no"
            first_order = ST.paper.resolve_trade(partial_leg["first_order_id"], outcome)
            leg_pnl = float(first_order.pnl) if first_order else 0.0
            _sync_trade_row(
                partial_leg["pair_id"],
                leg_pnl,
                "LEG1_WIN" if leg_pnl >= 0 else "LEG1_LOSS",
                note=f"leg1 expiry | {partial_leg['first_side'].upper()} only | ref={partial_leg['reference_price']:.2f} final={spot['price']:.2f}",
            )
            with ST._lock:
                c5.partial_legs.pop(asset, None)
                c5.reference_spot[asset] = float(spot["price"])
                c5.reference_source[asset] = "post_expiry_spot"
                partial_leg = None

        with ST._lock:
            open_exists = asset in c5.open_trade_ids
            partial_leg = c5.partial_legs.get(asset)
        total_budget_pct = ST.capital * (allocation_pct / 100.0) * (trade_size_pct / 100.0)
        total_budget = min(float(total_budget_pct), float(c5.max_trade_usd))
        total_cost = max(signal["sum_yes_no"], 1e-6) if signal else None
        trend_bias_used = None
        if signal:
            trend_bias_used = signal.get("leg1_bias") if c5.leg1_directional else "cheapest_leg"
        window_running = bool(mkt.get("start_ts") and mkt.get("end_ts") and mkt.get("start_ts") <= now_ts <= mkt.get("end_ts"))
        if signal and not ST.kill_switch and total_budget >= 1.0 and window_running:
            if signal.get("actionable") and not open_exists and not partial_leg and total_cost and total_cost < 1.0:
                yes_budget = total_budget * (yes_buy / total_cost)
                no_budget = total_budget * (no_buy / total_cost)
                yes_order = ST.paper.place_order(
                    strategy="coin5min",
                    platform="polymarket",
                    market_id=mkt["market_id"],
                    question=mkt["question"],
                    side="yes",
                    market_price=yes_buy,
                    size=yes_budget,
                    asset=asset,
                    direction="above",
                    edge=signal["edge"],
                )
                no_order = ST.paper.place_order(
                    strategy="coin5min",
                    platform="polymarket",
                    market_id=mkt["market_id"],
                    question=mkt["question"],
                    side="no",
                    market_price=no_buy,
                    size=no_budget,
                    asset=asset,
                    direction="below",
                    edge=signal["edge"],
                )
                if yes_order and no_order:
                    pair_id = f"C5ARB-{int(time.time())}-{asset}"
                    row = _record_trade_row(
                        pair_id,
                        "coin5min",
                        asset,
                        "yes+no",
                        signal["sum_yes_no"],
                        total_budget,
                        "OPEN",
                        pnl=0.0,
                        edge=signal["edge"],
                        note=f"SIMUL YES {yes_buy:.4f} + NO {no_buy:.4f} | ref={ref_price:.2f} | {interval_min}m | budget=${total_budget:.2f}",
                    )
                    with ST._lock:
                        ST.recent_trades.insert(0, row)
                        c5.open_trade_ids[asset] = pair_id
                        c5.open_trade_meta[asset] = {
                            "market_id": mkt["market_id"],
                            "yes_order_id": yes_order.id,
                            "no_order_id": no_order.id,
                            "reference_price": ref_price,
                            "entry_spot": spot["price"],
                            "entry_ts": time.time(),
                            "expiry_ts": float(mkt.get("end_ts") or (time.time() + mkt.get("interval_min", interval_min) * 60)),
                            "market_question": mkt["question"],
                            "mode": "simultaneous",
                        }
                        open_exists = True
            elif partial_leg and not open_exists:
                hedge_side = partial_leg["hedge_side"]
                hedge_price_now = no_buy if hedge_side == "no" else yes_buy
                locked_cost = partial_leg["first_price"] + hedge_price_now
                locked_edge = 1.0 - locked_cost
                if hedge_price_now <= partial_leg["hedge_trigger"] and locked_cost < 1.0:
                    hedge_budget = max(1.0, float(partial_leg.get("remaining_budget", total_budget * 0.5)))
                    hedge_order = ST.paper.place_order(
                        strategy="coin5min",
                        platform="polymarket",
                        market_id=mkt["market_id"],
                        question=mkt["question"],
                        side=hedge_side,
                        market_price=hedge_price_now,
                        size=hedge_budget,
                        asset=asset,
                        direction="above" if hedge_side == "yes" else "below",
                        edge=locked_edge,
                    )
                    if hedge_order:
                        yes_order_id = partial_leg["first_order_id"] if partial_leg["first_side"] == "yes" else hedge_order.id
                        no_order_id = partial_leg["first_order_id"] if partial_leg["first_side"] == "no" else hedge_order.id
                        _sync_trade_row(
                            partial_leg["pair_id"],
                            0.0,
                            "OPEN",
                            note=(
                                f"LEGGED {partial_leg['first_side'].upper()} {partial_leg['first_price']:.4f} -> "
                                f"{hedge_side.upper()} {hedge_price_now:.4f} | locked {locked_cost:.4f} | edge {locked_edge*100:.2f}%"
                            ),
                        )
                        with ST._lock:
                            c5.partial_legs.pop(asset, None)
                            c5.open_trade_ids[asset] = partial_leg["pair_id"]
                            c5.open_trade_meta[asset] = {
                                "market_id": mkt["market_id"],
                                "yes_order_id": yes_order_id,
                                "no_order_id": no_order_id,
                                "reference_price": partial_leg["reference_price"],
                                "entry_spot": partial_leg["entry_spot"],
                                "entry_ts": partial_leg["entry_ts"],
                                "expiry_ts": partial_leg["expiry_ts"],
                                "market_question": mkt["question"],
                                "mode": "legged",
                            }
                            open_exists = True
                            partial_leg = None
            elif not partial_leg and not open_exists:
                plan = _coin_first_leg_plan(signal, time_left, c5)
                if plan:
                    first_budget = total_budget * 0.5
                    first_order = ST.paper.place_order(
                        strategy="coin5min",
                        platform="polymarket",
                        market_id=mkt["market_id"],
                        question=mkt["question"],
                        side=plan["side"],
                        market_price=plan["first_price"],
                        size=first_budget,
                        asset=asset,
                        direction=signal.get("direction"),
                        edge=signal.get("edge"),
                    )
                    if first_order:
                        pair_id = f"C5LEG-{int(time.time())}-{asset}"
                        row = _record_trade_row(
                            pair_id,
                            "coin5min",
                            asset,
                            f"{plan['side']}_leg",
                            plan["first_price"],
                            first_budget,
                            "LEG1_OPEN",
                            pnl=0.0,
                            edge=signal.get("edge"),
                            note=(
                                f"LEG1 {plan['side'].upper()} {plan['first_price']:.4f} | wait {plan['hedge_side'].upper()} <= "
                                f"{plan['hedge_trigger']:.4f} | max {c5.max_leg_hold_sec}s | budget=${first_budget:.2f}"
                            ),
                        )
                        with ST._lock:
                            ST.recent_trades.insert(0, row)
                            c5.partial_legs[asset] = {
                                "pair_id": pair_id,
                                "market_id": mkt["market_id"],
                                "market_question": mkt["question"],
                                "first_side": plan["side"],
                                "first_price": plan["first_price"],
                                "first_order_id": first_order.id,
                                "hedge_side": plan["hedge_side"],
                                "hedge_trigger": plan["hedge_trigger"],
                                "remaining_budget": total_budget - first_budget,
                                "entry_spot": spot["price"],
                                "reference_price": ref_price,
                                "entry_ts": time.time(),
                                "expiry_ts": min(float(mkt.get("end_ts") or (time.time() + mkt.get("interval_min", interval_min) * 60)), time.time() + c5.max_leg_hold_sec),
                            }
                            partial_leg = c5.partial_legs.get(asset)

        with ST._lock:
            leg_state = "OPEN_PAIR" if asset in c5.open_trade_ids else (f"LEG1_{partial_leg['first_side'].upper()}" if partial_leg else "FLAT")
            hedge_trigger = partial_leg.get("hedge_trigger") if partial_leg else None
            if asset in c5.signals:
                c5.signals[asset].update({
                    "leg_state": leg_state,
                    "hedge_trigger": hedge_trigger,
                    "legging_enabled": c5.legging_enabled,
                    "leg1_directional": c5.leg1_directional,
                    "max_leg_hold_sec": c5.max_leg_hold_sec,
                    "max_first_leg_price": c5.max_first_leg_price,
                    "max_sum_yes_no": c5.max_sum_yes_no,
                    "budget_pct": round(float(trade_size_pct), 2),
                    "budget_cap_usd": round(float(c5.max_trade_usd), 2),
                    "budget_effective_usd": round(float(total_budget), 2),
                    "trend_bias_used": trend_bias_used,
                })
            if asset in c5.market_meta:
                c5.market_meta[asset].update({
                    "leg_state": leg_state,
                    "hedge_trigger": round(float(hedge_trigger), 4) if hedge_trigger is not None else None,
                    "leg1_directional": c5.leg1_directional,
                    "max_leg_hold_sec": c5.max_leg_hold_sec,
                    "max_first_leg_price": c5.max_first_leg_price,
                    "max_sum_yes_no": c5.max_sum_yes_no,
                    "budget_pct": round(float(trade_size_pct), 2),
                    "budget_cap_usd": round(float(c5.max_trade_usd), 2),
                    "budget_effective_usd": round(float(total_budget), 2),
                    "trend_bias_used": trend_bias_used,
                })

    with ST._lock:
        c5.markets = selected_markets
        ST.live_markets = list(selected_markets)
        if not assets:
            c5.status = "stopped"
        elif any(len(ST.price_history[a]) < c5.warmup_required for a in assets):
            c5.status = "warming"
        elif not selected_markets:
            c5.status = "waiting_market"
            c5.last_action = f"Warmup prêt – en attente d'un marché Polymarket {interval_min}m détectable pour {', '.join(waiting_assets or assets)}"
        elif any(not bool((m.get('start_ts') and m.get('end_ts') and m.get('start_ts') <= time.time() <= m.get('end_ts'))) for m in selected_markets):
            c5.status = "next_market"
            c5.last_action = "Marché suivant détecté – préchargement du carnet et attente d'ouverture"
        else:
            c5.status = "running"
        c5.last_update_ts = time.time()

def _run_betmiss_cycle():
    with ST._lock:
        bm = ST.betmiss
        if not bm.active:
            return
        if bm.paused:
            bm.status = "paused"
            return
        bm.status = "connecting" if bm.status == "stopped" else bm.status
        min_edge = bm.min_edge_pct / 100.0
        allocation_pct = bm.allocation_pct
        trade_size_pct = bm.trade_size_pct

    pm = _poll_polymarket_markets()
    ks = _poll_kalshi_markets()

    candidate_pairs: List[Tuple[float, Dict, Dict, str]] = []
    for a in pm:
        for b in ks:
            score, reason = _betmiss_match_score(a, b)
            if score < 0.52:
                continue
            if a.get("asset") != "?" and b.get("asset") != "?" and a.get("asset") != b.get("asset"):
                continue
            candidate_pairs.append((score, a, b, reason))

    candidate_pairs.sort(key=lambda x: x[0], reverse=True)
    opportunities: List[Dict] = []
    seen_pairs = set()
    for score, a, b, reason in candidate_pairs[:200]:
        pair_key = (a.get("market_id"), b.get("market_id"))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        combos = [
            ("pm_yes_ks_no", a["yes_price"] + b["no_price"], "polymarket", "kalshi", a, b),
            ("ks_yes_pm_no", b["yes_price"] + a["no_price"], "kalshi", "polymarket", b, a),
        ]
        for label, total_cost, buy_yes_on, buy_no_on, yes_mkt, no_mkt in combos:
            edge = 1.0 - total_cost
            if edge <= min_edge:
                continue
            opp = {
                "event": yes_mkt["question"],
                "asset": yes_mkt.get("asset") or no_mkt.get("asset") or "?",
                "buy_yes_on": buy_yes_on,
                "buy_no_on": buy_no_on,
                "yes_price": round(yes_mkt["yes_price"], 4),
                "no_price": round(no_mkt["no_price"], 4),
                "edge": round(edge, 4),
                "label": label,
                "match_score": round(score, 3),
                "match_reason": reason,
            }
            opportunities.append(opp)

    opportunities.sort(key=lambda x: x["edge"], reverse=True)
    opportunities = opportunities[:12]

    with ST._lock:
        bm.opportunities = opportunities
        bm.last_update_ts = time.time()
        bm.status = "running" if opportunities or (pm and ks) else "connecting"

    # auto paper-trade best opportunity if no identical event already logged recently
    if opportunities:
        best = opportunities[0]
        size = ST.capital * (allocation_pct / 100.0) * (trade_size_pct / 100.0)
        guaranteed_pnl = size * best["edge"]
        row_id = f"BM-{int(time.time())}-{best['asset']}"
        recent_same = False
        with ST._lock:
            for row in ST.recent_trades[:10]:
                if row.get("id") == row_id:
                    recent_same = True
                    break
        if not recent_same and size >= 1.0:
            row = _record_trade_row(
                row_id,
                "betmiss",
                best["asset"],
                "arb",
                best["yes_price"] + best["no_price"],
                size,
                "WIN",
                pnl=guaranteed_pnl,
                edge=best["edge"],
                note=f"YES {best['buy_yes_on']} / NO {best['buy_no_on']}",
            )
            with ST._lock:
                ST.betmiss_realized_pnl += guaranteed_pnl
                ST.recent_trades.insert(0, row)


def _runtime_loop():
    while not ST._stop_event.is_set():
        try:
            any_active = False
            with ST._lock:
                any_active = ST.betmiss.active or ST.coin5min.active or bool(ST.coin5min.open_trade_meta)

            if ST.betmiss.active:
                _run_betmiss_cycle()
            if ST.coin5min.active or ST.coin5min.open_trade_meta:
                _run_coin5min_cycle()

            if not any_active:
                with ST._lock:
                    ST.last_tick_ts = time.time()
                time.sleep(1.0)
                continue

            with ST._lock:
                ST.last_tick_ts = time.time()
        except Exception as exc:
            logger.exception("runtime loop error: %s", exc)
            with ST._lock:
                ST.platforms["polymarket"]["errors"] += 1
            ST.alert("CRIT", "SYS", f"Erreur runtime: {exc}")
        time.sleep(2.0)


# ============================================================
# Public API for dashboard
# ============================================================

def start_workers():
    with ST._lock:
        if ST._started:
            return
        ST._started = True
    init_db()
    threading.Thread(target=_runtime_loop, daemon=True).start()
    ST.alert("INFO", "SYS", "Workers lancés. Aucune donnée live ne sera chargée avant Start.")


def get_bt_history() -> pd.DataFrame:
    try:
        conn = sqlite3.connect(DB)
        df = pd.read_sql(
            "SELECT strategy,total_pnl,n_trades,win_rate,sharpe,max_drawdown,roi FROM backtest_results ORDER BY created_at DESC LIMIT 10",
            conn,
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def get_pnl_curve() -> List[Dict]:
    s = ST.snap()
    trades = sorted(s["recent_trades"], key=lambda x: x.get("timestamp", 0))
    curve = []
    cum = 0.0
    for t in trades:
        cum += t.get("pnl", 0.0)
        curve.append({"time": t.get("time", ""), "pnl": round(cum, 2), "strategy": t.get("strategy", "")})
    return curve

