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
import math
import httpx
import numpy as np
import pandas as pd

_SQLITE_CONNECT_ORIG = sqlite3.connect
def _sqlite_connect_threadsafe(*args, **kwargs):
    kwargs.setdefault('check_same_thread', False)
    return _SQLITE_CONNECT_ORIG(*args, **kwargs)
sqlite3.connect = _sqlite_connect_threadsafe

from paper_trading import PaperTradingEngine
from execution_engine import ExecutionEngine, ExecMode, build_engine_from_env

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
    target_sum_yes_no: float = 0.985
    min_entry_time_left_sec: int = 45
    no_new_entry_last_sec: int = 30
    order_requote_sec: int = 8
    max_trade_usd: float = 20.0
    trend_lookback_min: int = 20
    taker_fee_bps: float = 30.0
    extra_slippage_bps: float = 8.0
    depth_fill_ratio: float = 0.5
    min_prob_edge: float = 0.06
    stop_adverse_move_bps: float = 18.0
    max_daily_drawdown_pct: float = 8.0
    max_consecutive_losses: int = 10
    pause_duration_sec: int = 1800  # Durée de pause en secondes (par défaut 30min)
    realized_fee_drag: float = 0.0
    realized_slippage_drag: float = 0.0
    consecutive_losses: int = 0
    peak_equity: float = 0.0
    risk_pause_until_ts: Optional[float] = None
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
    # --- Action #4 (fill control + analysis log) ---
    realistic_fill_enabled: bool = True  # re-check live book before each paper fill
    fill_price_tolerance: float = 0.005  # accept asks within this much of intended
    analysis_log_enabled: bool = False
    analysis_log_path: str = "coin5min_analysis.jsonl"
    analysis_events_count: int = 0
    fill_rejected_count: int = 0
    fill_accepted_count: int = 0


# ============================================================
# Thread-safe live state
# ============================================================

class LiveState:

    def auto_alerts(self):
        """Déclenche des alertes automatiques sur PnL, connexion, positions anormales, uniquement pour les stratégies actives et sans doublons."""
        s = self.snap()
        pf = s.get("portfolio", {})
        # PnL drawdown
        if pf.get("total_pnl", 0) < -abs(self.capital) * 0.15:
            self._alert_once("WARN", "PNL", f"Drawdown important: PnL={pf.get('total_pnl', 0):.2f}")
        # PnL négatif persistant
        if pf.get("total_pnl", 0) < 0:
            self._alert_once("INFO", "PNL", f"PnL négatif: {pf.get('total_pnl', 0):.2f}")

        # Plateformes à surveiller selon stratégies actives
        platforms_needed = set()
        if self.coin5min.active:
            platforms_needed.update(["binance", "coinbase", "polymarket"])
        if self.betmiss.active:
            platforms_needed.update(["polymarket", "kalshi"])

        # Connexion serveur lost (uniquement plateformes utiles et pas déjà alertées)
        for plat, platinfo in s.get("platforms", {}).items():
            if plat not in platforms_needed:
                continue
            if platinfo.get("status") == "disconnected":
                self._alert_once("ERROR", plat.upper(), f"Connexion perdue avec {plat}")

        # Position anormale (trade ouvert > 30min)
        now = time.time()
        for meta in s.get("coin5min", {}).get("open_trade_meta", {}).values():
            entry = meta.get("entry_ts")
            if entry and now - entry > 1800:
                self._alert_once("WARN", "COIN5MIN", f"Position ouverte anormalement longue: {meta.get('market_id','?')} depuis {int((now-entry)/60)} min")

    def _alert_once(self, level: str, strat: str, msg: str):
        """Ajoute une alerte seulement si elle n'est pas déjà la dernière du même type/message."""
        if self.alerts and self.alerts[0]["level"] == level and self.alerts[0]["strat"] == strat and self.alerts[0]["msg"] == msg:
            return
        self.alert(level, strat, msg)

    def save_trades_to_csv(self, filename: str = "coin5min_trades.csv", hours: int = 2):
        """Sauvegarde les trades des 2 dernières heures dans un CSV."""
        import csv
        cutoff = time.time() - hours * 3600
        with self._lock:
            rows = [t for t in self.recent_trades if t.get("timestamp", 0) >= cutoff]
        if not rows:
            return
        keys = list(rows[0].keys())
        with open(filename, "w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    def set_execution_config(self, mode: str = "paper", max_order_usd: float = 3.0, private_key: str = "", funder: str = "", signature_type: int = 2, chain_id: int = 137) -> Tuple[bool, str]:
        """Configure et (re)connecte l'execution engine avec tous les paramètres live/paper."""
        mode = mode.lower().strip()
        if mode not in ("paper", "live"):
            return False, f"Mode invalide: {mode}. Utilise 'paper' ou 'live'."
        with self._lock:
            self.exec_mode = mode
            self.max_order_usd = min(50.0, max(0.5, float(max_order_usd)))
            try:
                self.execution = build_engine_from_env(
                    mode=mode,
                    max_order_usd=self.max_order_usd,
                )
                # Si live, on injecte les credentials
                if mode == "live" and self.execution:
                    if hasattr(self.execution, "_private_key"):
                        self.execution._private_key = private_key
                    if hasattr(self.execution, "_funder"):
                        self.execution._funder = funder
                    if hasattr(self.execution, "_signature_type"):
                        self.execution._signature_type = int(signature_type)
                    if hasattr(self.execution, "_chain_id"):
                        self.execution._chain_id = int(chain_id)
                ok = self.execution.connect() if self.execution else False
                if not ok:
                    return False, "Connexion à l'execution engine échouée."
            except Exception as exc:
                self.execution = None
                return False, f"Erreur lors de la configuration: {exc}"
        self.alert("INFO", "EXEC", f"Execution engine configuré: mode={mode}, max_order_usd={self.max_order_usd}, live={mode=='live'}")
        return True, f"Execution engine configuré: mode={mode}, max_order_usd={self.max_order_usd}, live={mode=='live'}"
    def __init__(self):
        self._lock = threading.RLock()
        self._started = False
        self._stop_event = threading.Event()

        self.mode = "paper_live"
        self.exec_mode = "paper"  # "paper" or "live" — controls real order routing
        self.max_order_usd = 3.0  # hard cap per order for live mode
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
        if hasattr(self.paper, "slippage_bps"):
            self.paper.slippage_bps = 0.0
        self.betmiss_realized_pnl = 0.0
        self.coin5min_realism_drag = 0.0
        self.coin5min.peak_equity = self.capital

        # --- Real execution engine (paper by default, switchable to live) ---
        self.execution: Optional[ExecutionEngine] = None
        self._init_execution_engine()

        self.feeds = PublicFeeds(timeout=6.0)

    def _init_execution_engine(self):
        """Build execution engine from env vars. Safe to call multiple times."""
        try:
            self.execution = build_engine_from_env(
                mode=self.exec_mode,
                max_order_usd=self.max_order_usd,
            )
            self.execution.connect()
            logger.info("Execution engine initialised: mode=%s, max=$%.2f", self.exec_mode, self.max_order_usd)
        except Exception as exc:
            logger.warning("Execution engine init failed (paper fallback): %s", exc)
            self.execution = None

    def set_exec_mode(self, mode: str, max_order_usd: Optional[float] = None) -> Tuple[bool, str]:
        """Switch between paper and live execution. Thread-safe."""
        mode = mode.lower().strip()
        if mode not in ("paper", "live"):
            return False, f"Mode invalide: {mode}. Utilise 'paper' ou 'live'."
        with self._lock:
            if max_order_usd is not None:
                self.max_order_usd = min(50.0, max(0.5, float(max_order_usd)))
            old_mode = self.exec_mode
            self.exec_mode = mode
            # Cancel all live orders before switching
            if self.execution and old_mode == "live":
                self.execution.cancel_all_orders()
            self._init_execution_engine()
        self.alert("WARN" if mode == "live" else "INFO", "EXEC",
                   f"Execution mode: {old_mode} → {mode} | max=${self.max_order_usd:.2f}")
        return True, f"Mode d'exécution changé en {mode}."

    # ---------- state snapshots ----------
    def snap(self) -> Dict:
        with self._lock:
            metrics = self.paper.get_metrics()
            total_pnl = metrics["total_pnl"] + self.betmiss_realized_pnl - self.coin5min_realism_drag
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
                    "partial_legs": {k: dict(v) for k, v in self.coin5min.partial_legs.items()},
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
                "execution": {
                    "exec_mode": self.exec_mode,
                    "max_order_usd": self.max_order_usd,
                    "connected": self.execution.is_connected if self.execution else False,
                    "killed": self.execution._killed if self.execution else False,
                    "live_orders": self.execution.list_orders(states=["OPEN", "PARTIAL", "POSTING"]) if self.execution else [],
                    "recent_alerts": self.execution.get_alerts(limit=20) if self.execution else [],
                    "balance_usdc": self.execution.get_balance() if self.execution and self.exec_mode == "live" else None,
                },
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
            if hasattr(self.paper, "slippage_bps"):
                self.paper.slippage_bps = 0.0
            self.betmiss_realized_pnl = 0.0
            self.coin5min_realism_drag = 0.0
            self.coin5min.realized_fee_drag = 0.0
            self.coin5min.realized_slippage_drag = 0.0
            self.coin5min.consecutive_losses = 0
            self.coin5min.peak_equity = self.capital
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

            # Reset execution engine (cancel any live orders, re-init)
            if self.execution:
                self.execution.cancel_all_orders()
            self._init_execution_engine()

        self.alert("INFO", "SYS", f"Session réinitialisée | capital=${self.capital:,.0f} | BM {self.allocations['betmiss']:.0f}% | C5 {self.allocations['coin5min']:.0f}% | exec={self.exec_mode}")
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
        realistic_fill_enabled: Optional[bool] = None,
        analysis_log_enabled: Optional[bool] = None,
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
            if realistic_fill_enabled is not None and bool(realistic_fill_enabled) != self.coin5min.realistic_fill_enabled:
                self.coin5min.realistic_fill_enabled = bool(realistic_fill_enabled)
                changed = True
            if analysis_log_enabled is not None and bool(analysis_log_enabled) != self.coin5min.analysis_log_enabled:
                self.coin5min.analysis_log_enabled = bool(analysis_log_enabled)
                if self.coin5min.analysis_log_enabled:
                    self.coin5min.analysis_events_count = 0  # reset counter on enable
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
                # Also kill the real execution engine
                if self.execution:
                    self.execution.kill()
                msg = "KILL SWITCH ACTIVÉ – toutes les stratégies et ordres live sont stoppés."
            else:
                if self.execution:
                    self.execution.unkill()
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

    def set_coin5min_risk_params(self, max_consecutive_losses: Optional[int] = None, pause_duration_sec: Optional[int] = None):
        with self._lock:
            if max_consecutive_losses is not None:
                self.coin5min.max_consecutive_losses = int(max_consecutive_losses)
            if pause_duration_sec is not None:
                self.coin5min.pause_duration_sec = int(pause_duration_sec)
        self.alert("INFO", "COIN5MIN", f"Risk params mis à jour: max_consecutive_losses={self.coin5min.max_consecutive_losses}, pause_duration_sec={self.coin5min.pause_duration_sec}s")

    def resume_coin5min_trading(self):
        with self._lock:
            self.coin5min.risk_pause_until_ts = None
        self.alert("INFO", "COIN5MIN", "Trading relancé manuellement (pause annulée)")


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
    """Pick the best Polymarket short-window market for `asset` at the requested interval.

    Action #2 hardening (revised):
      * Strictly reject already-expired markets (no 10s grace).
      * Accept the market only if we can verify it's a genuine `interval_min` window:
          either (a) the interval is parseable from the text and matches, OR
                 (b) `end_ts - start_ts` is within tolerance of `interval_min * 60`.
        At least one verification must pass — this stops long "Above $X by EOD" markets
        from being treated as 5-min ones.
    """
    candidates = []
    now = time.time()
    expected_span = interval_min * 60
    span_lo = expected_span * 0.5
    span_hi = expected_span * 1.5
    rejects = 0
    for m in markets:
        if not _is_coin_window_market(m, asset, interval_min):
            continue
        text_bag = f"{m.get('question','')} {m.get('slug','')} {m.get('event_title','')} {m.get('event_slug','')}"
        m_interval = _parse_interval_minutes(text_bag)
        mm = dict(m)
        mm["interval_min"] = m_interval or interval_min
        start_ts = mm.get("start_ts")
        end_ts = mm.get("end_ts")

        # strict expiry: any market whose window has already closed is unusable
        if end_ts and end_ts <= now:
            continue

        # interval verification: trust text parse OR span check
        ok_text = (m_interval == interval_min)
        ok_span = False
        if start_ts and end_ts:
            span = float(end_ts) - float(start_ts)
            ok_span = (span_lo <= span <= span_hi)
        if not (ok_text or ok_span):
            rejects += 1
            logger.debug(
                "[c5/select] skip %s market_id=%s slug=%s: parsed_interval=%s start=%s end=%s",
                asset, mm.get("market_id"), mm.get("slug") or mm.get("event_slug"),
                m_interval, start_ts, end_ts,
            )
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
        if rejects:
            logger.info(
                "[c5/select] %s interval=%dm: %d markets rejected (no interval match), 0 candidates",
                asset, interval_min, rejects,
            )
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



def _normalize_book_side(rows, descending: bool) -> List[Dict]:
    norm = []
    for row in rows or []:
        if isinstance(row, dict):
            price = _safe_float(row.get("price"), np.nan)
            size = _safe_float(row.get("size"), 0.0)
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            price = _safe_float(row[0], np.nan)
            size = _safe_float(row[1], 0.0)
        else:
            continue
        if not np.isfinite(price) or price <= 0 or price > 1.0:
            continue
        if size <= 0:
            continue
        norm.append({"price": float(price), "size": float(size)})
    norm.sort(key=lambda x: x["price"], reverse=descending)
    return norm


def _best_executable_from_book(book: Dict, mark_price: float, side_name: str) -> Dict:
    bids = _normalize_book_side(book.get("bids", []), descending=True)
    asks = _normalize_book_side(book.get("asks", []), descending=False)

    best_bid = bids[0]["price"] if bids else None
    best_ask = asks[0]["price"] if asks else None
    bid_size = bids[0]["size"] if bids else None
    ask_size = asks[0]["size"] if asks else None

    spread = None
    if best_bid is not None and best_ask is not None:
        spread = max(0.0, float(best_ask) - float(best_bid))

    # Filter clearly aberrant best ask / bid against mark.
    sane_ask = best_ask
    sane_bid = best_bid
    if mark_price is not None and np.isfinite(mark_price):
        if sane_ask is not None and abs(sane_ask - mark_price) > 0.18:
            sane_ask = None
            ask_size = None
        if sane_bid is not None and abs(sane_bid - mark_price) > 0.18:
            sane_bid = None
            bid_size = None

    # Aggressive paper proxy:
    # if executable ask is missing/aberrant, fall back to a tiny premium over mark
    # to avoid dead markets showing 0.99/0.99 forever.
    exec_buy = sane_ask
    if exec_buy is None and mark_price is not None and np.isfinite(mark_price):
        exec_buy = min(0.995, max(0.005, float(mark_price) + 0.01))

    exec_sell = sane_bid
    if exec_sell is None and mark_price is not None and np.isfinite(mark_price):
        exec_sell = min(0.995, max(0.005, float(mark_price) - 0.01))

    return {
        "best_bid": sane_bid,
        "best_ask": sane_ask,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "exec_buy": exec_buy,
        "exec_sell": exec_sell,
        "spread": spread,
        "raw_best_bid": best_bid,
        "raw_best_ask": best_ask,
    }


def _compute_frequency_opportunity(
    yes_mark: float,
    no_mark: float,
    yes_exec: Optional[float],
    no_exec: Optional[float],
    time_left_sec: Optional[float],
    c5: Coin5MinState,
    trend_bps: float,
    bias: str,
    p_up: Optional[float] = None,
    p_down: Optional[float] = None,
) -> Dict:
    exec_sum = None
    if yes_exec is not None and no_exec is not None:
        exec_sum = float(yes_exec) + float(no_exec)

    mark_sum = float(yes_mark) + float(no_mark)
    trade_cap = float(c5.target_sum_yes_no)
    wait_cap = min(0.999, float(c5.max_sum_yes_no) + 0.005)

    if time_left_sec is None:
        time_score = 0.0
    elif time_left_sec <= float(c5.no_new_entry_last_sec):
        time_score = 0.0
    elif time_left_sec >= 180:
        time_score = 1.0
    elif time_left_sec >= 60:
        time_score = 0.70
    else:
        time_score = 0.35

    trend_score = min(1.0, abs(float(trend_bps)) / 20.0)
    prob_edge = 0.0
    if p_up is not None and p_down is not None:
        prob_edge = max(abs(float(p_up) - 0.5), abs(float(p_down) - 0.5)) * 2.0
    spread_proxy = None
    if yes_exec is not None and no_exec is not None:
        spread_proxy = abs(float(yes_exec) - float(yes_mark)) + abs(float(no_exec) - float(no_mark))
    spread_score = max(0.0, 1.0 - (spread_proxy or 0.0) / 0.08)

    if exec_sum is None:
        sum_score = 0.0
    elif exec_sum <= trade_cap:
        sum_score = 1.0
    elif exec_sum <= wait_cap:
        sum_score = max(0.0, 1.0 - (exec_sum - trade_cap) / max(1e-6, wait_cap - trade_cap))
    else:
        sum_score = 0.0

    opp_score = round(0.45 * sum_score + 0.15 * trend_score + 0.10 * spread_score + 0.15 * time_score + 0.15 * min(1.0, prob_edge), 4)

    if time_left_sec is not None and time_left_sec <= float(c5.no_new_entry_last_sec):
        decision = "SKIP"
        reason = "last_30s_block"
    elif time_left_sec is not None and time_left_sec < float(c5.min_entry_time_left_sec):
        decision = "SKIP"
        reason = "too_late"
    elif exec_sum is not None and exec_sum <= trade_cap and prob_edge >= float(c5.min_prob_edge):
        decision = "TRADE"
        reason = "exec_sum_below_target"
    elif exec_sum is not None and exec_sum < 1.0:
        decision = "WAIT"
        reason = "watch_limit_fill"
    else:
        decision = "SKIP"
        reason = "exec_sum_too_high"

    return {
        "mark_sum_yes_no": round(mark_sum, 4),
        "exec_sum_yes_no": round(exec_sum, 4) if exec_sum is not None else None,
        "target_sum_yes_no": round(trade_cap, 4),
        "opp_score": opp_score,
        "sum_score": round(sum_score, 4),
        "trend_score_component": round(trend_score, 4),
        "prob_score": round(min(1.0, prob_edge), 4),
        "spread_score": round(spread_score, 4),
        "time_score": round(time_score, 4),
        "decision": decision,
        "decision_reason": reason,
        "edge_exec": (1.0 - exec_sum) if exec_sum is not None else None,
        "bias": bias,
    }


def _choose_leg1_frequency(signal: Dict, c5: Coin5MinState) -> Optional[Dict]:
    yes_exec = signal.get("yes_exec_buy")
    no_exec = signal.get("no_exec_buy")
    if yes_exec is None and no_exec is None:
        return None

    p_up = float(signal.get("p_up", 0.5) or 0.5)
    p_down = float(signal.get("p_down", 0.5) or 0.5)
    yes_ev = p_up - float(yes_exec) if yes_exec is not None else -999.0
    no_ev = p_down - float(no_exec) if no_exec is not None else -999.0

    time_left = signal.get("time_left_sec")
    if time_left is not None:
        if float(time_left) <= float(c5.no_new_entry_last_sec):
            return None
        if float(time_left) < float(c5.min_entry_time_left_sec):
            return None

    target_edge = max(0.005, 1.0 - float(c5.target_sum_yes_no))
    min_prob_edge = float(c5.min_prob_edge)

    bias = signal.get("leg1_bias") or signal.get("direction") or "above"

    candidates = []
    if yes_exec is not None and yes_exec <= c5.max_first_leg_price and yes_ev >= min_prob_edge:
        score = yes_ev + (0.02 if bias == "above" else 0.0)
        candidates.append(("yes", float(yes_exec), score))
    if no_exec is not None and no_exec <= c5.max_first_leg_price and no_ev >= min_prob_edge:
        score = no_ev + (0.02 if bias == "below" else 0.0)
        candidates.append(("no", float(no_exec), score))

    if not candidates:
        return None

    side, px, _score = max(candidates, key=lambda x: x[2])
    return {
        "side": side,
        "first_price": px,
        "hedge_side": "no" if side == "yes" else "yes",
        "hedge_trigger": max(0.01, 1.0 - px - target_edge),
        "leg1_ev": round(yes_ev if side == "yes" else no_ev, 4),
    }

def _resolve_single_leg(order: Optional[object], outcome: str) -> float:
    if order is None:
        return 0.0
    if getattr(order, 'status', None) in ('resolved_win', 'resolved_loss'):
        return float(getattr(order, 'pnl', 0.0) or 0.0)
    resolved = ST.paper.resolve_trade(order.id, outcome)
    return float(getattr(resolved, 'pnl', 0.0) or 0.0)



def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))


def _estimate_direction_probability(asset: str, spot: float, ref_price: float, trend: Dict, time_left_sec: Optional[float]) -> Dict:
    dist_pct = 0.0 if not ref_price else (float(spot) - float(ref_price)) / float(ref_price)
    trend_bps = float(trend.get("trend_bps", 0.0) or 0.0)
    vol_bps = max(1e-6, float(trend.get("vol_bps", 0.0) or 0.0))
    time_factor = 1.0
    if time_left_sec is not None:
        if time_left_sec < 60:
            time_factor = 1.2
        elif time_left_sec > 240:
            time_factor = 0.85
    z = (dist_pct * 1800.0 + (trend_bps / max(6.0, vol_bps * 1.5))) * time_factor
    p_up = _sigmoid(z)
    return {
        "p_up": float(p_up),
        "p_down": float(1.0 - p_up),
        "dist_ref_pct": float(dist_pct),
    }


def _cap_size_by_depth(size_usd: float, price: float, level_size: Optional[float], depth_fill_ratio: float) -> float:
    if level_size is None or price is None or price <= 0:
        return max(0.0, float(size_usd) * 0.35)
    max_usd = float(level_size) * float(price) * max(0.1, float(depth_fill_ratio))
    return max(0.0, min(float(size_usd), max_usd))


def _execution_adjustment(price: float, size_usd: float, spread: Optional[float], fee_bps: float, extra_slippage_bps: float) -> Dict:
    px = max(0.01, min(0.99, float(price)))
    spread_cost = 0.0 if spread is None else max(0.0, float(spread)) * 0.20
    size_impact = max(0.0, float(size_usd)) * (float(extra_slippage_bps) / 10000.0) / max(px, 0.05)
    adj_price = min(0.99, px + spread_cost + size_impact)
    fee_usd = float(size_usd) * (float(fee_bps) / 10000.0)
    slippage_usd = max(0.0, adj_price - px) * (float(size_usd) / max(adj_price, 0.05))
    return {
        "exec_price": float(adj_price),
        "fee_usd": float(fee_usd),
        "slippage_usd": float(slippage_usd),
    }


def _risk_guard_coin5min(c5: Coin5MinState, capital: float, current_total_pnl: float) -> Tuple[bool, str]:
    eq = float(capital) + float(current_total_pnl) - float(c5.realized_fee_drag) - float(c5.realized_slippage_drag)
    c5.peak_equity = max(float(c5.peak_equity or capital), eq)
    dd = 0.0 if c5.peak_equity <= 0 else (c5.peak_equity - eq) / c5.peak_equity * 100.0
    now = time.time()
    # Si une pause est active, on vérifie si elle est terminée
    if c5.risk_pause_until_ts and now < float(c5.risk_pause_until_ts):
        return False, "risk_pause"
    # Drawdown max journalier
    if dd >= float(c5.max_daily_drawdown_pct):
        c5.risk_pause_until_ts = now + getattr(c5, "pause_duration_sec", 1800)
        if hasattr(ST, "alert"):
            ST.alert("WARN", "COIN5MIN", f"PAUSE max_drawdown: drawdown={dd:.2f}% ({getattr(c5, 'pause_duration_sec', 1800)//60}min)")
        return False, "max_drawdown"
    # Pertes consécutives
    if int(c5.consecutive_losses) >= int(c5.max_consecutive_losses):
        c5.risk_pause_until_ts = now + getattr(c5, "pause_duration_sec", 1800)
        if hasattr(ST, "alert"):
            ST.alert("WARN", "COIN5MIN", f"PAUSE consecutive_losses: {c5.consecutive_losses} pertes consécutives ({getattr(c5, 'pause_duration_sec', 1800)//60}min)")
        return False, f"consecutive_losses ({c5.consecutive_losses})"
    return True, ""


def _apply_realism_drag(pnl: float, fee_usd: float, slippage_usd: float) -> float:
    return float(pnl) - float(fee_usd) - float(slippage_usd)

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

    def _sane_ref(val: float, spot: float) -> bool:
        if not np.isfinite(val) or val <= 0:
            return False
        lo = max(100.0, float(spot) * 0.5)
        hi = float(spot) * 1.5
        return lo <= float(val) <= hi

    try:
        ref_px = ST.feeds.get_polymarket_price_to_beat_by_slug(slug)
        if ref_px is not None and _sane_ref(float(ref_px), float(fallback_price)):
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
    # Sauvegarde automatique CSV après chaque trade
    try:
        ST.save_trades_to_csv()
    except Exception as e:
        logger.warning(f"Erreur sauvegarde CSV: {e}")


def _route_live_order(asset: str, mkt: Dict, side: str, exec_price: float, budget_usd: float, signal: Optional[Dict], pair_id: str):
    """
    Route an order to the real execution engine (if live mode).
    Paper engine is ALWAYS called separately — this is additive.
    Does nothing in paper mode.
    """
    if not ST.execution or ST.exec_mode != "live":
        return
    if not ST.execution.is_connected:
        ST.alert("WARN", "EXEC", f"Live engine not connected — skipping {asset} {side}")
        return

    token_key = "token_yes" if side.lower() == "yes" else "token_no"
    token_id = mkt.get(token_key, "")
    if not token_id:
        ST.alert("WARN", "EXEC", f"No token_id for {asset} {side} — skipping live order")
        return

    live_budget = min(float(budget_usd), float(ST.max_order_usd))
    if live_budget < 0.10:
        return

    try:
        resp = ST.execution.place_limit_order(
            symbol=asset,
            token_id=token_id,
            side="BUY",
            price=exec_price,
            size=live_budget,
            order_type="GTC",
            tags={
                "strategy": "coin5min",
                "asset": asset,
                "leg": side,
                "pair_id": pair_id,
                "edge": str(round(signal.get("edge", 0.0), 4)) if signal else "0",
            },
        )
        mode_tag = "[LIVE]" if ST.exec_mode == "live" else "[PAPER]"
        ST.alert(
            "INFO" if resp.state in ("FILLED", "OPEN") else "WARN",
            "EXEC",
            f"{mode_tag} {resp.state} {side.upper()} {asset} ${live_budget:.2f} @ {exec_price:.4f} | {resp.clob_order_id or 'no-id'}",
        )
    except Exception as exc:
        ST.alert("CRIT", "EXEC", f"Live order failed {asset} {side}: {exc}")


def _realistic_fill_check(token_id: Optional[str], intended_price: float, intended_size_usd: float, side: str, c5: Coin5MinState) -> Optional[Dict]:
    """Re-fetch the live L2 book and decide whether a paper order would have filled.

    Returns:
        None  -> reject the fill (price drifted away, or no liquidity)
        dict  -> {exec_price, filled_size_usd, ask_at_check, ask_size_at_check, raw_book}
    """
    if not token_id or not c5.realistic_fill_enabled:
        # Either no token to check against, or fill control is disabled → accept as-is.
        return {
            "exec_price": float(intended_price),
            "filled_size_usd": float(intended_size_usd),
            "ask_at_check": None,
            "ask_size_at_check": None,
            "raw_book": None,
            "fill_mode": "no_check",
        }
    try:
        raw = ST.feeds.get_polymarket_orderbook(token_id)
    except Exception as exc:
        logger.info("[c5/fill] %s book fetch failed: %s — falling back to no-check fill", side, exc)
        return {
            "exec_price": float(intended_price),
            "filled_size_usd": float(intended_size_usd),
            "ask_at_check": None,
            "ask_size_at_check": None,
            "raw_book": None,
            "fill_mode": "no_check_fetch_error",
        }
    asks = _normalize_book_side(raw.get("asks", []), descending=False)
    if not asks:
        return None  # empty book, cannot fill at all
    best_ask = float(asks[0]["price"])
    best_ask_size_shares = float(asks[0]["size"])  # number of YES/NO shares at the level
    # convert shares to USD at this price
    best_ask_size_usd = best_ask_size_shares * best_ask
    # is the ask still within tolerance of what the bot intended?
    tol = float(c5.fill_price_tolerance or 0.005)
    if best_ask > (intended_price + tol):
        return None  # price moved up beyond tolerance → no fill
    # accepted: pay the worse of (intended, current ask)
    exec_price = max(float(intended_price), best_ask)
    # cap fill size by the depth at the top level (depth_fill_ratio of it, like the cycle does)
    cap_usd = best_ask_size_usd * float(c5.depth_fill_ratio or 0.5)
    filled = min(float(intended_size_usd), cap_usd)
    return {
        "exec_price": float(exec_price),
        "filled_size_usd": float(filled),
        "ask_at_check": best_ask,
        "ask_size_at_check": best_ask_size_shares,
        "raw_book": {"asks": asks[:5], "bids": _normalize_book_side(raw.get("bids", []), descending=True)[:5]},
        "fill_mode": "checked",
    }


def _log_analysis_event(event_type: str, payload: Dict) -> None:
    """Append one JSON line to the analysis log file. No-op if disabled."""
    c5 = ST.coin5min
    if not c5.analysis_log_enabled:
        return
    path = c5.analysis_log_path or "coin5min_analysis.jsonl"
    rec = {
        "event": event_type,
        "ts": time.time(),
        "ts_iso": datetime.utcfromtimestamp(time.time()).isoformat() + "Z",
        **payload,
    }
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        with ST._lock:
            c5.analysis_events_count += 1
    except Exception as exc:
        logger.warning("[c5/analysis] failed to write event %s: %s", event_type, exc)


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
    # diag: summarize the raw pool so we can see whether Polymarket gave us anything
    by_asset_count: Dict[str, int] = {}
    for _m in markets or []:
        a = _m.get("asset") or "?"
        by_asset_count[a] = by_asset_count.get(a, 0) + 1
    logger.info(
        "[c5/poll] interval=%dm assets=%s pooled=%d by_asset=%s",
        interval_min, assets, len(markets or []), by_asset_count,
    )
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
                # Action #2: rotate out stale market context so the next valid
                # window forces a fresh ref_px fetch (market_changed=True path).
                stale_market_id = c5.current_market_ids.pop(asset, None)
                stale_slug = (c5.market_meta.get(asset) or {}).get("slug")
                c5.reference_spot.pop(asset, None)
                c5.reference_source.pop(asset, None)
                if stale_slug:
                    c5.ref_cache.pop(stale_slug, None)
                c5.market_meta[asset] = {
                    "asset": asset,
                    "status": "waiting_market",
                    "message": f"Aucun marché Polymarket {interval_min}m trouvé pour {asset}",
                    "candidate_slugs": _candidate_coin_slugs(asset, interval_min),
                    "search_queries": [f"{asset} Up or Down", f"{asset} Up or Down {interval_min} Minutes"],
                    "spot": float(spot["price"]),
                }
                c5.signals.pop(asset, None)
            if stale_market_id:
                logger.info(
                    "[c5/rotate] asset=%s no valid market found, dropped stale market_id=%s slug=%s",
                    asset, stale_market_id, stale_slug,
                )
            continue

        selected_markets.append(mkt)
        time_left = _time_left_sec(mkt)

        with ST._lock:
            prev_market_id = c5.current_market_ids.get(asset)
            market_changed = (prev_market_id != mkt["market_id"])
            if market_changed:
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

        # --- DIAG #1: trace ref_px / market_id / slug each tick (Action #1) ---
        logger.info(
            "[c5/tick] asset=%s market_id=%s slug=%s prev=%s changed=%s "
            "start=%s end=%s time_left=%.1fs ref_px=%.6f ref_src=%s spot=%.6f spot_src=%s",
            asset,
            mkt.get("market_id"),
            mkt.get("slug") or mkt.get("event_slug"),
            prev_market_id,
            market_changed,
            mkt.get("start_ts"),
            mkt.get("end_ts"),
            float(time_left) if time_left is not None else -1.0,
            float(ref_price),
            ref_source,
            float(spot["price"]),
            spot.get("source"),
        )

        with ST._lock:
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
                # expose token ids so the dashboard can re-fetch the L2 book directly
                "token_yes": mkt.get("token_yes"),
                "token_no": mkt.get("token_no"),
            }

        yes_mark = float(mkt.get("yes_price", 0.5))
        no_mark = float(mkt.get("no_price", 0.5))
        yes_bid = yes_ask = no_bid = no_ask = None
        yes_bid_size = yes_ask_size = no_bid_size = no_ask_size = None
        yes_exec_buy = no_exec_buy = None
        bid_depth = ask_depth = 0.0
        # L2 ladders (top N) for the dashboard view
        yes_bid_levels: List[Dict] = []
        yes_ask_levels: List[Dict] = []
        no_bid_levels: List[Dict] = []
        no_ask_levels: List[Dict] = []
        N_LEVELS = 8
        try:
            if mkt.get("token_yes"):
                book_yes = ST.feeds.get_polymarket_orderbook(mkt["token_yes"])
                yes_book = _best_executable_from_book(book_yes, yes_mark, "yes")
                yes_bid = yes_book["best_bid"]
                yes_ask = yes_book["best_ask"]
                yes_bid_size = yes_book["bid_size"]
                yes_ask_size = yes_book["ask_size"]
                yes_exec_buy = yes_book["exec_buy"]
                yes_bid_levels = _normalize_book_side(book_yes.get("bids", []), descending=True)[:N_LEVELS]
                yes_ask_levels = _normalize_book_side(book_yes.get("asks", []), descending=False)[:N_LEVELS]
                bid_depth += sum(float(x.get("size", 0)) for x in yes_bid_levels)
                ask_depth += sum(float(x.get("size", 0)) for x in yes_ask_levels)
            if mkt.get("token_no"):
                book_no = ST.feeds.get_polymarket_orderbook(mkt["token_no"])
                no_book = _best_executable_from_book(book_no, no_mark, "no")
                no_bid = no_book["best_bid"]
                no_ask = no_book["best_ask"]
                no_bid_size = no_book["bid_size"]
                no_ask_size = no_book["ask_size"]
                no_exec_buy = no_book["exec_buy"]
                no_bid_levels = _normalize_book_side(book_no.get("bids", []), descending=True)[:N_LEVELS]
                no_ask_levels = _normalize_book_side(book_no.get("asks", []), descending=False)[:N_LEVELS]
                bid_depth += sum(float(x.get("size", 0)) for x in no_bid_levels)
                ask_depth += sum(float(x.get("size", 0)) for x in no_ask_levels)
        except Exception:
            pass

        yes_buy = yes_exec_buy if yes_exec_buy is not None else yes_mark
        no_buy = no_exec_buy if no_exec_buy is not None else no_mark
        with ST._lock:
            c5.orderbooks[asset] = {
                "yes_best_bid": yes_bid,
                "yes_best_ask": yes_ask,
                "yes_exec_buy": yes_exec_buy,
                "yes_bid_size": yes_bid_size,
                "yes_ask_size": yes_ask_size,
                "no_best_bid": no_bid,
                "no_best_ask": no_ask,
                "no_exec_buy": no_exec_buy,
                "no_bid_size": no_bid_size,
                "no_ask_size": no_ask_size,
                "bid_depth": bid_depth,
                "ask_depth": ask_depth,
                "mark_sum_yes_no": round(yes_mark + no_mark, 4),
                "exec_sum_yes_no": round(yes_buy + no_buy, 4),
                # L2 snapshots for the dashboard
                "yes_bid_levels": yes_bid_levels,
                "yes_ask_levels": yes_ask_levels,
                "no_bid_levels": no_bid_levels,
                "no_ask_levels": no_ask_levels,
                "snap_ts": time.time(),
            }
            ref_price = c5.reference_spot.get(asset, float(spot["price"]))

        prices = [p for _ts, p in ST.price_history[asset]]
        trend = _spot_trend_stats(ST.price_history[asset], lookback_sec=max(300, int(c5.trend_lookback_min) * 60))
        dir_prob = _estimate_direction_probability(asset, float(spot["price"]), float(ref_price), trend, time_left)
        signal = _coin_signal_from_prices(prices, yes_buy, no_buy, ref_price, min_edge, c5.max_sum_yes_no)
        if signal:
            directional_bias = signal.get('direction')
            if c5.leg1_directional and abs(float(trend.get('trend_bps', 0.0))) >= 5.0 and trend.get('trend_dir') in ('above', 'below'):
                directional_bias = trend.get('trend_dir')

            opp = _compute_frequency_opportunity(
                yes_mark=yes_mark,
                no_mark=no_mark,
                yes_exec=yes_buy,
                no_exec=no_buy,
                time_left_sec=time_left,
                c5=c5,
                trend_bps=float(trend.get('trend_bps', 0.0)),
                bias=directional_bias,
                p_up=float(dir_prob.get("p_up", 0.5)),
                p_down=float(dir_prob.get("p_down", 0.5)),
            )

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
                "yes_mark_price": round(float(yes_mark), 4),
                "no_mark_price": round(float(no_mark), 4),
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "yes_exec_buy": round(float(yes_buy), 4),
                "no_bid": no_bid,
                "no_ask": no_ask,
                "no_exec_buy": round(float(no_buy), 4),
                "spot_source": spot.get("source"),
                "max_trade_usd": round(float(c5.max_trade_usd), 2),
                "p_up": round(float(dir_prob.get("p_up", 0.5)), 4),
                "p_down": round(float(dir_prob.get("p_down", 0.5)), 4),
                "dist_ref_pct": round(float(dir_prob.get("dist_ref_pct", 0.0)), 6),
                **opp,
            })
            with ST._lock:
                c5.signals[asset] = signal
                c5.market_meta[asset].update({
                    "yes_ask": round(yes_buy, 4),
                    "no_ask": round(no_buy, 4),
                    "mark_sum_yes_no": round(yes_mark + no_mark, 4),
                    "exec_sum_yes_no": round(yes_buy + no_buy, 4),
                    "sum_yes_no": round(signal["sum_yes_no"], 4),
                    "edge_pct": round((signal.get("edge_exec") if signal.get("edge_exec") is not None else signal["edge"]) * 100.0, 3),
                    "direction": signal.get("direction"),
                    "leg1_bias": signal.get("leg1_bias"),
                    "trend_bps": signal.get("trend_bps"),
                    "trend_dir": signal.get("trend_dir"),
                    "max_trade_usd": round(float(c5.max_trade_usd), 2),
                    "max_sum_yes_no": round(signal.get("max_sum_yes_no", c5.max_sum_yes_no), 4),
                    "target_sum_yes_no": round(float(c5.target_sum_yes_no), 4),
                    "opp_score": signal.get("opp_score"),
                    "decision": signal.get("decision"),
                    "decision_reason": signal.get("decision_reason"),
                })

        hist_row = {
            "ts": time.time(),
            "spot": round(float(spot["price"]), 6),
            "ref_px": round(float(ref_price), 6),
            "yes_mark": round(float(yes_mark), 6),
            "no_mark": round(float(no_mark), 6),
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
            fee_drag = float(open_meta.get("fee_drag", 0.0))
            slip_drag = float(open_meta.get("slippage_drag", 0.0))
            net_pnl = _apply_realism_drag(pair_pnl, fee_drag, slip_drag)
            status = "WIN" if net_pnl >= 0 else "LOSS"
            _sync_trade_row(
                open_id,
                net_pnl,
                status,
                note=f"expiry | ref={open_meta['reference_price']:.2f} final={spot['price']:.2f} outcome={outcome.upper()} | fee={fee_drag:.2f} slip={slip_drag:.2f}",
            )
            _log_analysis_event("trade_resolve_simul", {
                "asset": asset,
                "pair_id": open_id,
                "market_id": open_meta.get("market_id"),
                "outcome": outcome,
                "ref_price": float(open_meta["reference_price"]),
                "final_spot": float(spot["price"]),
                "yes_entry_price": open_meta.get("yes_entry_price"),
                "no_entry_price": open_meta.get("no_entry_price"),
                "yes_size_usd": open_meta.get("yes_size_usd"),
                "no_size_usd": open_meta.get("no_size_usd"),
                "pair_pnl_gross": pair_pnl,
                "fee_drag": fee_drag,
                "slippage_drag": slip_drag,
                "net_pnl": net_pnl,
                "status": status,
                "duration_sec": now_ts - float(open_meta.get("entry_ts") or now_ts),
            })
            with ST._lock:
                ST.coin5min_realism_drag += fee_drag + slip_drag
                c5.realized_fee_drag += fee_drag
                c5.realized_slippage_drag += slip_drag
                c5.consecutive_losses = 0 if net_pnl >= 0 else (int(c5.consecutive_losses) + 1)
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
            fee_drag = float(partial_leg.get("fee_drag", 0.0))
            slip_drag = float(partial_leg.get("slippage_drag", 0.0))
            # additional adverse selection penalty when left with single leg to maturity
            net_pnl = _apply_realism_drag(leg_pnl, fee_drag, slip_drag + 0.15)
            _sync_trade_row(
                partial_leg["pair_id"],
                net_pnl,
                "LEG1_WIN" if net_pnl >= 0 else "LEG1_LOSS",
                note=f"leg1 expiry | {partial_leg['first_side'].upper()} only | ref={partial_leg['reference_price']:.2f} final={spot['price']:.2f} | fee={fee_drag:.2f} slip={slip_drag+0.15:.2f}",
            )
            _log_analysis_event("trade_resolve_leg1", {
                "asset": asset,
                "pair_id": partial_leg.get("pair_id"),
                "market_id": partial_leg.get("market_id"),
                "outcome": outcome,
                "first_side": partial_leg.get("first_side"),
                "first_price": partial_leg.get("first_price"),
                "first_size_usd": partial_leg.get("first_size_usd"),
                "hedge_side": partial_leg.get("hedge_side"),
                "hedge_trigger": partial_leg.get("hedge_trigger"),
                "ref_price": float(partial_leg["reference_price"]),
                "final_spot": float(spot["price"]),
                "leg_pnl_gross": leg_pnl,
                "fee_drag": fee_drag,
                "slippage_drag": slip_drag + 0.15,
                "net_pnl": net_pnl,
                "status": "LEG1_WIN" if net_pnl >= 0 else "LEG1_LOSS",
                "duration_sec": now_ts - float(partial_leg.get("entry_ts") or now_ts),
            })
            with ST._lock:
                ST.coin5min_realism_drag += fee_drag + slip_drag + 0.15
                c5.realized_fee_drag += fee_drag
                c5.realized_slippage_drag += slip_drag + 0.15
                c5.consecutive_losses = 0 if net_pnl >= 0 else (int(c5.consecutive_losses) + 1)
                c5.partial_legs.pop(asset, None)
                c5.reference_spot[asset] = float(spot["price"])
                c5.reference_source[asset] = "post_expiry_spot"
                partial_leg = None

        with ST._lock:
            open_exists = asset in c5.open_trade_ids
            partial_leg = c5.partial_legs.get(asset)
        total_budget_pct = ST.capital * (allocation_pct / 100.0) * (trade_size_pct / 100.0)
        total_budget = min(float(total_budget_pct), float(c5.max_trade_usd))
        risk_ok, risk_reason = _risk_guard_coin5min(c5, ST.capital, ST.paper.get_metrics().get("total_pnl", 0.0) + ST.betmiss_realized_pnl)
        total_cost = max(signal["sum_yes_no"], 1e-6) if signal else None
        trend_bias_used = None
        if signal:
            trend_bias_used = signal.get("leg1_bias") if c5.leg1_directional else "cheapest_leg"
        window_running = bool(mkt.get("start_ts") and mkt.get("end_ts") and mkt.get("start_ts") <= now_ts <= mkt.get("end_ts"))
        if signal and not ST.kill_switch and total_budget >= 1.0 and window_running and risk_ok:
            if signal.get("decision") == "TRADE" and not open_exists and not partial_leg and total_cost and total_cost < 1.0:
                yes_order = None
                no_order = None
                yes_budget = total_budget * (yes_buy / total_cost)
                no_budget = total_budget * (no_buy / total_cost)
                yes_budget = _cap_size_by_depth(yes_budget, yes_buy, yes_ask_size, c5.depth_fill_ratio)
                no_budget = _cap_size_by_depth(no_budget, no_buy, no_ask_size, c5.depth_fill_ratio)
                yes_adj = _execution_adjustment(yes_buy, yes_budget, c5.orderbooks.get(asset, {}).get("yes_best_ask", None) - c5.orderbooks.get(asset, {}).get("yes_best_bid", None) if c5.orderbooks.get(asset, {}).get("yes_best_ask", None) is not None and c5.orderbooks.get(asset, {}).get("yes_best_bid", None) is not None else None, c5.taker_fee_bps, c5.extra_slippage_bps)
                no_adj = _execution_adjustment(no_buy, no_budget, c5.orderbooks.get(asset, {}).get("no_best_ask", None) - c5.orderbooks.get(asset, {}).get("no_best_bid", None) if c5.orderbooks.get(asset, {}).get("no_best_ask", None) is not None and c5.orderbooks.get(asset, {}).get("no_best_bid", None) is not None else None, c5.taker_fee_bps, c5.extra_slippage_bps)

                # --- Action #4: re-check live book before paper-filling ---
                fill_yes = _realistic_fill_check(mkt.get("token_yes"), yes_adj["exec_price"], yes_budget, "yes", c5)
                fill_no = _realistic_fill_check(mkt.get("token_no"), no_adj["exec_price"], no_budget, "no", c5)
                if fill_yes is None or fill_no is None:
                    with ST._lock:
                        c5.fill_rejected_count += 1
                    logger.info(
                        "[c5/fill-reject] %s simul: yes=%s no=%s (intended yes %.4f / no %.4f)",
                        asset,
                        "OK" if fill_yes else "REJECT",
                        "OK" if fill_no else "REJECT",
                        yes_adj["exec_price"], no_adj["exec_price"],
                    )
                    _log_analysis_event("fill_rejected_simul", {
                        "asset": asset,
                        "market_id": mkt.get("market_id"),
                        "slug": mkt.get("slug"),
                        "intended_yes_price": float(yes_adj["exec_price"]),
                        "intended_no_price": float(no_adj["exec_price"]),
                        "intended_yes_size_usd": float(yes_budget),
                        "intended_no_size_usd": float(no_budget),
                        "fill_yes": fill_yes,
                        "fill_no": fill_no,
                        "ref_px": float(ref_price),
                        "spot": float(spot["price"]),
                        "signal": {k: signal.get(k) for k in ("p_up", "p_down", "edge", "exec_sum_yes_no", "direction", "trend_bps")},
                    })
                else:
                    with ST._lock:
                        c5.fill_accepted_count += 1
                    yes_exec_actual = float(fill_yes["exec_price"])
                    no_exec_actual = float(fill_no["exec_price"])
                    yes_size_actual = float(fill_yes["filled_size_usd"])
                    no_size_actual = float(fill_no["filled_size_usd"])
                    yes_order = ST.paper.place_order(
                        strategy="coin5min",
                        platform="polymarket",
                        market_id=mkt["market_id"],
                        question=mkt["question"],
                        side="yes",
                        market_price=yes_exec_actual,
                        size=yes_size_actual,
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
                        market_price=no_exec_actual,
                        size=no_size_actual,
                        asset=asset,
                        direction="below",
                        edge=signal["edge"],
                    )
                    # overwrite adjusted exec into the existing *_adj dicts so the open_meta below uses the live fill
                    yes_adj["exec_price"] = yes_exec_actual
                    no_adj["exec_price"] = no_exec_actual
                    yes_budget = yes_size_actual
                    no_budget = no_size_actual
                    _log_analysis_event("trade_open_simultaneous", {
                        "asset": asset,
                        "market_id": mkt.get("market_id"),
                        "slug": mkt.get("slug"),
                        "yes_fill_price": yes_exec_actual,
                        "no_fill_price": no_exec_actual,
                        "yes_size_usd": yes_size_actual,
                        "no_size_usd": no_size_actual,
                        "exec_sum": yes_exec_actual + no_exec_actual,
                        "edge": 1.0 - (yes_exec_actual + no_exec_actual),
                        "fee_drag": yes_adj["fee_usd"] + no_adj["fee_usd"],
                        "slippage_drag": yes_adj["slippage_usd"] + no_adj["slippage_usd"],
                        "ref_px": float(ref_price),
                        "spot": float(spot["price"]),
                        "time_left_sec": signal.get("time_left_sec"),
                        "book_yes_at_fill": fill_yes.get("raw_book"),
                        "book_no_at_fill": fill_no.get("raw_book"),
                        "signal": {k: signal.get(k) for k in ("p_up", "p_down", "edge", "exec_sum_yes_no", "direction", "trend_bps", "decision", "decision_reason", "opp_score")},
                    })
                if yes_order and no_order and risk_ok:
                    pair_id = f"C5ARB-{int(time.time())}-{asset}"
                    total_fee_drag = yes_adj["fee_usd"] + no_adj["fee_usd"]
                    total_slippage_drag = yes_adj["slippage_usd"] + no_adj["slippage_usd"]
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
                    # --- LIVE EXECUTION: route orders to Polymarket CLOB ---
                    _route_live_order(asset, mkt, "yes", yes_adj["exec_price"], yes_budget, signal, pair_id)
                    _route_live_order(asset, mkt, "no", no_adj["exec_price"], no_budget, signal, pair_id)
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
                            "fee_drag": total_fee_drag,
                            "slippage_drag": total_slippage_drag,
                            # entry prices/sizes for the dashboard order-on-book view
                            "yes_entry_price": float(yes_adj["exec_price"]),
                            "no_entry_price": float(no_adj["exec_price"]),
                            "yes_size_usd": float(yes_budget),
                            "no_size_usd": float(no_budget),
                        }
                        open_exists = True
            elif partial_leg and not open_exists:
                hedge_side = partial_leg["hedge_side"]
                hedge_price_now = no_buy if hedge_side == "no" else yes_buy
                locked_cost = partial_leg["first_price"] + hedge_price_now
                locked_edge = 1.0 - locked_cost
                if hedge_price_now <= partial_leg["hedge_trigger"] and locked_cost < 1.0:
                    hedge_order = None
                    hedge_budget = max(1.0, float(partial_leg.get("remaining_budget", total_budget * 0.5)))
                    hedge_budget = _cap_size_by_depth(hedge_budget, hedge_price_now, no_ask_size if hedge_side == "no" else yes_ask_size, c5.depth_fill_ratio)
                    hedge_adj = _execution_adjustment(hedge_price_now, hedge_budget, None, c5.taker_fee_bps, c5.extra_slippage_bps)
                    hedge_token = mkt.get("token_no") if hedge_side == "no" else mkt.get("token_yes")
                    fill_hedge = _realistic_fill_check(hedge_token, hedge_adj["exec_price"], hedge_budget, hedge_side, c5)
                    if fill_hedge is None:
                        with ST._lock:
                            c5.fill_rejected_count += 1
                        logger.info(
                            "[c5/fill-reject] %s hedge %s: price %.4f drifted away",
                            asset, hedge_side, hedge_adj["exec_price"],
                        )
                        _log_analysis_event("fill_rejected_hedge", {
                            "asset": asset,
                            "market_id": mkt.get("market_id"),
                            "hedge_side": hedge_side,
                            "intended_price": float(hedge_adj["exec_price"]),
                            "intended_size_usd": float(hedge_budget),
                            "first_price": partial_leg.get("first_price"),
                            "first_side": partial_leg.get("first_side"),
                            "ref_px": float(ref_price),
                            "spot": float(spot["price"]),
                        })
                    else:
                        with ST._lock:
                            c5.fill_accepted_count += 1
                        hedge_exec_actual = float(fill_hedge["exec_price"])
                        hedge_size_actual = float(fill_hedge["filled_size_usd"])
                        hedge_adj["exec_price"] = hedge_exec_actual
                        hedge_budget = hedge_size_actual
                        hedge_order = ST.paper.place_order(
                            strategy="coin5min",
                            platform="polymarket",
                            market_id=mkt["market_id"],
                            question=mkt["question"],
                            side=hedge_side,
                            market_price=hedge_exec_actual,
                            size=hedge_size_actual,
                            asset=asset,
                            direction="above" if hedge_side == "yes" else "below",
                            edge=locked_edge,
                        )
                        _log_analysis_event("trade_open_hedge", {
                            "asset": asset,
                            "market_id": mkt.get("market_id"),
                            "first_side": partial_leg.get("first_side"),
                            "first_price": partial_leg.get("first_price"),
                            "hedge_side": hedge_side,
                            "hedge_fill_price": hedge_exec_actual,
                            "hedge_size_usd": hedge_size_actual,
                            "locked_cost": partial_leg["first_price"] + hedge_exec_actual,
                            "locked_edge": 1.0 - (partial_leg["first_price"] + hedge_exec_actual),
                            "book_at_fill": fill_hedge.get("raw_book"),
                            "ref_px": float(ref_price),
                            "spot": float(spot["price"]),
                        })
                    if hedge_order and risk_ok:
                        yes_order_id = partial_leg["first_order_id"] if partial_leg["first_side"] == "yes" else hedge_order.id
                        no_order_id = partial_leg["first_order_id"] if partial_leg["first_side"] == "no" else hedge_order.id
                        # --- LIVE EXECUTION: hedge leg ---
                        _route_live_order(asset, mkt, hedge_side, hedge_adj["exec_price"], hedge_budget, signal, partial_leg["pair_id"])
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
                plan = _choose_leg1_frequency(signal, c5)
                if plan:
                    first_order = None
                    first_budget = total_budget * 0.5
                    first_budget = _cap_size_by_depth(first_budget, plan["first_price"], yes_ask_size if plan["side"] == "yes" else no_ask_size, c5.depth_fill_ratio)
                    first_adj = _execution_adjustment(plan["first_price"], first_budget, None, c5.taker_fee_bps, c5.extra_slippage_bps)
                    first_token = mkt.get("token_yes") if plan["side"] == "yes" else mkt.get("token_no")
                    fill_first = _realistic_fill_check(first_token, first_adj["exec_price"], first_budget, plan["side"], c5)
                    if fill_first is None:
                        with ST._lock:
                            c5.fill_rejected_count += 1
                        logger.info(
                            "[c5/fill-reject] %s leg1 %s: price %.4f drifted away",
                            asset, plan["side"], first_adj["exec_price"],
                        )
                        _log_analysis_event("fill_rejected_leg1", {
                            "asset": asset,
                            "market_id": mkt.get("market_id"),
                            "side": plan["side"],
                            "intended_price": float(first_adj["exec_price"]),
                            "intended_size_usd": float(first_budget),
                            "ref_px": float(ref_price),
                            "spot": float(spot["price"]),
                            "signal": {k: signal.get(k) for k in ("p_up", "p_down", "edge", "direction", "trend_bps")},
                        })
                    else:
                        with ST._lock:
                            c5.fill_accepted_count += 1
                        first_exec_actual = float(fill_first["exec_price"])
                        first_size_actual = float(fill_first["filled_size_usd"])
                        first_adj["exec_price"] = first_exec_actual
                        first_budget = first_size_actual
                        plan["first_price"] = first_exec_actual  # honor the worse-of price
                        first_order = ST.paper.place_order(
                            strategy="coin5min",
                            platform="polymarket",
                            market_id=mkt["market_id"],
                            question=mkt["question"],
                            side=plan["side"],
                            market_price=first_exec_actual,
                            size=first_size_actual,
                            asset=asset,
                            direction=signal.get("direction"),
                            edge=signal.get("edge"),
                        )
                        _log_analysis_event("trade_open_leg1", {
                            "asset": asset,
                            "market_id": mkt.get("market_id"),
                            "slug": mkt.get("slug"),
                            "side": plan["side"],
                            "first_fill_price": first_exec_actual,
                            "first_size_usd": first_size_actual,
                            "hedge_side": plan["hedge_side"],
                            "hedge_trigger": plan["hedge_trigger"],
                            "book_at_fill": fill_first.get("raw_book"),
                            "ref_px": float(ref_price),
                            "spot": float(spot["price"]),
                            "time_left_sec": signal.get("time_left_sec"),
                            "signal": {k: signal.get(k) for k in ("p_up", "p_down", "edge", "direction", "trend_bps", "decision", "decision_reason", "opp_score")},
                        })
                    if first_order and risk_ok:
                        pair_id = f"C5LEG-{int(time.time())}-{asset}"
                        # --- LIVE EXECUTION: first leg ---
                        _route_live_order(asset, mkt, plan["side"], first_adj["exec_price"], first_budget, signal, pair_id)
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
                                "first_size_usd": float(first_budget),
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
                    "target_sum_yes_no": round(float(c5.target_sum_yes_no), 4),
                    "opp_score": signal.get("opp_score") if signal else None,
                    "decision": signal.get("decision") if signal else None,
                    "target_sum_yes_no": round(float(c5.target_sum_yes_no), 4),
                    "opp_score": signal.get("opp_score") if signal else None,
                    "decision": signal.get("decision") if signal else None,
                    "p_up": signal.get("p_up") if signal else None,
                    "p_down": signal.get("p_down") if signal else None,
                    "risk_ok": risk_ok,
                    "risk_reason": risk_reason,
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
                    "target_sum_yes_no": round(float(c5.target_sum_yes_no), 4),
                    "opp_score": signal.get("opp_score") if signal else None,
                    "decision": signal.get("decision") if signal else None,
                    "p_up": signal.get("p_up") if signal else None,
                    "p_down": signal.get("p_down") if signal else None,
                    "risk_ok": risk_ok,
                    "risk_reason": risk_reason,
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

            # --- Refresh live orders on CLOB ---
            if ST.execution and ST.exec_mode == "live" and ST.execution.is_connected:
                try:
                    for order in ST.execution.list_orders(states=["OPEN", "PARTIAL", "POSTING"]):
                        ST.execution.refresh_order(order["client_order_id"])
                except Exception as exc:
                    logger.warning("Live order refresh error: %s", exc)

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
    with ST._lock:
        ST.paper = PaperTradingEngine(initial_capital=ST.capital, db_path=":memory:")
        if hasattr(ST.paper, "slippage_bps"):
            ST.paper.slippage_bps = 0.0
    threading.Thread(target=_runtime_loop, daemon=True).start()
    exec_status = "connected" if (ST.execution and ST.execution.is_connected) else "not connected"
    ST.alert("INFO", "SYS", f"Workers lancés | exec_mode={ST.exec_mode} | engine={exec_status} | max_order=${ST.max_order_usd:.2f}")


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

