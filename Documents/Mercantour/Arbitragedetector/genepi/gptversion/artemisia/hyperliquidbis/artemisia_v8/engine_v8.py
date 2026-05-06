"""
engine_v8.py — Artemisia v8 engine for S7 Aggressive Maker Scalping.

Architecture:
  - Main thread: polling loop every 500ms
  - Background thread: WebSocket orderbook (OrderbookManager)
  - Background thread: KillSwitch watchdog

Separate from v7 — runs fully independently.
Paper mode by default. Use --live flag ONLY after 14-day paper validation.

Usage:
  python engine_v8.py --paper --coins BTC,ETH,SOL,HYPE
  python engine_v8.py --paper                           # all coins from config
  python engine_v8.py --kill-switch                     # emergency stop
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import requests

# ---- Internal imports ----
from execution.orderbook_manager import OrderbookManager
from execution.high_freq_executor import HighFreqExecutor, OpenPosition
from risk.aggressive_kill_switch import AggressiveKillSwitch
from strategies.s7_aggressive_maker_scalping import (
    S7AggressiveMakerScalping, TPStopDecision,
)
from monitoring.pnl_tracker_realtime import PnLTracker

log = logging.getLogger(__name__)

HL_API       = "https://api.hyperliquid.xyz/info"
POLL_INTERVAL = 0.5    # 500ms main loop
CANDLE_REFRESH = 60    # refresh 1m candles every 60s

# Default symbols if not specified via CLI
DEFAULT_SYMBOLS = ["BTC", "ETH", "SOL", "HYPE", "AVAX", "WLD", "ARB", "OP"]


# ---------------------------------------------------------------------------
# 1m candle fetch
# ---------------------------------------------------------------------------

def fetch_1m_candles(symbol: str, n: int = 50) -> dict:
    """Fetch n 1-minute candles from Hyperliquid REST. Returns {h, l, c} arrays."""
    try:
        end_ms   = int(time.time() * 1000)
        start_ms = end_ms - n * 60 * 1000
        r = requests.post(HL_API, json={
            "type": "candleSnapshot",
            "req": {
                "coin": symbol, "interval": "1m",
                "startTime": start_ms, "endTime": end_ms,
            }
        }, timeout=5)
        r.raise_for_status()
        data = r.json()
        if not data:
            return {}
        highs  = [float(c["h"]) for c in data]
        lows   = [float(c["l"]) for c in data]
        closes = [float(c["c"]) for c in data]
        volumes= [float(c["v"]) for c in data]
        return {"h": highs, "l": lows, "c": closes, "v": volumes}
    except Exception as e:
        log.debug("Candle fetch failed for %s: %s", symbol, e)
        return {}


def volume_5m(candle_data: dict) -> float:
    """Sum of last 5 minutes of volume in USD (uses close as price proxy)."""
    if not candle_data:
        return 0.0
    closes  = candle_data.get("c", [])
    volumes = candle_data.get("v", [])
    n = min(5, len(closes), len(volumes))
    if n == 0:
        return 0.0
    return sum(closes[-n + i] * volumes[-n + i] for i in range(n))


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ArtemiasV8Engine:

    def __init__(self, config_path: str = "config_s7.json", paper: bool = True,
                 symbols: Optional[list[str]] = None):
        cfg_file = Path(__file__).parent / config_path
        with open(cfg_file) as f:
            self.cfg = json.load(f)

        self.paper   = paper
        self.symbols = [s.upper() for s in (symbols or DEFAULT_SYMBOLS)]
        self.equity  = self.cfg["capital"]["initial_usdc"]
        self._running = False

        s7_params = self.cfg.get("strategies", {}).get("s7", {})
        self.strategy = S7AggressiveMakerScalping(params=s7_params)

        risk_cfg = self.cfg.get("risk", {})
        self.ks = AggressiveKillSwitch(
            initial_capital=self.equity,
            daily_dd_pct=risk_cfg.get("max_dd_daily_pct",    3.0),
            total_dd_pct=risk_cfg.get("max_dd_total_pct",    6.0),
            max_positions=risk_cfg.get("max_open_positions",  6),
            max_notional_mult=risk_cfg.get("max_notional_mult", 4.0),
            network_timeout_s=risk_cfg.get("network_timeout_s", 30.0),
            max_trades_per_hour=risk_cfg.get("max_trades_per_hour", 30),
            max_loss_streak=risk_cfg.get("max_loss_streak",  5),
            btc_move_5m_pct=risk_cfg.get("btc_move_5m_pct",  1.5),
            close_all_callback=self._close_all_positions,
        )

        log_cfg = self.cfg.get("logging", {})
        self.executor = HighFreqExecutor(
            paper=paper,
            trade_log_path=log_cfg.get("trade_log", "logs/fills_s7.csv"),
            on_position_open=self._on_position_open,
            on_position_close=self._on_position_close,
        )

        self.tracker = PnLTracker(
            log_path=log_cfg.get("metrics_log", "logs/metrics_s7.csv"),
            equity=self.equity,
        )

        self.obm = OrderbookManager(self.symbols)

        # Strategy state
        self._blacklist: dict[str, float] = {}    # {symbol: until_ts}
        self._recent_stops: dict[str, list] = {}  # {symbol: [ts, ...]}
        self._candles: dict[str, dict]  = {}      # {symbol: {h,l,c,v}}
        self._last_candle_refresh: float = 0.0
        self._last_dashboard_ts:   float = 0.0

        dashboard_freq = log_cfg.get("terminal_frequency_seconds", 60)
        self._dashboard_interval = dashboard_freq

        log.info(
            "ArtemiasV8Engine | paper=%s | symbols=%s | equity=%.2f",
            paper, self.symbols, self.equity,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        self.ks.start()
        self.obm.start()
        self._running = True

        log.info("Engine v8 running. Ctrl+C to stop.")
        # Brief warm-up
        time.sleep(2)
        self._refresh_candles()

        try:
            while self._running:
                loop_start = time.time()
                try:
                    self._poll()
                except Exception as e:
                    log.error("Poll error: %s", e, exc_info=True)

                elapsed = time.time() - loop_start
                sleep_t = max(0, POLL_INTERVAL - elapsed)
                time.sleep(sleep_t)

        except KeyboardInterrupt:
            log.info("Shutting down v8...")
        finally:
            self._running = False
            self.executor.cancel_all()
            mids = self._get_mids()
            self.executor.close_all_market(mids, "shutdown")
            self.ks.stop()
            self.obm.stop()

    # ------------------------------------------------------------------
    # Poll loop (runs every 500ms)
    # ------------------------------------------------------------------

    def _poll(self):
        now = time.time()

        # 1. Record WS heartbeat (resets network watchdog)
        if self.obm.connected:
            self.ks.record_ws_heartbeat()

        # 2. Refresh 1m candles periodically
        if now - self._last_candle_refresh > CANDLE_REFRESH:
            self._refresh_candles()
            self._last_candle_refresh = now

        # 3. Collect current book snapshots
        books = {s: self.obm.get_book(s) for s in self.symbols}

        # Feed BTC price to vol guard
        btc_book = books.get("BTC")
        if btc_book and btc_book.mid:
            self.ks.update_btc_price(btc_book.mid)

        # 4. Check exits for all open positions
        mids      = {s: b.mid       for s, b in books.items() if b.mid}
        best_bids = {s: b.best_bid  for s, b in books.items() if b.best_bid}
        best_asks = {s: b.best_ask  for s, b in books.items() if b.best_ask}

        to_close = self.executor.check_exits(mids, best_bids, best_asks)
        for pos, exit_price, reason in to_close:
            net_pnl = self.executor.close_position(pos, exit_price, reason)
            if reason == "stop_loss":
                self._blacklist = self.strategy.check_blacklist(
                    pos.symbol, self._blacklist, self._recent_stops
                )

        # 5. Check fills on pending orders
        for sym in self.symbols:
            book = books.get(sym)
            if not book or book.best_bid is None:
                continue
            if self.obm.is_stale(sym, max_age_s=5.0):
                continue

            self.executor.check_fills(
                symbol=sym,
                best_bid=book.best_bid,
                best_ask=book.best_ask,
                mid=book.mid,
                tp_stop_callback=self._make_tp_stop_callback(sym),
            )

        # 6. Place / refresh quotes (only if KS allows)
        can_open, reason = self.ks.can_open_position()
        if not can_open:
            if reason not in ("max_positions=6",):
                log.debug("Cannot open: %s", reason)
        else:
            self._place_quotes(books)

        # 7. Dashboard
        if now - self._last_dashboard_ts >= self._dashboard_interval:
            self._print_dashboard()
            self._last_dashboard_ts = now

    # ------------------------------------------------------------------
    # Quote placement
    # ------------------------------------------------------------------

    def _place_quotes(self, books: dict):
        already_quoting = self.executor.symbols_with_pending()
        already_in_pos  = self.executor.symbols_with_positions()

        for sym in self.symbols:
            # Don't double-quote
            if sym in already_quoting or sym in already_in_pos:
                continue

            book = books.get(sym)
            if not book or book.best_bid is None or book.best_ask is None:
                continue
            if self.obm.is_stale(sym, max_age_s=3.0):
                continue

            candle = self._candles.get(sym, {})
            atr_1m = self.strategy.atr_from_candles(
                candle.get("h", []), candle.get("l", []), candle.get("c", [])
            )
            vol_5m = volume_5m(candle)
            trades = self.obm.get_trades(sym, seconds=30)

            tradable, why_not = self.strategy.is_tradable(
                symbol=sym,
                best_bid=book.best_bid,
                best_ask=book.best_ask,
                volume_5m_usd=vol_5m,
                atr_1m=atr_1m,
                blacklist=self._blacklist,
            )
            if not tradable:
                log.debug("Skip %s: %s", sym, why_not)
                continue

            quote = self.strategy.compute_quotes(
                symbol=sym,
                best_bid=book.best_bid,
                best_ask=book.best_ask,
                imbalance=book.imbalance(self.strategy.params["imbalance_levels"]),
                trades_30s=trades,
                equity=self.equity,
            )
            if quote is None:
                continue

            self.executor.place_quotes(quote)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _make_tp_stop_callback(self, symbol: str):
        """Returns a closure that computes TP/stop given a fill."""
        def callback(fill) -> TPStopDecision:
            candle = self._candles.get(symbol, {})
            atr_1m = self.strategy.atr_from_candles(
                candle.get("h", []), candle.get("l", []), candle.get("c", [])
            )
            book = self.obm.get_book(symbol)
            fair = (book.best_bid + book.best_ask) / 2 if (book.best_bid and book.best_ask) \
                   else fill.price
            return self.strategy.compute_tp_stop(
                entry_price=fill.price,
                filled_side=fill.side,
                atr_1m=atr_1m,
                fair_value=fair,
            )
        return callback

    def _on_position_open(self, pos: OpenPosition):
        self.ks.register_open_position()

    def _on_position_close(self, pos: OpenPosition, net_pnl: float, reason: str):
        hold_s = time.time() - pos.entry_ts
        self.equity += net_pnl
        self.ks.update_equity(self.equity)
        self.ks.register_close_position()
        self.ks.record_trade(net_pnl)
        self.tracker.record_trade(net_pnl, hold_s, reason)

    def _close_all_positions(self, reason: str = "kill_switch"):
        log.critical("EMERGENCY CLOSE ALL: %s", reason)
        self.executor.cancel_all()
        mids = self._get_mids()
        self.executor.close_all_market(mids, reason)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_candles(self):
        for sym in self.symbols:
            data = fetch_1m_candles(sym, n=50)
            if data:
                self._candles[sym] = data
        log.debug("Candles refreshed for %d symbols", len(self._candles))

    def _get_mids(self) -> dict[str, float]:
        try:
            r = requests.post(HL_API, json={"type": "allMids"}, timeout=5)
            r.raise_for_status()
            return {k: float(v) for k, v in r.json().items()}
        except Exception as e:
            log.warning("allMids fetch failed: %s", e)
            return {}

    def _print_dashboard(self):
        open_pos = self.executor.open_positions
        pos_detail = ", ".join(
            f"{p.symbol}({p.side[0]})${p.notional_usd:.0f}"
            f"@{p.entry_price:.4g}+{time.time()-p.entry_ts:.0f}s"
            for p in open_pos
        ) or "none"

        now = time.time()
        bl_detail = ", ".join(
            f"{s}({v-now:.0f}s)" for s, v in self._blacklist.items()
            if v > now
        ) or "none"

        snap = self.tracker.tick(
            open_positions=len(open_pos),
            quotes_active=len(self.executor.pending_orders),
            reconnections=self.obm.reconnections,
            blacklisted_coins=sum(1 for v in self._blacklist.values() if v > now),
        )

        dashboard = self.tracker.get_dashboard(
            snap=snap,
            open_pos_detail=pos_detail,
            blacklist_detail=bl_detail,
            ks_status=self.ks.status_dict(),
            equity=self.equity,
        )
        print(f"\n{dashboard}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Artemisia v8 — S7 Maker Scalping")
    parser.add_argument("--config", default="config_s7.json")
    parser.add_argument("--paper",  action="store_true", default=True)
    parser.add_argument("--live",   action="store_true",
                        help="WARNING: real money. Only after 14-day paper validation.")
    parser.add_argument("--coins",  type=str, default="",
                        help="Comma-separated list, e.g. BTC,ETH,SOL")
    parser.add_argument("--kill-switch", action="store_true",
                        help="Send kill signal to running instance (not implemented yet)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/engine_v8.log"),
        ],
    )

    if args.kill_switch:
        print("Manual kill switch not yet implemented. Stop the process with Ctrl+C.")
        sys.exit(0)

    if args.live:
        print("WARNING: LIVE mode selected. Real money at risk.")
        print("CONFIRM: type 'CONFIRMED LIVE' to proceed:")
        confirm = input().strip()
        if confirm != "CONFIRMED LIVE":
            print("Aborted.")
            sys.exit(0)

    symbols = [s.strip().upper() for s in args.coins.split(",")] if args.coins else None
    paper   = not args.live

    cfg_path = Path(__file__).parent / args.config
    if not cfg_path.exists():
        print(f"Config not found: {args.config}")
        sys.exit(1)

    engine = ArtemiasV8Engine(
        config_path=args.config,
        paper=paper,
        symbols=symbols,
    )
    engine.run()
