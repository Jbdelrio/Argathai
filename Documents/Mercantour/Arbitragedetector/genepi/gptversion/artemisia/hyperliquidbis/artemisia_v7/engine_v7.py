"""
engine_v7.py — Artemisia v7 Live Engine
Non-negotiable constraints enforced:
- 60s watchdog: no poll -> close all
- Max 3 simultaneous positions
- Daily DD kill at 2%, total DD at 8%
- All orders attempted as maker first
- State persisted every 60s
"""
import os
import sys
import json
import time
import logging
import threading
import requests
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from risk.kill_switch import KillSwitch
from risk.position_sizer import size_usd, regime_dd_factor
from regime.hmm_detector import RegimeDetector, REGIME_NAMES
from strategies.s1_funding_arb import FundingArbStrategy
from strategies.s2_maker_mr import MakerMRStrategy
from strategies.s3_pairs_kalman import PairsKalmanStrategy
from strategies.s4_regime_router import RegimeRouter
from data.universe import get_universe, get_funding_snapshot
from data.downloader import download_all_candles, download_all_funding

log = logging.getLogger(__name__)

HL_API = "https://api.hyperliquid.xyz/info"
STATE_FILE = "v7_state.json"
POLL_INTERVAL = 15    # seconds between market data polls
CANDLE_BARS   = 200   # bars to keep in memory per symbol for signal generation


class Position:
    __slots__ = ("symbol", "strategy", "side", "size_usd", "entry_price",
                 "stop_price", "tp_price", "entry_time", "max_hold_s", "meta")

    def __init__(self, symbol, strategy, side, size_usd, entry_price,
                 stop_price=None, tp_price=None, max_hold_s=28800, meta=None):
        self.symbol     = symbol
        self.strategy   = strategy
        self.side       = side
        self.size_usd   = size_usd
        self.entry_price = entry_price
        self.stop_price  = stop_price
        self.tp_price    = tp_price
        self.entry_time  = time.time()
        self.max_hold_s  = max_hold_s
        self.meta        = meta or {}


class ArtemiasV7Engine:
    """
    Main live engine for Artemisia v7.
    """

    def __init__(self, config_path: str = "config.json", paper: bool = True):
        cfg_file = Path(__file__).parent / config_path
        with open(cfg_file) as f:
            self.cfg = json.load(f)

        self.paper        = paper
        self.equity       = self.cfg["capital"]["initial_usdc"]
        self.state_file   = self.cfg.get("logging", {}).get("state_file", "v7_state.json")
        # Instance label derived from config filename (A vs B)
        self.instance_tag = "B" if "config_b" in config_path.lower() else "A"
        self.start_time = time.time()

        # Live metrics counters
        self.trades_today      = 0
        self.pnl_today         = 0.0
        self.signals_generated = 0
        self.signals_rejected  = {"hmm": 0, "funding": 0, "sizing": 0, "other": 0}
        self._last_metrics_log = 0.0
        self._today_date       = time.strftime("%Y-%m-%d", time.gmtime())

        # Kill switch
        self.ks = KillSwitch(
            initial_capital=self.equity,
            daily_dd_pct=self.cfg["risk"]["daily_dd_kill_pct"] * 100,
            total_dd_pct=self.cfg["risk"]["total_dd_hard_stop_pct"] * 100,
            max_positions=self.cfg["capital"]["max_simultaneous_positions"],
            watchdog_interval_s=self.cfg["risk"]["watchdog_interval_s"],
            network_timeout_s=self.cfg["risk"]["network_timeout_s"],
            close_all_callback=self._close_all_positions,
        )

        # Strategies
        self.router = RegimeRouter(params=None, config=self.cfg)

        # State
        self.positions: list[Position] = []
        self.candle_buffer: dict[str, list] = {}  # {sym: [candle_dict, ...]}
        self.funding_buffer: dict[str, list] = {}
        self._lock = threading.Lock()
        self._running = False
        # Cooldown: {(symbol, strategy): last_close_time}
        self._last_close_time: dict[tuple, float] = {}

        # Logging
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        fh = logging.FileHandler(self.cfg["logging"]["file"])
        fh.setLevel(logging.INFO)
        logging.getLogger().addHandler(fh)

        log.info("ArtemiasV7Engine init | paper=%s | equity=%.2f", paper, self.equity)

    # ------------------------------------------------------------------
    # Live metrics
    # ------------------------------------------------------------------

    def log_live_metrics(self):
        """Log key live metrics every terminal_frequency_seconds (default 60s)."""
        now = time.time()
        freq = self.cfg.get("logging", {}).get("terminal_frequency_seconds", 60)
        if now - self._last_metrics_log < freq:
            return
        self._last_metrics_log = now

        # Daily reset at UTC midnight
        today = time.strftime("%Y-%m-%d", time.gmtime())
        if today != self._today_date:
            self.trades_today = 0
            self.pnl_today    = 0.0
            self._today_date  = today

        uptime_h = (now - self.start_time) / 3600
        regime_name = REGIME_NAMES.get(self.router._last_regime, "?")
        n_open = len(self.positions)
        total_rejected = sum(self.signals_rejected.values())

        # Show open positions detail
        pos_detail = ", ".join(
            f"{p.symbol}({p.side[0].upper()})" for p in self.positions
        ) or "—"

        log.info(
            "\n" + "=" * 72 + "\n"
            "[INSTANCE %s] uptime=%.1fh | regime=%s | equity=$%.2f\n"
            "  Open(%d): %s\n"
            "  Today: trades=%d  pnl=%+.2f%%\n"
            "  Signals: generated=%d  rejected=%d "
            "(hmm=%d funding=%d sizing=%d other=%d)\n"
            + "=" * 72,
            self.instance_tag, uptime_h, regime_name, self.equity,
            n_open, pos_detail,
            self.trades_today, self.pnl_today / max(self.equity, 1) * 100,
            self.signals_generated, total_rejected,
            self.signals_rejected["hmm"], self.signals_rejected["funding"],
            self.signals_rejected["sizing"], self.signals_rejected["other"],
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Start the engine."""
        self.ks.start()
        self._running = True
        self._load_state()

        # Initial data download
        self._refresh_universe()

        log.info("Engine running. Ctrl+C to stop.")
        last_state_save = time.time()
        last_universe_refresh = time.time()

        try:
            while self._running:
                loop_start = time.time()

                if self.ks.is_killed:
                    log.warning("Kill switch active, sleeping 60s")
                    time.sleep(60)
                    continue

                try:
                    self._poll_and_act()
                    self.ks.record_poll()
                except Exception as e:
                    log.error("Poll error: %s", e, exc_info=True)

                # Periodic tasks
                now = time.time()
                if now - last_state_save > 60:
                    self._save_state()
                    last_state_save = now

                if now - last_universe_refresh > 3600 * self.cfg["universe"]["refresh_interval_h"]:
                    self._refresh_universe()
                    last_universe_refresh = now

                self.log_live_metrics()

                elapsed = time.time() - loop_start
                sleep_time = max(0, POLL_INTERVAL - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            log.info("Shutting down...")
        finally:
            self._running = False
            self.ks.stop()
            self._save_state()

    def stop(self):
        self._running = False

    # ------------------------------------------------------------------
    # Core poll loop
    # ------------------------------------------------------------------

    def _poll_and_act(self):
        """One iteration of the main loop."""
        # 1. Fetch current market data
        mids = self._fetch_mids()
        if not mids:
            log.warning("No market data received")
            return

        # 2. Update equity estimate from open positions
        self._update_equity_estimate(mids)

        # 3. Check existing positions for stop/TP/max_hold
        self._check_exit_conditions(mids)

        # 3b. EXPLOSION regime: close ALL positions immediately, bypass can_open gate
        if self.router._last_regime == 2 and self.positions:
            log.warning("EXPLOSION regime active — closing all %d open positions",
                        len(self.positions))
            with self._lock:
                pos_copy = list(self.positions)
            for pos in pos_copy:
                self._close_position(pos, mids.get(pos.symbol, pos.entry_price),
                                     "explosion_regime")
            return

        # 4. Generate new signals
        can_open, reason = self.ks.can_open_position()
        if not can_open:
            log.debug("Cannot open position: %s", reason)
            return

        signals = self._generate_signals(mids)

        # 5. Execute signals
        for sig in signals:
            can_open, reason = self.ks.can_open_position()
            if not can_open:
                break
            self._execute_signal(sig, mids)

    def _fetch_mids(self) -> dict[str, float]:
        """Fetch current mid prices."""
        try:
            r = requests.post(HL_API, json={"type": "allMids"}, timeout=5)
            r.raise_for_status()
            return {k: float(v) for k, v in r.json().items()}
        except Exception as e:
            log.warning("allMids fetch failed: %s", e)
            return {}

    def _generate_signals(self, mids: dict[str, float]) -> list:
        """Generate signals from the regime router."""
        signals = []
        # Use a representative symbol's candles for regime detection
        main_sym = "BTC"
        candles_df = self._get_candle_df(main_sym)
        if candles_df is None or candles_df.empty:
            return signals

        funding_df = self._get_funding_df(main_sym)

        # Fit regime detector if not yet done (or retrain periodically)
        if not self.router._detector_fitted and len(candles_df) > 100:
            self.router.fit_regime_detector(candles_df, funding_df)

        open_pos_dicts = [
            {"symbol": p.symbol, "strategy": p.strategy,
             "side": p.side, "size_usd": p.size_usd}
            for p in self.positions
        ]

        # Give S3 the full candle snapshot so it can pick the best live pair
        live_candles = {sym: self._get_candle_df(sym) for sym in self.candle_buffer}
        live_candles = {k: v for k, v in live_candles.items()
                        if v is not None and not v.empty}
        self.router.s3.update_candles(live_candles)

        for sym in list(self.candle_buffer.keys()):
            candles_df = self._get_candle_df(sym)
            if candles_df is None or len(candles_df) < 50:
                continue

            funding_df = self._get_funding_df(sym)
            sigs = self.router.signal(candles_df, funding_df, None, self.equity, open_pos_dicts)

            for sig in sigs:
                if sig.symbol == "UNKNOWN":
                    sig.symbol = sym
                signals.append(sig)

        self.signals_generated += len(signals)
        return signals

    def _check_exit_conditions(self, mids: dict[str, float]):
        """Check stop/TP/max_hold for all open positions."""
        to_close = []
        now = time.time()

        with self._lock:
            for pos in self.positions:
                mid = mids.get(pos.symbol)
                if mid is None:
                    continue

                # Max hold time
                if now - pos.entry_time > pos.max_hold_s:
                    to_close.append((pos, "max_hold"))
                    continue

                # Stop loss
                if pos.stop_price is not None:
                    if pos.side == "long" and mid <= pos.stop_price:
                        to_close.append((pos, "stop_loss"))
                    elif pos.side == "short" and mid >= pos.stop_price:
                        to_close.append((pos, "stop_loss"))

                # Take profit
                if pos.tp_price is not None:
                    if pos.side == "long" and mid >= pos.tp_price:
                        to_close.append((pos, "take_profit"))
                    elif pos.side == "short" and mid <= pos.tp_price:
                        to_close.append((pos, "take_profit"))

        for pos, reason in to_close:
            self._close_position(pos, mids.get(pos.symbol, pos.entry_price), reason)

    def _execute_signal(self, sig, mids: dict[str, float]):
        """Execute a trading signal (paper or live)."""
        if sig.side == "close":
            # Find and close matching position
            with self._lock:
                for pos in list(self.positions):
                    if pos.symbol == sig.symbol:
                        self._close_position(pos, mids.get(sig.symbol, pos.entry_price), "signal_close")
            return

        if sig.size_usd < 10:
            log.debug("Signal size too small: $%.2f for %s", sig.size_usd, sig.symbol)
            return

        mid = mids.get(sig.symbol)
        if mid is None:
            log.warning("No price for %s", sig.symbol)
            return

        # Cooldown: don't re-enter the same (symbol, strategy) for 1h after a close
        strategy_name = sig.meta.get("strategy", "unknown") if sig.meta else "unknown"
        cooldown_key = (sig.symbol, strategy_name)
        cooldown_s = self.cfg.get("risk", {}).get("reentry_cooldown_s", 3600)
        last_close = self._last_close_time.get(cooldown_key, 0)
        if time.time() - last_close < cooldown_s:
            log.debug("Cooldown active for %s/%s (%.0fs remaining)",
                      sig.symbol, strategy_name,
                      cooldown_s - (time.time() - last_close))
            return

        # Re-anchor stop to actual execution mid using intended stop_loss_pct
        if sig.stop_price is not None and sig.meta:
            stop_pct = sig.meta.get("stop_loss_pct")
            if stop_pct:
                if sig.side == "long":
                    sig.stop_price = mid * (1 - stop_pct)
                else:
                    sig.stop_price = mid * (1 + stop_pct)

        # Risk check
        stop_dist = abs(mid - sig.stop_price) / mid if sig.stop_price else 0.005
        max_loss = stop_dist * sig.size_usd
        if max_loss > self.equity * self.cfg["risk"]["max_loss_per_trade_pct"]:
            log.info("Trade rejected: max loss $%.2f > $%.2f limit",
                     max_loss, self.equity * self.cfg["risk"]["max_loss_per_trade_pct"])
            self.signals_rejected["sizing"] += 1
            return

        if self.paper:
            log.info("[PAPER] %s %s $%.0f @ %.4f stop=%.4f tp=%.4f",
                     sig.side.upper(), sig.symbol, sig.size_usd, mid,
                     sig.stop_price or 0, sig.tp_price or 0)
        else:
            # Live order placement (requires wallet/API key setup)
            log.warning("Live order not implemented yet - use paper mode")
            return

        # Record position
        pos = Position(
            symbol=sig.symbol,
            strategy=sig.meta.get("strategy", "unknown"),
            side=sig.side,
            size_usd=sig.size_usd,
            entry_price=mid,
            stop_price=sig.stop_price,
            tp_price=sig.tp_price,
            max_hold_s=sig.meta.get("max_hold_hours", 8) * 3600,
            meta=sig.meta,
        )
        with self._lock:
            self.positions.append(pos)
        self.ks.register_open_position()

    def _close_position(self, pos: Position, exit_price: float, reason: str):
        """Close a position and record P&L."""
        if pos.side == "long":
            pnl = (exit_price - pos.entry_price) / pos.entry_price * pos.size_usd
        else:
            pnl = (pos.entry_price - exit_price) / pos.entry_price * pos.size_usd

        # Deduct fees
        fee = pos.size_usd * 3.8 / 10_000  # approximate
        net_pnl = pnl - fee

        hold_s = time.time() - pos.entry_time

        log.info("[CLOSE] %s %s %s: PnL=$%.2f (net=$%.2f) hold=%.0fs reason=%s",
                 pos.symbol, pos.side, pos.strategy, pnl, net_pnl, hold_s, reason)

        self.equity += net_pnl
        self.pnl_today += net_pnl
        self.trades_today += 1
        self.ks.update_equity(self.equity)

        with self._lock:
            if pos in self.positions:
                self.positions.remove(pos)
        self.ks.register_close_position()
        # Cooldown only after stop-loss (not after normal max_hold/funding exit)
        if reason == "stop_loss":
            self._last_close_time[(pos.symbol, pos.strategy)] = time.time()

        # Log to CSV
        self._log_trade(pos, exit_price, net_pnl, reason)

    def _close_all_positions(self, reason: str = "kill_switch"):
        """Close all open positions (called by kill switch)."""
        log.critical("CLOSING ALL POSITIONS: %s", reason)
        mids = self._fetch_mids()
        with self._lock:
            pos_copy = list(self.positions)
        for pos in pos_copy:
            exit_price = mids.get(pos.symbol, pos.entry_price)
            self._close_position(pos, exit_price, reason)

    def _update_equity_estimate(self, mids: dict[str, float]):
        """Estimate current equity from mark-to-market on open positions."""
        unrealised = 0.0
        with self._lock:
            for pos in self.positions:
                mid = mids.get(pos.symbol)
                if mid is None:
                    continue
                if pos.side == "long":
                    unrealised += (mid - pos.entry_price) / pos.entry_price * pos.size_usd
                else:
                    unrealised += (pos.entry_price - mid) / pos.entry_price * pos.size_usd

        # Don't update kill switch on every tick (only on realized P&L)
        # This is MTM only for display
        log.debug("MTM equity: %.2f (unrealised: %.2f)", self.equity + unrealised, unrealised)

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _refresh_universe(self):
        """Download/refresh candle and funding data for the universe."""
        log.info("Refreshing universe data...")
        cache_dir = self.cfg["data"]["cache_dir"]
        symbols = get_universe(
            min_volume=self.cfg["universe"]["min_volume_usd_24h"],
            top_n=self.cfg["universe"]["top_n"],
            cache_file=f"{cache_dir}/universe.json",
        )
        log.info("Universe: %d symbols", len(symbols))

        # Download recent candles for signal generation (shorter window)
        # Full 52-day download is done by run_wf_backtest.py pre-live
        interval = self.cfg["data"]["candle_interval"]
        days = min(7, self.cfg["data"]["candle_days"])  # 7 days for live

        candles_map = download_all_candles(
            symbols, interval=interval, days=days,
            cache_dir=cache_dir, cache_max_age_h=1.0,
        )
        for sym, df in candles_map.items():
            if not df.empty:
                self.candle_buffer[sym] = df.tail(CANDLE_BARS).to_dict("records")

        funding_map = download_all_funding(
            symbols, rows=100, cache_dir=cache_dir, cache_max_age_h=0.5,
        )
        for sym, df in funding_map.items():
            if not df.empty:
                self.funding_buffer[sym] = df.tail(200).to_dict("records")

        # Refresh S3 cointegrated pair selection with latest candle data
        candles_for_s3 = {
            sym: self._get_candle_df(sym)
            for sym in self.candle_buffer
        }
        candles_for_s3 = {k: v for k, v in candles_for_s3.items()
                          if v is not None and not v.empty}
        if candles_for_s3:
            self.router.s3.refresh_active_pairs(candles_for_s3)

    def _get_candle_df(self, sym: str):
        import pandas as pd
        data = self.candle_buffer.get(sym)
        if not data:
            return None
        df = pd.DataFrame(data)
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"])
        return df

    def _get_funding_df(self, sym: str):
        import pandas as pd
        data = self.funding_buffer.get(sym)
        if not data:
            return None
        df = pd.DataFrame(data)
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"])
        return df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_state(self):
        state = {
            "equity":     self.equity,
            "paper":      self.paper,
            "positions":  [
                {
                    "symbol":      p.symbol,
                    "strategy":    p.strategy,
                    "side":        p.side,
                    "size_usd":    p.size_usd,
                    "entry_price": p.entry_price,
                    "stop_price":  p.stop_price,
                    "tp_price":    p.tp_price,
                    "entry_time":  p.entry_time,
                }
                for p in self.positions
            ],
            "kill_status": self.ks.status_dict(),
            "timestamp":   time.time(),
        }
        try:
            Path(self.state_file).write_text(json.dumps(state, indent=2, default=str))
        except Exception as e:
            log.error("State save failed: %s", e)

    def _load_state(self):
        if not Path(self.state_file).exists():
            return
        try:
            state = json.loads(Path(self.state_file).read_text())
            self.equity = state.get("equity", self.equity)
            self.ks.state.current_equity = self.equity
            log.info("State loaded: equity=%.2f, %d positions",
                     self.equity, len(state.get("positions", [])))
        except Exception as e:
            log.warning("State load failed: %s", e)

    def _log_trade(self, pos: Position, exit_price: float, net_pnl: float, reason: str):
        import csv
        trade_log = self.cfg["logging"].get("trade_log", "logs/trades_v7.csv")
        Path(trade_log).parent.mkdir(exist_ok=True)
        write_header = not Path(trade_log).exists()
        with open(trade_log, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp", "symbol", "strategy", "side",
                             "size_usd", "entry", "exit", "net_pnl", "reason"])
            w.writerow([
                time.strftime("%Y-%m-%dT%H:%M:%S"),
                pos.symbol, pos.strategy, pos.side,
                round(pos.size_usd, 2), round(pos.entry_price, 6),
                round(exit_price, 6), round(net_pnl, 4), reason
            ])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Artemisia v7 Engine")
    parser.add_argument("--config", default="config.json",
                        help="Path to config file (default: config.json)")
    parser.add_argument("--live", action="store_true",
                        help="Run in live mode (default: paper)")
    args = parser.parse_args()

    cfg_path = Path(__file__).parent / args.config
    if not cfg_path.exists():
        print(f"{args.config} not found. Run from artemisia_v7/ directory.")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    engine = ArtemiasV7Engine(config_path=args.config, paper=not args.live)
    engine.run()
