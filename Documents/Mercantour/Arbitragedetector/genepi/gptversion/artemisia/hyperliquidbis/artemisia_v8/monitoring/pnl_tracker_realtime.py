"""
pnl_tracker_realtime.py — Minute-by-minute metrics logging for S7.

Logs one CSV row per minute to logs/metrics_s7.csv.
Exposes get_dashboard() for the terminal display.
"""
import csv
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class MinuteSnapshot:
    ts: float
    open_positions: int
    quotes_active: int
    fills_count: int
    pnl_minute: float
    pnl_hour: float
    pnl_day: float
    win_rate_day: float
    avg_hold_s: float
    win_count_day: int
    loss_count_day: int
    stop_count_day: int
    tp_count_day: int
    reconnections: int
    blacklisted_coins: int


class PnLTracker:
    """
    Tracks per-trade and per-minute PnL for S7.
    Thread-safe; called from the main engine loop.
    """

    def __init__(self, log_path: str = "logs/metrics_s7.csv", equity: float = 500.0):
        self.log_path = log_path
        self.initial_equity = equity

        # Minute buckets
        self._minute_pnl: deque = deque(maxlen=60)      # last 60 minutes
        self._current_min_pnl: float = 0.0
        self._current_min_fills: int = 0
        self._current_min_start: float = time.time()

        # Day stats
        self._day_start: float = time.time()
        self._day_pnl: float = 0.0
        self._wins: int = 0
        self._losses: int = 0
        self._stops: int = 0
        self._tps: int = 0
        self._hold_times: deque = deque(maxlen=500)

        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Called on every trade close
    # ------------------------------------------------------------------

    def record_trade(self, net_pnl: float, hold_s: float, reason: str):
        self._day_pnl += net_pnl
        self._current_min_pnl += net_pnl
        self._current_min_fills += 1
        self._hold_times.append(hold_s)

        if net_pnl > 0:
            self._wins += 1
        else:
            self._losses += 1

        if reason == "stop_loss":
            self._stops += 1
        elif reason == "take_profit":
            self._tps += 1

    # ------------------------------------------------------------------
    # Called every ~60s from engine loop
    # ------------------------------------------------------------------

    def tick(
        self,
        open_positions: int,
        quotes_active: int,
        reconnections: int,
        blacklisted_coins: int,
    ) -> MinuteSnapshot:
        now = time.time()
        elapsed = now - self._current_min_start

        # Flush current minute if >= 60s
        if elapsed >= 60:
            self._minute_pnl.append(self._current_min_pnl)
            self._current_min_pnl  = 0.0
            self._current_min_fills = 0
            self._current_min_start = now

        # Rolling 1h PnL
        pnl_hour = sum(self._minute_pnl)

        total_trades = self._wins + self._losses
        win_rate = self._wins / total_trades if total_trades > 0 else 0.0
        avg_hold = sum(self._hold_times) / len(self._hold_times) if self._hold_times else 0.0

        snap = MinuteSnapshot(
            ts=now,
            open_positions=open_positions,
            quotes_active=quotes_active,
            fills_count=self._current_min_fills,
            pnl_minute=self._current_min_pnl,
            pnl_hour=pnl_hour,
            pnl_day=self._day_pnl,
            win_rate_day=win_rate,
            avg_hold_s=avg_hold,
            win_count_day=self._wins,
            loss_count_day=self._losses,
            stop_count_day=self._stops,
            tp_count_day=self._tps,
            reconnections=reconnections,
            blacklisted_coins=blacklisted_coins,
        )

        self._write_csv(snap)

        # Daily reset at UTC midnight
        if now - self._day_start > 86400:
            self._reset_daily(now)

        return snap

    # ------------------------------------------------------------------
    # Dashboard string
    # ------------------------------------------------------------------

    def get_dashboard(
        self,
        snap: MinuteSnapshot,
        open_pos_detail: str,
        blacklist_detail: str,
        ks_status: dict,
        equity: float,
    ) -> str:
        total_trades = snap.win_count_day + snap.loss_count_day
        wr_str   = f"{snap.win_rate_day*100:.1f}%" if total_trades > 0 else "—"
        hold_str = f"{snap.avg_hold_s:.0f}s" if snap.avg_hold_s > 0 else "—"

        # Risk bars (ASCII)
        def bar(val_pct: float, limit_pct: float, width: int = 10) -> str:
            ratio = min(val_pct / max(limit_pct, 0.001), 1.0)
            filled = round(ratio * width)
            return "▓" * filled + "░" * (width - filled)

        daily_dd   = ks_status.get("daily_dd_pct", 0.0)
        total_dd   = ks_status.get("total_dd_pct", 0.0)
        trades_h   = ks_status.get("trades_today", 0)
        max_tph    = 30

        # Suspension info
        susp_parts = []
        if ks_status.get("rampage_remaining", 0) > 0:
            susp_parts.append(f"RAMPAGE {ks_status['rampage_remaining']:.0f}s")
        if ks_status.get("streak_remaining", 0) > 0:
            susp_parts.append(f"STREAK {ks_status['streak_remaining']:.0f}s")
        if ks_status.get("volguard_remaining", 0) > 0:
            susp_parts.append(f"VOLGUARD {ks_status['volguard_remaining']:.0f}s")
        susp_str = " | ".join(susp_parts) if susp_parts else "none"

        lines = [
            "┌" + "─" * 57 + "┐",
            "│ S7 AGGRESSIVE MAKER SCALPING  [PAPER]" + " " * 19 + "│",
            "├" + "─" * 57 + "┤",
            f"│ Equity: ${equity:.2f}  │  Quotes active: {snap.quotes_active:<4}  │  Pos: {snap.open_positions:<2}        │",
            f"│ PnL today: ${snap.pnl_day:+.4f}  │  Trades: {total_trades:<4}  │  WR: {wr_str:<7}  │",
            f"│ PnL 1h: ${snap.pnl_hour:+.4f}  │  TP: {snap.tp_count_day}  Stop: {snap.stop_count_day}  Avg hold: {hold_str:<6} │",
            "├" + "─" * 57 + "┤",
            f"│ Open positions: {open_pos_detail:<41}│",
            "├" + "─" * 57 + "┤",
            f"│ Daily DD:  {bar(daily_dd, 3.0)} {daily_dd:.2f}% / 3.0%         │",
            f"│ Total DD:  {bar(total_dd, 6.0)} {total_dd:.2f}% / 6.0%         │",
            f"│ Trade rate:{bar(trades_h % 30, max_tph)} {trades_h % 30}/{max_tph} per hour        │",
            "├" + "─" * 57 + "┤",
            f"│ Suspend: {susp_str:<48}│",
            f"│ Blacklist: {blacklist_detail:<46}│",
            "└" + "─" * 57 + "┘",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_csv(self, snap: MinuteSnapshot):
        try:
            write_header = not Path(self.log_path).exists()
            with open(self.log_path, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow([
                        "ts", "open_positions", "quotes_active",
                        "fills_min", "pnl_min", "pnl_hour", "pnl_day",
                        "win_rate", "avg_hold_s", "wins", "losses",
                        "stops", "tps", "reconnections", "blacklisted",
                    ])
                w.writerow([
                    time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(snap.ts)),
                    snap.open_positions, snap.quotes_active,
                    snap.fills_count,
                    round(snap.pnl_minute, 6), round(snap.pnl_hour, 6),
                    round(snap.pnl_day, 6),
                    round(snap.win_rate_day, 4), round(snap.avg_hold_s, 1),
                    snap.win_count_day, snap.loss_count_day,
                    snap.stop_count_day, snap.tp_count_day,
                    snap.reconnections, snap.blacklisted_coins,
                ])
        except Exception as e:
            log.error("Metrics CSV write failed: %s", e)

    def _reset_daily(self, now: float):
        self._day_start = now
        self._day_pnl   = 0.0
        self._wins      = 0
        self._losses    = 0
        self._stops     = 0
        self._tps       = 0
        self._hold_times.clear()
        self._minute_pnl.clear()
        log.info("PnL tracker daily reset")
