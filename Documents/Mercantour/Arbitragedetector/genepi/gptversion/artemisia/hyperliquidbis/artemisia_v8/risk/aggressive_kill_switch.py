"""
aggressive_kill_switch.py — Stricter risk management for S7 (8x leverage, 150 trades/day).

Key differences vs v7 KillSwitch:
  - Daily DD hard stop: 3% (vs 4% v7)
  - Total DD hard stop: 6% (vs 8% v7)
  - Network timeout: 30s (vs 60s v7)
  - Anti-rampage: > 30 trades/h → suspend 5 min
  - Loss streak: 5 consecutive losses → suspend 30 min
  - BTC move guard: > 1.5% in 5 min → close all, pause 15 min
  - Max notional: 4× capital
"""
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

log = logging.getLogger(__name__)


@dataclass
class AggressiveRiskState:
    initial_capital: float = 500.0
    current_equity: float = 500.0
    daily_start_equity: float = 500.0
    session_peak_equity: float = 500.0

    # Kill flags
    daily_dd_breached: bool = False
    total_dd_breached: bool = False
    network_kill: bool = False
    manual_kill: bool = False

    # Suspension flags (temporary, auto-clear)
    rampage_suspended_until: float = 0.0    # timestamp
    streak_suspended_until: float = 0.0
    volatility_paused_until: float = 0.0

    # Counters
    open_positions: int = 0
    trades_today: int = 0
    last_poll_ts: float = field(default_factory=time.time)
    day_start_ts: float = field(default_factory=time.time)

    # Anti-rampage: recent trade timestamps
    recent_trade_ts: deque = field(default_factory=lambda: deque(maxlen=200))
    # Loss streak
    consecutive_losses: int = 0

    def is_hard_killed(self) -> bool:
        return (self.daily_dd_breached or self.total_dd_breached
                or self.network_kill or self.manual_kill)

    def is_suspended(self) -> bool:
        now = time.time()
        return (now < self.rampage_suspended_until
                or now < self.streak_suspended_until
                or now < self.volatility_paused_until)

    def daily_dd_pct(self) -> float:
        if self.daily_start_equity <= 0:
            return 0.0
        return (self.daily_start_equity - self.current_equity) / self.daily_start_equity * 100

    def total_dd_pct(self) -> float:
        if self.session_peak_equity <= 0:
            return 0.0
        return (self.session_peak_equity - self.current_equity) / self.session_peak_equity * 100


class AggressiveKillSwitch:
    """
    Kill switch tuned for S7: high frequency, 8x leverage, aggressive stops.

    Suspension reasons (temporary, not hard kill):
      - RAMPAGE : > max_trades_per_hour in last 60 min → suspend 5 min
      - STREAK  : max_loss_streak consecutive losses → suspend 30 min
      - VOLGUARD: BTC moved > btc_move_5m_pct in 5 min → pause 15 min

    Hard kill reasons (permanent until restart):
      - DAILY_DD: daily drawdown >= 3%
      - TOTAL_DD: total drawdown >= 6%
      - NETWORK : no WebSocket message for 30s
      - MANUAL  : manual override
    """

    def __init__(
        self,
        initial_capital: float = 500.0,
        daily_dd_pct: float = 3.0,
        total_dd_pct: float = 6.0,
        max_positions: int = 6,
        max_notional_mult: float = 4.0,
        network_timeout_s: float = 30.0,
        watchdog_interval_s: float = 5.0,
        max_trades_per_hour: int = 30,
        max_loss_streak: int = 5,
        btc_move_5m_pct: float = 1.5,
        close_all_callback: Optional[Callable] = None,
    ):
        self.max_daily_dd_pct   = daily_dd_pct
        self.max_total_dd_pct   = total_dd_pct
        self.max_positions      = max_positions
        self.max_notional       = initial_capital * max_notional_mult
        self.network_timeout_s  = network_timeout_s
        self.watchdog_interval  = watchdog_interval_s
        self.max_trades_per_hour = max_trades_per_hour
        self.max_loss_streak    = max_loss_streak
        self.btc_move_5m_pct    = btc_move_5m_pct
        self.close_all_callback = close_all_callback

        self.state = AggressiveRiskState(
            initial_capital=initial_capital,
            current_equity=initial_capital,
            daily_start_equity=initial_capital,
            session_peak_equity=initial_capital,
        )

        # BTC price history for vol guard (5-min window)
        self._btc_prices: deque = deque(maxlen=500)   # (ts, price)

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="ks-watchdog"
        )
        self._thread.start()
        log.info(
            "AggressiveKillSwitch started | daily=%.1f%% total=%.1f%% "
            "net_timeout=%ds rampage=%d/h streak=%d btc_guard=%.1f%%",
            self.max_daily_dd_pct, self.max_total_dd_pct,
            self.network_timeout_s, self.max_trades_per_hour,
            self.max_loss_streak, self.btc_move_5m_pct,
        )

    def stop(self):
        self._running = False

    # ------------------------------------------------------------------
    # Public API (called by engine on every event)
    # ------------------------------------------------------------------

    def record_ws_heartbeat(self):
        """Call on every incoming WebSocket message."""
        with self._lock:
            self.state.last_poll_ts = time.time()

    def update_equity(self, new_equity: float):
        with self._lock:
            self.state.current_equity = new_equity
            if new_equity > self.state.session_peak_equity:
                self.state.session_peak_equity = new_equity
            self._check_dd_limits()

    def update_btc_price(self, price: float):
        """Feed BTC mid price to monitor 5-min volatility guard."""
        with self._lock:
            self._btc_prices.append((time.time(), price))

    def record_trade(self, pnl: float):
        """Call after every fill close."""
        with self._lock:
            self.state.trades_today += 1
            self.state.recent_trade_ts.append(time.time())

            if pnl < 0:
                self.state.consecutive_losses += 1
                if self.state.consecutive_losses >= self.max_loss_streak:
                    resume = time.time() + 1800  # 30 min
                    self.state.streak_suspended_until = resume
                    self.state.consecutive_losses = 0
                    log.warning(
                        "LOSS STREAK %d consecutive losses → suspend 30 min",
                        self.max_loss_streak,
                    )
            else:
                self.state.consecutive_losses = 0

            # Rampage check: count trades in last 60 min
            cutoff = time.time() - 3600
            recent = sum(1 for ts in self.state.recent_trade_ts if ts > cutoff)
            if recent > self.max_trades_per_hour:
                resume = time.time() + 300  # 5 min
                self.state.rampage_suspended_until = resume
                log.warning("RAMPAGE: %d trades/h > %d limit → suspend 5 min",
                            recent, self.max_trades_per_hour)

    def register_open_position(self):
        with self._lock:
            self.state.open_positions += 1

    def register_close_position(self):
        with self._lock:
            self.state.open_positions = max(0, self.state.open_positions - 1)

    def can_open_position(self, notional_usd: float = 0.0) -> tuple[bool, str]:
        """
        Returns (True, "") if a new position is allowed.
        Checks hard kills first, then soft suspensions, then limits.
        """
        with self._lock:
            if self.state.is_hard_killed():
                return False, self._hard_kill_reason()

            now = time.time()
            if now < self.state.rampage_suspended_until:
                secs = self.state.rampage_suspended_until - now
                return False, f"rampage suspend ({secs:.0f}s remaining)"
            if now < self.state.streak_suspended_until:
                secs = self.state.streak_suspended_until - now
                return False, f"loss streak suspend ({secs:.0f}s remaining)"
            if now < self.state.volatility_paused_until:
                secs = self.state.volatility_paused_until - now
                return False, f"vol guard pause ({secs:.0f}s remaining)"

            if self.state.open_positions >= self.max_positions:
                return False, f"max_positions={self.max_positions}"

            return True, ""

    def kill(self, reason: str = "manual"):
        with self._lock:
            self.state.manual_kill = True
        log.critical("MANUAL KILL: %s", reason)
        self._execute_close_all(reason)

    def reset_daily(self):
        with self._lock:
            self.state.daily_start_equity = self.state.current_equity
            self.state.daily_dd_breached  = False
            self.state.trades_today       = 0
            self.state.consecutive_losses = 0
            self.state.day_start_ts       = time.time()
        log.info("Daily reset: equity=%.2f", self.state.current_equity)

    @property
    def is_killed(self) -> bool:
        return self.state.is_hard_killed()

    def status_dict(self) -> dict:
        with self._lock:
            s = self.state
            now = time.time()
            return {
                "equity":             s.current_equity,
                "daily_dd_pct":       s.daily_dd_pct(),
                "total_dd_pct":       s.total_dd_pct(),
                "open_positions":     s.open_positions,
                "trades_today":       s.trades_today,
                "consecutive_losses": s.consecutive_losses,
                "hard_killed":        s.is_hard_killed(),
                "suspended":          s.is_suspended(),
                "rampage_remaining":  max(0, s.rampage_suspended_until - now),
                "streak_remaining":   max(0, s.streak_suspended_until - now),
                "volguard_remaining": max(0, s.volatility_paused_until - now),
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _hard_kill_reason(self) -> str:
        s = self.state
        reasons = []
        if s.daily_dd_breached:
            reasons.append(f"daily DD {s.daily_dd_pct():.2f}%")
        if s.total_dd_breached:
            reasons.append(f"total DD {s.total_dd_pct():.2f}%")
        if s.network_kill:
            reasons.append("network timeout")
        if s.manual_kill:
            reasons.append("manual kill")
        return "KILL: " + ", ".join(reasons)

    def _check_dd_limits(self):
        """Must be called with self._lock held."""
        daily = self.state.daily_dd_pct()
        total = self.state.total_dd_pct()

        if daily >= self.max_daily_dd_pct and not self.state.daily_dd_breached:
            self.state.daily_dd_breached = True
            log.critical("DAILY DD BREACHED: %.2f%% >= %.1f%%", daily, self.max_daily_dd_pct)
            threading.Thread(
                target=self._execute_close_all,
                args=(f"daily DD {daily:.2f}%",), daemon=True
            ).start()

        if total >= self.max_total_dd_pct and not self.state.total_dd_breached:
            self.state.total_dd_breached = True
            log.critical("TOTAL DD BREACHED: %.2f%% >= %.1f%%", total, self.max_total_dd_pct)
            threading.Thread(
                target=self._execute_close_all,
                args=(f"total DD {total:.2f}%",), daemon=True
            ).start()

    def _check_btc_vol_guard(self):
        """Check 5-min BTC move. Called from watchdog. Lock NOT held."""
        with self._lock:
            prices = list(self._btc_prices)

        if len(prices) < 2:
            return

        now = time.time()
        cutoff = now - 300  # 5 min
        recent = [(ts, px) for ts, px in prices if ts >= cutoff]
        if len(recent) < 2:
            return

        px_old = recent[0][1]
        px_new = recent[-1][1]
        if px_old <= 0:
            return

        move_pct = abs(px_new - px_old) / px_old * 100
        if move_pct >= self.btc_move_5m_pct:
            with self._lock:
                already_paused = time.time() < self.state.volatility_paused_until
            if not already_paused:
                pause_until = time.time() + 900  # 15 min
                with self._lock:
                    self.state.volatility_paused_until = pause_until
                log.warning(
                    "VOL GUARD: BTC moved %.2f%% in 5 min (>= %.1f%%) → pause 15 min",
                    move_pct, self.btc_move_5m_pct,
                )
                self._execute_close_all(f"BTC vol guard ({move_pct:.1f}% in 5m)")

    def _watchdog_loop(self):
        while self._running:
            time.sleep(self.watchdog_interval)

            with self._lock:
                since = time.time() - self.state.last_poll_ts
                killed = self.state.network_kill
                day_age = time.time() - self.state.day_start_ts

            if since > self.network_timeout_s and not killed:
                log.critical("NETWORK TIMEOUT: %.0fs without WS message (limit=%ds)",
                             since, self.network_timeout_s)
                with self._lock:
                    self.state.network_kill = True
                self._execute_close_all(f"network timeout {since:.0f}s")

            self._check_btc_vol_guard()

            if day_age > 86400:
                self.reset_daily()

    def _execute_close_all(self, reason: str):
        log.critical("CLOSE ALL: %s", reason)
        if self.close_all_callback:
            try:
                self.close_all_callback(reason=reason)
            except Exception as e:
                log.error("close_all_callback error: %s", e)
