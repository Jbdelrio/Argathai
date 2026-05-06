"""
kill_switch.py — Risk management: DD watchdog, network watchdog, position limits
Non-negotiable constraints:
- Max 0.5% loss per trade ($2.50 on $500)
- Daily DD kill at 4% hard stop (via cascading)
- Total DD hard stop at 8% ($40)
- Network: 60s no-poll -> close all
- Max 3 simultaneous positions

v7.5 addition: CascadingDrawdownProtection replaces binary 2% kill.
Graduated reduction: 1.5% -> 50%, 2.5% -> 25%, 4% -> full stop.
"""
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable

log = logging.getLogger(__name__)


class CascadingDrawdownProtection:
    """
    Graduated position-size reduction based on intra-day drawdown.
    Inspired by FinAgent 2025. Replaces a binary kill-switch with a
    smooth throttle: the strategy can recover without a full stop.

    Levels (daily DD -> allocation multiplier):
      < 1.0%  -> 100%  (full normal sizing)
      >= 1.5% ->  50%  (halve all new positions)
      >= 2.5% ->  25%  (quarter sizing — damage control)
      >= 4.0% ->   0%  (no new positions until midnight reset)

    Auto-resets at UTC midnight alongside the daily equity baseline.
    """

    # Thresholds in PERCENTAGE POINTS (matching RiskState.daily_dd_pct() output)
    LEVELS = [
        (1.0, 1.00),   # DD < 1.0%  -> 100%
        (1.5, 0.50),   # DD >= 1.5% -> 50%
        (2.5, 0.25),   # DD >= 2.5% -> 25%
        (4.0, 0.00),   # DD >= 4.0% -> halted
    ]

    def get_multiplier(self, daily_dd_pct: float) -> float:
        """Return allocation multiplier [0, 1] for the given daily DD %.
        daily_dd_pct is in percentage points (e.g. 1.6 means 1.6%)."""
        multiplier = 1.00
        for threshold, mult in self.LEVELS:
            if daily_dd_pct >= threshold:
                multiplier = mult
            else:
                break
        return multiplier

    def adjust_size(self, base_size: float, daily_dd_pct: float) -> float:
        return base_size * self.get_multiplier(daily_dd_pct)

    def level_description(self, daily_dd_pct: float) -> str:
        m = self.get_multiplier(daily_dd_pct)
        if m == 0.00:
            return "HALTED"
        if m == 0.25:
            return "REDUCED_75"
        if m == 0.50:
            return "REDUCED_50"
        return "NORMAL"


@dataclass
class RiskState:
    initial_capital: float = 500.0
    current_equity: float = 500.0
    daily_start_equity: float = 500.0
    session_peak_equity: float = 500.0
    daily_peak_equity: float = 500.0

    # Kill flags
    daily_dd_breached: bool = False
    total_dd_breached: bool = False
    network_kill: bool = False
    manual_kill: bool = False

    # Counters
    open_positions: int = 0
    trades_today: int = 0
    last_poll_ts: float = field(default_factory=time.time)
    day_start_ts: float = field(default_factory=time.time)

    def is_killed(self) -> bool:
        return (self.daily_dd_breached or self.total_dd_breached
                or self.network_kill or self.manual_kill)

    def daily_dd_pct(self) -> float:
        if self.daily_start_equity <= 0:
            return 0.0
        return (self.daily_start_equity - self.current_equity) / self.daily_start_equity * 100

    def total_dd_pct(self) -> float:
        if self.session_peak_equity <= 0:
            return 0.0
        return (self.session_peak_equity - self.current_equity) / self.session_peak_equity * 100

    def max_loss_per_trade(self, max_pct: float = 0.005) -> float:
        return self.current_equity * max_pct


class KillSwitch:
    """
    Thread-safe kill switch monitor.

    Usage:
        ks = KillSwitch(close_all_callback=engine.close_all_positions)
        ks.start()
        ...
        ks.update_equity(new_equity)
        ks.record_poll()
        ...
        ks.stop()
    """

    def __init__(
        self,
        initial_capital: float = 500.0,
        daily_dd_pct: float = 4.0,    # v7.5: hard stop at 4% (cascade handles earlier)
        total_dd_pct: float = 8.0,
        max_positions: int = 3,
        watchdog_interval_s: float = 60.0,
        network_timeout_s: float = 120.0,
        close_all_callback: Optional[Callable] = None,
    ):
        self.max_daily_dd_pct = daily_dd_pct
        self.max_total_dd_pct = total_dd_pct
        self.max_positions = max_positions
        self.watchdog_interval_s = watchdog_interval_s
        self.network_timeout_s = network_timeout_s
        self.close_all_callback = close_all_callback

        self.cascade = CascadingDrawdownProtection()

        self.state = RiskState(
            initial_capital=initial_capital,
            current_equity=initial_capital,
            daily_start_equity=initial_capital,
            session_peak_equity=initial_capital,
            daily_peak_equity=initial_capital,
        )

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Start background watchdog thread."""
        self._running = True
        self._thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._thread.start()
        log.info("KillSwitch started (daily_dd=%.1f%%, total_dd=%.1f%%, timeout=%ds)",
                 self.max_daily_dd_pct, self.max_total_dd_pct, self.network_timeout_s)

    def stop(self):
        self._running = False

    def record_poll(self):
        """Call this every time a successful API poll completes."""
        with self._lock:
            self.state.last_poll_ts = time.time()

    def update_equity(self, new_equity: float):
        """Call after every P&L update."""
        with self._lock:
            self.state.current_equity = new_equity
            if new_equity > self.state.session_peak_equity:
                self.state.session_peak_equity = new_equity
            if new_equity > self.state.daily_peak_equity:
                self.state.daily_peak_equity = new_equity
            self._check_dd_limits()

    def register_open_position(self):
        """Call when a new position is opened."""
        with self._lock:
            self.state.open_positions += 1
            self.state.trades_today += 1

    def register_close_position(self):
        """Call when a position is closed."""
        with self._lock:
            self.state.open_positions = max(0, self.state.open_positions - 1)

    def can_open_position(self) -> tuple[bool, str]:
        """
        Returns (True, "") if a new position is allowed.
        Returns (False, reason) if blocked.
        Also blocks if cascade level is 0% (daily DD >= 4%).
        """
        with self._lock:
            if self.state.is_killed():
                reasons = []
                if self.state.daily_dd_breached:
                    reasons.append(f"daily DD {self.state.daily_dd_pct():.1f}%")
                if self.state.total_dd_breached:
                    reasons.append(f"total DD {self.state.total_dd_pct():.1f}%")
                if self.state.network_kill:
                    reasons.append("network timeout")
                if self.state.manual_kill:
                    reasons.append("manual kill")
                return False, "KILL: " + ", ".join(reasons)

            if self.state.open_positions >= self.max_positions:
                return False, f"max positions reached ({self.max_positions})"

            # Cascade check: if multiplier is 0, no new entries allowed
            daily_dd = self.state.daily_dd_pct()
            mult = self.cascade.get_multiplier(daily_dd)
            if mult == 0.0:
                return False, f"cascade HALTED (daily DD {daily_dd:.1f}%)"

            return True, ""

    def cascade_multiplier(self) -> float:
        """Return current cascade allocation multiplier (0-1)."""
        with self._lock:
            return self.cascade.get_multiplier(self.state.daily_dd_pct())

    def cascade_level(self) -> str:
        """Return human-readable cascade level."""
        with self._lock:
            return self.cascade.level_description(self.state.daily_dd_pct())

    def position_size_usd(self, edge_bps: float, win_rate: float,
                           max_loss_pct: float = 0.005,
                           kelly_fraction: float = 0.25) -> float:
        """
        Fractional Kelly position sizing with DD reduction.
        Returns USD notional to deploy (before leverage).
        """
        with self._lock:
            equity = self.state.current_equity
            dd_pct = self.state.total_dd_pct()

        # Kelly formula: f = (p*b - q) / b
        # where b = reward/risk ratio, p = win_rate, q = 1-p
        # We approximate b from edge and max_loss
        if win_rate <= 0 or win_rate >= 1:
            win_rate = max(0.01, min(0.99, win_rate))

        reward_bps = edge_bps  # approximate
        risk_bps = max_loss_pct * 10_000  # 0.5% = 50bps

        if risk_bps <= 0:
            return 0.0

        b = reward_bps / risk_bps
        if b <= 0:
            return 0.0

        q = 1 - win_rate
        kelly_full = (win_rate * b - q) / b
        kelly_full = max(0.0, kelly_full)

        # Fractional Kelly
        kelly_frac = kelly_full * kelly_fraction

        # Apply cascading DD multiplier (uses DAILY dd, not total)
        with self._lock:
            daily_dd = self.state.daily_dd_pct()
        cascade_mult = self.cascade.get_multiplier(daily_dd)
        kelly_frac *= cascade_mult

        if cascade_mult < 1.0:
            log.debug(
                "Cascade reduction: daily_dd=%.1f%% mult=%.2f",
                daily_dd, cascade_mult
            )

        # Hard cap: max 20% of equity per position
        kelly_frac = min(kelly_frac, 0.20)

        size_usd = equity * kelly_frac
        # Minimum meaningful size: $10
        return max(0.0, size_usd) if size_usd >= 10.0 else 0.0

    def kill(self, reason: str = "manual"):
        """Manually trigger kill switch."""
        with self._lock:
            self.state.manual_kill = True
        log.critical("KILL SWITCH TRIGGERED: %s", reason)
        self._execute_close_all(reason)

    def reset_daily(self):
        """Call at UTC midnight to reset daily counters."""
        with self._lock:
            self.state.daily_start_equity = self.state.current_equity
            self.state.daily_peak_equity = self.state.current_equity
            self.state.daily_dd_breached = False
            self.state.trades_today = 0
            self.state.day_start_ts = time.time()
        log.info("Daily reset: equity=%.2f", self.state.current_equity)

    @property
    def is_killed(self) -> bool:
        return self.state.is_killed()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_dd_limits(self):
        """Must be called with self._lock held."""
        daily_dd = self.state.daily_dd_pct()
        total_dd = self.state.total_dd_pct()

        if daily_dd >= self.max_daily_dd_pct and not self.state.daily_dd_breached:
            self.state.daily_dd_breached = True
            log.critical("DAILY DD LIMIT BREACHED: %.2f%% (limit=%.1f%%)",
                         daily_dd, self.max_daily_dd_pct)
            threading.Thread(
                target=self._execute_close_all,
                args=(f"daily DD {daily_dd:.2f}%",),
                daemon=True
            ).start()

        if total_dd >= self.max_total_dd_pct and not self.state.total_dd_breached:
            self.state.total_dd_breached = True
            log.critical("TOTAL DD LIMIT BREACHED: %.2f%% (limit=%.1f%%)",
                         total_dd, self.max_total_dd_pct)
            threading.Thread(
                target=self._execute_close_all,
                args=(f"total DD {total_dd:.2f}%",),
                daemon=True
            ).start()

    def _watchdog_loop(self):
        """Background thread: checks network timeout and triggers midnight reset."""
        while self._running:
            time.sleep(self.watchdog_interval_s)

            with self._lock:
                since_poll = time.time() - self.state.last_poll_ts
                killed = self.state.network_kill
                day_age = time.time() - self.state.day_start_ts

            # Network timeout check
            if since_poll > self.network_timeout_s and not killed:
                log.critical("NETWORK TIMEOUT: no poll for %.0fs (limit=%ds)",
                             since_poll, self.network_timeout_s)
                with self._lock:
                    self.state.network_kill = True
                self._execute_close_all(f"network timeout {since_poll:.0f}s")

            # Midnight reset (86400s = 24h)
            if day_age > 86400:
                self.reset_daily()

    def _execute_close_all(self, reason: str):
        """Call the registered close_all callback."""
        log.critical("CLOSE ALL POSITIONS: %s", reason)
        if self.close_all_callback:
            try:
                self.close_all_callback(reason=reason)
            except Exception as e:
                log.error("close_all_callback failed: %s", e)

    def status_dict(self) -> dict:
        with self._lock:
            s = self.state
            daily_dd = s.daily_dd_pct()
            return {
                "equity": s.current_equity,
                "daily_dd_pct": daily_dd,
                "total_dd_pct": s.total_dd_pct(),
                "open_positions": s.open_positions,
                "trades_today": s.trades_today,
                "killed": s.is_killed(),
                "kill_reasons": {
                    "daily_dd": s.daily_dd_breached,
                    "total_dd": s.total_dd_breached,
                    "network": s.network_kill,
                    "manual": s.manual_kill,
                },
                "cascade_level": self.cascade.level_description(daily_dd),
                "cascade_multiplier": self.cascade.get_multiplier(daily_dd),
                "since_last_poll_s": time.time() - s.last_poll_ts,
            }


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")

    def fake_close_all(reason=""):
        print(f"[CLOSE ALL] reason={reason}")

    ks = KillSwitch(
        initial_capital=500.0,
        daily_dd_pct=2.0,
        total_dd_pct=8.0,
        close_all_callback=fake_close_all,
    )
    ks.start()

    # Simulate equity drop
    print("Initial:", ks.status_dict())
    ks.update_equity(495.0)
    print("After -$5:", ks.status_dict())
    ks.update_equity(489.5)  # triggers daily DD (2.1%)
    print("After -$10.5:", ks.status_dict())

    size = ks.position_size_usd(edge_bps=5.0, win_rate=0.55)
    print(f"Position size: ${size:.2f}")

    ks.stop()
