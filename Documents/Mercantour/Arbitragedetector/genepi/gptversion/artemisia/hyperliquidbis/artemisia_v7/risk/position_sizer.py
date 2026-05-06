"""
position_sizer.py — Fractional Kelly position sizing with regime awareness.
Plugs into KillSwitch.position_size_usd() but can also be used standalone.
"""
import logging
import numpy as np

log = logging.getLogger(__name__)

# Fee model (bps)
MAKER_REBATE_BPS = 0.3
TAKER_FEE_BPS = 2.5
SLIPPAGE_BPS = 0.8
ROUNDTRIP_MAKER_BPS = (TAKER_FEE_BPS - MAKER_REBATE_BPS) * 2 + SLIPPAGE_BPS * 2  # ~4.8
ROUNDTRIP_TAKER_BPS = TAKER_FEE_BPS * 2 + SLIPPAGE_BPS * 2                        # ~6.6


def net_edge_bps(gross_edge_bps: float, order_type: str = "maker") -> float:
    """Subtract round-trip costs from gross edge."""
    roundtrip = ROUNDTRIP_MAKER_BPS if order_type == "maker" else ROUNDTRIP_TAKER_BPS
    return gross_edge_bps - roundtrip


def kelly_fraction(win_rate: float,
                   avg_win_bps: float,
                   avg_loss_bps: float) -> float:
    """
    Full Kelly fraction.
    f* = (p * b - q) / b   where b = avg_win / avg_loss
    Returns fraction in [0, 1].
    """
    if avg_loss_bps <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    b = avg_win_bps / avg_loss_bps
    q = 1.0 - win_rate
    f = (win_rate * b - q) / b
    return max(0.0, min(1.0, f))


def size_usd(equity: float,
             win_rate: float,
             avg_win_bps: float,
             avg_loss_bps: float,
             kelly_frac: float = 0.25,
             dd_reduction_factor: float = 1.0,
             max_pct: float = 0.20,
             min_usd: float = 10.0) -> float:
    """
    Compute position size in USD.

    Args:
        equity: Current account equity in USD
        win_rate: Fraction [0,1]
        avg_win_bps: Average winning trade in basis points
        avg_loss_bps: Average losing trade magnitude in basis points
        kelly_frac: Fractional Kelly multiplier (default 0.25 = quarter-Kelly)
        dd_reduction_factor: Multiplier in [0,1] applied when in drawdown
        max_pct: Hard cap as fraction of equity (default 20%)
        min_usd: Minimum size; returns 0 if below this

    Returns:
        USD notional position size (pre-leverage)
    """
    f_full = kelly_fraction(win_rate, avg_win_bps, avg_loss_bps)
    f = f_full * kelly_frac * dd_reduction_factor
    f = min(f, max_pct)

    size = equity * f
    if size < min_usd:
        log.debug("Position size $%.2f below min $%.0f, returning 0", size, min_usd)
        return 0.0

    log.debug(
        "Sizing: equity=$%.2f WR=%.1f%% win=%.1fbps loss=%.1fbps "
        "full_kelly=%.3f frac=%.3f dd_factor=%.2f -> $%.2f",
        equity, win_rate * 100, avg_win_bps, avg_loss_bps,
        f_full, f, dd_reduction_factor, size
    )
    return size


def atr_stop_distance(atr: float, multiplier: float = 2.0) -> float:
    """Price distance for ATR-based stop."""
    return atr * multiplier


def check_trade_risk(entry_price: float,
                     stop_price: float,
                     position_size_usd: float,
                     equity: float,
                     max_loss_pct: float = 0.005) -> tuple[bool, float]:
    """
    Verify that the trade's max loss doesn't exceed max_loss_pct of equity.

    Returns:
        (ok, actual_loss_pct)
    """
    if entry_price <= 0 or position_size_usd <= 0:
        return False, 0.0

    price_move_pct = abs(entry_price - stop_price) / entry_price
    max_loss_usd = position_size_usd * price_move_pct
    actual_loss_pct = max_loss_usd / equity

    ok = actual_loss_pct <= max_loss_pct
    if not ok:
        log.debug(
            "Trade risk check FAIL: loss=$%.2f (%.2f%% > %.2f%% limit)",
            max_loss_usd, actual_loss_pct * 100, max_loss_pct * 100
        )
    return ok, actual_loss_pct


def regime_dd_factor(total_dd_pct: float,
                     threshold_pct: float = 4.0,
                     min_factor: float = 0.25) -> float:
    """
    Returns a factor in [min_factor, 1.0] based on current drawdown.
    - Below threshold: factor = 1.0 (full sizing)
    - At 2x threshold (8%): factor = min_factor
    Interpolated linearly between.
    """
    if total_dd_pct <= threshold_pct:
        return 1.0
    max_dd = threshold_pct * 2.0
    factor = 1.0 - (total_dd_pct - threshold_pct) / (max_dd - threshold_pct) * (1.0 - min_factor)
    return max(min_factor, min(1.0, factor))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Example: funding arb with 8bps gross edge, 65% win rate
    gross = 8.0
    net = net_edge_bps(gross, "maker")
    print(f"Net edge after fees: {net:.1f}bps")

    sz = size_usd(
        equity=500.0,
        win_rate=0.65,
        avg_win_bps=8.0,
        avg_loss_bps=4.0,
        kelly_frac=0.25,
    )
    print(f"Position size: ${sz:.2f}")

    # With 5% drawdown
    dd_f = regime_dd_factor(5.0)
    sz_dd = size_usd(
        equity=460.0,
        win_rate=0.65,
        avg_win_bps=8.0,
        avg_loss_bps=4.0,
        kelly_frac=0.25,
        dd_reduction_factor=dd_f,
    )
    print(f"Position size at 5% DD: ${sz_dd:.2f} (factor={dd_f:.2f})")

    # Risk check
    ok, loss_pct = check_trade_risk(
        entry_price=100.0,
        stop_price=99.4,   # 0.6% stop
        position_size_usd=sz,
        equity=500.0,
        max_loss_pct=0.005,
    )
    print(f"Risk check: ok={ok}, loss={loss_pct*100:.3f}%")
