"""
metrics.py — Backtest performance metrics
Sharpe, Sortino, Calmar, Profit Factor, Monte Carlo CI
"""
import numpy as np
import pandas as pd
from typing import Optional


def sharpe_ratio(returns: np.ndarray,
                 annualisation: float = 252 * 96,  # 15-min bars per year
                 risk_free: float = 0.0) -> float:
    """
    Annualised Sharpe ratio from per-bar returns.
    annualisation = bars per year (96 bars/day * 252 days for 15m).
    """
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / annualisation
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(annualisation))


def sortino_ratio(returns: np.ndarray,
                  annualisation: float = 252 * 96,
                  risk_free: float = 0.0) -> float:
    """Sortino ratio (penalises only downside volatility)."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / annualisation
    downside = excess[excess < 0]
    if len(downside) == 0:
        return np.inf
    downside_std = np.std(downside, ddof=1)
    if downside_std == 0:
        return 0.0
    return float(np.mean(excess) / downside_std * np.sqrt(annualisation))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum drawdown as a positive fraction (e.g. 0.05 = 5%)."""
    if len(equity_curve) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    dd = (peak - equity_curve) / np.where(peak > 0, peak, 1)
    return float(np.max(dd))


def calmar_ratio(returns: np.ndarray,
                 annualisation: float = 252 * 96) -> float:
    """Calmar = annualised return / max drawdown."""
    equity = np.cumprod(1 + returns)
    ann_return = float(np.mean(returns) * annualisation)
    mdd = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    return ann_return / mdd


def profit_factor(pnl_series: np.ndarray) -> float:
    """Sum of wins / sum of losses (magnitude)."""
    wins = pnl_series[pnl_series > 0].sum()
    losses = abs(pnl_series[pnl_series < 0].sum())
    if losses == 0:
        return np.inf if wins > 0 else 0.0
    return float(wins / losses)


def win_rate(pnl_series: np.ndarray) -> float:
    """Fraction of trades with positive P&L."""
    if len(pnl_series) == 0:
        return 0.0
    return float(np.sum(pnl_series > 0) / len(pnl_series))


def avg_win_loss_ratio(pnl_series: np.ndarray) -> float:
    """Average win / average loss magnitude."""
    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series < 0]
    if len(wins) == 0 or len(losses) == 0:
        return 0.0
    return float(wins.mean() / abs(losses.mean()))


def trades_per_day(n_trades: int, n_bars: int, bars_per_day: int = 96) -> float:
    """Trades per calendar day."""
    days = n_bars / bars_per_day
    if days == 0:
        return 0.0
    return n_trades / days


def compute_all_metrics(pnl_series: np.ndarray,
                        n_bars: int,
                        initial_capital: float = 500.0,
                        bars_per_day: int = 96,
                        annualisation: Optional[float] = None) -> dict:
    """
    Compute all metrics from a per-trade P&L array.

    Sharpe is computed using TRADE-FREQUENCY annualisation (per-trade returns
    annualised by trade_count/period * 252).  This is correct for sparse strategies
    and avoids the artificial inflation that occurs when trades are distributed onto
    dense bar grids where most bars are zero (suppresses variance, inflates Sharpe).

    Args:
        pnl_series: Array of trade P&L in USD
        n_bars: Number of price bars in the backtest period
        initial_capital: Starting capital in USD
        bars_per_day: Number of bars per trading day (96 for 15m)
        annualisation: Override for annualisation factor (pass bars_per_day*252 to
                       keep bar-based Sharpe for strategies that trade every bar)

    Returns:
        dict with all metrics
    """
    n_trades = len(pnl_series)
    if n_trades == 0:
        return _empty_metrics()

    # Build equity curve from trades
    equity = initial_capital + np.cumsum(pnl_series)
    equity = np.insert(equity, 0, initial_capital)

    total_return_pct = (equity[-1] - initial_capital) / initial_capital * 100

    # Trade-based annualisation: avoids Sharpe inflation for sparse strategies.
    # For a strategy making T trades over D days, annualise by (T/D)*252.
    # Falls back to caller-supplied annualisation if provided.
    if annualisation is not None:
        ann_factor = annualisation
    else:
        n_days = max(n_bars / bars_per_day, 1.0)
        ann_factor = (n_trades / n_days) * 252.0

    # Per-trade returns (as fraction of initial capital)
    per_trade_returns = pnl_series / max(initial_capital, 1.0)

    return {
        "n_trades":         n_trades,
        "total_return_pct": round(total_return_pct, 3),
        "sharpe":           round(sharpe_ratio(per_trade_returns, ann_factor), 3),
        "sortino":          round(sortino_ratio(per_trade_returns, ann_factor), 3),
        "calmar":           round(calmar_ratio(per_trade_returns, ann_factor), 3),
        "max_dd_pct":       round(max_drawdown(equity) * 100, 3),
        "profit_factor":    round(profit_factor(pnl_series), 3),
        "win_rate":         round(win_rate(pnl_series) * 100, 2),
        "avg_win_loss":     round(avg_win_loss_ratio(pnl_series), 3),
        "trades_per_day":   round(trades_per_day(n_trades, n_bars, bars_per_day), 2),
        "final_equity":     round(equity[-1], 2),
        "avg_trade_pnl":    round(float(np.mean(pnl_series)), 4),
    }


def _empty_metrics() -> dict:
    return {
        "n_trades": 0, "total_return_pct": 0.0, "sharpe": 0.0, "sortino": 0.0,
        "calmar": 0.0, "max_dd_pct": 0.0, "profit_factor": 0.0, "win_rate": 0.0,
        "avg_win_loss": 0.0, "trades_per_day": 0.0, "final_equity": 0.0,
        "avg_trade_pnl": 0.0,
    }


def monte_carlo_sharpe(pnl_series: np.ndarray,
                       n_sims: int = 1000,
                       n_bars: int = 10_000,
                       initial_capital: float = 500.0,
                       bars_per_day: int = 96,
                       seed: int = 42) -> dict:
    """
    Bootstrap Monte Carlo to get Sharpe distribution.
    Resamples trades with replacement n_sims times.

    Returns:
        {p5, p25, p50, p75, p95, mean, std}
    """
    if len(pnl_series) < 10:
        return {"p5": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0,
                "mean": 0.0, "std": 0.0}

    rng = np.random.default_rng(seed)
    sharpes = []

    n = len(pnl_series)

    for _ in range(n_sims):
        sample = rng.choice(pnl_series, size=n, replace=True)
        # No annualisation override — use trade-frequency-based Sharpe (consistent
        # with compute_all_metrics default) so MC distribution matches point estimate.
        m = compute_all_metrics(sample, n_bars, initial_capital, bars_per_day)
        sharpes.append(m["sharpe"])

    sharpes = np.array(sharpes)
    return {
        "p5":   round(float(np.percentile(sharpes, 5)), 3),
        "p25":  round(float(np.percentile(sharpes, 25)), 3),
        "p50":  round(float(np.percentile(sharpes, 50)), 3),
        "p75":  round(float(np.percentile(sharpes, 75)), 3),
        "p95":  round(float(np.percentile(sharpes, 95)), 3),
        "mean": round(float(np.mean(sharpes)), 3),
        "std":  round(float(np.std(sharpes)), 3),
    }


def passes_oos_criteria(metrics: dict,
                        min_sharpe: float = 1.5,
                        min_fold_sharpe: float = 0.5,
                        max_dd_pct: float = 6.0,
                        min_profit_factor: float = 1.3,
                        min_trades: int = 30,
                        mc_p5_sharpe: float = 0.0) -> tuple[bool, list[str]]:
    """
    Check if a strategy passes all OOS validation criteria.

    Returns:
        (passes: bool, failures: list of failed criteria strings)
    """
    failures = []

    n_trades = metrics.get("n_trades", 0)
    sharpe   = metrics.get("sharpe", 0)
    dd       = metrics.get("max_dd_pct", 100)

    # ── Sanity guards (physically impossible results) ───────────────────────
    # Renaissance Medallion ≈ 2.5 annualised. 20.0 gives enormous headroom for
    # short-window crypto strategies but blocks look-ahead / fee-omission artifacts.
    if sharpe > 20.0:
        failures.append(
            f"sharpe {sharpe:.2f} > 20.0 — physically impossible in real markets "
            f"(likely look-ahead bias or annualisation bug)"
        )

    # Zero drawdown with more than a handful of trades is not credible:
    # even a delta-neutral arb has inter-trade MTM fluctuations.
    if dd < 0.1 and n_trades >= 10:
        failures.append(
            f"max_dd {dd:.3f}% with {n_trades} trades — zero-DD is not credible "
            f"(possible look-ahead, too-short test window, or synthetic pair)"
        )
    # ── Standard OOS criteria ───────────────────────────────────────────────
    if sharpe < min_sharpe:
        failures.append(f"sharpe {sharpe:.2f} < {min_sharpe}")

    if dd > max_dd_pct:
        failures.append(f"max_dd {dd:.1f}% > {max_dd_pct}%")

    if metrics.get("profit_factor", 0) < min_profit_factor:
        failures.append(f"PF {metrics.get('profit_factor', 0):.2f} < {min_profit_factor}")

    if n_trades < min_trades:
        failures.append(f"trades {n_trades} < {min_trades}")

    mc_val = metrics.get("mc_p5_sharpe")
    if mc_val is not None and mc_val < mc_p5_sharpe:
        failures.append(f"MC p5 sharpe {mc_val:.3f} < {mc_p5_sharpe}")

    return len(failures) == 0, failures


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(0)
    # Simulate 200 trades with slight positive edge
    pnl = rng.normal(loc=0.5, scale=3.0, size=200)

    m = compute_all_metrics(pnl, n_bars=9600, initial_capital=500.0)
    print("Metrics:")
    for k, v in m.items():
        print(f"  {k}: {v}")

    mc = monte_carlo_sharpe(pnl, n_sims=500, n_bars=9600)
    print("\nMonte Carlo Sharpe distribution:")
    for k, v in mc.items():
        print(f"  {k}: {v}")

    ok, failures = passes_oos_criteria(m)
    print(f"\nPasses OOS: {ok}")
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
