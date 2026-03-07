"""
Risk & Performance Metrics
===========================
VaR, CVaR, Sharpe, Sortino, Information Ratio, MaxDD,
Calmar, Win Rate, Profit Factor, Unrealized PnL tracking.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional


def sharpe_ratio(returns: np.ndarray, rf: float = 0.0, periods: int = 252) -> float:
    excess = returns - rf / periods
    if excess.std() < 1e-12:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(periods))


def sortino_ratio(returns: np.ndarray, rf: float = 0.0, periods: int = 252) -> float:
    excess = returns - rf / periods
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    dd_std = downside.std()
    if dd_std < 1e-12:
        return 0.0
    return float(excess.mean() / dd_std * np.sqrt(periods))


def information_ratio(returns: np.ndarray, benchmark: np.ndarray, periods: int = 252) -> float:
    active = returns - benchmark
    if active.std() < 1e-12:
        return 0.0
    return float(active.mean() / active.std() * np.sqrt(periods))


def max_drawdown(equity: np.ndarray) -> dict:
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    max_dd = dd.min()
    max_dd_idx = dd.argmin()
    peak_idx = np.argmax(equity[:max_dd_idx + 1]) if max_dd_idx > 0 else 0
    recovery_idx = None
    if max_dd_idx < len(equity) - 1:
        after = equity[max_dd_idx:]
        rec = np.where(after >= equity[peak_idx])[0]
        if len(rec) > 0:
            recovery_idx = max_dd_idx + rec[0]
    return {
        'max_dd': float(max_dd),
        'max_dd_pct': float(max_dd / (equity[peak_idx] + 1e-12)),
        'peak_idx': int(peak_idx),
        'trough_idx': int(max_dd_idx),
        'recovery_idx': recovery_idx,
        'duration_to_trough': int(max_dd_idx - peak_idx),
    }


def calmar_ratio(returns: np.ndarray, equity: np.ndarray, periods: int = 252) -> float:
    dd = max_drawdown(equity)
    if abs(dd['max_dd_pct']) < 1e-12:
        return 0.0
    ann_return = returns.mean() * periods
    return float(ann_return / abs(dd['max_dd_pct']))


def value_at_risk(returns: np.ndarray, confidence: float = 0.95) -> float:
    return float(np.percentile(returns, (1 - confidence) * 100))


def conditional_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    var = value_at_risk(returns, confidence)
    return float(returns[returns <= var].mean()) if (returns <= var).any() else var


def profit_factor(pnls: np.ndarray) -> float:
    gains = pnls[pnls > 0].sum()
    losses = abs(pnls[pnls < 0].sum())
    if losses < 1e-12:
        return float('inf') if gains > 0 else 0.0
    return float(gains / losses)


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    if avg_loss < 1e-12:
        return 0.0
    b = avg_win / avg_loss
    return float(win_rate - (1 - win_rate) / b)


def compute_all_metrics(trades_df: pd.DataFrame, capital_usd: float = 1500.0,
                        trades_per_day: float = 1.0) -> dict:
    """
    Compute full risk report from trades DataFrame.
    Expected columns: pnl_net_pct, pnl_usd, duration_sec, exit_reason, label
    """
    if trades_df.empty:
        return {'n_trades': 0}

    pnls_pct = trades_df['pnl_net_pct'].values
    pnls_usd = trades_df['pnl_usd'].values if 'pnl_usd' in trades_df else pnls_pct * capital_usd

    # Equity curve
    eq_usd = capital_usd + np.cumsum(pnls_usd)
    eq_full = np.concatenate([[capital_usd], eq_usd])

    # Per-trade returns (as fraction of capital)
    rets = pnls_usd / capital_usd

    # Approximate annualization: trades_per_day * 365
    periods = int(trades_per_day * 365)

    wins = pnls_usd > 0
    losses = pnls_usd < 0
    avg_win = pnls_usd[wins].mean() if wins.any() else 0
    avg_loss = abs(pnls_usd[losses].mean()) if losses.any() else 0

    dd = max_drawdown(eq_full)

    return {
        # ── Counts ──
        'n_trades': len(trades_df),
        'win_rate': float(wins.mean()),
        'tp_rate': float(trades_df['label'].mean()) if 'label' in trades_df else None,

        # ── PnL ──
        'total_pnl_usd': float(pnls_usd.sum()),
        'total_pnl_pct': float(pnls_pct.sum() * 100),
        'avg_pnl_usd': float(pnls_usd.mean()),
        'avg_pnl_pct': float(pnls_pct.mean() * 100),
        'median_pnl_usd': float(np.median(pnls_usd)),
        'std_pnl_usd': float(pnls_usd.std()),

        # ── Risk-adjusted ──
        'sharpe': sharpe_ratio(rets, periods=periods),
        'sortino': sortino_ratio(rets, periods=periods),
        'calmar': calmar_ratio(rets, eq_full, periods=periods),
        'profit_factor': profit_factor(pnls_usd),
        'kelly_f': kelly_fraction(wins.mean(), avg_win, avg_loss),

        # ── Drawdown ──
        'max_dd_usd': dd['max_dd'],
        'max_dd_pct': dd['max_dd_pct'] * 100,
        'dd_duration': dd['duration_to_trough'],

        # ── VaR ──
        'var_95_usd': value_at_risk(pnls_usd, 0.95),
        'var_99_usd': value_at_risk(pnls_usd, 0.99),
        'cvar_95_usd': conditional_var(pnls_usd, 0.95),

        # ── Execution ──
        'avg_duration_min': float(trades_df['duration_sec'].mean() / 60) if 'duration_sec' in trades_df else None,
        'avg_win_usd': float(avg_win),
        'avg_loss_usd': float(avg_loss),
        'win_loss_ratio': float(avg_win / (avg_loss + 1e-12)),
        'expectancy_usd': float(pnls_usd.mean()),

        # ── Exit breakdown ──
        'exit_TP': int((trades_df['exit_reason'] == 'TP').sum()) if 'exit_reason' in trades_df else None,
        'exit_SL': int((trades_df['exit_reason'] == 'SL').sum()) if 'exit_reason' in trades_df else None,
        'exit_TIME': int((trades_df['exit_reason'] == 'TIME').sum()) if 'exit_reason' in trades_df else None,

        # ── Capital ──
        'initial_capital': capital_usd,
        'final_capital': float(eq_full[-1]),
        'return_pct': float((eq_full[-1] / capital_usd - 1) * 100),

        # ── Equity curve ──
        'equity_curve': eq_full.tolist(),
    }
