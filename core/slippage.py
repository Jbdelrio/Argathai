"""
Almgren-Chriss Slippage Model
==============================
Estimates execution cost (slippage + market impact) for a given trade.

Reference: Almgren & Chriss (2001) "Optimal Execution of Portfolio Transactions"
Adapted from: https://hedgenordic.com/2025/05/how-to-deal-with-slippage/

For a retail trader with ~1500$ positions, permanent impact is negligible.
We focus on TEMPORARY impact + spread cost + volatility slippage.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class AlmgrenChrissParams:
    """Parameters for slippage estimation."""
    sigma_1s: float = 0.0001      # 1-second volatility (log-returns std)
    eta: float = 2.5e-7           # temporary impact coefficient
    gamma: float = 2.5e-8         # permanent impact coefficient (negligible for retail)
    spread_bps: float = 1.0       # typical bid-ask spread in bps
    risk_aversion: float = 1e-6   # λ in Almgren-Chriss


def estimate_slippage(
    qty_usd: float,
    price: float,
    adv_usd: float,                # average daily volume in USD
    sigma_daily: float = 0.02,     # daily volatility (~2% for BTC)
    execution_time_sec: float = 5, # how long to fill the order
    params: AlmgrenChrissParams = None,
) -> dict:
    """
    Estimate total execution cost for a single trade.

    For a 1500$ position on BTC:
    - Market impact is negligible (< 0.001%)
    - Spread cost dominates (~0.5-1 bps)
    - Volatility slippage depends on execution speed

    Returns dict with breakdown in % and USD.
    """
    if params is None:
        params = AlmgrenChrissParams()

    qty_coins = qty_usd / price
    T = execution_time_sec / 86400  # in days
    n_steps = max(1, int(execution_time_sec))

    # Participation rate (what fraction of volume we are)
    daily_vol_coins = adv_usd / price
    participation = qty_coins / (daily_vol_coins * T + 1e-12)

    # 1) Spread cost (half spread, always paid)
    spread_cost_pct = params.spread_bps / 10000 / 2

    # 2) Temporary market impact (Almgren-Chriss)
    # η * (q/τ) where q = qty per step, τ = time per step
    trade_rate = qty_coins / (T * 86400 + 1e-12)  # coins per second
    temp_impact_pct = params.eta * trade_rate / (price + 1e-12)

    # 3) Permanent impact (negligible for retail)
    perm_impact_pct = params.gamma * qty_coins / (price + 1e-12)

    # 4) Volatility slippage (timing risk)
    # σ * √T * risk_aversion factor
    vol_slippage_pct = sigma_daily * np.sqrt(T) * 0.5

    # 5) Total
    total_pct = spread_cost_pct + temp_impact_pct + perm_impact_pct + vol_slippage_pct
    total_usd = total_pct * qty_usd

    return {
        'total_pct': total_pct,
        'total_usd': total_usd,
        'spread_pct': spread_cost_pct,
        'temp_impact_pct': temp_impact_pct,
        'perm_impact_pct': perm_impact_pct,
        'vol_slippage_pct': vol_slippage_pct,
        'participation_rate': participation,
        'is_negligible_impact': participation < 0.001,
    }


def slippage_for_backtest(qty_usd: float, price: float, sigma_1s: float,
                           adv_usd: float = 5e9) -> float:
    """
    Quick slippage estimate for backtesting (returns % to add to fees).
    BTC ADV ~5B$ → retail 1500$ is ~0.00003% of daily volume.
    """
    result = estimate_slippage(qty_usd, price, adv_usd,
                                sigma_daily=sigma_1s * np.sqrt(86400))
    return result['total_pct']


# ── Convenience for different market conditions ──────────────────────────

def slippage_calm(qty_usd: float, price: float) -> float:
    """Low vol regime (σ_daily ~ 1.5%)."""
    return slippage_for_backtest(qty_usd, price, sigma_1s=0.00007)

def slippage_normal(qty_usd: float, price: float) -> float:
    """Normal vol (σ_daily ~ 2.5%)."""
    return slippage_for_backtest(qty_usd, price, sigma_1s=0.0001)

def slippage_volatile(qty_usd: float, price: float) -> float:
    """High vol (σ_daily ~ 5%)."""
    return slippage_for_backtest(qty_usd, price, sigma_1s=0.0002)
