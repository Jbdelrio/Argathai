"""
Position Sizing — Risk-Based + Fractional Kelly
=================================================
Capital: 1500$ | Leverage: configurable
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class SizingConfig:
    capital_usd: float = 1500.0
    leverage: float = 3.0
    risk_per_trade: float = 0.02      # 2% of capital
    max_position_pct: float = 0.30    # 30% of leveraged capital
    kelly_fraction: float = 0.25      # use 25% of full Kelly

    @property
    def buying_power(self) -> float:
        return self.capital_usd * self.leverage


def size_from_risk(capital: float, sl_pct: float, risk_pct: float = 0.02,
                   leverage: float = 3.0, max_pct: float = 0.30) -> float:
    """
    Risk-based position sizing.
    Risk = capital * risk_pct = max loss per trade
    Position = Risk / SL%
    Capped at max_pct of leveraged capital.
    """
    risk_usd = capital * risk_pct
    pos_from_risk = risk_usd / (sl_pct + 1e-12)
    max_pos = capital * leverage * max_pct
    return min(pos_from_risk, max_pos)


def size_from_kelly(capital: float, win_rate: float, avg_win: float,
                    avg_loss: float, fraction: float = 0.25,
                    leverage: float = 3.0) -> float:
    """
    Kelly criterion position sizing (fractional).
    f* = p - q/b where p=win_rate, q=1-p, b=avg_win/avg_loss
    Use fraction of Kelly (typically 25%) for safety.
    """
    if avg_loss < 1e-12:
        return capital * leverage * 0.10  # minimal

    b = avg_win / avg_loss
    kelly = win_rate - (1 - win_rate) / b
    kelly = max(0, min(kelly, 1))  # clamp

    f = kelly * fraction
    return capital * leverage * f


def compute_position(capital: float, sl_pct: float, price: float,
                     config: SizingConfig = None) -> dict:
    """Full position sizing computation."""
    if config is None:
        config = SizingConfig(capital_usd=capital)

    pos_usd = size_from_risk(capital, sl_pct, config.risk_per_trade,
                              config.leverage, config.max_position_pct)
    qty_coin = pos_usd / price
    risk_usd = capital * config.risk_per_trade
    margin_required = pos_usd / config.leverage

    return {
        'position_usd': pos_usd,
        'qty_coin': qty_coin,
        'risk_usd': risk_usd,
        'margin_required': margin_required,
        'leverage_used': pos_usd / capital,
        'pct_of_capital': pos_usd / (capital * config.leverage),
    }
