"""
BaseStrategy — Abstract base for all Agarthai strategies.
==========================================================
Every strategy (baudouin4, innocent3, future ones) inherits from this.
Ensures consistent interface for both backtest and live engines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
import yaml
import logging
import numpy as np
import pandas as pd
from datetime import datetime


@dataclass
class Signal:
    """Trading signal produced by a strategy."""
    timestamp: datetime
    direction: int          # -1=short, 0=flat, 1=long
    confidence: float       # 0-1
    entry_price: float
    tp_price: float
    sl_price: float
    metadata: Dict = field(default_factory=dict)  # strategy-specific data


@dataclass
class TradeResult:
    """Result of an executed trade."""
    signal: Signal
    entry_price: float
    exit_price: float
    pnl_usd: float
    pnl_pct: float
    fees_usd: float
    slippage_usd: float
    exit_reason: str        # 'TP', 'SL', 'TIME', 'SIGNAL', 'MANUAL'
    duration_sec: int
    position_usd: float


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    To create a new strategy:
    1. Create strategies/my_strategy/strategy.py
    2. Create a class inheriting BaseStrategy
    3. Implement generate_signal() and compute_features()
    4. Add to config/strategies.yaml
    """

    def __init__(self, strategy_name: str, params_path: str = None):
        self.name = strategy_name
        self.params = self._load_params(params_path) if params_path else {}
        self.exchange_client = None
        self.real_time_mode = False
        self.logger = self._setup_logger()
        self.trade_history: List[TradeResult] = []
        self.capital_allocated: float = 0.0

    def _load_params(self, path: str) -> dict:
        p = Path(path)
        if p.exists():
            with open(p, 'r') as f:
                return yaml.safe_load(f) or {}
        self.logger.warning(f"Params file not found: {path}")
        return {}

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"agarthai.{self.name}")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            # Console handler
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter(
                f'[%(asctime)s] [{self.name}] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            logger.addHandler(ch)
            # File handler
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            fh = logging.FileHandler(
                log_dir / f"{datetime.now().strftime('%Y-%m-%d')}_{self.name}.log"
            )
            fh.setFormatter(logging.Formatter(
                f'[%(asctime)s] [{self.name}] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            logger.addHandler(fh)
        return logger

    def set_exchange_client(self, client):
        self.exchange_client = client
        self.logger.info(f"Exchange client set: {client.__class__.__name__}")

    def set_real_time_mode(self, mode: bool):
        self.real_time_mode = mode
        self.logger.info(f"Mode: {'LIVE' if mode else 'BACKTEST'}")

    def set_capital(self, capital_usd: float):
        self.capital_allocated = capital_usd
        self.logger.info(f"Capital allocated: ${capital_usd:.0f}")

    def get_warmup_sec(self) -> int:
        """Return warmup seconds required before strategy can trade.
        Override in subclasses if warmup differs from params['warmup'].
        Live engine reads live_warmup_sec first (faster paper-trading start).
        """
        return int(self.params.get('live_warmup_sec', self.params.get('warmup', 3600)))

    # ─── Abstract methods (MUST implement) ─────────────────────────────

    @abstractmethod
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute strategy-specific features from raw 1s data."""
        pass

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        """
        Generate a trading signal from featured data.
        Returns Signal or None if no signal.
        """
        pass

    # ─── Optional overrides ────────────────────────────────────────────

    def on_trade_open(self, signal: Signal):
        """Called when a trade is opened. Override for custom logic."""
        self.logger.info(
            f"Trade opened: {'LONG' if signal.direction == 1 else 'SHORT'} "
            f"@ {signal.entry_price:.2f} | conf={signal.confidence:.2f}"
        )

    def on_trade_close(self, result: TradeResult):
        """Called when a trade is closed. Override for custom logic."""
        self.trade_history.append(result)
        self.logger.info(
            f"Trade closed: {result.exit_reason} | "
            f"PnL=${result.pnl_usd:.2f} ({result.pnl_pct*100:.3f}%)"
        )

    def get_status(self) -> Dict:
        """Return strategy status for GUI display."""
        n = len(self.trade_history)
        pnls = [t.pnl_usd for t in self.trade_history]
        return {
            'name': self.name,
            'mode': 'LIVE' if self.real_time_mode else 'BACKTEST',
            'capital': self.capital_allocated,
            'n_trades': n,
            'total_pnl': sum(pnls) if pnls else 0,
            'win_rate': np.mean([p > 0 for p in pnls]) if pnls else 0,
            'last_signal': self.trade_history[-1].signal.direction if self.trade_history else 0,
        }
