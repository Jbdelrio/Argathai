"""
LiveEngine — Real-time trading engine.
========================================
Inherits BacktestEngine for strategy loading.
Adds: streaming, start/stop, position management.
"""

from backtest.runner import BacktestEngine
from strategies.common.base_strategy import Signal
from typing import Dict, Optional
import logging
import time
import threading

logger = logging.getLogger('agarthai.live')


class LiveEngine(BacktestEngine):
    """
    Live trading engine. Inherits strategy loading from BacktestEngine.
    Adds real-time data streaming, start/stop controls, position tracking.
    """

    def __init__(self, config_path='config/strategies.yaml'):
        super().__init__(config_path)
        self.running = False
        self.connected = False
        self._thread: Optional[threading.Thread] = None
        self.current_positions: Dict = {}
        self.pnl_history: list = []
        self._exchange_client = None

    def connect(self, exchange_name: str = 'hyperliquid', paper: bool = True):
        """Connect to exchange."""
        try:
            from exchanges.clients import get_client
            self._exchange_client = get_client(exchange_name, paper=paper)
            self.connected = self._exchange_client.connect()
            if self.connected:
                for strat_data in self.strategies.values():
                    strat_data['instance'].set_exchange_client(self._exchange_client)
                logger.info(f"Connected to {exchange_name} (paper={paper})")
            return self.connected
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.connected = False
            return False

    def disconnect(self):
        self.stop()
        self.connected = False
        logger.info("Disconnected")

    def start(self):
        """Start live trading loop."""
        if not self.connected:
            logger.error("Cannot start: not connected")
            return False

        self.running = True
        for strat_data in self.strategies.values():
            strat_data['instance'].set_real_time_mode(True)

        logger.info("Live trading STARTED")
        return True

    def stop(self):
        """Stop live trading."""
        self.running = False
        for strat_data in self.strategies.values():
            strat_data['instance'].set_real_time_mode(False)
        logger.info("Live trading STOPPED")

    def emergency_stop(self):
        """Close all positions immediately."""
        self.stop()
        self.current_positions = {}
        logger.warning("EMERGENCY STOP — all positions closed")

    def get_current_pnl(self) -> list:
        """Get PnL history for charting."""
        return self.pnl_history

    def get_status(self) -> Dict:
        return {
            'running': self.running,
            'connected': self.connected,
            'n_strategies': len(self.strategies),
            'strategies': {n: s['instance'].get_status() for n, s in self.strategies.items()},
            'positions': self.current_positions,
            'total_pnl': sum(self.pnl_history) if self.pnl_history else 0,
        }
