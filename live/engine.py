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
        # Per-strategy runtime state
        self.strategy_runtime: Dict[str, Dict] = {}
        for name, strat_data in self.strategies.items():
            # Use get_warmup_sec() which prefers live_warmup_sec over warmup
            warmup = strat_data['instance'].get_warmup_sec()
            self.strategy_runtime[name] = {
                'active': False,
                'warmup_required_sec': warmup,
                'started_at': None,
                'buffered_sec': 0,
            }

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
        """Start all strategies in live trading loop."""
        if not self.connected:
            logger.error("Cannot start: not connected")
            return False

        for name in self.strategies.keys():
            self.start_strategy(name)
        return True

    def stop(self):
        """Stop all strategies."""
        for name in self.strategies.keys():
            self.stop_strategy(name)
        self.running = False
        logger.info("Live trading STOPPED")
    def start_strategy(self, strategy_name: str) -> bool:
        """Start one strategy independently."""
        if not self.connected:
            logger.error("Cannot start strategy: not connected")
            return False
        if strategy_name not in self.strategies:
            logger.error(f"Unknown strategy: {strategy_name}")
            return False

        self.strategies[strategy_name]['instance'].set_real_time_mode(True)
        rt = self.strategy_runtime[strategy_name]
        rt['active'] = True
        rt['started_at'] = time.time()
        rt['buffered_sec'] = 0
        self.running = True
        logger.info(f"Strategy STARTED: {strategy_name}")
        return True

    def stop_strategy(self, strategy_name: str) -> bool:
        """Stop one strategy independently."""
        if strategy_name not in self.strategies:
            logger.error(f"Unknown strategy: {strategy_name}")
            return False
        self.strategies[strategy_name]['instance'].set_real_time_mode(False)
        rt = self.strategy_runtime[strategy_name]
        rt['active'] = False
        rt['started_at'] = None
        rt['buffered_sec'] = 0

        # global running flag: true if any strategy is active
        self.running = any(r['active'] for r in self.strategy_runtime.values())
        logger.info(f"Strategy STOPPED: {strategy_name}")
        return True

    def update_runtime(self):
        """Update warmup/buffered seconds for active strategies."""
        now = time.time()
        for name, rt in self.strategy_runtime.items():
            if rt['active'] and rt['started_at'] is not None:
                elapsed = int(now - rt['started_at'])
                rt['buffered_sec'] = min(elapsed, int(rt['warmup_required_sec']))

    def get_available_coins(self) -> list:
        if self._exchange_client and self.connected:
            try:
                return self._exchange_client.get_available_coins()
            except Exception:
                return []
        return []

    def emergency_stop(self):
        """Close all positions immediately."""
        self.stop()
        self.current_positions = {}
        logger.warning("EMERGENCY STOP — all positions closed")

    def get_current_pnl(self) -> list:
        """Get PnL history for charting."""
        return self.pnl_history

    def get_status(self) -> Dict:
        self.update_runtime()
        return {
            'running': self.running,
            'connected': self.connected,
            'n_strategies': len(self.strategies),
            'strategies': {n: s['instance'].get_status() for n, s in self.strategies.items()},
            'strategy_runtime': self.strategy_runtime,
            'positions': self.current_positions,
            'total_pnl': sum(self.pnl_history) if self.pnl_history else 0,
            'available_coins': self.get_available_coins(),
        }
