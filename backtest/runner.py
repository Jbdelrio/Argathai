"""
BacktestEngine — Walk-forward backtest with dynamic strategy loading.
======================================================================
Loads strategies from config/strategies.yaml, runs walk-forward on each.
"""

import importlib
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger('agarthai.backtest')


class BacktestEngine:
    """
    Orchestrates backtesting across multiple strategies.
    Strategies are loaded dynamically from config/strategies.yaml.
    """

    def __init__(self, config_path: str = 'config/strategies.yaml'):
        self.config = self._load_config(config_path)
        self.strategies = {}
        self.results = {}
        self._load_strategies()

    def _load_config(self, path: str) -> dict:
        p = Path(path)
        if not p.exists():
            logger.warning(f"Config not found: {path}, using defaults")
            return {'active_strategies': {}, 'capital_usd': 1500}
        with open(p, 'r') as f:
            return yaml.safe_load(f)

    def _load_strategies(self):
        """Dynamically import and instantiate each active strategy."""
        total_capital = self.config.get('capital_usd', 1500)

        for name, weight in self.config.get('active_strategies', {}).items():
            try:
                module = importlib.import_module(f'strategies.{name}.strategy')
                # Convention: class name = strategy name capitalized
                class_name = name.replace('_', ' ').title().replace(' ', '')
                # Try exact name first, then capitalized
                strategy_class = getattr(module, class_name, None)
                if strategy_class is None:
                    # Fallback: get first BaseStrategy subclass in module
                    from strategies.common.base_strategy import BaseStrategy
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and issubclass(attr, BaseStrategy)
                                and attr is not BaseStrategy):
                            strategy_class = attr
                            break

                if strategy_class is None:
                    logger.error(f"No strategy class found in strategies.{name}.strategy")
                    continue

                params_path = f'strategies/{name}/params.yaml'
                instance = strategy_class(name, params_path)
                instance.set_capital(total_capital * weight)

                self.strategies[name] = {
                    'instance': instance,
                    'weight': weight,
                    'capital': total_capital * weight,
                }
                logger.info(f"Loaded strategy: {name} (weight={weight:.0%}, "
                           f"capital=${total_capital * weight:.0f})")

            except Exception as e:
                logger.error(f"Failed to load strategy '{name}': {e}")

    def run(self, data: pd.DataFrame, data_pair: pd.DataFrame = None) -> Dict:
        """
        Run backtest on all loaded strategies.

        data: primary asset DataFrame (BTC 1s)
        data_pair: secondary asset for pair strategies (ETH 1s)
        """
        self.results = {}

        for name, strat_data in self.strategies.items():
            strategy = strat_data['instance']
            strategy.set_real_time_mode(False)

            logger.info(f"Running backtest: {name}")

            try:
                # Compute features
                df_feat = strategy.compute_features(data)

                # Generate signals across the dataset
                signals = []
                step = strategy.params.get('decision_step', 600)
                start = strategy.params.get('warmup', 3600)

                for t in range(start, len(df_feat), step):
                    sig = strategy.generate_signal(df_feat.iloc[:t+1])
                    if sig is not None:
                        sig.metadata['idx'] = t
                        signals.append(sig)

                self.results[name] = {
                    'signals': signals,
                    'n_signals': len(signals),
                    'strategy': strategy,
                    'status': strategy.get_status(),
                }

                logger.info(f"  {name}: {len(signals)} signals generated")

            except Exception as e:
                logger.error(f"  {name} failed: {e}")
                self.results[name] = {'signals': [], 'n_signals': 0, 'error': str(e)}

        return self.results

    def get_all_metrics(self) -> Dict:
        """Collect metrics from all strategies."""
        metrics = {}
        for name, result in self.results.items():
            strategy = result.get('strategy')
            if strategy:
                metrics[name] = strategy.get_status()
        return metrics

    def get_strategy_names(self) -> List[str]:
        return list(self.strategies.keys())
