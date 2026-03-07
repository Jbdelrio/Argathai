import unittest
from pathlib import Path

import yaml


class TestConfigFiles(unittest.TestCase):
    def test_strategies_config_exists_and_has_active_strategies(self):
        p = Path('config/strategies.yaml')
        self.assertTrue(p.exists())
        data = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
        self.assertIn('active_strategies', data)
        self.assertGreaterEqual(len(data['active_strategies']), 1)

    def test_exchanges_config_exists_and_has_two_exchanges(self):
        p = Path('config/exchanges.yaml')
        self.assertTrue(p.exists())
        data = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
        self.assertIn('hyperliquid', data)
        self.assertIn('bitget_futures', data)


if __name__ == '__main__':
    unittest.main()
