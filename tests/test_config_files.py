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


    def test_settings_has_history_fetch_config(self):
        p = Path('config/settings.yaml')
        self.assertTrue(p.exists())
        data = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
        self.assertIn('history_fetch', data)
        self.assertIn('enabled', data['history_fetch'])

    def test_settings_history_fetch_if_present_is_well_formed(self):
        """Backward-compatible: history_fetch is optional in some user configs."""
        p = Path('config/settings.yaml')
        self.assertTrue(p.exists())
        data = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
        hf = data.get('history_fetch')
        if hf is not None:
            self.assertIn('enabled', hf)


if __name__ == '__main__':
    unittest.main()