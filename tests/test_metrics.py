import unittest

import pandas as pd

from backtest.metrics import compute_all_metrics


class TestMetrics(unittest.TestCase):
    def test_compute_all_metrics_basic(self):
        trades = pd.DataFrame([
            {'pnl_net_pct': 0.01, 'pnl_usd': 10.0, 'duration_sec': 120, 'exit_reason': 'TP', 'label': 1},
            {'pnl_net_pct': -0.005, 'pnl_usd': -5.0, 'duration_sec': 90, 'exit_reason': 'SL', 'label': 0},
        ])
        m = compute_all_metrics(trades, capital_usd=1500.0)
        self.assertEqual(m['n_trades'], 2)
        self.assertAlmostEqual(m['total_pnl_usd'], 5.0)
        self.assertIn('equity_curve', m)


if __name__ == '__main__':
    unittest.main()
