import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.exchange_fetcher import ExchangeHistoricalFetcher


class TestExchangeFetcher(unittest.TestCase):
    def test_trades_to_1s_aggregation(self):
        trades = pd.DataFrame([
            {'timestamp': pd.Timestamp('2026-02-01T00:00:00.100Z'), 'price': 100.0, 'amount': 1.0, 'side': 'buy', 'id': '1'},
            {'timestamp': pd.Timestamp('2026-02-01T00:00:00.500Z'), 'price': 101.0, 'amount': 2.0, 'side': 'sell', 'id': '2'},
            {'timestamp': pd.Timestamp('2026-02-01T00:00:01.100Z'), 'price': 102.0, 'amount': 1.5, 'side': 'buy', 'id': '3'},
        ])

        out = ExchangeHistoricalFetcher.trades_to_1s(trades, symbol='BTC/USDT:USDT')
        self.assertEqual(len(out), 2)
        self.assertIn('vwap', out.columns)
        self.assertIn('ofi_proxy', out.columns)
        self.assertAlmostEqual(float(out.iloc[0]['qty']), 3.0)
        self.assertAlmostEqual(float(out.iloc[0]['buy_qty']), 1.0)
        self.assertAlmostEqual(float(out.iloc[0]['sell_qty']), 2.0)


if __name__ == '__main__':
    unittest.main()
