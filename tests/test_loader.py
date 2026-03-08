import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import _candidate_paths, prepare


class TestLoader(unittest.TestCase):
    def test_candidate_paths_include_local_default(self):
        paths = _candidate_paths('btc')
        self.assertTrue(any(str(p).endswith('data/binance_spot/btc_1s.csv.gz') for p in paths))

    def test_prepare_generates_required_columns(self):
        df = pd.DataFrame({
            'timestamp': ['2026-01-01 00:00:00', '2026-01-01 00:00:01'],
            'last': [100.0, 101.0],
        })
        out = prepare(df)
        self.assertIn('ret_1s', out.columns)
        self.assertIn('log_price', out.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(out['timestamp']))


if __name__ == '__main__':
    unittest.main()
