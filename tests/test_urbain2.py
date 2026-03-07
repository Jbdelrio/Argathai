import unittest

import numpy as np
import pandas as pd

from strategies.urbain2.strategy import Urbain2


class TestUrbain2(unittest.TestCase):
    def test_compute_features_adds_expected_columns(self):
        n = 9000
        rng = np.random.default_rng(42)
        price = 30000 * np.exp(np.cumsum(rng.normal(0, 1e-4, n)))
        ret = np.concatenate([[0.0], np.diff(np.log(price))])

        df = pd.DataFrame({
            'timestamp': pd.date_range('2026-01-01', periods=n, freq='1s'),
            'last': price,
            'ret_1s': ret,
            'qty': rng.exponential(0.3, n),
            'open_interest': 1e6 + np.cumsum(rng.normal(0, 500, n)),
            'funding_rate': rng.normal(0, 1e-4, n),
            'spread_bps': np.clip(rng.normal(6.0, 1.0, n), 2, 20),
            'btc_ret_1h': pd.Series(ret).rolling(3600, min_periods=300).sum().fillna(0),
            'eth_ret_1h': pd.Series(ret).rolling(2400, min_periods=300).sum().fillna(0),
            'alt_pca1': pd.Series(ret).rolling(1800, min_periods=300).mean().fillna(0),
        })

        s = Urbain2()
        out = s.compute_features(df)

        for col in ['u_resid', 'm_resid', 'o_oi', 'phi_funding', 'l_liq', 'regime_gate', 's_star', 'cost_proxy']:
            self.assertIn(col, out.columns)

    def test_generate_signal_runs_without_error(self):
        n = 9000
        rng = np.random.default_rng(7)
        price = 2000 * np.exp(np.cumsum(rng.normal(0, 2e-4, n)))
        ret = np.concatenate([[0.0], np.diff(np.log(price))])

        df = pd.DataFrame({
            'timestamp': pd.date_range('2026-01-01', periods=n, freq='1s'),
            'last': price,
            'ret_1s': ret,
            'qty': rng.exponential(0.2, n),
            'open_interest': 5e5 + np.cumsum(rng.normal(0, 300, n)),
            'funding_rate': rng.normal(0, 8e-5, n),
            'spread_bps': np.clip(rng.normal(5.5, 1.2, n), 1, 20),
        })

        s = Urbain2()
        feat = s.compute_features(df)
        sig = s.generate_signal(feat)
        self.assertTrue(sig is None or hasattr(sig, 'direction'))


if __name__ == '__main__':
    unittest.main()