"""
Innocent3 — Dynamic Cointegration Pairs Trading
=================================================
Trades spread between BTC-ETH using OU mean-reversion,
filtered by OFI divergence (microstructure-informed entry).
"""

import numpy as np
import pandas as pd
from typing import Optional
from strategies.common.base_strategy import BaseStrategy, Signal
from datetime import datetime


class Innocent3(BaseStrategy):

    def __init__(self, name='innocent3', params_path='strategies/innocent3/params.yaml'):
        super().__init__(name, params_path)
        self._pair_data = None

    def set_pair_data(self, df_y: pd.DataFrame):
        """Set the second leg data (ETH) for pair trading."""
        self._pair_data = df_y

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute cointegration spread features.
        df = BTC (primary), self._pair_data = ETH (secondary)
        """
        df = df.copy()
        if self._pair_data is None:
            self.logger.warning("No pair data set — call set_pair_data(eth_df)")
            df['spread_z'] = 0; df['ofi_div'] = 0; df['half_life'] = 999
            return df

        N = min(len(df), len(self._pair_data))
        log_x = np.log(df['last'].values[:N])        # BTC
        log_y = np.log(self._pair_data['last'].values[:N])  # ETH

        ofi_x = df['ofi_proxy'].values[:N] if 'ofi_proxy' in df else np.zeros(N)
        ofi_y = self._pair_data['ofi_proxy'].values[:N] if 'ofi_proxy' in self._pair_data else np.zeros(N)

        # In live mode use a shorter cointegration window so beta is computable
        # with a buffer of ~25h instead of 7 days.
        # live_coint_window must satisfy: N > live_coint_window (beta loop needs iterations).
        if self.real_time_mode:
            win = int(self.params.get('live_coint_window',
                                      self.params.get('coint_window', 604800)))
        else:
            win = int(self.params.get('coint_window', 604800))
        eps = 1e-12

        # Rolling hedge ratio β
        beta = np.full(N, np.nan)
        for t in range(min(win, N), N, 60):  # compute every 60s
            s = max(0, t - win)
            x, y = log_x[s:t], log_y[s:t]
            cov_xy = np.cov(x, y)[0, 1]
            var_x = np.var(x)
            if var_x > eps:
                b = cov_xy / var_x
                beta[t:min(t+60, N)] = b

        # Spread
        spread = log_y - np.nan_to_num(beta) * log_x

        # OU params (rolling AR(1))
        ou_win = 3600
        kappa = np.full(N, np.nan)
        mu = np.full(N, np.nan)
        sigma_ou = np.full(N, np.nan)

        for t in range(ou_win, N, 60):
            s_slice = spread[t-ou_win:t]
            valid = ~np.isnan(s_slice)
            if valid.sum() < 100: continue
            s_v = s_slice[valid]
            y_ar, x_ar = s_v[1:], s_v[:-1]
            if len(y_ar) < 50: continue
            b_ar = np.sum((x_ar - x_ar.mean())*(y_ar - y_ar.mean())) / (np.sum((x_ar - x_ar.mean())**2) + eps)
            if 0 < b_ar < 1:
                k = -np.log(b_ar)
                m = (y_ar.mean() - b_ar * x_ar.mean()) / (1 - b_ar)
                sig = (y_ar - b_ar * x_ar - (1-b_ar)*m).std()
                kappa[t:min(t+60, N)] = k
                mu[t:min(t+60, N)] = m
                sigma_ou[t:min(t+60, N)] = sig

        # Z-score
        z = (spread - np.nan_to_num(mu)) / (np.nan_to_num(sigma_ou) + eps)

        # OFI divergence
        def rsum(x, w):
            N = len(x)
            out = np.full(N, np.nan)
            if N < w:
                return out
            cs = np.cumsum(np.concatenate([[0], x]))
            valid = N - w + 1
            out[w-1:] = cs[w:w + valid] - cs[:valid]
            return out

        ofi_cum_y = rsum(ofi_y, 600)
        ofi_cum_x = rsum(ofi_x, 600)
        spread_ofi = ofi_cum_y - np.nan_to_num(beta) * ofi_cum_x

        s_ofi = pd.Series(spread_ofi)
        med = s_ofi.rolling(3600, min_periods=600).median()
        mad = (s_ofi - med).abs().rolling(3600, min_periods=600).median() * 1.4826
        ofi_div = ((s_ofi - med) / (mad + eps)).values

        # Half-life
        half_life = np.log(2) / (np.nan_to_num(kappa) + eps)
        half_life = np.clip(half_life, 60, 86400)

        df = df.iloc[:N].copy()
        df['spread'] = spread; df['spread_z'] = z
        df['beta'] = beta; df['kappa'] = kappa
        df['mu'] = mu; df['sigma_ou'] = sigma_ou
        df['ofi_div'] = ofi_div; df['half_life'] = half_life

        return df

    def generate_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        """Generate spread trading signal."""
        t = len(df) - 1
        # In live mode use live_warmup_sec (default 1 day) instead of 7-day backtest gate.
        if self.real_time_mode:
            warmup_gate = int(self.params.get('live_warmup_sec',
                                              self.params.get('warmup', 604800)))
        else:
            warmup_gate = int(self.params.get('warmup', 604800))
        if t < warmup_gate:
            return None

        z = df['spread_z'].iloc[t]
        ofi_d = df['ofi_div'].iloc[t]
        hl = df['half_life'].iloc[t]

        if any(np.isnan(x) for x in [z, ofi_d, hl]):
            return None

        entry_z = self.params.get('entry_z', 2.0)
        min_hl = self.params.get('min_half_life', 300)
        max_hl = self.params.get('max_half_life', 21600)
        ofi_min = self.params.get('ofi_divergence_min', 1.5)

        # Check half-life is in viable range
        if hl < min_hl or hl > max_hl:
            return None

        # Z-score threshold
        if abs(z) < entry_z:
            return None

        # OFI filter: one-sided flow pushing spread
        if abs(ofi_d) < ofi_min:
            return None
        if np.sign(ofi_d) != np.sign(z):
            return None  # info-driven, not flow-driven

        direction = -int(np.sign(z))  # contrarian
        price = df['last'].iloc[t]

        return Signal(
            timestamp=df['timestamp'].iloc[t] if 'timestamp' in df else datetime.now(),
            direction=direction,
            confidence=min(1.0, abs(z) / (entry_z * 2)),
            entry_price=price,
            tp_price=price,  # spread trade — TP is z reverting
            sl_price=price,  # spread trade — SL is z expanding
            metadata={'z': z, 'ofi_div': ofi_d, 'half_life': hl,
                      'beta': df['beta'].iloc[t]},
        )
