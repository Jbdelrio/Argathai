"""
Baudouin4 — TIER-Q6h-D Strategy
=================================
Transient Impact Exhaustion Reversal.
Fades impulse moves after absorption detected.
"""

import numpy as np
import pandas as pd
from typing import Optional
from strategies.common.base_strategy import BaseStrategy, Signal
from datetime import datetime


class Baudouin4(BaseStrategy):
    """
    TIER-Q6h-D: After a violent impulse (price + volume + activity),
    wait for stabilization + absorption (flow continues but price stalls),
    then fade the move.
    """

    def __init__(self, name='baudouin4', params_path='strategies/baudouin4/params.yaml'):
        super().__init__(name, params_path)
        self._state = 'IDLE'
        self._impulse_dir = 0
        self._impulse_start = 0
        self._consec_exhaust = 0
        self._last_signal_idx = -999999

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute TIER features: I, C, Λ, A, S, D, E*."""
        df = df.copy()
        ws = self.params.get('w_s', 60)
        wm = self.params.get('w_m', 600)
        wl = self.params.get('w_l', 3600)
        eps = 1e-12
        N = len(df)

        r = df['ret_1s'].fillna(0).values if 'ret_1s' in df else np.zeros(N)
        v = df['qty'].fillna(0).values if 'qty' in df else np.ones(N)
        q = df['ofi_proxy'].fillna(0).values if 'ofi_proxy' in df else np.zeros(N)
        n = df['n_trades'].fillna(0).values if 'n_trades' in df else np.ones(N)

        def rsum(x, w):
            N = len(x)
            out = np.full(N, np.nan)
            if N < w:
                return out  # not enough data — return all NaN
            cs = np.cumsum(np.concatenate([[0], x]))
            valid = N - w + 1
            out[w-1:] = cs[w:w + valid] - cs[:valid]
            return out

        def rsumsq(x, w):
            N = len(x)
            out = np.full(N, np.nan)
            if N < w:
                return out  # not enough data — return all NaN
            cs = np.cumsum(np.concatenate([[0], x**2]))
            valid = N - w + 1
            out[w-1:] = cs[w:w + valid] - cs[:valid]
            return out

        R_ws = rsum(r, ws); V_ws = rsum(v, ws); Q_ws = rsum(q, ws)
        N_ws = rsum(n, ws)
        sigma_ws = np.sqrt(np.maximum(rsumsq(r, ws), 0))
        sigma_wm = np.sqrt(np.maximum(rsumsq(r, wm), 0))
        sigma_wl = np.sqrt(np.maximum(rsumsq(r, wl), 0))

        med_V = pd.Series(V_ws).rolling(wl, min_periods=wl//2).median().values
        med_N = pd.Series(N_ws).rolling(wl, min_periods=wl//2).median().values

        I = (np.abs(R_ws)/(sigma_wl+eps)) * (V_ws/(med_V+eps)) * (N_ws/(med_N+eps))
        d = np.sign(R_ws)
        C = np.abs(Q_ws) / (V_ws + eps)
        Lambda = np.abs(R_ws) / (np.abs(Q_ws) + eps)
        delta = self.params.get('delta_absorb', 180)
        A = np.full(N, np.nan)
        if delta < N: A[delta:] = Lambda[:-delta] - Lambda[delta:]
        S = sigma_ws / (sigma_wm + eps)

        def rz(x, w):
            s = pd.Series(x); med = s.rolling(w, min_periods=w//4).median()
            mad = (s-med).abs().rolling(w, min_periods=w//4).median() * 1.4826
            return ((s-med)/(mad+eps)).values

        zw = self.params.get('z_win', 3600)
        E = rz(I, zw) + rz(C, zw) + rz(A, zw) + rz(-S, zw)

        # In live mode use a shorter calibration window so quantiles stabilise faster.
        # live_calib_lookback default 86400 (24h) → min_periods = 6h instead of 42h.
        if self.real_time_mode:
            cb = int(self.params.get('live_calib_lookback',
                                     self.params.get('calib_lookback', 7*86400)))
        else:
            cb = int(self.params.get('calib_lookback', 7*86400))

        def rq(x, w, q_val):
            return pd.Series(x).rolling(w, min_periods=w//4).quantile(q_val).values

        df['I'] = I; df['d'] = d; df['S'] = S; df['E_star'] = E
        df['tau_I'] = rq(I, cb, self.params.get('q_impulse', 0.95))
        df['tau_I_down'] = rq(I, cb, self.params.get('q_impulse_down', 0.80))
        df['tau_S'] = rq(S, cb, self.params.get('q_stab', 0.35))
        df['tau_E'] = rq(E, cb, self.params.get('q_exhaust', 0.92))

        return df

    def generate_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        """Run state machine on latest data point, return Signal if entry."""
        t = len(df) - 1
        if t < self.params.get('w_l', 3600):
            return None

        I_t = df['I'].iloc[t]
        S_t = df['S'].iloc[t]
        E_t = df['E_star'].iloc[t]
        d_t = df['d'].iloc[t]
        tau_I = df['tau_I'].iloc[t]
        tau_I_down = df['tau_I_down'].iloc[t]
        tau_S = df['tau_S'].iloc[t]
        tau_E = df['tau_E'].iloc[t]

        if any(np.isnan(x) for x in [I_t, tau_I, S_t, E_t]):
            return None

        cooldown = self.params.get('cooldown_sec', 1800)
        if t - self._last_signal_idx < cooldown:
            return None

        # State machine
        if self._state == 'IDLE':
            if I_t > tau_I:
                self._state = 'IMPULSE'
                self._impulse_dir = int(d_t) if not np.isnan(d_t) else 0
                self._impulse_start = t
                self._consec_exhaust = 0

        elif self._state == 'IMPULSE':
            if I_t < tau_I_down and S_t < tau_S:
                self._state = 'STAB'
                self._consec_exhaust = 0
            elif t - self._impulse_start > 7200:
                self._state = 'IDLE'

        elif self._state == 'STAB':
            if I_t > tau_I:
                self._state = 'IMPULSE'
                self._impulse_dir = int(d_t) if not np.isnan(d_t) else 0
                self._impulse_start = t
                self._consec_exhaust = 0
                return None

            if E_t > tau_E:
                self._consec_exhaust += 1
            else:
                self._consec_exhaust = 0

            persist_k = self.params.get('persist_k', 2)
            if self._consec_exhaust >= persist_k and self._impulse_dir != 0:
                direction = -self._impulse_dir
                price = df['last'].iloc[t]
                tp_pct = self.params.get('tp', 0.0075)
                sl_pct = self.params.get('sl', 0.0035)

                signal = Signal(
                    timestamp=df['timestamp'].iloc[t] if 'timestamp' in df else datetime.now(),
                    direction=direction,
                    confidence=min(1.0, E_t / (tau_E * 2 + 1e-12)),
                    entry_price=price,
                    tp_price=price * (1 + tp_pct * direction),
                    sl_price=price * (1 - sl_pct * direction),
                    metadata={'E_star': E_t, 'impulse_dir': self._impulse_dir},
                )

                self._last_signal_idx = t
                self._state = 'IDLE'
                self._consec_exhaust = 0
                self.on_trade_open(signal)
                return signal

            elif t - self._impulse_start > 10800:
                self._state = 'IDLE'
                self._consec_exhaust = 0

        return None
