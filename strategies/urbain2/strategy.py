"""
Urbain2 — Residual Cross-Sectional Rotation (single-symbol implementation)
===========================================================================
Idiosyncratic continuation signal after common-factor neutralization,
confirmed by OI, penalized by funding crowding, and filtered by costs/uncertainty.
"""

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from strategies.common.base_strategy import BaseStrategy, Signal


class Urbain2(BaseStrategy):
    """Residual momentum strategy with cost-aware and uncertainty-aware filtering."""

    def __init__(self, name='urbain2', params_path='strategies/urbain2/params.yaml'):
        super().__init__(name, params_path)
        self._last_signal_idx = -999999

    @staticmethod
    def _robust_z(x: pd.Series, window: int, eps: float = 1e-12) -> pd.Series:
        med = x.rolling(window, min_periods=max(5, window // 4)).median()
        mad = (x - med).abs().rolling(window, min_periods=max(5, window // 4)).median() * 1.4826
        return (x - med) / (mad + eps)

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        eps = 1e-12

        # Base returns
        if 'ret_1s' not in df.columns:
            df['ret_1s'] = np.log(df['last']).diff().fillna(0)
        r = df['ret_1s'].fillna(0)

        # Common factors (if unavailable, fallback to smooth market mode proxy)
        btc = df['btc_ret_1h'] if 'btc_ret_1h' in df.columns else r.rolling(3600, min_periods=300).mean().fillna(0)
        eth = df['eth_ret_1h'] if 'eth_ret_1h' in df.columns else r.rolling(1800, min_periods=300).mean().fillna(0)
        alt = df['alt_pca1'] if 'alt_pca1' in df.columns else (0.5 * btc + 0.5 * eth)

        reg_w = int(self.params.get('reg_window', 7200))
        X = np.column_stack([btc.values, eth.values, alt.values])
        y = r.values

        residual = np.full(len(df), np.nan)
        for t in range(reg_w, len(df)):
            xs = X[t - reg_w:t]
            ys = y[t - reg_w:t]
            if np.isnan(xs).any() or np.isnan(ys).any():
                continue
            try:
                beta, *_ = np.linalg.lstsq(xs, ys, rcond=None)
                residual[t] = y[t] - float(np.dot(X[t], beta))
            except np.linalg.LinAlgError:
                continue

        df['u_resid'] = residual

        # Residual momentum
        hm = int(self.params.get('momentum_window', 3600))
        sig_u = pd.Series(residual).rolling(hm, min_periods=max(20, hm // 5)).std()
        m = pd.Series(residual).rolling(hm, min_periods=max(20, hm // 5)).sum() / (sig_u * np.sqrt(max(hm, 1)) + eps)
        df['m_resid'] = m

        # OI confirmation
        oi = df['open_interest'] if 'open_interest' in df.columns else pd.Series(np.nan, index=df.index)
        d_oi = np.log(oi.replace(0, np.nan)).diff()
        df['o_oi'] = self._robust_z(d_oi, int(self.params.get('oi_window', 1440)))

        # Funding crowding term
        funding = df['funding_rate'] if 'funding_rate' in df.columns else pd.Series(0.0, index=df.index)
        zf = self._robust_z(funding.fillna(0), int(self.params.get('funding_window', 1440))).fillna(0)
        eta = float(self.params.get('funding_eta', 0.15))
        phi = np.sign(m.fillna(0)) * zf - eta * (zf**2)
        df['phi_funding'] = phi

        # Liquidity quality
        adv = df['quote_volume'] if 'quote_volume' in df.columns else (
            df['qty'] * df['last'] if {'qty', 'last'}.issubset(df.columns) else pd.Series(0.0, index=df.index)
        )
        spread = df['spread_bps'] if 'spread_bps' in df.columns else pd.Series(6.0, index=df.index)
        impact_proxy = pd.Series(r).abs().rolling(300, min_periods=60).mean() * 1e4
        z_adv = self._robust_z(np.log(adv.replace(0, np.nan)), int(self.params.get('liq_window', 1440))).fillna(0)
        z_spread = self._robust_z(spread.fillna(spread.median()), int(self.params.get('liq_window', 1440))).fillna(0)
        z_impact = self._robust_z(impact_proxy.fillna(impact_proxy.median()), int(self.params.get('liq_window', 1440))).fillna(0)
        df['l_liq'] = z_adv - z_spread - z_impact

        # Regime gate (residual dispersion high, BTC move not extreme)
        disp = pd.Series(residual).rolling(int(self.params.get('disp_window', 1800)), min_periods=100).std()
        disp_q = disp.rolling(int(self.params.get('regime_ref_window', 7200)), min_periods=300).quantile(0.6)
        btc_abs = pd.Series(btc).abs()
        btc_q = btc_abs.rolling(int(self.params.get('regime_ref_window', 7200)), min_periods=300).quantile(0.9)
        gate = ((disp > disp_q) & (btc_abs < btc_q)).astype(float)
        df['regime_gate'] = gate

        # Uncertainty proxy and cost proxy
        q_unc = pd.Series(residual).rolling(int(self.params.get('uncertainty_window', 1800)), min_periods=100).std().fillna(np.inf)
        fee_rt = float(self.params.get('taker_fee_rt', 0.0009))
        spread_rt = spread.fillna(6.0) / 10000.0
        slippage_rt = float(self.params.get('slippage_bps', 1.5)) / 10000.0
        cost = fee_rt + spread_rt + slippage_rt

        # Final score
        t1 = float(self.params.get('theta_m', 1.0))
        t2 = float(self.params.get('theta_oi', 0.6))
        t3 = float(self.params.get('theta_funding', 0.4))
        t4 = float(self.params.get('theta_liq', 0.5))
        l_unc = float(self.params.get('lambda_uncertainty', 0.75))
        raw = gate * (t1 * m.fillna(0) + t2 * df['o_oi'].fillna(0) + t3 * phi.fillna(0) + t4 * df['l_liq'].fillna(0))
        df['s_star'] = raw - l_unc * q_unc
        df['cost_proxy'] = cost

        return df

    def generate_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        t = len(df) - 1
        warmup = int(self.params.get('warmup', 7200))
        if t < warmup:
            return None

        s_star = df['s_star'].iloc[t] if 's_star' in df.columns else np.nan
        cost = df['cost_proxy'].iloc[t] if 'cost_proxy' in df.columns else np.nan
        gate = df['regime_gate'].iloc[t] if 'regime_gate' in df.columns else 0
        oi_conf = df['o_oi'].iloc[t] if 'o_oi' in df.columns else 0

        if any(np.isnan(x) for x in [s_star, cost]) or gate <= 0:
            return None

        cooldown = int(self.params.get('cooldown_sec', 4 * 3600))
        if t - self._last_signal_idx < cooldown:
            return None

        tau = float(self.params.get('tau', 1.0))
        kappa = float(self.params.get('kappa_cost', 3.0))
        threshold = tau + kappa * cost

        if abs(s_star) <= threshold:
            return None

        # OI must confirm direction unless intentionally disabled
        oi_floor = float(self.params.get('oi_confirmation_floor', 0.0))
        direction = int(np.sign(s_star))
        if direction > 0 and oi_conf < oi_floor:
            return None
        if direction < 0 and oi_conf > -oi_floor:
            return None

        price = float(df['last'].iloc[t])
        tp_pct = float(self.params.get('tp_pct', 0.010))
        sl_pct = float(self.params.get('sl_pct', 0.006))

        self._last_signal_idx = t

        return Signal(
            timestamp=df['timestamp'].iloc[t] if 'timestamp' in df.columns else datetime.now(),
            direction=direction,
            confidence=float(min(1.0, max(0.0, abs(s_star) / (threshold + 1e-12) - 1))),
            entry_price=price,
            tp_price=price * (1 + direction * tp_pct),
            sl_price=price * (1 - direction * sl_pct),
            metadata={
                's_star': float(s_star),
                'threshold': float(threshold),
                'oi_conf': float(oi_conf),
                'regime_gate': float(gate),
                'strategy': 'urbain2',
            },
        )
