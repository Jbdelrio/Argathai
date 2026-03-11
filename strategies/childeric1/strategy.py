"""
Childeric1 — Cross-Sectional Residual Reversion (single-stream compatible)
=======================================================================

Purpose
-------
This implementation is designed to plug directly into the user's existing
BaseStrategy framework, which appears to call strategies on a *single*
instrument DataFrame. The original Childeric1 concept was a cross-altcoin
stat-arb model; this version preserves the same logic in a format that can
run immediately inside the bot:

1. Detect a short-horizon dislocation in the traded altcoin.
2. Normalize it by recent volatility and, when available, a market/basket proxy.
3. Require stabilization / reversal confirmation in flow and very-short momentum.
4. Fade the move with controlled TP/SL and cooldown.

Required columns
----------------
- last OR close            : last traded price
- timestamp (optional)     : datetime-like timestamp

Strongly recommended columns
----------------------------
- ret_1s                   : 1-second return; derived from `last` if absent
- qty                      : traded quantity per row
- ofi_proxy                : order-flow imbalance proxy
- n_trades                 : number of trades on the row / bucket

Optional columns (used automatically if present)
------------------------------------------------
- market_ret_1s / basket_ret_1s / sector_ret_1s : market factor proxy
- spread_bps / spread                           : spread cost proxy
- funding_rate / oi_delta                       : regime filters

Notes
-----
- If a market factor column exists, Childeric1 trades a true *residual*.
- If not, it falls back to self-normalized mean reversion on the instrument.
- No external model object is required. A future LSTM/meta-model can be
  plugged in by writing a probability/confidence column before signal gen.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from strategies.common.base_strategy import BaseStrategy, Signal


class Childeric1(BaseStrategy):
    """
    Single-stream implementation of Childeric1.

    Conceptually, the strategy fades short-horizon overreactions after a
    stabilization phase. It is inspired by cross-sectional stat-arb, but it is
    implemented so it can run today inside a per-symbol strategy engine.
    """

    DEFAULTS = {
        # Warmup / calibration
        'warmup': 3600,
        'live_warmup_sec': 1800,
        'cooldown_sec': 600,

        # Feature windows (assuming ~1s rows)
        'w_fast': 30,
        'w_mid': 120,
        'w_slow': 900,
        'w_vol': 300,
        'w_mad': 1800,
        'w_flow': 60,
        'w_activity': 300,

        # Entry logic
        'entry_z': 2.2,
        'rearm_z': 1.0,
        'confirm_ret_z': 0.35,
        'confirm_flow_flip': True,
        'min_activity_z': -0.5,
        'max_vol_ratio': 2.5,
        'max_spread_bps': 20.0,

        # Optional market-factor residual beta
        'beta_win': 600,
        'beta_clip': 5.0,

        # Risk / exits
        'tp_sigma': 1.40,
        'sl_sigma': 0.90,
        'tp_pct_min': 0.0025,
        'sl_pct_min': 0.0018,
        'tp_pct_max': 0.0100,
        'sl_pct_max': 0.0075,
        'max_hold_sec': 900,

        # Regime filters
        'max_abs_funding': 0.0020,
        'max_abs_oi_z': 3.0,
    }

    def __init__(self, name: str = 'childeric1', params_path: str = None):
        # Avoid BaseStrategy logger issue when params file does not exist:
        # pass params_path=None by default, then merge defaults.
        super().__init__(name, params_path=None if params_path is None else params_path)
        self.params = {**self.DEFAULTS, **(self.params or {})}

        self._state = 'IDLE'
        self._shock_dir = 0              # +1 price shock up, -1 price shock down
        self._shock_idx = -999999
        self._last_signal_idx = -999999
        self._confirm_count = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pick_col(df: pd.DataFrame, names, default=None):
        for c in names:
            if c in df.columns:
                return c
        return default

    @staticmethod
    def _rolling_mad_z(series: pd.Series, window: int, eps: float = 1e-12) -> pd.Series:
        med = series.rolling(window, min_periods=max(10, window // 4)).median()
        mad = (series - med).abs().rolling(window, min_periods=max(10, window // 4)).median()
        mad = mad * 1.4826
        return (series - med) / (mad + eps)

    @staticmethod
    def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
        n = len(arr)
        out = np.full(n, np.nan)
        if window <= 0 or n < window:
            return out
        cs = np.cumsum(np.concatenate([[0.0], arr]))
        valid = n - window + 1
        out[window - 1:] = cs[window: window + valid] - cs[:valid]
        return out

    @staticmethod
    def _ewm_std(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False, min_periods=max(5, span // 5)).std(bias=False)

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if len(df) == 0:
            return df

        eps = 1e-12
        pcol = self._pick_col(df, ['last', 'close', 'price', 'mid'])
        if pcol is None:
            raise ValueError("Childeric1 requires one of columns: last, close, price, mid")

        # ------------------------------------------------------------------
        # Core market series
        # ------------------------------------------------------------------
        px = pd.to_numeric(df[pcol], errors='coerce').astype(float)
        df['last'] = px

        if 'ret_1s' not in df.columns:
            df['ret_1s'] = np.log(px / px.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            df['ret_1s'] = pd.to_numeric(df['ret_1s'], errors='coerce').fillna(0.0)

        if 'qty' not in df.columns:
            df['qty'] = 0.0
        else:
            df['qty'] = pd.to_numeric(df['qty'], errors='coerce').fillna(0.0)

        if 'ofi_proxy' not in df.columns:
            df['ofi_proxy'] = 0.0
        else:
            df['ofi_proxy'] = pd.to_numeric(df['ofi_proxy'], errors='coerce').fillna(0.0)

        if 'n_trades' not in df.columns:
            df['n_trades'] = (df['qty'] > 0).astype(float)
        else:
            df['n_trades'] = pd.to_numeric(df['n_trades'], errors='coerce').fillna(0.0)

        r = df['ret_1s']
        q = df['qty']
        ofi = df['ofi_proxy']
        ntr = df['n_trades']

        w_fast = int(self.params['w_fast'])
        w_mid = int(self.params['w_mid'])
        w_slow = int(self.params['w_slow'])
        w_vol = int(self.params['w_vol'])
        w_mad = int(self.params['w_mad'])
        w_flow = int(self.params['w_flow'])
        w_activity = int(self.params['w_activity'])
        beta_win = int(self.params['beta_win'])

        # ------------------------------------------------------------------
        # Fast / slow returns and volatility
        # ------------------------------------------------------------------
        df['ret_fast'] = r.rolling(w_fast, min_periods=max(5, w_fast // 3)).sum()
        df['ret_mid'] = r.rolling(w_mid, min_periods=max(10, w_mid // 3)).sum()
        df['ret_slow'] = r.rolling(w_slow, min_periods=max(30, w_slow // 3)).sum()

        vol_fast = self._ewm_std(r, span=max(10, w_fast))
        vol_mid = self._ewm_std(r, span=max(20, w_vol))
        vol_slow = self._ewm_std(r, span=max(50, w_slow))
        df['vol_fast'] = vol_fast
        df['vol_mid'] = vol_mid
        df['vol_slow'] = vol_slow
        df['vol_ratio'] = vol_fast / (vol_slow + eps)

        # ------------------------------------------------------------------
        # Optional market factor (preferred) else self trend proxy
        # ------------------------------------------------------------------
        mcol = self._pick_col(df, ['market_ret_1s', 'basket_ret_1s', 'sector_ret_1s'])
        if mcol is not None:
            market_r = pd.to_numeric(df[mcol], errors='coerce').fillna(0.0)
            df['market_ret_1s'] = market_r
        else:
            # Fallback: use a smoothed self-return proxy; weaker than true market factor,
            # but still gives a local "relative-to-recent-state" residual.
            market_r = r.ewm(span=max(20, w_mid), adjust=False).mean().fillna(0.0)
            df['market_ret_1s'] = market_r

        cov = r.rolling(beta_win, min_periods=max(20, beta_win // 4)).cov(market_r)
        var_m = market_r.rolling(beta_win, min_periods=max(20, beta_win // 4)).var()
        beta = (cov / (var_m + eps)).clip(-self.params['beta_clip'], self.params['beta_clip']).fillna(0.0)
        df['beta_mkt'] = beta
        df['resid_1s'] = r - beta * market_r
        df['resid_fast'] = df['resid_1s'].rolling(w_fast, min_periods=max(5, w_fast // 3)).sum()
        df['resid_mid'] = df['resid_1s'].rolling(w_mid, min_periods=max(10, w_mid // 3)).sum()
        df['z_resid'] = self._rolling_mad_z(df['resid_fast'].fillna(0.0), w_mad)

        # ------------------------------------------------------------------
        # Flow, activity, and stabilization features
        # ------------------------------------------------------------------
        df['flow_fast'] = ofi.rolling(w_flow, min_periods=max(5, w_flow // 3)).sum()
        df['flow_slow'] = ofi.rolling(w_slow, min_periods=max(30, w_slow // 4)).sum()
        df['flow_per_qty'] = df['flow_fast'] / (q.rolling(w_flow, min_periods=max(5, w_flow // 3)).sum() + 1.0)
        df['z_flow'] = self._rolling_mad_z(df['flow_fast'].fillna(0.0), w_mad)

        trade_intensity = ntr.rolling(w_fast, min_periods=max(5, w_fast // 3)).sum()
        df['trade_intensity'] = trade_intensity
        df['activity_z'] = self._rolling_mad_z(trade_intensity.fillna(0.0), w_activity)

        # 3s / 10s micro momentum used as reversal confirmation
        df['ret_3s'] = r.rolling(3, min_periods=1).sum()
        df['ret_10s'] = r.rolling(10, min_periods=1).sum()
        df['ret_3s_z'] = self._rolling_mad_z(df['ret_3s'].fillna(0.0), max(60, w_fast * 6))

        # Price extension vs local mean
        ema_mid = px.ewm(span=max(20, w_mid), adjust=False, min_periods=max(5, w_mid // 4)).mean()
        df['dist_ema'] = np.log(px / ema_mid).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df['z_dist'] = self._rolling_mad_z(df['dist_ema'], w_mad)

        # ------------------------------------------------------------------
        # Spread / fee proxy filters
        # ------------------------------------------------------------------
        if 'spread_bps' in df.columns:
            df['spread_bps_eff'] = pd.to_numeric(df['spread_bps'], errors='coerce').fillna(np.nan)
        elif 'spread' in df.columns:
            spr = pd.to_numeric(df['spread'], errors='coerce').fillna(np.nan)
            df['spread_bps_eff'] = (spr / (px.abs() + eps)) * 1e4
        elif {'bid', 'ask'}.issubset(df.columns):
            bid = pd.to_numeric(df['bid'], errors='coerce')
            ask = pd.to_numeric(df['ask'], errors='coerce')
            df['spread_bps_eff'] = ((ask - bid) / (px.abs() + eps)) * 1e4
        else:
            df['spread_bps_eff'] = np.nan

        # ------------------------------------------------------------------
        # Optional regime filters: funding and open interest changes
        # ------------------------------------------------------------------
        if 'funding_rate' in df.columns:
            df['funding_rate'] = pd.to_numeric(df['funding_rate'], errors='coerce').fillna(0.0)
        else:
            df['funding_rate'] = 0.0

        if 'oi_delta' in df.columns:
            oid = pd.to_numeric(df['oi_delta'], errors='coerce').fillna(0.0)
            df['oi_delta'] = oid
            df['oi_z'] = self._rolling_mad_z(oid, max(300, w_mid * 3)).fillna(0.0)
        else:
            df['oi_delta'] = 0.0
            df['oi_z'] = 0.0

        # ------------------------------------------------------------------
        # Composite alpha + simple meta-score
        # ------------------------------------------------------------------
        # Higher alpha_raw means "too up" => candidate short.
        # Lower alpha_raw means "too down" => candidate long.
        df['alpha_raw'] = 0.65 * df['z_resid'].fillna(0.0) + 0.35 * df['z_dist'].fillna(0.0)

        # Confirmation score rewards weakening of the original impulse.
        # For a positive shock, we want negative short-term return and/or flow cooling.
        flow_sign = np.sign(df['flow_fast'].fillna(0.0))
        alpha_sign = np.sign(df['alpha_raw'].fillna(0.0))
        df['flow_against_alpha'] = -alpha_sign * flow_sign  # +1 when flow opposes prior extension
        df['meta_score'] = (
            0.50 * np.clip(np.abs(df['alpha_raw']) / max(self.params['entry_z'], eps), 0, 2)
            + 0.25 * np.clip(df['flow_against_alpha'], -1, 1)
            + 0.25 * np.clip(-alpha_sign * df['ret_3s_z'].fillna(0.0), -2, 2)
        )
        df['meta_score'] = df['meta_score'].clip(0, 1)

        return df

    def generate_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        t = len(df) - 1
        if t < int(self.params['warmup']):
            return None

        row = df.iloc[t]
        if row.isna().all():
            return None

        cooldown = int(self.params['cooldown_sec'])
        if t - self._last_signal_idx < cooldown:
            return None

        alpha = float(row.get('alpha_raw', np.nan))
        z_resid = float(row.get('z_resid', np.nan))
        ret_3s_z = float(row.get('ret_3s_z', np.nan))
        activity_z = float(row.get('activity_z', np.nan))
        vol_ratio = float(row.get('vol_ratio', np.nan))
        spread_bps = float(row.get('spread_bps_eff', np.nan))
        funding = float(row.get('funding_rate', 0.0))
        oi_z = float(row.get('oi_z', 0.0))

        if any(np.isnan(x) for x in [alpha, z_resid, ret_3s_z, activity_z, vol_ratio]):
            return None

        # Hard regime / cost filters
        if activity_z < float(self.params['min_activity_z']):
            return None
        if vol_ratio > float(self.params['max_vol_ratio']):
            return None
        if not np.isnan(spread_bps) and spread_bps > float(self.params['max_spread_bps']):
            return None
        if abs(funding) > float(self.params['max_abs_funding']):
            return None
        if abs(oi_z) > float(self.params['max_abs_oi_z']):
            return None

        entry_z = float(self.params['entry_z'])
        rearm_z = float(self.params['rearm_z'])
        confirm_ret_z = float(self.params['confirm_ret_z'])
        confirm_flow_flip = bool(self.params['confirm_flow_flip'])

        # +1 means upward dislocation to fade via SHORT later.
        current_shock_dir = int(np.sign(alpha))

        # ----------------------------
        # State machine
        # ----------------------------
        if self._state == 'IDLE':
            if abs(alpha) >= entry_z:
                self._state = 'SHOCK'
                self._shock_dir = current_shock_dir
                self._shock_idx = t
                self._confirm_count = 0
            return None

        if self._state == 'SHOCK':
            # If extension already mean-reverted too much, re-arm and wait for next one.
            if abs(alpha) <= rearm_z:
                self._state = 'IDLE'
                self._shock_dir = 0
                self._confirm_count = 0
                return None

            # If a new larger shock occurs in the opposite direction, restart.
            if current_shock_dir != 0 and current_shock_dir != self._shock_dir and abs(alpha) >= entry_z:
                self._shock_dir = current_shock_dir
                self._shock_idx = t
                self._confirm_count = 0
                return None

            # Confirmation rules:
            #  - the very-short momentum should now oppose the prior shock
            #  - optionally flow should also stop supporting the shock
            price_reversal = (-self._shock_dir * ret_3s_z) > confirm_ret_z
            flow_fast = float(row.get('flow_fast', 0.0))
            flow_ok = True
            if confirm_flow_flip:
                flow_ok = (-self._shock_dir * np.sign(flow_fast)) >= 0

            if price_reversal and flow_ok:
                self._confirm_count += 1
            else:
                self._confirm_count = 0

            # Need a little persistence so we do not fade the first noisy tick.
            if self._confirm_count < 2:
                # Expire stale shock states.
                if t - self._shock_idx > int(self.params['max_hold_sec']):
                    self._state = 'IDLE'
                    self._shock_dir = 0
                    self._confirm_count = 0
                return None

            # Emit entry: fade the shock.
            direction = -self._shock_dir
            if direction == 0:
                return None

            price = float(row['last'])
            sigma = float(max(row.get('vol_mid', np.nan), row.get('vol_fast', np.nan), 1e-6))

            # Convert sigma of log-return into bounded TP/SL percentages.
            tp_pct = float(np.clip(self.params['tp_sigma'] * sigma * np.sqrt(60),
                                   self.params['tp_pct_min'], self.params['tp_pct_max']))
            sl_pct = float(np.clip(self.params['sl_sigma'] * sigma * np.sqrt(60),
                                   self.params['sl_pct_min'], self.params['sl_pct_max']))

            confidence = float(np.clip(row.get('meta_score', 0.5), 0.05, 1.0))

            signal = Signal(
                timestamp=row['timestamp'] if 'timestamp' in df.columns else datetime.now(),
                direction=direction,
                confidence=confidence,
                entry_price=price,
                tp_price=price * (1 + tp_pct * direction),
                sl_price=price * (1 - sl_pct * direction),
                metadata={
                    'alpha_raw': alpha,
                    'z_resid': z_resid,
                    'z_dist': float(row.get('z_dist', np.nan)),
                    'ret_3s_z': ret_3s_z,
                    'flow_fast': flow_fast,
                    'activity_z': activity_z,
                    'vol_ratio': vol_ratio,
                    'spread_bps_eff': None if np.isnan(spread_bps) else spread_bps,
                    'beta_mkt': float(row.get('beta_mkt', 0.0)),
                    'shock_dir': self._shock_dir,
                    'state': 'FADE_ENTRY',
                },
            )

            self._last_signal_idx = t
            self._state = 'IDLE'
            self._shock_dir = 0
            self._confirm_count = 0
            self.on_trade_open(signal)
            return signal

        return None