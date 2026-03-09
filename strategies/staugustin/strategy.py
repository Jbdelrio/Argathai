"""
Staugustin — Liquidity Release Detection Strategy
===================================================
Detects the transition from absorbed aggressive flow to impacting flow,
then accompanies the continuation on a mid-frequency horizon.

Architecture (3 layers):
  1. Microstructure aggregation : 1s raw → 1m bars
  2. Liquidity release features : 10 signals on 1m bars
  3. Mid-frequency execution    : 5min decisions, 15-60min holds

Features:
  X_t   — signed aggressive flow (buy - sell, normalized)
  P_t   — smoothed microstructural pressure (EMA of X + β·OFI)
  Π_t   — flow persistence (fraction of recent bars coherent with P)
  Comp  — prior compression (-z of short/long vol ratio)
  I_t   — marginal flow impact (return per unit of flow)
  ΔI_t  — impact acceleration (I minus its EMA)
  A_t   — participation acceleration (Δlog(1+N_trades))
  E_t   — directional efficiency (|Σr| / Σ|r|)
  M_t   — illiquidity / Amihud proxy (|R| / V)
  S_t   — composite score (weighted z-scores)

v1: Rule-based (no Ridge). Entry when structural gates pass AND |S_t| > τ.
"""

import numpy as np
import pandas as pd
from typing import Optional
from strategies.common.base_strategy import BaseStrategy, Signal
from datetime import datetime


class Staugustin(BaseStrategy):
    """
    Liquidity Release Detection.

    Enters when persistent aggressive flow transitions from being absorbed
    to actually printing price (impact acceleration + directional efficiency).
    Exits when microstructural edge degrades or vol-stops are hit.
    """

    def __init__(self, name: str = 'staugustin',
                 params_path: str = 'strategies/staugustin/params.yaml'):
        super().__init__(name, params_path)
        self._last_signal_idx: int = -999999
        self._bars_1m: Optional[pd.DataFrame] = None

    # ─── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _ema(x: np.ndarray, span: int) -> np.ndarray:
        """Exponential moving average (pandas, vectorised)."""
        return pd.Series(x).ewm(span=span, min_periods=max(1, span // 2)).mean().values

    @staticmethod
    def _rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
        """Fast rolling sum via cumsum trick (NaN-safe)."""
        n = len(x)
        out = np.full(n, np.nan)
        if n < w:
            return out
        # Replace NaN with 0 for cumsum; mark windows containing NaN
        x_clean = np.nan_to_num(x, nan=0.0)
        cs = np.cumsum(np.concatenate([[0.0], x_clean]))
        out[w - 1:] = cs[w:] - cs[:n - w + 1]
        # Invalidate any window that overlapped a NaN in the input
        nan_flag = np.isnan(x).astype(float)
        nan_cs = np.cumsum(np.concatenate([[0.0], nan_flag]))
        has_nan = nan_cs[w:] - nan_cs[:n - w + 1]
        out[w - 1:] = np.where(has_nan > 0, np.nan, out[w - 1:])
        return out

    @staticmethod
    def _robust_z(x: np.ndarray, w: int) -> np.ndarray:
        """Robust rolling z-score: (x − median) / (1.4826·MAD + ε)."""
        eps = 1e-12
        s = pd.Series(x)
        med = s.rolling(w, min_periods=max(1, w // 4)).median()
        mad = (s - med).abs().rolling(w, min_periods=max(1, w // 4)).median() * 1.4826
        return ((s - med) / (mad + eps)).values

    # ─── Layer 1 : 1s → 1m aggregation ──────────────────────────────────

    def _aggregate_1m(self, df: pd.DataFrame, n_bars: int, agg: int) -> pd.DataFrame:
        """Reshape raw 1s arrays into 1-minute bars."""
        N = n_bars * agg

        ret = df['ret_1s'].fillna(0).values[:N]
        qty = df['qty'].fillna(0).values[:N]
        ofi = df['ofi_proxy'].fillna(0).values[:N]
        ntrades = df['n_trades'].fillna(0).values[:N]
        last = df['last'].ffill().values[:N]

        # Buy / sell decomposition
        if 'buy_qty' in df.columns and 'sell_qty' in df.columns:
            buy = df['buy_qty'].fillna(0).values[:N]
            sell = df['sell_qty'].fillna(0).values[:N]
        else:
            # Fallback: ofi ≈ buy − sell, qty ≈ buy + sell
            buy = np.maximum(0.0, (qty + ofi) / 2.0)
            sell = np.maximum(0.0, (qty - ofi) / 2.0)

        # Reshape (n_bars, agg) and aggregate
        ret_2d = ret.reshape(n_bars, agg)
        qty_2d = qty.reshape(n_bars, agg)
        buy_2d = buy.reshape(n_bars, agg)
        sell_2d = sell.reshape(n_bars, agg)
        ntrades_2d = ntrades.reshape(n_bars, agg)
        ofi_2d = ofi.reshape(n_bars, agg)
        last_2d = last.reshape(n_bars, agg)

        bars = pd.DataFrame({
            'R': ret_2d.sum(axis=1),                   # log-return
            'Q': qty_2d.sum(axis=1),                   # total quantity
            'B': buy_2d.sum(axis=1),                   # buy volume
            'S_vol': sell_2d.sum(axis=1),               # sell volume
            'N_trades': ntrades_2d.sum(axis=1),         # trade count
            'OF': ofi_2d.sum(axis=1),                   # OFI
            'V': (last_2d * qty_2d).sum(axis=1),        # dollar volume
            'close': last_2d[:, -1],                    # close price
        })
        return bars

    # ─── Layer 2 : Feature computation on 1m bars ───────────────────────

    def _compute_raw_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Build the 10 staugustin features on 1-minute bars."""
        p = self.params
        eps = 1e-12
        n = len(bars)

        L_Q  = p.get('L_Q', 10)
        L_f  = p.get('L_f', 5)
        H_pi = p.get('H_pi', 5)
        h_s  = p.get('h_s', 5)
        h_l  = p.get('h_l', 30)
        h_I  = p.get('h_I', 5)
        L_I  = p.get('L_I', 5)
        h_E  = p.get('h_E', 5)
        beta = p.get('beta_ofi', 0.5)

        R        = bars['R'].values
        Q        = bars['Q'].values
        B        = bars['B'].values
        S_vol    = bars['S_vol'].values
        N_trades = bars['N_trades'].values
        OF       = bars['OF'].values
        V        = bars['V'].values

        # ── 4.1  Signed aggressive flow  X_t = D_t / EMA(Q) ──────────
        D = B - S_vol
        ema_Q = self._ema(Q, L_Q)
        X = D / (ema_Q + eps)

        # ── 4.2  Smoothed pressure  P = EMA(X) + β·EMA(OF̃) ─────────
        OF_norm = OF / (ema_Q + eps)
        P_pressure = self._ema(X, L_f) + beta * self._ema(OF_norm, L_f)

        # ── 4.3  Flow persistence  Π_t ∈ [0,1] ──────────────────────
        sign_X = np.sign(X)
        sign_P = np.sign(P_pressure)
        match = np.zeros(n)
        for j in range(1, H_pi + 1):
            # At index t (≥ j): compare sign(X[t-j]) with sign(P[t])
            match[j:] += (sign_X[:-j] == sign_P[j:]).astype(float)
        Pi = np.full(n, np.nan)
        Pi[H_pi:] = match[H_pi:] / H_pi

        # ── 4.4  Compression  Comp = −z(RV_short / RV_long) ─────────
        RV_s = self._rolling_sum(R ** 2, h_s)
        RV_l = self._rolling_sum(R ** 2, h_l)
        C_ratio = RV_s / (RV_l + eps)
        # Comp computed after z-scoring step (needs robust_z)

        # ── 4.5  Marginal flow impact  I_t ───────────────────────────
        sum_R     = self._rolling_sum(R, h_I)
        sum_absX  = self._rolling_sum(np.abs(X), h_I)
        I_impact  = np.sign(P_pressure) * sum_R / (sum_absX + eps)

        # ── 4.6  Impact acceleration  ΔI_t = I − EMA(I) ─────────────
        ema_I   = self._ema(I_impact, L_I)
        delta_I = I_impact - ema_I

        # ── 4.7  Participation acceleration  A_t = Δlog(1+N) ─────────
        log_N = np.log1p(N_trades)
        A_part = np.empty(n)
        A_part[0] = np.nan
        A_part[1:] = np.diff(log_N)

        # ── 4.8  Directional efficiency  E_t ∈ [0,1] ────────────────
        abs_sum_R = np.abs(self._rolling_sum(R, h_E))
        sum_abs_R = self._rolling_sum(np.abs(R), h_E)
        E_eff = abs_sum_R / (sum_abs_R + eps)

        # ── 4.9  Illiquidity (Amihud)  M_t = |R| / V ────────────────
        M_illiq = np.abs(R) / (V + eps)

        # ── Realized vol for stops ───────────────────────────────────
        sigma_short = np.sqrt(np.maximum(RV_s / max(h_s, 1), 0.0))

        # Store on bars DataFrame
        bars['X']             = X
        bars['pressure']      = P_pressure
        bars['persistence']   = Pi
        bars['C_ratio']       = C_ratio
        bars['impact']        = I_impact
        bars['delta_impact']  = delta_I
        bars['participation'] = A_part
        bars['efficiency']    = E_eff
        bars['illiquidity']   = M_illiq
        bars['sigma_short']   = sigma_short
        bars['close_1m']      = bars['close'].values
        bars['V_dollar']      = V

        return bars

    def _apply_z_scores(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Robust rolling z-scores on raw features."""
        zw = self.params.get('z_window', 288)

        bars['z_pressure']      = self._robust_z(bars['pressure'].values, zw)
        bars['z_impact']        = self._robust_z(bars['impact'].values, zw)
        bars['z_delta_impact']  = self._robust_z(bars['delta_impact'].values, zw)
        bars['z_participation'] = self._robust_z(bars['participation'].values, zw)
        bars['z_efficiency']    = self._robust_z(bars['efficiency'].values, zw)
        bars['z_illiquidity']   = self._robust_z(bars['illiquidity'].values, zw)

        # Compression = −z(C_ratio) : high → vol was compressed before
        bars['compression'] = -self._robust_z(bars['C_ratio'].values, zw)

        return bars

    def _compute_score(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Composite score S_t (weighted z-scores)."""
        p = self.params
        w1 = p.get('w_pressure', 0.30)
        w2 = p.get('w_impact', 0.20)
        w3 = p.get('w_delta_impact', 0.20)
        w4 = p.get('w_participation', 0.10)
        w5 = p.get('w_efficiency', 0.10)
        w6 = p.get('w_compression', 0.10)
        w7 = p.get('w_illiquidity', 0.15)

        Pi = bars['persistence'].values

        S = (  w1 * bars['z_pressure'].values * Pi
             + w2 * bars['z_impact'].values
             + w3 * bars['z_delta_impact'].values
             + w4 * bars['z_participation'].values
             + w5 * bars['z_efficiency'].values
             + w6 * bars['compression'].values
             - w7 * bars['z_illiquidity'].values)

        bars['S_score'] = S
        return bars

    def _compute_cost(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Round-trip cost barrier  c_RT = 2·c_side."""
        p = self.params
        fee  = p.get('fee_side', 0.0001)
        lam1 = p.get('lambda1', 0.15)
        lam2 = p.get('lambda2', 0.5)
        notional = p.get('min_notional', 250)
        eps = 1e-12

        V = bars['V_dollar'].values
        M = bars['illiquidity'].values

        c_side = fee + lam1 * notional / (V + eps) + lam2 * M
        bars['cost_rt'] = 2.0 * c_side
        return bars

    # ─── Main compute_features  (BaseStrategy interface) ─────────────────

    _FEATURE_COLS = [
        'z_pressure', 'persistence', 'compression',
        'z_impact', 'z_delta_impact', 'z_participation',
        'efficiency', 'z_efficiency', 'z_illiquidity',
        'S_score', 'cost_rt', 'sigma_short', 'close_1m',
    ]

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate 1s → 1m, compute 10 features, merge back to 1s.

        Feature lag: each 1s row receives features from the **previous
        completed** 1m bar (bar_idx = floor(i/agg) − 1) to avoid any
        look-ahead bias.
        """
        df = df.copy()
        agg    = self.params.get('agg_interval', 60)
        N      = len(df)
        n_bars = N // agg

        min_bars_needed = self.params.get('h_l', 30) + self.params.get('z_window', 288)

        if n_bars < min_bars_needed:
            for col in self._FEATURE_COLS:
                df[col] = np.nan
            return df

        # Layer 1 — aggregate
        bars = self._aggregate_1m(df, n_bars, agg)

        # Layer 2 — features
        bars = self._compute_raw_features(bars)
        bars = self._apply_z_scores(bars)
        bars = self._compute_score(bars)
        bars = self._compute_cost(bars)

        self._bars_1m = bars

        # Map features back to 1s with 1-bar lag (no look-ahead)
        # Seconds 0..agg-1 → bar_idx = -1 → NaN
        # Seconds agg..2*agg-1 → bar_idx = 0 → features of first complete bar
        bar_idx = np.arange(N) // agg - 1

        for col in self._FEATURE_COLS:
            if col not in bars.columns:
                df[col] = np.nan
                continue
            vals = bars[col].values
            col_data = np.full(N, np.nan)
            valid = bar_idx >= 0
            col_data[valid] = vals[np.minimum(bar_idx[valid], n_bars - 1)]
            df[col] = col_data

        return df

    # ─── Signal generation  (BaseStrategy interface) ─────────────────────

    def generate_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        """
        Check the 8 structural gates + score threshold.
        Return Signal(long/short) or None.
        """
        t = len(df) - 1
        warmup = self.get_warmup_sec()

        if t < warmup:
            return None

        # Cooldown
        cooldown = self.params.get('cooldown_sec', 900)
        if t - self._last_signal_idx < cooldown:
            return None

        # Latest features (from last completed 1m bar)
        row = df.iloc[t]

        z_P   = row.get('z_pressure',      np.nan)
        Pi    = row.get('persistence',      np.nan)
        Comp  = row.get('compression',      np.nan)
        z_I   = row.get('z_impact',         np.nan)
        z_dI  = row.get('z_delta_impact',   np.nan)
        z_A   = row.get('z_participation',  np.nan)
        E     = row.get('efficiency',       np.nan)
        z_M   = row.get('z_illiquidity',    np.nan)
        S     = row.get('S_score',          np.nan)
        c_RT  = row.get('cost_rt',          np.nan)
        sigma = row.get('sigma_short',      np.nan)

        needed = [z_P, Pi, Comp, z_I, z_dI, z_A, E, z_M, S, c_RT, sigma]
        if any(np.isnan(x) for x in needed):
            return None

        # ── Thresholds ───────────────────────────────────────────────
        p = self.params
        theta_P  = p.get('theta_P',  1.0)
        theta_pi = p.get('theta_pi', 0.70)
        theta_C  = p.get('theta_C',  0.0)
        theta_I  = p.get('theta_I',  0.50)
        theta_dI = p.get('theta_dI', 0.50)
        theta_A  = p.get('theta_A',  0.0)
        theta_E  = p.get('theta_E',  0.55)
        theta_M  = p.get('theta_M',  1.0)
        tau      = p.get('tau',      1.75)

        # ── Entry gates — LONG ───────────────────────────────────────
        long_ok = (
            z_P  > theta_P
            and Pi   > theta_pi
            and Comp > theta_C
            and z_I  > theta_I
            and z_dI > theta_dI
            and z_A  > theta_A
            and E    > theta_E
            and z_M  < theta_M
            and S    > tau
        )

        # ── Entry gates — SHORT (symmetric) ──────────────────────────
        short_ok = (
            z_P  < -theta_P
            and Pi   > theta_pi
            and Comp > theta_C
            and z_I  < -theta_I
            and z_dI < -theta_dI
            and z_A  > theta_A
            and E    > theta_E
            and z_M  < theta_M
            and S    < -tau
        )

        if not long_ok and not short_ok:
            return None

        direction = 1 if long_ok else -1
        price = df['last'].iloc[t]

        # ── Volatility-based stops ───────────────────────────────────
        k_SL = p.get('k_SL', 1.5)
        k_TP = p.get('k_TP', 2.5)
        sigma_val = max(float(sigma), 1e-8)

        sl_pct = k_SL * sigma_val
        tp_pct = k_TP * sigma_val

        # Clamp to sane bounds (0.2 % – 3 %)
        sl_pct = min(max(sl_pct, 0.002), 0.03)
        tp_pct = min(max(tp_pct, 0.003), 0.03)

        # ── Build Signal ─────────────────────────────────────────────
        signal = Signal(
            timestamp=(df['timestamp'].iloc[t]
                       if 'timestamp' in df.columns else datetime.now()),
            direction=direction,
            confidence=min(1.0, abs(float(S)) / (tau * 2 + 1e-12)),
            entry_price=price,
            tp_price=price * (1 + tp_pct * direction),
            sl_price=price * (1 - sl_pct * direction),
            metadata={
                'S_score':        float(S),
                'z_pressure':     float(z_P),
                'persistence':    float(Pi),
                'compression':    float(Comp),
                'z_impact':       float(z_I),
                'z_delta_impact': float(z_dI),
                'z_participation': float(z_A),
                'efficiency':     float(E),
                'z_illiquidity':  float(z_M),
                'cost_rt':        float(c_RT),
                'sigma':          float(sigma_val),
                'sl_pct':         float(sl_pct),
                'tp_pct':         float(tp_pct),
            },
        )

        self._last_signal_idx = t
        self.on_trade_open(signal)
        return signal
