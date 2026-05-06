"""
s3_pairs_kalman.py — Pairs Stat Arb with dynamic Kalman hedge ratio
Logic:
- Find cointegrated pairs (ADF test p < 0.05, corr > 0.75)
- Track spread z-score via Kalman filter (time-varying beta)
- Entry: |z| > 2.0 (long underperformer, short outperformer)
- Exit: |z| < 0.5 OR |z| > 3.5 (stop)
"""
import logging
import numpy as np
import pandas as pd
from typing import Optional

try:
    from .base import BaseStrategy, Signal
    from ..regime.kalman_pair import KalmanHedgeRatio, find_cointegrated_pairs, test_cointegration
except ImportError:
    from strategies.base import BaseStrategy, Signal
    from regime.kalman_pair import KalmanHedgeRatio, find_cointegrated_pairs, test_cointegration

log = logging.getLogger(__name__)


class PairsKalmanStrategy(BaseStrategy):

    NAME = "s3_pairs_kalman"

    DEFAULT_PARAMS = {
        "entry_zscore":    2.0,
        "exit_zscore":     0.5,
        "stop_zscore":     3.5,
        "adf_pvalue_max":  0.05,   # standard: at least 95% confidence cointegrated
        "min_correlation": 0.80,   # stricter than original 0.75, avoids spurious pairs
        "lookback_bars":   2880,   # 30 days * 96 bars/day
        "kalman_trans_cov":  1e-4,
        "kalman_obs_cov":    1.0,
        "zscore_window":   20,
        "position_pct":    0.10,   # Per leg (conservative sizing)
        "max_pairs":       5,
    }

    PARAM_GRID = [
        {
            "entry_zscore":    ez,
            "exit_zscore":     xz,
            "stop_zscore":     sz,
            "adf_pvalue_max":  0.05,
            "min_correlation": mc,
            "lookback_bars":   2880,
            "kalman_trans_cov": tc,
            "kalman_obs_cov":   1.0,
            "zscore_window":   zw,
            "position_pct":    0.10,
            "max_pairs":       5,
        }
        for ez  in [1.5, 2.0, 2.5]
        for xz  in [0.3, 0.5, 0.7]
        for sz  in [3.0, 3.5, 4.0]
        for mc  in [0.75, 0.80, 0.85]
        for tc  in [1e-5, 1e-4, 1e-3]
        for zw  in [15, 20, 30]
        if xz < ez < sz
    ]

    def __init__(self, params=None, pair: Optional[tuple[str, str]] = None):
        super().__init__(params)
        # Legacy single-pair mode (kept for backtest compatibility)
        self.pair = pair
        self._khr: Optional[KalmanHedgeRatio] = None
        # Multi-pair live mode
        self.active_pairs: list[dict] = []       # [{pair, khr, adf_pvalue, corr, half_life}]
        self._live_candles: dict[str, pd.DataFrame] = {}  # latest candles per symbol

    def backtest(self,
                 candles: pd.DataFrame,
                 params: Optional[dict] = None,
                 candles_B: Optional[pd.DataFrame] = None) -> list[float]:
        """
        Backtest pairs stat arb.

        For walk-forward validation, candles is symA and candles_B is symB.
        If candles_B is None, we use a shifted version of candles for testing only.
        """
        p = {**self.params, **(params or {})}

        if candles.empty:
            return []

        closes_A = candles["close"].values.astype(float)

        if candles_B is not None and not candles_B.empty:
            # Align lengths
            min_len = min(len(closes_A), len(candles_B))
            closes_A = closes_A[-min_len:]
            closes_B = candles_B["close"].values.astype(float)[-min_len:]
        else:
            # Fallback: synthetic pair for single-asset test
            rng = np.random.default_rng(0)
            closes_B = closes_A * 0.8 + rng.normal(0, 0.5, len(closes_A))

        n = len(closes_A)
        if n < p["zscore_window"] * 3:
            return []

        capital  = 500.0
        pos_size = capital * p["position_pct"]

        khr = KalmanHedgeRatio(
            transition_cov=p["kalman_trans_cov"],
            observation_cov=p["kalman_obs_cov"],
            zscore_window=p["zscore_window"],
        )

        # Use first 20% as warmup for Kalman
        warmup = max(p["zscore_window"] * 3, n // 5)
        khr.fit(closes_A[:warmup], closes_B[:warmup])

        # Online trading simulation
        trades      = []
        in_trade    = False
        long_A      = True   # True = long A / short B
        entry_z     = 0.0
        entry_beta  = 1.0    # hedge ratio at entry time (for correct fee on leg B)

        for i in range(warmup, n):
            beta, intercept, z_score = khr.update(closes_A[i], closes_B[i])

            if not in_trade:
                if abs(z_score) >= p["entry_zscore"]:
                    in_trade   = True
                    entry_z    = z_score
                    entry_beta = beta
                    # z < 0: spread below mean → A underpriced → long A, short B
                    # z > 0: spread above mean → A overpriced → short A, long B
                    long_A = (z_score < 0)

            else:
                exit_trade = False
                if abs(z_score) <= p["exit_zscore"]:
                    exit_trade = True
                elif abs(z_score) >= p["stop_zscore"]:
                    exit_trade = True

                if exit_trade:
                    # z_change > 0 when spread moves up (back toward mean from below)
                    z_change_signed = z_score - entry_z

                    spreads = khr.spread_series()
                    if len(spreads) > p["zscore_window"]:
                        spread_std = np.std(spreads[-p["zscore_window"]:])
                    else:
                        spread_std = np.std(spreads) if len(spreads) > 1 else 1.0

                    # Gross P&L: long A profits when spread rises (z_change > 0)
                    sign = 1.0 if long_A else -1.0
                    pnl_bps  = sign * z_change_signed * spread_std / max(closes_A[i], 1) * 10_000
                    gross_pnl = pos_size * pnl_bps / 10_000

                    # Fees: leg A (pos_size) + leg B (pos_size * |beta|), maker roundtrip
                    fee = pos_size * (1.0 + abs(entry_beta)) * 4.8 / 10_000
                    net_pnl = gross_pnl - fee
                    trades.append(net_pnl)
                    in_trade = False

        return trades

    # ------------------------------------------------------------------
    # Multi-pair live mode
    # ------------------------------------------------------------------

    def update_candles(self, candles_dict: dict[str, pd.DataFrame]):
        """Store latest per-symbol candles. Call before signal() each tick."""
        self._live_candles = candles_dict

    def refresh_active_pairs(self, candles_dict: dict[str, pd.DataFrame]) -> int:
        """
        Test all symbol pairs for cointegration + half-life validity.
        Call at startup and every 24h. Fits a fresh KalmanHedgeRatio per valid pair.

        Filters:
          - Correlation > min_correlation
          - ADF p-value < adf_pvalue_max on OLS spread
          - Mean-reversion half-life in [4, 96] bars (1h to 24h in 15m bars)
        """
        p = self.params
        # 7-day live download = ~672 bars (96 bars/day × 7).
        # Require at least 1/8 of the full lookback or 200, whichever is larger.
        min_bars = max(p.get("lookback_bars", 2880) // 8, 200)

        symbols = [
            s for s, df in candles_dict.items()
            if df is not None and not df.empty and len(df) >= min_bars
        ]

        candidates = []
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                sym_a, sym_b = symbols[i], symbols[j]
                df_a = candles_dict[sym_a]
                df_b = candles_dict[sym_b]

                min_len = min(len(df_a), len(df_b))
                closes_a = df_a["close"].values.astype(float)[-min_len:]
                closes_b = df_b["close"].values.astype(float)[-min_len:]

                # Correlation check (cheap — run first to skip most pairs)
                try:
                    corr = float(np.corrcoef(closes_a, closes_b)[0, 1])
                except Exception:
                    continue
                if abs(corr) < p.get("min_correlation", 0.75):
                    continue

                # ADF cointegration test on OLS spread
                p_val, is_coint = test_cointegration(closes_a, closes_b)
                if not is_coint or p_val > p.get("adf_pvalue_max", 0.05):
                    continue

                # Half-life check: must mean-revert in 1.5h–12h (6–48 bars at 15m)
                try:
                    beta_ols = float(np.polyfit(closes_b, closes_a, 1)[0])
                    spread_ols = closes_a - beta_ols * closes_b
                except Exception:
                    continue
                half_life = self._compute_half_life(spread_ols)
                if not (6.0 <= half_life <= 48.0):
                    continue

                candidates.append({
                    "pair": (sym_a, sym_b),
                    "adf_pvalue": p_val,
                    "correlation": corr,
                    "half_life": half_life,
                    "closes_a": closes_a,
                    "closes_b": closes_b,
                })

        # Take top max_pairs sorted by ADF p-value (most cointegrated first)
        candidates.sort(key=lambda x: x["adf_pvalue"])
        selected = candidates[:p.get("max_pairs", 5)]

        # Fit Kalman filter for each selected pair
        new_active = []
        for c in selected:
            khr = KalmanHedgeRatio(
                transition_cov=p.get("kalman_trans_cov", 1e-4),
                observation_cov=p.get("kalman_obs_cov", 1.0),
                zscore_window=p.get("zscore_window", 20),
            )
            khr.fit(c["closes_a"], c["closes_b"])
            new_active.append({
                "pair": c["pair"],
                "khr": khr,
                "adf_pvalue": c["adf_pvalue"],
                "correlation": c["correlation"],
                "half_life": c["half_life"],
            })

        self.active_pairs = new_active
        log.info(
            "S3 pairs refresh: %d valid pairs out of %d candidates tested "
            "(%d symbols)",
            len(self.active_pairs), len(candidates), len(symbols),
        )
        for info in self.active_pairs:
            log.info(
                "  Pair %s/%s  adf_p=%.4f  corr=%.3f  half_life=%.1f bars",
                info["pair"][0], info["pair"][1],
                info["adf_pvalue"], info["correlation"], info["half_life"],
            )
        return len(self.active_pairs)

    def _compute_half_life(self, spread: np.ndarray) -> float:
        """OLS regression Δspread = α + β*spread_lag → half_life = -ln(2)/β."""
        if len(spread) < 10:
            return np.inf
        spread_lag = spread[:-1]
        spread_diff = np.diff(spread)
        try:
            beta = float(np.polyfit(spread_lag, spread_diff, 1)[0])
        except Exception:
            return np.inf
        if beta >= 0:
            return np.inf
        return float(-np.log(2) / beta)

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def signal(self,
               candles: pd.DataFrame,
               funding_df: Optional[pd.DataFrame],
               params: Optional[dict],
               equity: float,
               open_positions: list[dict]) -> list[Signal]:
        """
        If active_pairs are set (populated by refresh_active_pairs), use
        multi-pair mode: pick the pair with highest |z-score| that exceeds
        the entry threshold.

        Falls back to legacy single-pair mode when active_pairs is empty.
        """
        p = {**self.params, **(params or {})}

        # Multi-pair mode
        if self.active_pairs and self._live_candles:
            return self._signal_multi_pair(p, equity, open_positions)

        # Legacy single-pair mode (used in backtests / when pair is pre-set)
        if self._khr is None or candles.empty:
            return []

        existing = [pos for pos in open_positions if pos.get("strategy") == self.NAME]
        if existing:
            return []

        pos_size = equity * p["position_pct"]
        if pos_size < 10:
            return []

        z = self._khr.zscore()
        if abs(z) < p["entry_zscore"]:
            return []

        coin_a = candles.get("coin", "BTC") if hasattr(candles, "get") else "BTC"
        coin_b = self.pair[1] if self.pair else "ETH"
        beta, _ = self._khr.latest_params()

        return self._build_pair_signals(coin_a, coin_b, beta, z, pos_size, p)

    def _signal_multi_pair(self,
                           p: dict,
                           equity: float,
                           open_positions: list[dict]) -> list[Signal]:
        """Select the pair with highest |z-score| above entry threshold."""
        existing = [pos for pos in open_positions if pos.get("strategy") == self.NAME]
        if existing:
            return []

        pos_size = equity * p.get("position_pct", 0.15)
        if pos_size < 10:
            return []

        best_z = 0.0
        best: Optional[tuple] = None

        for pair_info in self.active_pairs:
            sym_a, sym_b = pair_info["pair"]
            df_a = self._live_candles.get(sym_a)
            df_b = self._live_candles.get(sym_b)
            if df_a is None or df_b is None or df_a.empty or df_b.empty:
                continue

            price_a = float(df_a["close"].iloc[-1])
            price_b = float(df_b["close"].iloc[-1])
            beta, intercept, z = pair_info["khr"].update(price_a, price_b)

            if abs(z) >= p.get("entry_zscore", 2.0) and abs(z) > abs(best_z):
                best_z = z
                best = (sym_a, sym_b, beta, z)

        if best is None:
            return []

        sym_a, sym_b, beta, z = best
        return self._build_pair_signals(sym_a, sym_b, beta, z, pos_size, p)

    def _build_pair_signals(self,
                            sym_a: str,
                            sym_b: str,
                            beta: float,
                            z: float,
                            pos_size: float,
                            p: dict) -> list[Signal]:
        meta_base = {"strategy": self.NAME, "zscore": z, "beta": beta,
                     "pair": (sym_a, sym_b)}
        if z < 0:
            # A underpriced → long A, short B
            return [
                Signal(symbol=sym_a, side="long",  size_usd=pos_size,
                       entry_type="maker", meta=meta_base),
                Signal(symbol=sym_b, side="short", size_usd=pos_size * beta,
                       entry_type="maker", meta=meta_base),
            ]
        else:
            # A overpriced → short A, long B
            return [
                Signal(symbol=sym_a, side="short", size_usd=pos_size,
                       entry_type="maker", meta=meta_base),
                Signal(symbol=sym_b, side="long",  size_usd=pos_size * beta,
                       entry_type="maker", meta=meta_base),
            ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rng = np.random.default_rng(42)
    n = 3000
    ts = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    pB = 100 + np.cumsum(rng.normal(0, 0.5, n))
    pA = 0.8 * pB + 5.0 + rng.normal(0, 0.4, n)

    candles_A = pd.DataFrame({
        "ts": ts, "open": pA, "high": pA * 1.001,
        "low": pA * 0.999, "close": pA, "volume": rng.uniform(500, 2000, n)
    })
    candles_B = pd.DataFrame({
        "ts": ts, "open": pB, "high": pB * 1.001,
        "low": pB * 0.999, "close": pB, "volume": rng.uniform(500, 2000, n)
    })

    strat = PairsKalmanStrategy()
    trades = strat.backtest(candles_A, candles_B=candles_B)
    print(f"Trades: {len(trades)}")
    if trades:
        pnl = np.array(trades)
        print(f"Win rate: {(pnl > 0).mean()*100:.1f}%")
        print(f"Total P&L: ${pnl.sum():.2f}")
