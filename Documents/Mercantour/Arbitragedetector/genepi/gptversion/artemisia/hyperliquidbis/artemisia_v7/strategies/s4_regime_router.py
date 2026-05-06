"""
s4_regime_router.py — HMM Regime Router
Routes signals from S1/S2/S3 based on current detected market regime.
Regime 0 (CALM)      -> S1 + S2 + S3 all active
Regime 1 (NORMAL)    -> S2 + S3 active, S1 active if funding > threshold
Regime 2 (EXPLOSION) -> ALL strategies paused, positions closed
"""
import logging
import numpy as np
import pandas as pd
from typing import Optional

try:
    from .base import BaseStrategy, Signal
    from .s1_funding_arb import FundingArbStrategy
    from .s2_maker_mr import MakerMRStrategy
    from .s3_pairs_kalman import PairsKalmanStrategy
    from ..regime.hmm_detector import RegimeDetector, REGIME_NAMES
except ImportError:
    from strategies.base import BaseStrategy, Signal
    from strategies.s1_funding_arb import FundingArbStrategy
    from strategies.s2_maker_mr import MakerMRStrategy
    from strategies.s3_pairs_kalman import PairsKalmanStrategy
    from regime.hmm_detector import RegimeDetector, REGIME_NAMES

log = logging.getLogger(__name__)

# Regime -> allowed strategy names
REGIME_ALLOWED = {
    0: ["s1_funding_arb", "s2_maker_mr", "s3_pairs_kalman"],  # CALM: all
    1: ["s2_maker_mr", "s3_pairs_kalman"],                      # NORMAL: no S1 unless high funding
    2: [],                                                        # EXPLOSION: none
}

# In NORMAL regime, allow S1 if funding exceeds this threshold
S1_NORMAL_THRESHOLD = 50.0  # annual %


class RegimeRouter(BaseStrategy):
    """
    Meta-strategy that wraps S1/S2/S3 and gates signals by regime.
    In EXPLOSION regime, emits 'close' signals for all open positions.
    """

    NAME = "s4_regime_router"

    DEFAULT_PARAMS = {
        "n_regimes": 3,
        "hysteresis_steps": 8,
        "min_duration_candles": 8,
        "confidence_threshold": 0.7,
        "gmm_retrain_bars": 2304,
        "s1_normal_threshold": S1_NORMAL_THRESHOLD,
        "explosion_close_all": True,
    }

    PARAM_GRID = []  # Regime router doesn't need param grid; sub-strategies have their own

    def __init__(self, params=None, config: Optional[dict] = None):
        super().__init__(params)
        self.config = config or {}

        # Sub-strategies (will be initialised with validated params from config)
        s_cfg = self.config.get("strategies", {})
        self.s1 = FundingArbStrategy(s_cfg.get("s1_funding_arb", {}))
        self.s2 = MakerMRStrategy(s_cfg.get("s2_maker_mr", {}))
        self.s3 = PairsKalmanStrategy(s_cfg.get("s3_pairs_kalman", {}))

        s4_cfg = self.config.get("strategies", {}).get("s4_regime_router", {})
        self.detector = RegimeDetector(
            n_regimes=self.params.get("n_regimes", 3),
            hysteresis_steps=self.params.get("hysteresis_steps", 8),
            min_duration_candles=s4_cfg.get(
                "min_duration_candles", self.params.get("min_duration_candles", 8)),
            confidence_threshold=s4_cfg.get(
                "confidence_threshold", self.params.get("confidence_threshold", 0.7)),
        )
        self._detector_fitted = False
        self._last_regime = 1  # Default NORMAL

        # Build per-regime allowed lists from config (overrides hardcoded defaults)
        self._regime_allowed = dict(REGIME_ALLOWED)
        if s4_cfg:
            if "calm_strategies" in s4_cfg:
                self._regime_allowed[0] = s4_cfg["calm_strategies"]
            if "normal_strategies" in s4_cfg:
                self._regime_allowed[1] = s4_cfg["normal_strategies"]
            if "explosion_strategies" in s4_cfg:
                self._regime_allowed[2] = s4_cfg["explosion_strategies"]

    def fit_regime_detector(self,
                             candles: pd.DataFrame,
                             funding_df: Optional[pd.DataFrame] = None):
        """Fit/retrain the GMM on historical data."""
        self.detector.fit(candles, funding_df)
        self._detector_fitted = True
        log.info("Regime detector fitted on %d candles", len(candles))

    def current_regime(self, recent_candles: pd.DataFrame,
                       funding_df: Optional[pd.DataFrame] = None) -> int:
        if not self._detector_fitted:
            return 1  # NORMAL default
        regime = self.detector.current_regime(recent_candles, funding_df)
        if regime != self._last_regime:
            log.info("Regime change: %s -> %s",
                     REGIME_NAMES.get(self._last_regime, "?"),
                     REGIME_NAMES.get(regime, "?"))
            self._last_regime = regime
        return regime

    def backtest(self,
                 candles: pd.DataFrame,
                 params: Optional[dict] = None) -> list[float]:
        """
        Backtest regime router by running S2 (most backtestable single-asset strategy)
        with regime filtering. S1/S3 require multi-asset data.
        """
        p = {**self.params, **(params or {})}
        if candles.empty:
            return []

        # Fit regime detector on first 60% of data
        n = len(candles)
        fit_end = int(n * 0.6)

        if fit_end > 50:
            self.detector.fit(candles.iloc[:fit_end])
            self._detector_fitted = True

        # Get regime predictions for full dataset
        regimes = self.detector.predict(candles)

        # Run S2 with regime filter: only trade in CALM/NORMAL (regime 0, 1)
        trades = []
        closes = candles["close"].values.astype(float)
        highs  = candles["high"].values.astype(float)
        lows   = candles["low"].values.astype(float)

        bb_p   = self.s2.params.get("bb_period", 20)
        bb_std = self.s2.params.get("bb_std", 2.0)
        rsi_p  = self.s2.params.get("rsi_period", 14)
        atr_p  = self.s2.params.get("atr_period", 14)
        stop_m = self.s2.params.get("stop_atr_mult", 2.0)
        pos_pct = self.s2.params.get("position_pct", 0.15)
        rsi_os = self.s2.params.get("rsi_oversold", 35)
        rsi_ob = self.s2.params.get("rsi_overbought", 65)

        capital  = 500.0
        pos_size = capital * pos_pct

        bb_upper, bb_mid, bb_lower = self.s2._bollinger(closes, bb_p, bb_std)
        rsi  = self.s2._rsi(closes, rsi_p)
        atr  = self.s2._atr(highs, lows, closes, atr_p)
        bb_width = (bb_upper - bb_lower) / np.where(bb_mid > 0, bb_mid, 1)

        warmup   = max(bb_p, rsi_p, atr_p) + 2
        in_trade = False
        side_str = "long"
        entry_p  = 0.0
        stop_p   = 0.0
        tp_p     = 0.0

        for i in range(warmup, n):
            regime = regimes[i]

            # Close position if entering EXPLOSION
            if in_trade and regime == 2:
                pnl_pct = ((closes[i] - entry_p) / entry_p
                           if side_str == "long"
                           else (entry_p - closes[i]) / entry_p)
                fee = pos_size * 3.8 / 10_000
                trades.append(pnl_pct * pos_size - fee)
                in_trade = False
                continue

            if not in_trade:
                if regime == 2:  # No entries in EXPLOSION
                    continue
                if bb_width[i] < self.s2.params.get("min_bb_width_pct", 0.002):
                    continue
                if closes[i] < bb_lower[i] and rsi[i] < rsi_os:
                    in_trade = True; side_str = "long"
                    entry_p = closes[i]; stop_p = entry_p - stop_m * atr[i]; tp_p = bb_mid[i]
                elif closes[i] > bb_upper[i] and rsi[i] > rsi_ob:
                    in_trade = True; side_str = "short"
                    entry_p = closes[i]; stop_p = entry_p + stop_m * atr[i]; tp_p = bb_mid[i]
            else:
                if side_str == "long":
                    hit_stop = lows[i] <= stop_p
                    hit_tp   = highs[i] >= tp_p
                    pnl_pct  = (closes[i] - entry_p) / entry_p
                else:
                    hit_stop = highs[i] >= stop_p
                    hit_tp   = lows[i] <= tp_p
                    pnl_pct  = (entry_p - closes[i]) / entry_p

                if hit_stop or hit_tp:
                    if hit_stop and not hit_tp:
                        actual_exit = stop_p
                        if side_str == "long":
                            pnl_pct = (actual_exit - entry_p) / entry_p
                        else:
                            pnl_pct = (entry_p - actual_exit) / entry_p
                    fee = pos_size * 3.8 / 10_000
                    trades.append(pnl_pct * pos_size - fee)
                    in_trade = False

        return trades

    def signal(self,
               candles: pd.DataFrame,
               funding_df: Optional[pd.DataFrame],
               params: Optional[dict],
               equity: float,
               open_positions: list[dict]) -> list[Signal]:
        """
        Generate signals, filtered by current regime.
        In EXPLOSION: emit close signals for all open positions.
        """
        p = {**self.params, **(params or {})}
        regime = self.current_regime(candles, funding_df)
        signals = []

        if regime == 2:  # EXPLOSION
            if p.get("explosion_close_all", True):
                for pos in open_positions:
                    signals.append(Signal(
                        symbol=pos.get("symbol", "BTC"),
                        side="close",
                        size_usd=pos.get("size_usd", 0),
                        entry_type="taker",  # Use taker for fast exit in explosion
                        meta={"strategy": self.NAME, "reason": "explosion_regime"}
                    ))
            log.warning("EXPLOSION regime: no new entries, closing all positions")
            return signals

        allowed = self._regime_allowed.get(regime, [])

        if "s1_funding_arb" in allowed and self.s1.params.get("enabled", True):
            s1_sigs = self.s1.signal(candles, funding_df, None, equity, open_positions)
            signals.extend(s1_sigs)

        if "s2_maker_mr" in allowed and self.s2.params.get("enabled", True):
            s2_sigs = self.s2.signal(candles, funding_df, None, equity, open_positions)
            signals.extend(s2_sigs)

        if "s3_pairs_kalman" in allowed and self.s3.params.get("enabled", True):
            s3_sigs = self.s3.signal(candles, funding_df, None, equity, open_positions)
            signals.extend(s3_sigs)

        if signals:
            log.info("Regime %s: %d signals generated", REGIME_NAMES.get(regime, "?"), len(signals))

        return signals


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rng = np.random.default_rng(42)
    n = 5000
    ts = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")

    # Simulate regime changes
    vol = np.ones(n)
    vol[1000:1200] = 5.0  # explosion
    vol[2500:2700] = 5.0

    close = 100 + np.cumsum(rng.normal(0, 0.2, n) * vol)
    candles = pd.DataFrame({
        "ts": ts, "open": close, "high": close + abs(rng.normal(0, 0.3, n)) * vol,
        "low": close - abs(rng.normal(0, 0.3, n)) * vol,
        "close": close, "volume": rng.uniform(100, 1000, n) * vol,
    })

    router = RegimeRouter()
    trades = router.backtest(candles)
    print(f"Trades (S2 + regime filter): {len(trades)}")
    if trades:
        pnl = np.array(trades)
        print(f"Win rate: {(pnl > 0).mean()*100:.1f}%")
        print(f"Total P&L: ${pnl.sum():.2f}")
