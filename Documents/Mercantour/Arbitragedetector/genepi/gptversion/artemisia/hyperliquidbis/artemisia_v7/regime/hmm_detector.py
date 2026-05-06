"""
hmm_detector.py — Market regime detection via Gaussian Mixture Model with hysteresis.
hmmlearn is NOT available; using sklearn GaussianMixture + temporal hysteresis.

3 regimes:
  0 = CALM     (low vol, trending funding, range-bound)
  1 = NORMAL   (moderate vol, directional)
  2 = EXPLOSION (high vol, spike, liquidation cascade)
"""
import logging
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

REGIME_NAMES = {0: "CALM", 1: "NORMAL", 2: "EXPLOSION"}
N_REGIMES = 3
HYSTERESIS_STEPS = 3   # Must see same regime for N steps before switching


def _compute_features(candles: pd.DataFrame,
                      funding_df: pd.DataFrame | None = None,
                      window: int = 20) -> pd.DataFrame:
    """
    Compute regime features from candle data.

    Features:
    - vol_norm: normalised rolling volatility (ATR / close)
    - volume_ratio: current volume / rolling mean volume
    - price_trend: signed trend strength (close vs EMA)
    - hl_range: high-low range ratio (spike indicator)
    - funding_premium: absolute funding premium (if available)
    """
    df = candles.copy()
    closes = df["close"].values.astype(float)
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    vols   = df["volume"].values.astype(float)

    n = len(df)
    feat = pd.DataFrame(index=df.index)

    # 1. Normalised ATR
    tr = np.maximum(highs - lows,
           np.maximum(abs(highs - np.roll(closes, 1)),
                      abs(lows  - np.roll(closes, 1))))
    tr[0] = highs[0] - lows[0]
    atr = pd.Series(tr).ewm(span=window, adjust=False).mean().values
    feat["vol_norm"] = atr / np.where(closes > 0, closes, 1)

    # 2. Volume ratio
    vol_ma = pd.Series(vols).rolling(window, min_periods=1).mean().values
    feat["volume_ratio"] = np.where(vol_ma > 0, vols / vol_ma, 1.0)

    # 3. Price trend: (close - EMA) / ATR
    ema = pd.Series(closes).ewm(span=window, adjust=False).mean().values
    feat["price_trend"] = (closes - ema) / np.where(atr > 0, atr, 1)

    # 4. High-low spike indicator: (H-L)/close relative to ATR
    hl = (highs - lows) / np.where(closes > 0, closes, 1)
    feat["hl_range"] = hl / feat["vol_norm"].values

    # 5. Funding premium (if available)
    if funding_df is not None and not funding_df.empty:
        # Align funding to candle timestamps
        funding_df = funding_df.set_index("ts").reindex(df["ts"], method="ffill")
        feat["funding_premium"] = funding_df["premium"].abs().values
    else:
        feat["funding_premium"] = 0.0

    # Drop warmup rows
    feat = feat.iloc[window:].copy()
    # Replace inf/nan
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feat


class RegimeDetector:
    """
    GMM-based regime detector with hysteresis smoothing.

    Usage:
        det = RegimeDetector()
        det.fit(candles_df, funding_df)
        regimes = det.predict(candles_df)
        current = det.current_regime(candles_df.tail(50))
    """

    def __init__(self,
                 n_regimes: int = N_REGIMES,
                 hysteresis_steps: int = HYSTERESIS_STEPS,
                 min_duration_candles: int = 8,
                 confidence_threshold: float = 0.7,
                 feature_window: int = 20,
                 random_state: int = 42):
        self.n_regimes = n_regimes
        self.hysteresis_steps = hysteresis_steps
        self.min_duration_candles = min_duration_candles
        self.confidence_threshold = confidence_threshold
        self.feature_window = feature_window
        self.random_state = random_state

        self.gmm: GaussianMixture | None = None
        self.scaler: StandardScaler | None = None
        self._regime_labels: dict = {}   # GMM cluster -> CALM/NORMAL/EXPLOSION
        self._last_regime: int = 1       # NORMAL by default
        self._regime_buffer: list = []   # For batch hysteresis (predict())
        # Stateful sticky state for live current_regime() calls
        self._pending_regime: int | None = None
        self._pending_count: int = 0

    def fit(self,
            candles: pd.DataFrame,
            funding_df: pd.DataFrame | None = None) -> "RegimeDetector":
        """
        Fit GMM on historical candle data.
        Also assigns regime labels (low-vol cluster -> CALM, high-vol -> EXPLOSION).
        """
        feat = _compute_features(candles, funding_df, self.feature_window)

        if len(feat) < self.n_regimes * 10:
            log.warning("Too few feature rows (%d) to fit GMM", len(feat))
            return self

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(feat.values)

        self.gmm = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type="full",
            n_init=5,
            random_state=self.random_state,
        )
        self.gmm.fit(X)

        # Assign semantic labels based on vol_norm feature (index 0)
        # The cluster with lowest mean vol_norm = CALM
        # Highest = EXPLOSION
        means_vol = self.gmm.means_[:, 0]  # vol_norm column (after scaling, order preserved)
        sorted_idx = np.argsort(means_vol)  # ascending vol order
        if self.n_regimes == 3:
            self._regime_labels = {
                sorted_idx[0]: 0,  # CALM
                sorted_idx[1]: 1,  # NORMAL
                sorted_idx[2]: 2,  # EXPLOSION
            }
        else:
            self._regime_labels = {i: i for i in range(self.n_regimes)}

        log.info("GMM fitted on %d samples (%d regimes). Labels: %s",
                 len(X), self.n_regimes, self._regime_labels)
        # Reset sticky state after refit so we don't carry stale pending counts
        self._pending_regime = None
        self._pending_count = 0
        return self

    def predict(self,
                candles: pd.DataFrame,
                funding_df: pd.DataFrame | None = None,
                apply_hysteresis: bool = True) -> np.ndarray:
        """
        Predict regime for each candle bar (after warmup).
        Returns array of regime ints aligned to candles (NaN for warmup rows).
        """
        if self.gmm is None:
            log.warning("GMM not fitted, returning NORMAL regime")
            return np.ones(len(candles), dtype=int)

        feat = _compute_features(candles, funding_df, self.feature_window)
        if feat.empty:
            return np.ones(len(candles), dtype=int)

        X = self.scaler.transform(feat.values)
        raw_labels = self.gmm.predict(X)

        # Map to semantic labels
        semantic = np.array([self._regime_labels.get(r, 1) for r in raw_labels])

        if apply_hysteresis:
            semantic = self._apply_hysteresis(semantic)

        # Pad warmup rows with NORMAL
        padding = np.ones(len(candles) - len(semantic), dtype=int)
        return np.concatenate([padding, semantic])

    def current_regime(self,
                       recent_candles: pd.DataFrame,
                       funding_df: pd.DataFrame | None = None) -> int:
        """
        Stateful live regime update. Safe to call on every tick (every 15s).

        Processes only the most recent candle bar. Requires min_duration_candles
        consecutive bars pointing to the same new regime AND confidence >=
        confidence_threshold before switching. This prevents the HMM from
        oscillating on tick-level noise.

        Typical behaviour: at most 1 regime change per 2h (8 × 15m bars).
        """
        if self.gmm is None or self.scaler is None:
            return self._last_regime

        feat = _compute_features(recent_candles, funding_df, self.feature_window)
        if feat.empty:
            return self._last_regime

        # Score only the last bar
        X_last = self.scaler.transform(feat.values[-1:])
        probas = self.gmm.predict_proba(X_last)[0]
        most_likely_raw = int(np.argmax(probas))
        most_likely = self._regime_labels.get(most_likely_raw, 1)
        confidence = float(probas[most_likely_raw])

        # Low confidence — stay in current regime, reset any pending switch
        if confidence < self.confidence_threshold:
            self._pending_regime = None
            self._pending_count = 0
            return self._last_regime

        # Confirmed in current regime — reset pending switch
        if most_likely == self._last_regime:
            self._pending_regime = None
            self._pending_count = 0
            return self._last_regime

        # Accumulate confirmation count for candidate new regime
        if most_likely == self._pending_regime:
            self._pending_count += 1
        else:
            self._pending_regime = most_likely
            self._pending_count = 1

        # Commit switch only after min_duration_candles consecutive confirmations
        if self._pending_count >= self.min_duration_candles:
            old = self._last_regime
            self._last_regime = self._pending_regime
            self._pending_regime = None
            self._pending_count = 0
            log.info(
                "Regime STABLE change: %s -> %s "
                "(confidence=%.2f, confirmed over %d candles)",
                REGIME_NAMES.get(old, "?"),
                REGIME_NAMES.get(self._last_regime, "?"),
                confidence,
                self.min_duration_candles,
            )

        return self._last_regime

    def current_regime_name(self, recent_candles: pd.DataFrame,
                            funding_df: pd.DataFrame | None = None) -> str:
        return REGIME_NAMES.get(self.current_regime(recent_candles, funding_df), "UNKNOWN")

    def _apply_hysteresis(self, labels: np.ndarray) -> np.ndarray:
        """
        Apply temporal hysteresis: only switch regime if the new regime
        persists for at least `hysteresis_steps` consecutive bars.
        """
        if len(labels) == 0:
            return labels

        smoothed = labels.copy()
        current = labels[0]
        buffer = [labels[0]]

        for i in range(1, len(labels)):
            buffer.append(labels[i])
            if len(buffer) > self.hysteresis_steps:
                buffer.pop(0)

            # Switch if last N bars all agree on a new regime
            if len(buffer) == self.hysteresis_steps and len(set(buffer)) == 1:
                current = buffer[0]
            smoothed[i] = current

        return smoothed

    def save(self, path: str):
        """Serialize fitted detector."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info("RegimeDetector saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "RegimeDetector":
        with open(path, "rb") as f:
            det = pickle.load(f)
        log.info("RegimeDetector loaded from %s", path)
        return det

    def strategy_allowed(self, regime: int, strategy_name: str,
                         config: dict | None = None) -> bool:
        """
        Check if a strategy is allowed in the current regime.
        Based on config.json strategies.s4_regime_router settings.
        """
        if config is None:
            # Default: all strategies allowed in CALM/NORMAL, none in EXPLOSION
            if regime == 2:  # EXPLOSION
                return False
            return True

        router_cfg = config.get("strategies", {}).get("s4_regime_router", {})
        regime_map = {
            0: router_cfg.get("calm_strategies", []),
            1: router_cfg.get("normal_strategies", []),
            2: router_cfg.get("explosion_strategies", []),
        }
        allowed = regime_map.get(regime, [])
        return strategy_name in allowed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Generate synthetic candles for testing
    n = 5000
    rng = np.random.default_rng(42)
    ts = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")

    # Simulate 3 regimes
    regime_seq = np.zeros(n, dtype=int)
    regime_seq[1000:1500] = 2  # EXPLOSION
    regime_seq[2000:2500] = 2
    regime_seq[3500:4000] = 1  # NORMAL

    close = 100 + np.cumsum(rng.normal(0, 0.1, n))
    vol_mult = np.where(regime_seq == 2, 5.0, np.where(regime_seq == 1, 2.0, 1.0))
    high = close + abs(rng.normal(0, 0.5, n)) * vol_mult
    low  = close - abs(rng.normal(0, 0.5, n)) * vol_mult
    volume = 1000 * vol_mult * (1 + rng.uniform(0, 0.5, n))

    candles = pd.DataFrame({
        "ts": ts, "open": close, "high": high, "low": low,
        "close": close, "volume": volume,
    })

    det = RegimeDetector(n_regimes=3, hysteresis_steps=3)
    det.fit(candles)
    preds = det.predict(candles)

    print("Predicted regime distribution:")
    for r, name in REGIME_NAMES.items():
        count = np.sum(preds == r)
        print(f"  {name}: {count} bars ({count/len(preds)*100:.1f}%)")

    print(f"\nCurrent regime: {det.current_regime_name(candles.tail(100))}")
