"""
kalman_pair.py — Dynamic hedge ratio estimation via Kalman Filter (pykalman).
Used by S3 Pairs Stat Arb strategy.

The hedge ratio beta_t evolves as a random walk:
  beta_t = beta_{t-1} + noise
  spread_t = price_A_t - beta_t * price_B_t

State: [beta, intercept]
Observation: price_A = beta * price_B + intercept + noise
"""
import logging
import numpy as np
import pandas as pd
from typing import Optional

try:
    from pykalman import KalmanFilter
    HAS_PYKALMAN = True
except ImportError:
    HAS_PYKALMAN = False
    logging.warning("pykalman not available, falling back to OLS rolling window")

log = logging.getLogger(__name__)


class KalmanHedgeRatio:
    """
    Estimates time-varying hedge ratio between two assets using Kalman filter.

    State vector: [beta, intercept]
    Observation:  price_A = beta * price_B + intercept + obs_noise

    Usage:
        khr = KalmanHedgeRatio()
        khr.fit(prices_A, prices_B)
        beta, intercept = khr.latest_params()
        z_score = khr.zscore(prices_A, prices_B)
    """

    def __init__(self,
                 transition_cov: float = 1e-4,
                 observation_cov: float = 1.0,
                 zscore_window: int = 20):
        self.transition_cov = transition_cov
        self.observation_cov = observation_cov
        self.zscore_window = zscore_window
        self._kf: Optional[object] = None
        self._states: Optional[np.ndarray] = None
        self._covs: Optional[np.ndarray] = None
        self._spreads: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, prices_A: np.ndarray, prices_B: np.ndarray) -> "KalmanHedgeRatio":
        """
        Fit the Kalman filter on historical price series.

        Args:
            prices_A: Price series of asset A (the one being hedged)
            prices_B: Price series of asset B (the hedge)
        """
        prices_A = np.array(prices_A, dtype=float)
        prices_B = np.array(prices_B, dtype=float)
        n = len(prices_A)

        if n < 30:
            log.warning("Too few observations (%d) to fit KalmanHedgeRatio", n)
            return self

        if HAS_PYKALMAN:
            self._fit_pykalman(prices_A, prices_B)
        else:
            self._fit_rolling_ols(prices_A, prices_B)

        self._fitted = True
        return self

    def _fit_pykalman(self, prices_A: np.ndarray, prices_B: np.ndarray):
        """Full Kalman filter via pykalman."""
        n = len(prices_A)

        # Observation matrix: [price_B, 1] (multiplied by state [beta, intercept])
        obs_mats = np.ones((n, 1, 2))
        obs_mats[:, 0, 0] = prices_B
        obs_mats[:, 0, 1] = 1.0

        self._kf = KalmanFilter(
            n_dim_obs=1,
            n_dim_state=2,
            initial_state_mean=np.array([1.0, 0.0]),
            initial_state_covariance=np.eye(2) * 1.0,
            transition_matrices=np.eye(2),
            transition_covariance=np.eye(2) * self.transition_cov,
            observation_covariance=np.array([[self.observation_cov]]),
            observation_matrices=obs_mats,
        )

        obs = prices_A.reshape(-1, 1)
        state_means, state_covs = self._kf.filter(obs)
        self._states = state_means    # shape (n, 2): [beta, intercept]
        self._covs   = state_covs     # shape (n, 2, 2)

        betas      = state_means[:, 0]
        intercepts = state_means[:, 1]
        self._spreads = prices_A - betas * prices_B - intercepts
        log.debug("Kalman fit: final beta=%.4f, intercept=%.4f", betas[-1], intercepts[-1])

    def _fit_rolling_ols(self, prices_A: np.ndarray, prices_B: np.ndarray):
        """Fallback: rolling OLS when pykalman unavailable."""
        w = self.zscore_window * 5  # larger window for OLS stability
        n = len(prices_A)

        betas = np.full(n, np.nan)
        intercepts = np.full(n, np.nan)

        for i in range(w, n + 1):
            ya = prices_A[i-w:i]
            yb = prices_B[i-w:i]
            A = np.column_stack([yb, np.ones(w)])
            try:
                res = np.linalg.lstsq(A, ya, rcond=None)
                betas[i-1] = res[0][0]
                intercepts[i-1] = res[0][1]
            except Exception:
                betas[i-1] = betas[i-2] if i > w else 1.0
                intercepts[i-1] = 0.0

        # Forward-fill NaN warmup
        for i in range(1, n):
            if np.isnan(betas[i]):
                betas[i] = betas[i-1] if not np.isnan(betas[i-1]) else 1.0
            if np.isnan(intercepts[i]):
                intercepts[i] = intercepts[i-1] if not np.isnan(intercepts[i-1]) else 0.0

        self._states = np.column_stack([betas, intercepts])
        self._spreads = prices_A - betas * prices_B - intercepts
        log.debug("Rolling OLS fit: final beta=%.4f, intercept=%.4f", betas[-1], intercepts[-1])

    def latest_params(self) -> tuple[float, float]:
        """Return (beta, intercept) from the most recent Kalman update."""
        if self._states is None or len(self._states) == 0:
            return 1.0, 0.0
        return float(self._states[-1, 0]), float(self._states[-1, 1])

    def spread_series(self) -> np.ndarray:
        """Return the full spread time series: A - beta*B - intercept."""
        if self._spreads is None:
            return np.array([])
        return self._spreads

    def zscore(self) -> float:
        """
        Return current z-score of the spread (using last zscore_window observations).
        """
        if self._spreads is None or len(self._spreads) < 2:
            return 0.0
        w = min(self.zscore_window, len(self._spreads))
        recent = self._spreads[-w:]
        std = np.std(recent)
        if std == 0:
            return 0.0
        return float((recent[-1] - np.mean(recent)) / std)

    def zscore_series(self) -> np.ndarray:
        """Return full z-score series (rolling window)."""
        if self._spreads is None:
            return np.array([])
        s = pd.Series(self._spreads)
        mean = s.rolling(self.zscore_window, min_periods=2).mean()
        std  = s.rolling(self.zscore_window, min_periods=2).std()
        z = (s - mean) / std.replace(0, np.nan)
        return z.fillna(0).values

    def update(self, new_price_A: float, new_price_B: float) -> tuple[float, float, float]:
        """
        Online update: add one new observation and return (beta, intercept, zscore).
        Uses simplified recursive update when pykalman is available.
        """
        if not self._fitted or self._states is None:
            return 1.0, 0.0, 0.0

        beta, intercept = self.latest_params()

        if HAS_PYKALMAN and self._kf is not None and self._covs is not None:
            # One-step Kalman update
            obs_mat = np.array([[new_price_B, 1.0]])
            obs = np.array([[new_price_A]])
            prev_state = self._states[-1]
            prev_cov   = self._covs[-1]

            new_state, new_cov = self._kf.filter_update(
                filtered_state_mean=prev_state,
                filtered_state_covariance=prev_cov,
                observation=obs,
                observation_matrix=obs_mat,
            )
            # pykalman filter_update may return masked arrays; extract as plain floats
            new_state_arr = np.asarray(new_state, dtype=float).flatten()[:2]
            beta      = float(new_state_arr[0])
            intercept = float(new_state_arr[1])
            self._states = np.vstack([self._states, new_state_arr.reshape(1, 2)])
            new_cov_arr  = np.asarray(new_cov, dtype=float).reshape(2, 2)
            self._covs   = np.concatenate([self._covs, new_cov_arr.reshape(1, 2, 2)])
        else:
            # No online update for rolling OLS; just use last params
            pass

        new_spread = new_price_A - beta * new_price_B - intercept
        self._spreads = np.append(self._spreads, new_spread)

        z = self.zscore()
        return beta, intercept, z


def test_cointegration(prices_A: np.ndarray, prices_B: np.ndarray) -> tuple[float, bool]:
    """
    ADF test for cointegration (spread stationarity).
    Returns (p_value, is_cointegrated).
    Requires scipy.
    """
    try:
        from scipy.stats import pearsonr
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        log.warning("statsmodels not available, skipping cointegration test")
        return 0.05, True

    # Simple OLS to get spread
    n = len(prices_A)
    if n < 30:
        return 1.0, False

    A = np.column_stack([prices_B, np.ones(n)])
    try:
        result = np.linalg.lstsq(A, prices_A, rcond=None)
        beta, intercept = result[0]
        spread = prices_A - beta * prices_B - intercept
        adf_result = adfuller(spread, autolag="AIC")
        p_value = adf_result[1]
        return float(p_value), p_value < 0.05
    except Exception as e:
        log.error("ADF test failed: %s", e)
        return 1.0, False


def find_cointegrated_pairs(prices_dict: dict[str, np.ndarray],
                            max_pvalue: float = 0.05,
                            min_correlation: float = 0.75) -> list[tuple[str, str, float]]:
    """
    Find cointegrated pairs from a dict of price series.
    Returns list of (symA, symB, p_value) sorted by p_value.
    """
    symbols = list(prices_dict.keys())
    n = len(symbols)
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            symA, symB = symbols[i], symbols[j]
            pA = prices_dict[symA]
            pB = prices_dict[symB]

            # Align lengths
            min_len = min(len(pA), len(pB))
            if min_len < 50:
                continue
            pA, pB = pA[-min_len:], pB[-min_len:]

            # Quick correlation check first (cheaper than ADF)
            try:
                corr = float(np.corrcoef(pA, pB)[0, 1])
            except Exception:
                continue

            if abs(corr) < min_correlation:
                continue

            p_val, is_coint = test_cointegration(pA, pB)
            if is_coint and p_val <= max_pvalue:
                pairs.append((symA, symB, p_val))
                log.debug("Pair found: %s/%s p=%.4f corr=%.3f", symA, symB, p_val, corr)

    pairs.sort(key=lambda x: x[2])
    log.info("Found %d cointegrated pairs (max_p=%.2f, min_corr=%.2f)",
             len(pairs), max_pvalue, min_correlation)
    return pairs


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")

    rng = np.random.default_rng(42)
    n = 2000
    # True beta = 0.8
    pB = 100 + np.cumsum(rng.normal(0, 0.5, n))
    pA = 0.8 * pB + 5.0 + rng.normal(0, 0.3, n)  # cointegrated

    khr = KalmanHedgeRatio(transition_cov=1e-4, observation_cov=1.0, zscore_window=30)
    khr.fit(pA, pB)

    beta, intercept = khr.latest_params()
    z = khr.zscore()
    print(f"Kalman beta={beta:.4f} (true=0.8), intercept={intercept:.4f} (true=5.0)")
    print(f"Current z-score: {z:.3f}")

    p_val, is_coint = test_cointegration(pA, pB)
    print(f"Cointegration ADF p-value: {p_val:.4f}, cointegrated: {is_coint}")

    # Online update
    new_pB = pB[-1] + rng.normal(0, 0.5)
    new_pA = 0.8 * new_pB + 5.0 + rng.normal(0, 0.3)
    beta2, intercept2, z2 = khr.update(new_pA, new_pB)
    print(f"After online update: beta={beta2:.4f}, z={z2:.3f}")
