"""
lstm_filter.py — Daily directional bias filter (CONSERVATIVE USE ONLY)
Based on Islas & García-Medina 2025. Used EXCLUSIVELY as a gating filter,
NOT as a primary signal.

Usage rules (from v7.5 spec):
  - P(up) > 0.55  -> allow longs, attenuate shorts (0.5x short multiplier)
  - P(up) < 0.45  -> attenuate longs (0.5x), allow shorts
  - 0.45-0.55     -> neutral: do not filter any direction

Deployment gate (mandatory before use):
  - Brier score on OOS data must be < 0.245 (better than random 50/50)
  - OOS accuracy must be >= 53%
  - Walk-forward on 90 days minimum

If these are not met: lstm_filter is bypassed (neutral mode).
"""
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logging.warning("torch not available — LSTM filter disabled")

log = logging.getLogger(__name__)

# Thresholds
LONG_THRESHOLD  = 0.55   # P(up) above this -> bias longs
SHORT_THRESHOLD = 0.45   # P(up) below this -> bias shorts

# Features computed from daily OHLCV + macro data
N_FEATURES = 10
LOOKBACK_DAYS = 30       # LSTM input window


def compute_daily_features(btc_ohlcv: pd.DataFrame,
                            eth_ohlcv: pd.DataFrame,
                            btc_funding: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Compute 10 daily features for LSTM input.

    Args:
        btc_ohlcv: Daily BTC OHLCV (columns: open, high, low, close, volume)
        eth_ohlcv: Daily ETH OHLCV
        btc_funding: Daily average funding (from hourly funding history)

    Returns:
        DataFrame with N_FEATURES columns, one row per day.
        NaN rows at the start (warmup).
    """
    if btc_ohlcv.empty or len(btc_ohlcv) < 8:
        return pd.DataFrame()

    df = pd.DataFrame(index=btc_ohlcv.index)

    btc_close = btc_ohlcv["close"].astype(float)
    eth_close = eth_ohlcv["close"].astype(float) if not eth_ohlcv.empty else btc_close

    # 1. BTC 1-day return
    df["btc_ret_1d"] = btc_close.pct_change(1)

    # 2. BTC 7-day return
    df["btc_ret_7d"] = btc_close.pct_change(7)

    # 3. BTC 30-day realised volatility (annualised)
    df["btc_vol_30d"] = btc_close.pct_change().rolling(30).std() * np.sqrt(365)

    # 4. ETH 1-day return
    df["eth_ret_1d"] = eth_close.pct_change(1)

    # 5. BTC/ETH 30-day correlation
    btc_ret = btc_close.pct_change()
    eth_ret = eth_close.pct_change()
    df["btc_eth_corr_30d"] = btc_ret.rolling(30).corr(eth_ret)

    # 6. BTC average daily funding (annualised %)
    if btc_funding is not None and not btc_funding.empty:
        # Resample hourly -> daily mean
        try:
            daily_fund = btc_funding.set_index("ts")["annual_pct"].resample("1D").mean()
            daily_fund = daily_fund.reindex(df.index, method="ffill")
            df["btc_funding_avg"] = daily_fund.values
        except Exception:
            df["btc_funding_avg"] = 0.0
    else:
        df["btc_funding_avg"] = 0.0

    # 7. Volume ratio (today vs 7-day avg)
    btc_vol = btc_ohlcv["volume"].astype(float)
    df["btc_vol_ratio"] = btc_vol / btc_vol.rolling(7).mean()

    # 8. BTC RSI (14-day)
    delta = btc_close.diff()
    gain  = delta.clip(lower=0).ewm(com=13).mean()
    loss  = (-delta).clip(lower=0).ewm(com=13).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["btc_rsi_14"] = (100 - 100 / (1 + rs)).fillna(50) / 100.0  # normalise to [0,1]

    # 9. BTC price relative to 20-day SMA (trend proxy)
    sma20 = btc_close.rolling(20).mean()
    df["btc_vs_sma20"] = (btc_close - sma20) / sma20

    # 10. High-low range ratio (intra-day volatility indicator)
    df["btc_hl_ratio"] = (btc_ohlcv["high"].astype(float) - btc_ohlcv["low"].astype(float)) / btc_close

    # Fill NaN with 0 (LSTM handles this better than NaN)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


class DailyDirectionFilter(nn.Module if HAS_TORCH else object):
    """
    Minimal LSTM: 10 features, 30-day lookback, hidden=32, output=P(up).
    Trained offline, loaded at runtime.
    """

    def __init__(self, n_features: int = N_FEATURES, hidden: int = 32):
        if HAS_TORCH:
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, num_layers=1, batch_first=True)
            self.fc   = nn.Linear(hidden, 1)
            self.sig  = nn.Sigmoid()

    def forward(self, x):
        if not HAS_TORCH:
            return None
        out, _ = self.lstm(x)
        return self.sig(self.fc(out[:, -1, :]))


class LSTMDirectionFilter:
    """
    Runtime wrapper for the LSTM filter.
    Loads a trained model or operates in neutral pass-through mode.

    Usage:
        filt = LSTMDirectionFilter(model_path="models/lstm_filter.pkl")
        filt.load()
        mult = filt.get_multipliers(daily_features_df)
        # mult = {"long_mult": 0.5-1.0, "short_mult": 0.5-1.0}
    """

    def __init__(self,
                 model_path: str = "models/lstm_filter.pt",
                 features_path: str = "models/lstm_scaler.pkl"):
        self.model_path    = Path(model_path)
        self.features_path = Path(features_path)
        self.model         = None
        self.scaler        = None
        self._trained      = False
        self._validated    = False   # Must pass Brier/accuracy gate before use

    def load(self) -> bool:
        """Load trained model from disk. Returns True if successful."""
        if not HAS_TORCH:
            log.warning("torch not available, LSTM filter in neutral mode")
            return False
        if not self.model_path.exists():
            log.info("LSTM model not found at %s — neutral mode", self.model_path)
            return False
        try:
            self.model = DailyDirectionFilter()
            self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
            self.model.eval()
            if self.features_path.exists():
                with open(self.features_path, "rb") as f:
                    self.scaler = pickle.load(f)
            self._trained = True
            log.info("LSTM filter loaded from %s", self.model_path)
            return True
        except Exception as e:
            log.error("LSTM load failed: %s", e)
            return False

    def predict_proba(self, features_df: pd.DataFrame) -> Optional[float]:
        """
        Return P(up) for next 24h given the last LOOKBACK_DAYS rows.
        Returns None if model not loaded or not validated.
        """
        if not self._trained or not self._validated:
            return None
        if not HAS_TORCH or self.model is None:
            return None
        if len(features_df) < LOOKBACK_DAYS:
            return None

        try:
            window = features_df.tail(LOOKBACK_DAYS).values.astype(np.float32)
            if self.scaler:
                window = self.scaler.transform(window)
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                prob = float(self.model(x).squeeze())
            return prob
        except Exception as e:
            log.error("LSTM predict failed: %s", e)
            return None

    def get_multipliers(self, features_df: pd.DataFrame) -> dict:
        """
        Return {"long_mult": float, "short_mult": float} based on P(up).
        If model unavailable / not validated: both = 1.0 (neutral).
        """
        prob = self.predict_proba(features_df)
        return _apply_filter(prob)

    def validate_and_enable(self, features_df: pd.DataFrame,
                             labels: np.ndarray) -> dict:
        """
        Run OOS validation. Must pass Brier score < 0.245 AND accuracy >= 53%.
        Enables the filter only if both criteria pass.

        Args:
            features_df: Full features DataFrame
            labels: Binary labels (1=up, 0=down) aligned to features_df

        Returns:
            {"brier": float, "accuracy": float, "enabled": bool}
        """
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import brier_score_loss, accuracy_score

        if not self._trained:
            return {"brier": 1.0, "accuracy": 0.0, "enabled": False}

        probas = []
        for i in range(LOOKBACK_DAYS, len(features_df)):
            window = features_df.iloc[i - LOOKBACK_DAYS:i]
            p = self.predict_proba(window)
            probas.append(p if p is not None else 0.5)

        probas = np.array(probas)
        y_true = np.array(labels[LOOKBACK_DAYS:])
        y_pred = (probas > 0.5).astype(int)

        brier   = float(brier_score_loss(y_true, probas))
        acc     = float(accuracy_score(y_true, y_pred))
        enabled = brier < 0.245 and acc >= 0.53

        log.info(
            "LSTM validation: Brier=%.4f (< 0.245), Accuracy=%.1f%% (>= 53%%) -> %s",
            brier, acc * 100, "ENABLED" if enabled else "DISABLED"
        )

        self._validated = enabled
        return {"brier": brier, "accuracy": acc, "enabled": enabled}

    def train(self,
              features_df: pd.DataFrame,
              labels: np.ndarray,
              epochs: int = 50,
              lr: float = 1e-3,
              val_split: float = 0.20) -> dict:
        """
        Train the LSTM on historical daily data.
        Returns training/validation loss history.

        Args:
            features_df: Feature matrix (rows = days)
            labels: Binary target (1=next day up, 0=down)
            epochs: Training epochs
            lr: Learning rate
            val_split: Fraction held out for validation

        Note: This should be run OFFLINE, not in the live engine.
        """
        if not HAS_TORCH:
            log.error("torch not available, cannot train LSTM")
            return {}

        from sklearn.preprocessing import StandardScaler

        assert len(features_df) == len(labels), "features and labels must be same length"

        n = len(features_df) - LOOKBACK_DAYS
        if n < 60:
            log.error("Too few samples (%d) to train LSTM meaningfully", n)
            return {}

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(features_df.values.astype(np.float32))

        # Build sequences
        X_seqs = np.array([X_scaled[i:i + LOOKBACK_DAYS] for i in range(n)])
        y_arr  = np.array(labels[LOOKBACK_DAYS:], dtype=np.float32)

        # Train/val split (temporal, not random)
        val_n   = int(n * val_split)
        X_train, X_val = X_seqs[:-val_n], X_seqs[-val_n:]
        y_train, y_val = y_arr[:-val_n],  y_arr[-val_n:]

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
        y_val_t   = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1)

        self.model = DailyDirectionFilter(N_FEATURES)
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.BCELoss()

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            self.model.train()
            optim.zero_grad()
            preds = self.model(X_train_t)
            loss  = loss_fn(preds, y_train_t)
            loss.backward()
            optim.step()

            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_val_t)
                val_loss  = loss_fn(val_preds, y_val_t)

            history["train_loss"].append(float(loss))
            history["val_loss"].append(float(val_loss))

            if (epoch + 1) % 10 == 0:
                log.info("Epoch %d/%d: train=%.4f val=%.4f",
                         epoch + 1, epochs, float(loss), float(val_loss))

        self._trained = True

        # Auto-validate after training
        with torch.no_grad():
            val_proba = self.model(X_val_t).squeeze().numpy()
        val_acc = float(((val_proba > 0.5) == y_val.astype(bool)).mean())
        log.info("Training complete. Val accuracy: %.1f%%", val_acc * 100)

        return history

    def save(self):
        """Persist model and scaler to disk."""
        if not HAS_TORCH or self.model is None:
            return
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        if self.scaler:
            with open(self.features_path, "wb") as f:
                pickle.dump(self.scaler, f)
        log.info("LSTM filter saved to %s", self.model_path)


def _apply_filter(prob: Optional[float]) -> dict:
    """Convert P(up) probability to sizing multipliers."""
    if prob is None:
        return {"long_mult": 1.0, "short_mult": 1.0, "prob_up": None, "mode": "neutral"}

    if prob > LONG_THRESHOLD:
        return {"long_mult": 1.0, "short_mult": 0.5,
                "prob_up": prob, "mode": "bullish"}
    elif prob < SHORT_THRESHOLD:
        return {"long_mult": 0.5, "short_mult": 1.0,
                "prob_up": prob, "mode": "bearish"}
    else:
        return {"long_mult": 1.0, "short_mult": 1.0,
                "prob_up": prob, "mode": "neutral"}


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not HAS_TORCH:
        print("torch not available")
        sys.exit(0)

    # Synthetic test: train on random data
    rng = np.random.default_rng(42)
    n_days = 200

    ts = pd.date_range("2024-01-01", periods=n_days, freq="1D")
    btc_close = 40000 + np.cumsum(rng.normal(50, 500, n_days))
    eth_close = 2500  + np.cumsum(rng.normal(3,  40,  n_days))

    btc_df = pd.DataFrame({
        "open": btc_close * 0.995, "high": btc_close * 1.01,
        "low": btc_close * 0.99,   "close": btc_close,
        "volume": rng.uniform(1e9, 3e9, n_days),
    }, index=ts)

    eth_df = pd.DataFrame({
        "open": eth_close, "high": eth_close * 1.01,
        "low": eth_close * 0.99, "close": eth_close,
        "volume": rng.uniform(5e8, 1e9, n_days),
    }, index=ts)

    features = compute_daily_features(btc_df, eth_df)
    print(f"Features shape: {features.shape}")
    print(f"Columns: {list(features.columns)}")

    # Labels: 1 if tomorrow's BTC return > 0
    labels = (btc_close[1:] > btc_close[:-1]).astype(int)
    labels = np.append(labels, 0)  # pad last day

    # Train
    filt = LSTMDirectionFilter(model_path="models/test_lstm.pt",
                                features_path="models/test_scaler.pkl")
    history = filt.train(features, labels, epochs=30)
    print(f"\nFinal val loss: {history['val_loss'][-1]:.4f}")

    # Validate
    result = filt.validate_and_enable(features, labels)
    print(f"Validation: {result}")

    # Get multipliers for last day
    mults = filt.get_multipliers(features)
    print(f"Multipliers: {mults}")
