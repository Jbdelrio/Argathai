"""
Artemisia Glacialis v6 -- 6 New Strategy Implementations
Each strategy is a class with:
  - NAME, PARAM_GRID, DEFAULT_PARAMS
  - precompute(candles: List[dict]) -> dict   (cached per symbol)
  - signal(idx, precomp, params) -> dict|None  (called per candle)

Signal dict schema:
  strategy, side ("long"|"short"), stop_dist (fraction), tp_dist (fraction),
  max_hold (candles), meta (dict)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# ── shared indicator helpers ────────────────────────────────────────────────

def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, p: int) -> np.ndarray:
    if len(c) < 2:
        return np.zeros(len(c))
    tr_h = h[1:] - l[1:]
    tr_hl = np.abs(h[1:] - c[:-1])
    tr_lc = np.abs(l[1:] - c[:-1])
    tr = np.maximum(tr_h, np.maximum(tr_hl, tr_lc))
    tr_full = np.concatenate([[tr[0] if len(tr) > 0 else 0], tr])
    return pd.Series(tr_full).rolling(p, min_periods=1).mean().values


def _ema(c: np.ndarray, p: int) -> np.ndarray:
    return pd.Series(c).ewm(span=p, adjust=False).mean().values


def _rsi(c: np.ndarray, p: int) -> np.ndarray:
    s = pd.Series(c)
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p-1, min_periods=p).mean()
    ls = (-d).clip(lower=0).ewm(com=p-1, min_periods=p).mean()
    rs = g / ls.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50).values


def _rolling_std(c: np.ndarray, p: int) -> np.ndarray:
    return pd.Series(c).rolling(p, min_periods=2).std(ddof=1).fillna(0).values


def _rolling_mean(c: np.ndarray, p: int) -> np.ndarray:
    return pd.Series(c).rolling(p, min_periods=1).mean().values


# ============================================================================
# STRATEGY 1 -- MEAN REVERSION POST-SPIKE
# ============================================================================
class SpikeReversion:
    NAME = "SPIKE_REV"

    PARAM_GRID = {
        "spike_lookback":   [3, 5, 7],
        "spike_threshold":  [2.0, 2.5, 3.0, 3.5, 4.0],
        "reversion_target": [0.30, 0.40, 0.50, 0.60],
        "stop_mult":        [1.0, 1.5, 2.0],
        "max_hold_candles": [5, 10, 15, 20],
        "atr_period":       [10, 14, 20],
    }
    # 3*5*4*3*4*3 = 2160 combos

    DEFAULT_PARAMS = {
        "spike_lookback": 3, "spike_threshold": 5.0,
        "reversion_target": 0.40, "stop_mult": 3.0,
        "max_hold_candles": 15, "atr_period": 14,
    }

    @staticmethod
    def precompute(candles: List[dict]) -> Optional[dict]:
        n = len(candles)
        if n < 30:
            return None
        c = np.array([x["c"] for x in candles])
        h = np.array([x["h"] for x in candles])
        l = np.array([x["l"] for x in candles])
        return {
            "c": c, "h": h, "l": l,
            "atr": {p: _atr(h, l, c, p) for p in [10, 14, 20]},
        }

    @staticmethod
    def signal(idx: int, pre: dict, p: dict) -> Optional[dict]:
        N = p["spike_lookback"]
        ap = p["atr_period"]
        if idx < N + ap:
            return None
        c = pre["c"]
        close, prev = c[idx], c[idx - N]
        if prev <= 0 or close <= 0:
            return None
        atr = pre["atr"][ap][idx]
        if atr <= 0:
            return None
        move = (close - prev) / prev
        atr_pct = atr / close
        ratio = abs(move) / atr_pct
        if ratio < p["spike_threshold"]:
            return None
        side = "short" if move > 0 else "long"
        return {
            "strategy": SpikeReversion.NAME, "side": side,
            "stop_dist": abs(move) * p["stop_mult"],
            "tp_dist": abs(move) * p["reversion_target"],
            "max_hold": p["max_hold_candles"],
            "meta": {"spike_ratio": round(ratio, 2), "move_pct": round(move * 100, 3)},
        }


# ============================================================================
# STRATEGY 2 -- FUNDING RATE HARVESTING (live-first, backtest-approximate)
# ============================================================================
class FundingHarvest:
    """
    Backtest approximation: when a coin has had sustained directional price
    pressure (proxy for high funding), go contrary to the trend with a
    mean-reversion entry. Real edge comes from the 8h funding payment.
    NOTE: For accurate backtest, funding rate history is needed. This
    implementation uses price-based proxies. Run live first.
    """
    NAME = "FUNDING_HARV"

    PARAM_GRID = {
        "trend_lookback":   [60, 120, 240],   # candles = 1h, 2h, 4h
        "trend_z_thresh":   [1.5, 2.0, 2.5],  # z-score of price deviation
        "max_hold_candles": [30, 60, 120, 240],
        "stop_pct":         [0.005, 0.008, 0.012],
        "tp_pct":           [0.003, 0.005, 0.008],
    }
    # 3*3*4*3*3 = 324 combos

    DEFAULT_PARAMS = {
        "trend_lookback": 120, "trend_z_thresh": 2.0,
        "max_hold_candles": 120, "stop_pct": 0.008, "tp_pct": 0.005,
    }

    @staticmethod
    def precompute(candles: List[dict]) -> Optional[dict]:
        n = len(candles)
        if n < 30:
            return None
        c = np.array([x["c"] for x in candles])
        return {"c": c}

    @staticmethod
    def signal(idx: int, pre: dict, p: dict) -> Optional[dict]:
        lb = p["trend_lookback"]
        if idx < lb + 5:
            return None
        c = pre["c"]
        window = c[idx - lb: idx + 1]
        mean = np.mean(window)
        std = np.std(window, ddof=1)
        if std <= 0:
            return None
        z = (c[idx] - mean) / std
        if abs(z) < p["trend_z_thresh"]:
            return None
        # Fade extreme deviation (proxy for funding imbalance)
        side = "short" if z > 0 else "long"
        return {
            "strategy": FundingHarvest.NAME, "side": side,
            "stop_dist": p["stop_pct"],
            "tp_dist": p["tp_pct"],
            "max_hold": p["max_hold_candles"],
            "meta": {"z_score": round(z, 2), "trend_lookback": lb},
        }


# ============================================================================
# STRATEGY 3 -- LIQUIDATION CASCADE (requires live OI; backtest via vol proxy)
# ============================================================================
class LiquidationCascade:
    """
    Real signal requires real-time OI from metaAndAssetCtxs.
    Backtest proxy: sudden large candle + high volume relative to recent vol
    = likely liquidation event. Direction: fade it.
    """
    NAME = "LIQ_CASCADE"

    PARAM_GRID = {
        "vol_lookback":     [10, 20, 30],      # candles for vol baseline
        "vol_z_thresh":     [2.5, 3.0, 3.5, 4.0],  # sudden vol spike
        "price_z_thresh":   [2.0, 2.5, 3.0],   # sudden price move
        "bounce_target":    [0.30, 0.40, 0.50],
        "stop_mult":        [1.0, 1.5, 2.0],
        "max_hold_candles": [3, 5, 10],
    }
    # 3*4*3*3*3*3 = 972 combos

    DEFAULT_PARAMS = {
        "vol_lookback": 20, "vol_z_thresh": 3.0, "price_z_thresh": 2.5,
        "bounce_target": 0.40, "stop_mult": 1.5, "max_hold_candles": 5,
    }

    @staticmethod
    def precompute(candles: List[dict]) -> Optional[dict]:
        n = len(candles)
        if n < 30:
            return None
        c = np.array([x["c"] for x in candles])
        h = np.array([x["h"] for x in candles])
        l = np.array([x["l"] for x in candles])
        # Candle range as vol proxy
        rng = (h - l) / np.maximum(c, 1e-10)
        rets = pd.Series(c).pct_change().fillna(0).values
        return {"c": c, "range": rng, "rets": rets}

    @staticmethod
    def signal(idx: int, pre: dict, p: dict) -> Optional[dict]:
        lb = p["vol_lookback"]
        if idx < lb + 5:
            return None
        c, rng, rets = pre["c"], pre["range"], pre["rets"]

        # Sudden range spike (liquidation proxy)
        hist_range = rng[idx - lb: idx]
        mean_r, std_r = np.mean(hist_range), np.std(hist_range, ddof=1)
        if std_r <= 0:
            return None
        range_z = (rng[idx] - mean_r) / std_r
        if range_z < p["vol_z_thresh"]:
            return None

        # Sudden price move
        hist_rets = rets[idx - lb: idx]
        mean_ret, std_ret = np.mean(hist_rets), np.std(hist_rets, ddof=1)
        if std_ret <= 0:
            return None
        ret_z = (rets[idx] - mean_ret) / std_ret
        if abs(ret_z) < p["price_z_thresh"]:
            return None

        side = "short" if rets[idx] > 0 else "long"
        move = abs(rets[idx])
        return {
            "strategy": LiquidationCascade.NAME, "side": side,
            "stop_dist": move * p["stop_mult"],
            "tp_dist": move * p["bounce_target"],
            "max_hold": p["max_hold_candles"],
            "meta": {"range_z": round(range_z, 2), "ret_z": round(ret_z, 2)},
        }


# ============================================================================
# STRATEGY 4 -- CROSS-SECTIONAL PAIRS MEAN REVERSION
# ============================================================================

SECTOR_PAIRS = [
    # L1 blockchains
    ("SOL", "AVAX"), ("SOL", "SUI"), ("AVAX", "SUI"),
    ("ETH", "SOL"), ("ETH", "AVAX"),
    # L2 / Scaling
    ("ARB", "OP"), ("ARB", "IMX"),
    # Meme
    ("DOGE", "SHIB"), ("DOGE", "PEPE"), ("PEPE", "WIF"),
    # DeFi
    ("LINK", "UNI"), ("AAVE", "MKR"),
    # AI / Data
    ("FET", "RENDER"), ("TAO", "FET"),
    # Infra
    ("ATOM", "DOT"), ("NEAR", "ALGO"),
    # Exchange tokens
    ("BNB", "OKB"), ("HYPE", "BNB"),
]


class PairsMeanReversion:
    NAME = "PAIRS_MR"

    PARAM_GRID = {
        "spread_window":    [30, 60, 120],
        "entry_z":          [1.5, 2.0, 2.5, 3.0],
        "exit_z":           [0.0, 0.3, 0.5],
        "max_hold_candles": [15, 30, 60],
        "leverage_per_leg": [3, 5],
    }
    # 3*4*3*3*2 = 216 combos

    DEFAULT_PARAMS = {
        "spread_window": 60, "entry_z": 2.0, "exit_z": 0.3,
        "max_hold_candles": 30, "leverage_per_leg": 3,
    }

    @staticmethod
    def precompute_pair(sym_a: str, sym_b: str,
                        candles_a: List[dict], candles_b: List[dict],
                        window: int) -> Optional[dict]:
        n = min(len(candles_a), len(candles_b))
        if n < window + 5:
            return None
        ca = np.array([x["c"] for x in candles_a[:n]])
        cb = np.array([x["c"] for x in candles_b[:n]])
        # Log-ratio spread
        spread = np.log(np.maximum(ca, 1e-10) / np.maximum(cb, 1e-10))
        s = pd.Series(spread)
        mean = s.rolling(window, min_periods=window//2).mean().values
        std  = s.rolling(window, min_periods=window//2).std(ddof=1).fillna(1).values
        zscore = np.where(std > 0, (spread - mean) / std, 0)
        return {
            "sym_a": sym_a, "sym_b": sym_b,
            "ca": ca, "cb": cb, "spread": spread, "zscore": zscore,
            "n": n,
        }

    @staticmethod
    def signal_pair(idx: int, pair_data: dict, p: dict) -> Optional[Tuple[dict, dict]]:
        """Returns (signal_a, signal_b) or None."""
        if idx < p["spread_window"] + 5:
            return None
        z = pair_data["zscore"][idx]
        if abs(z) < p["entry_z"]:
            return None
        # z > 0: A expensive vs B → short A, long B
        side_a = "short" if z > 0 else "long"
        side_b = "long"  if z > 0 else "short"
        tp_d = 0.005   # 0.5% per leg (spread convergence proxy)
        stop_d = 0.012
        sig_a = {
            "strategy": PairsMeanReversion.NAME, "side": side_a,
            "stop_dist": stop_d, "tp_dist": tp_d,
            "max_hold": p["max_hold_candles"],
            "pair_id": f"{pair_data['sym_a']}-{pair_data['sym_b']}",
            "meta": {"zscore": round(z, 2), "role": "A"},
        }
        sig_b = dict(sig_a)
        sig_b["side"] = side_b
        sig_b["meta"] = {"zscore": round(z, 2), "role": "B"}
        return sig_a, sig_b


# ============================================================================
# STRATEGY 5 -- MULTI-TIMEFRAME MOMENTUM CASCADE
# ============================================================================
class MomentumCascade:
    NAME = "MTF_MOM"

    PARAM_GRID = {
        "w1":              [0.15, 0.20, 0.25],
        "w5":              [0.35, 0.40, 0.45],
        "w15":             [0.35, 0.40, 0.45],
        "entry_threshold": [1.0, 1.5, 2.0, 2.5],
        "max_hold_candles":[10, 15, 20, 30],
        "vol_window":      [10, 20],
    }
    # 3*3*3*4*4*2 = 864 combos

    DEFAULT_PARAMS = {
        "w1": 0.20, "w5": 0.40, "w15": 0.40,
        "entry_threshold": 1.5, "max_hold_candles": 15, "vol_window": 20,
    }

    @staticmethod
    def precompute(candles: List[dict]) -> Optional[dict]:
        n = len(candles)
        if n < 50:
            return None
        c = np.array([x["c"] for x in candles])
        # Returns at 1, 5, 15 candle horizons
        r1 = pd.Series(c).pct_change(1).fillna(0).values
        r5 = pd.Series(c).pct_change(5).fillna(0).values
        r15 = pd.Series(c).pct_change(15).fillna(0).values
        return {"c": c, "r1": r1, "r5": r5, "r15": r15}

    @staticmethod
    def signal(idx: int, pre: dict, p: dict) -> Optional[dict]:
        vw = p["vol_window"]
        if idx < 20 + vw:
            return None
        r1, r5, r15 = pre["r1"], pre["r5"], pre["r15"]

        # Normalize by rolling vol
        v1  = np.std(r1[idx - vw: idx], ddof=1) or 1e-8
        v5  = np.std(r5[idx - vw: idx], ddof=1) or 1e-8
        v15 = np.std(r15[idx - vw: idx], ddof=1) or 1e-8

        s1  = r1[idx]  / v1
        s5  = r5[idx]  / v5
        s15 = r15[idx] / v15

        composite = p["w1"] * s1 + p["w5"] * s5 + p["w15"] * s15

        # Require ALL timeframes aligned
        th = p["entry_threshold"]
        if composite >= th and s1 > 0 and s5 > 0 and s15 > 0:
            side = "long"
        elif composite <= -th and s1 < 0 and s5 < 0 and s15 < 0:
            side = "short"
        else:
            return None

        # Dynamic TP/stop based on recent vol
        atr_proxy = v1 * pre["c"][idx] * 2  # rough 1-sigma in price units
        tp_d  = v1 * 2.0   # 2-sigma TP
        stop_d = v1 * 1.5   # 1.5-sigma stop

        return {
            "strategy": MomentumCascade.NAME, "side": side,
            "stop_dist": stop_d, "tp_dist": tp_d,
            "max_hold": p["max_hold_candles"],
            "meta": {
                "composite": round(composite, 3),
                "s1": round(s1, 2), "s5": round(s5, 2), "s15": round(s15, 2),
            },
        }


# ============================================================================
# STRATEGY 6 -- VOLATILITY REGIME SWITCH
# ============================================================================
class VolRegimeSwitch:
    NAME = "VOL_REGIME"

    PARAM_GRID = {
        "compression_threshold": [0.3, 0.4, 0.5, 0.6],
        "expansion_threshold":   [1.5, 2.0, 2.5, 3.0],
        "stop_vol_mult":         [2.0, 3.0, 4.0],
        "min_squeeze_candles":   [3, 5, 8, 10],
        "vol_short":             [3, 5, 8],
        "vol_medium":            [20, 30, 45],
    }
    # 4*4*3*4*3*3 = 1728 combos

    DEFAULT_PARAMS = {
        "compression_threshold": 0.4, "expansion_threshold": 2.0,
        "stop_vol_mult": 3.0, "min_squeeze_candles": 5,
        "vol_short": 5, "vol_medium": 30,
    }

    @staticmethod
    def precompute(candles: List[dict]) -> Optional[dict]:
        n = len(candles)
        if n < 60:
            return None
        c = np.array([x["c"] for x in candles])
        rets = pd.Series(c).pct_change().fillna(0).values
        return {"c": c, "rets": rets}

    @staticmethod
    def signal(idx: int, pre: dict, p: dict, prev_regime: str = "NORMAL") -> Optional[dict]:
        vs, vm = p["vol_short"], p["vol_medium"]
        if idx < vm + p["min_squeeze_candles"] + 5:
            return None
        rets = pre["rets"]

        vol_s = np.std(rets[idx - vs: idx], ddof=1) if vs > 1 else abs(rets[idx])
        vol_m = np.std(rets[idx - vm: idx], ddof=1) if vm > 1 else 1e-8
        if vol_m <= 0:
            return None

        ratio = vol_s / vol_m
        if ratio < p["compression_threshold"]:
            regime = "SQUEEZE"
        elif ratio > p["expansion_threshold"]:
            regime = "EXPLOSION"
        else:
            regime = "NORMAL"

        # Signal on SQUEEZE -> EXPLOSION transition
        if prev_regime == "SQUEEZE" and regime == "EXPLOSION":
            direction = rets[idx]
            if direction == 0:
                return None
            side = "long" if direction > 0 else "short"
            tp_d  = vol_s * p["stop_vol_mult"] * 2
            stop_d = vol_s * p["stop_vol_mult"]
            return {
                "strategy": VolRegimeSwitch.NAME, "side": side,
                "stop_dist": stop_d, "tp_dist": tp_d,
                "max_hold": 20,
                "meta": {"ratio": round(ratio, 3), "vol_s": round(vol_s * 100, 4)},
                "_regime": regime,
            }
        # Return regime change info via meta even when no trade
        return {"_regime": regime}


# ── Registry ─────────────────────────────────────────────────────────────────
ALL_STRATEGIES = {
    SpikeReversion.NAME:     SpikeReversion,
    FundingHarvest.NAME:     FundingHarvest,
    LiquidationCascade.NAME: LiquidationCascade,
    PairsMeanReversion.NAME: PairsMeanReversion,
    MomentumCascade.NAME:    MomentumCascade,
    VolRegimeSwitch.NAME:    VolRegimeSwitch,
}

VALIDATION_CRITERIA = {
    "sharpe":        1.0,    # annualized Sharpe > 1
    "win_rate":      52.0,   # > 52%
    "profit_factor": 1.15,   # gross profit / gross loss > 1.15
    "max_dd_pct":    10.0,   # max drawdown < 10%
    "trades_per_day": 3.0,   # at least 3 trades/day
    "edge_bps":      2.0,    # net edge > 2bp per trade (realistic on 4-day sample)
}
