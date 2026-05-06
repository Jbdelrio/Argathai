"""
Artemisia Glacialis v5 — Tick Poller + 1-Min Candle Aggregator

Polls every 2s for surveillance.
Aggregates into 1-min candles for signals.
Computes Bollinger Bands, momentum, vol on CANDLE scale.
"""
import time, logging, requests
import numpy as np
from collections import deque, defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
from config import Config, Candle

logger = logging.getLogger("g5.ticker")


class HLPoller:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.url = cfg.hl_info_url
        self._last = 0.0
        self._delay = cfg.tick_interval_ms / 1000

    def poll_mids(self) -> Dict[str, float]:
        elapsed = time.time() - self._last
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        self._last = time.time()
        try:
            r = requests.post(self.url, json={"type": "allMids"},
                              headers={"Content-Type": "application/json"}, timeout=5)
            r.raise_for_status()
            return {k: float(v) for k, v in r.json().items()}
        except Exception as e:
            logger.warning(f"Poll: {e}")
            return {}

    def get_meta_contexts(self) -> Tuple[List[str], Dict[str, dict]]:
        try:
            r = requests.post(self.url, json={"type": "metaAndAssetCtxs"},
                              headers={"Content-Type": "application/json"}, timeout=10)
            r.raise_for_status()
            data = r.json()
            if not data or len(data) < 2: return [], {}
            universe = data[0]["universe"]
            ctxs = {}
            for meta, ctx in zip(universe, data[1]):
                sym = meta["name"]
                try:
                    ctxs[sym] = {
                        "mid": float(ctx.get("midPx", 0) or 0),
                        "vol24h": float(ctx.get("dayNtlVlm", 0) or 0),
                        "funding": float(ctx.get("funding", 0) or 0),
                    }
                except: pass
            return [m["name"] for m in universe], ctxs
        except Exception as e:
            logger.error(f"Meta: {e}")
            return [], {}

    def test(self) -> Tuple[bool, str]:
        try:
            r = requests.post(self.url, json={"type": "meta"},
                              headers={"Content-Type": "application/json"}, timeout=10)
            r.raise_for_status()
            d = r.json()
            if d and "universe" in d:
                return True, f"OK — {len(d['universe'])} perps"
            return False, "Bad response"
        except Exception as e:
            return False, str(e)


class CandleStore:
    """
    Aggregates tick prices into 1-min candles.
    Computes Bollinger Bands, returns, volatility on candle scale.
    """
    def __init__(self, cfg: Config, symbols: List[str]):
        self.cfg = cfg
        self.symbols = symbols
        self.candle_num = 0

        # Current building candle per symbol
        self._building: Dict[str, dict] = {}  # sym → {open, high, low, close, ticks, start_time}

        # Completed candles
        self.candles: Dict[str, deque] = {s: deque(maxlen=cfg.max_candle_history) for s in symbols}

        # Latest prices (updated every tick)
        self.prices: Dict[str, float] = {}

        # Candle indicators (updated when candle closes)
        self.indicators: Dict[str, dict] = {}

        # Tick counter for candle boundary
        self._tick_count = 0
        self._candle_start = time.time()

    @property
    def n_candles(self):
        if not self.candles: return 0
        return min(len(v) for v in self.candles.values()) if self.candles else 0

    @property
    def is_warmed_up(self):
        return self.n_candles >= self.cfg.warmup_candles

    @property
    def warmup_pct(self):
        return min(self.n_candles / max(self.cfg.warmup_candles, 1) * 100, 100)

    def add_tick(self, prices: Dict[str, float]) -> bool:
        """Add a tick. Returns True if a candle just closed."""
        self._tick_count += 1
        now = time.time()
        candle_closed = False

        for sym in self.symbols:
            p = prices.get(sym, 0)
            if p <= 0:
                continue
            self.prices[sym] = p

            # Build candle
            if sym not in self._building:
                self._building[sym] = {
                    "open": p, "high": p, "low": p, "close": p,
                    "ticks": 0, "start": datetime.now(timezone.utc)
                }

            b = self._building[sym]
            b["high"] = max(b["high"], p)
            b["low"] = min(b["low"], p)
            b["close"] = p
            b["ticks"] += 1

        # Check if candle period elapsed
        if now - self._candle_start >= self.cfg.candle_seconds:
            self._close_candles()
            self._candle_start = now
            candle_closed = True

        return candle_closed

    def _close_candles(self):
        """Close current candles and compute indicators."""
        self.candle_num += 1
        now = datetime.now(timezone.utc)

        for sym in self.symbols:
            b = self._building.get(sym)
            if b is None or b["ticks"] == 0:
                continue

            candle = Candle(
                timestamp=now,
                open=b["open"], high=b["high"],
                low=b["low"], close=b["close"],
                n_ticks=b["ticks"]
            )
            self.candles[sym].append(candle)

            # Reset building candle
            self._building[sym] = {
                "open": b["close"], "high": b["close"],
                "low": b["close"], "close": b["close"],
                "ticks": 0, "start": now
            }

        # Recompute indicators
        self._compute_indicators()

    def _compute_indicators(self):
        """Compute candle-scale indicators for all symbols."""
        cfg = self.cfg

        # Get all close arrays
        all_returns = {}
        for sym in self.symbols:
            candle_list = list(self.candles.get(sym, []))
            if len(candle_list) < 5:
                continue

            closes = np.array([c.close for c in candle_list])
            highs = np.array([c.high for c in candle_list])
            lows = np.array([c.low for c in candle_list])
            n = len(closes)

            # Returns
            rets = np.diff(closes) / closes[:-1] if n > 1 else np.array([0])
            all_returns[sym] = rets

            # Volatility (1-min candle vol)
            vol = np.std(rets[-20:]) if len(rets) >= 20 else np.std(rets) if len(rets) > 2 else 0

            # ── Bollinger Bands ──
            bb_n = min(cfg.bb_period, n)
            if bb_n >= 10:
                bb_close = closes[-bb_n:]
                bb_mean = np.mean(bb_close)
                bb_std_val = np.std(bb_close)
                bb_upper = bb_mean + cfg.bb_std * bb_std_val
                bb_lower = bb_mean - cfg.bb_std * bb_std_val
                bb_width = (bb_upper - bb_lower) / bb_mean if bb_mean > 0 else 0

                # Bandwidth percentile (is current width narrow vs history?)
                if n >= 30:
                    widths = []
                    for i in range(bb_n, n):
                        w_close = closes[i-bb_n:i]
                        w_mean = np.mean(w_close)
                        w_std = np.std(w_close)
                        w = (w_mean + cfg.bb_std * w_std - (w_mean - cfg.bb_std * w_std)) / w_mean if w_mean > 0 else 0
                        widths.append(w)
                    if widths:
                        bb_pctile = sum(1 for w in widths if w < bb_width) / len(widths) * 100
                    else:
                        bb_pctile = 50
                else:
                    bb_pctile = 50

                # Price position relative to bands
                last = closes[-1]
                if bb_upper != bb_lower:
                    bb_position = (last - bb_lower) / (bb_upper - bb_lower)  # 0=lower, 1=upper
                else:
                    bb_position = 0.5
            else:
                bb_mean = bb_upper = bb_lower = closes[-1]
                bb_width = bb_pctile = 0
                bb_position = 0.5

            # ── Momentum (returns over lookback) ──
            mom_lb = cfg.mom_lookback
            if n > mom_lb:
                mom = closes[-1] / closes[-1 - mom_lb] - 1
            else:
                mom = 0

            # ── ATR (average true range) for stop sizing ──
            if n >= 3:
                tr_list = []
                for i in range(1, min(14, n)):
                    tr = max(
                        highs[-i] - lows[-i],
                        abs(highs[-i] - closes[-i-1]),
                        abs(lows[-i] - closes[-i-1])
                    )
                    tr_list.append(tr)
                atr = np.mean(tr_list) if tr_list else 0
                atr_pct = atr / closes[-1] if closes[-1] > 0 else 0
            else:
                atr = atr_pct = 0

            self.indicators[sym] = {
                "price": closes[-1],
                "vol": vol,               # 1-min candle vol
                "atr": atr,
                "atr_pct": atr_pct,
                "bb_mean": bb_mean,
                "bb_upper": bb_upper,
                "bb_lower": bb_lower,
                "bb_width": bb_width,
                "bb_pctile": bb_pctile,   # is BB squeezed?
                "bb_position": bb_position,
                "mom": mom,               # 5-min return
                "n_candles": n,
                "last_return": rets[-1] if len(rets) > 0 else 0,
            }

        # Cross-sectional momentum rankings
        mom_list = [(sym, ind.get("mom", 0)) for sym, ind in self.indicators.items()
                    if sym != "BTC" and np.isfinite(ind.get("mom", 0))]
        mom_list.sort(key=lambda x: x[1], reverse=True)
        for rank, (sym, mom_val) in enumerate(mom_list):
            if sym in self.indicators:
                self.indicators[sym]["mom_rank"] = rank + 1
                self.indicators[sym]["mom_total"] = len(mom_list)

    def get_prices(self) -> Dict[str, float]:
        return dict(self.prices)

    def get_funding_rates(self, ctxs: Dict[str, dict]) -> Dict[str, float]:
        return {s: ctxs.get(s, {}).get("funding", 0) for s in self.symbols if s in ctxs}

    def get_bb_squeeze_symbols(self) -> List[str]:
        """Return symbols currently in BB squeeze."""
        return [sym for sym, ind in self.indicators.items()
                if ind.get("bb_pctile", 50) <= self.cfg.bb_squeeze_pctile
                and sym != "BTC"]
