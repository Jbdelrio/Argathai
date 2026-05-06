"""
Artemisia Glacialis v5b — SCALPING Strategies
EMA Cross + RSI Extreme + VWAP Bounce = 15-40 trades/day
"""
import logging, numpy as np
from datetime import datetime, timezone
from typing import Dict, List
from config import Config, Signal
from ticker import CandleStore

logger = logging.getLogger("g5.strats")


def _rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _ema(data, period):
    if len(data) < period:
        return np.mean(data) if len(data) > 0 else 0
    alpha = 2 / (period + 1)
    ema = data[0]
    for p in data[1:]:
        ema = alpha * p + (1 - alpha) * ema
    return ema


class EMAScalper:
    """EMA(5) crosses EMA(15) → micro-trend change → scalp."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.name = "EMA"
        self._prev_state: Dict[str, str] = {}

    def scan(self, store: CandleStore) -> List[Signal]:
        if not self.cfg.enable_ema:
            return []
        signals = []
        cfg = self.cfg
        now = datetime.now(timezone.utc)
        fee_cost = cfg.roundtrip_cost_bps()

        for sym in store.symbols:
            if sym == "BTC": continue
            candle_list = list(store.candles.get(sym, []))
            if len(candle_list) < cfg.ema_slow + 2: continue

            closes = np.array([c.close for c in candle_list])
            ema_f = _ema(closes, cfg.ema_fast)
            ema_s = _ema(closes, cfg.ema_slow)
            price = closes[-1]
            atr_pct = store.indicators.get(sym, {}).get("atr_pct", 0)
            if atr_pct <= 0 or ema_s <= 0: continue

            state = "above" if ema_f > ema_s else "below"
            prev = self._prev_state.get(sym)
            self._prev_state[sym] = state
            if prev is None: continue

            side = None
            if prev == "below" and state == "above" and price > ema_f:
                side = "long"
            elif prev == "above" and state == "below" and price < ema_f:
                side = "short"
            if side is None: continue

            raw_edge_pct = atr_pct * cfg.ema_atr_mult
            raw_edge_bps = raw_edge_pct * 10000
            net_edge_bps = raw_edge_bps - fee_cost
            if net_edge_bps < cfg.min_edge_bps: continue

            lev = np.clip(net_edge_bps / 10, cfg.min_leverage, cfg.max_leverage)
            signals.append(Signal(
                timestamp=now, candle_num=store.candle_num,
                symbol=sym, side=side, strategy=self.name,
                edge_bps=round(net_edge_bps, 1), raw_edge_bps=round(raw_edge_bps, 1),
                leverage=round(lev, 1),
                reason=f"EMA: {sym} EMA{cfg.ema_fast}{'>' if side=='long' else '<'}EMA{cfg.ema_slow} "
                       f"ATR={atr_pct*100:.2f}% edge={net_edge_bps:.0f}bp",
                meta={"ema_fast": ema_f, "ema_slow": ema_s, "atr_pct": atr_pct},
            ))
        return signals


class RSIScalper:
    """RSI(14) < 28 → oversold → LONG bounce. RSI > 72 → overbought → SHORT."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.name = "RSI"

    def scan(self, store: CandleStore) -> List[Signal]:
        if not self.cfg.enable_rsi:
            return []
        signals = []
        cfg = self.cfg
        now = datetime.now(timezone.utc)
        fee_cost = cfg.roundtrip_cost_bps()

        for sym in store.symbols:
            if sym == "BTC": continue
            candle_list = list(store.candles.get(sym, []))
            if len(candle_list) < cfg.rsi_period + 2: continue

            closes = np.array([c.close for c in candle_list])
            rsi = _rsi(closes, cfg.rsi_period)
            atr_pct = store.indicators.get(sym, {}).get("atr_pct", 0)
            if atr_pct <= 0: continue

            side = None
            if rsi < cfg.rsi_oversold:
                side = "long"
            elif rsi > cfg.rsi_overbought:
                side = "short"
            if side is None: continue

            if side == "long":
                extremity = (cfg.rsi_oversold - rsi) / cfg.rsi_oversold
            else:
                extremity = (rsi - cfg.rsi_overbought) / (100 - cfg.rsi_overbought)

            raw_edge_pct = atr_pct * (0.8 + extremity)
            raw_edge_bps = raw_edge_pct * 10000
            net_edge_bps = raw_edge_bps - fee_cost
            if net_edge_bps < cfg.min_edge_bps: continue

            lev = np.clip(net_edge_bps / 10, cfg.min_leverage, cfg.max_leverage)
            signals.append(Signal(
                timestamp=now, candle_num=store.candle_num,
                symbol=sym, side=side, strategy=self.name,
                edge_bps=round(net_edge_bps, 1), raw_edge_bps=round(raw_edge_bps, 1),
                leverage=round(lev, 1),
                reason=f"RSI: {sym} RSI={rsi:.0f} → {side} "
                       f"ATR={atr_pct*100:.2f}% edge={net_edge_bps:.0f}bp",
                meta={"rsi": rsi, "atr_pct": atr_pct, "extremity": extremity},
            ))
        return signals


class VWAPScalper:
    """Price crosses SMA(20) = VWAP proxy → bounce trade."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.name = "VWAP"

    def scan(self, store: CandleStore) -> List[Signal]:
        if not self.cfg.enable_vwap:
            return []
        signals = []
        cfg = self.cfg
        now = datetime.now(timezone.utc)
        fee_cost = cfg.roundtrip_cost_bps()

        for sym in store.symbols:
            if sym == "BTC": continue
            candle_list = list(store.candles.get(sym, []))
            if len(candle_list) < cfg.vwap_period + 2: continue

            closes = np.array([c.close for c in candle_list])
            vwap = np.mean(closes[-cfg.vwap_period:])
            price = closes[-1]
            prev_price = closes[-2]
            atr_pct = store.indicators.get(sym, {}).get("atr_pct", 0)
            if atr_pct <= 0 or vwap <= 0: continue

            dist_pct = (price - vwap) / vwap
            side = None
            if prev_price < vwap and price > vwap and abs(dist_pct) < cfg.vwap_max_dist:
                side = "long"
            elif prev_price > vwap and price < vwap and abs(dist_pct) < cfg.vwap_max_dist:
                side = "short"
            if side is None: continue

            raw_edge_pct = atr_pct * cfg.vwap_atr_mult
            raw_edge_bps = raw_edge_pct * 10000
            net_edge_bps = raw_edge_bps - fee_cost
            if net_edge_bps < cfg.min_edge_bps: continue

            lev = np.clip(net_edge_bps / 10, cfg.min_leverage, cfg.max_leverage)
            signals.append(Signal(
                timestamp=now, candle_num=store.candle_num,
                symbol=sym, side=side, strategy=self.name,
                edge_bps=round(net_edge_bps, 1), raw_edge_bps=round(raw_edge_bps, 1),
                leverage=round(lev, 1),
                reason=f"VWAP: {sym} cross {'up' if side=='long' else 'dn'} "
                       f"dist={dist_pct*100:+.2f}% ATR={atr_pct*100:.2f}% edge={net_edge_bps:.0f}bp",
                meta={"vwap": vwap, "dist_pct": dist_pct, "atr_pct": atr_pct},
            ))
        return signals


class StrategyManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ema = EMAScalper(cfg)
        self.rsi = RSIScalper(cfg)
        self.vwap = VWAPScalper(cfg)
        self.signal_history: List[Signal] = []

    def scan_all(self, store: CandleStore,
                 funding_rates: Dict[str, float] = None) -> List[Signal]:
        all_sigs = self.ema.scan(store) + self.rsi.scan(store) + self.vwap.scan(store)

        best = {}
        for sig in all_sigs:
            if sig.symbol not in best or sig.edge_bps > best[sig.symbol].edge_bps:
                best[sig.symbol] = sig

        ranked = sorted(best.values(), key=lambda s: s.edge_bps, reverse=True)
        self.signal_history.extend(ranked)
        if len(self.signal_history) > 3000:
            self.signal_history = self.signal_history[-2000:]
        return ranked
