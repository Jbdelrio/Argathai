"""
Artemisia Glacialis v6 -- Multi-Strategy Live Engine
Loads validated params from g5_config.json, runs all 6 strategies.
Paper trading only (no real orders placed).

Usage: python engine_v2.py [--paper] [--config g5_config.json]
"""
import time, json, logging, threading, pathlib, sys
import numpy as np
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

from strategies_v2 import (
    SpikeReversion, FundingHarvest, LiquidationCascade,
    MomentumCascade, VolRegimeSwitch, PairsMeanReversion,
    SECTOR_PAIRS, ALL_STRATEGIES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("v6.engine")

HL_URL = "https://api.hyperliquid.xyz/info"
ROUNDTRIP_FEES = 0.00048
SLIPPAGE       = 0.00008


# ── Market data live feed ────────────────────────────────────────────────────

class LiveFeed:
    """Polls Hyperliquid every 2s for mid prices and builds 1-min candles."""

    def __init__(self, symbols: List[str], max_candles: int = 500):
        self.symbols    = symbols
        self.max_candles = max_candles
        self.prices: Dict[str, float] = {}
        self.candles: Dict[str, deque] = {s: deque(maxlen=max_candles) for s in symbols}
        self._building: Dict[str, dict] = {}
        self._candle_start = time.time()
        self.candle_num = 0
        self.lock = threading.Lock()

    def poll(self) -> Dict[str, float]:
        try:
            r = requests.post(HL_URL, json={"type": "allMids"},
                              headers={"Content-Type": "application/json"}, timeout=5)
            r.raise_for_status()
            mids = {k: float(v) for k, v in r.json().items()}
            with self.lock:
                for sym in self.symbols:
                    p = mids.get(sym, 0)
                    if p > 0:
                        self.prices[sym] = p
                        self._update_candle(sym, p)
            return mids
        except Exception as e:
            log.warning(f"Poll failed: {e}")
            return {}

    def _update_candle(self, sym: str, p: float):
        now = time.time()
        if sym not in self._building:
            self._building[sym] = {"o": p, "h": p, "l": p, "c": p, "t": now, "ticks": 0}
        b = self._building[sym]
        b["h"] = max(b["h"], p)
        b["l"] = min(b["l"], p)
        b["c"] = p
        b["ticks"] += 1

        # Close candle every 60s
        if now - self._candle_start >= 60:
            for s in self.symbols:
                if s in self._building and self._building[s]["ticks"] > 0:
                    b2 = self._building[s]
                    self.candles[s].append({
                        "t": int(now * 1000),
                        "o": b2["o"], "h": b2["h"],
                        "l": b2["l"], "c": b2["c"],
                    })
                    price_now = self.prices.get(s, b2["c"])
                    self._building[s] = {"o": price_now, "h": price_now,
                                         "l": price_now, "c": price_now,
                                         "t": now, "ticks": 0}
            self._candle_start = now
            self.candle_num += 1

    def get_candles(self, sym: str) -> List[dict]:
        with self.lock:
            return list(self.candles[sym])

    def get_prices(self) -> Dict[str, float]:
        with self.lock:
            return dict(self.prices)

    def is_warm(self, min_candles: int = 30) -> bool:
        with self.lock:
            return all(len(self.candles[s]) >= min_candles for s in self.symbols
                       if s in self.candles)

    def get_funding_rates(self) -> Dict[str, float]:
        try:
            r = requests.post(HL_URL, json={"type": "metaAndAssetCtxs"},
                              headers={"Content-Type": "application/json"}, timeout=15)
            r.raise_for_status()
            data = r.json()
            result = {}
            for meta, ctx in zip(data[0]["universe"], data[1]):
                sym = meta["name"]
                result[sym] = float(ctx.get("funding", 0) or 0)
            return result
        except Exception as e:
            log.warning(f"Funding fetch failed: {e}")
            return {}

    def get_open_interest(self) -> Dict[str, float]:
        try:
            r = requests.post(HL_URL, json={"type": "metaAndAssetCtxs"},
                              headers={"Content-Type": "application/json"}, timeout=15)
            r.raise_for_status()
            data = r.json()
            result = {}
            for meta, ctx in zip(data[0]["universe"], data[1]):
                sym = meta["name"]
                result[sym] = float(ctx.get("openInterest", 0) or 0)
            return result
        except Exception as e:
            log.warning(f"OI fetch failed: {e}")
            return {}


# ── Position manager ─────────────────────────────────────────────────────────

class PositionManager:
    def __init__(self, initial_capital: float, max_pos: int = 8):
        self.capital   = initial_capital
        self.available = initial_capital
        self.initial   = initial_capital
        self.peak      = initial_capital
        self.max_pos   = max_pos

        self.positions: Dict[str, dict] = {}
        self.closed:    List[dict]      = []
        self.trade_log: List[dict]      = []
        self.equity_curve: List[dict]   = []
        self.cooldowns: Dict[str, int]  = {}

        # Per-strategy counters
        self.strat_stats: Dict[str, dict] = {}

    def can_open(self, sym: str, candle_num: int) -> bool:
        return (
            sym not in self.positions
            and len(self.positions) < self.max_pos
            and self.available >= 5.0
            and candle_num >= self.cooldowns.get(sym, 0)
        )

    def open(self, sym: str, sig: dict, price: float,
             candle_num: int, leverage: float = 5.0, size_pct: float = 0.05) -> bool:
        if not self.can_open(sym, candle_num):
            return False

        entry = price * (1 + SLIPPAGE) if sig["side"] == "long" else price * (1 - SLIPPAGE)
        size  = float(np.clip(self.available * size_pct, 5.0, self.available * 0.25))

        if sig["side"] == "long":
            stop = entry * (1 - sig["stop_dist"])
            tp   = entry * (1 + sig["tp_dist"])
        else:
            stop = entry * (1 + sig["stop_dist"])
            tp   = entry * (1 - sig["tp_dist"])

        self.positions[sym] = {
            "side": sig["side"], "entry": entry, "stop": stop, "tp": tp,
            "max_hold": sig["max_hold"], "open_candle": candle_num,
            "size": size, "lev": leverage, "strategy": sig["strategy"],
            "open_time": datetime.now(timezone.utc).isoformat(),
            "meta": sig.get("meta", {}),
        }
        self.available -= size

        strat = sig["strategy"]
        if strat not in self.strat_stats:
            self.strat_stats[strat] = {"opens": 0, "trades": 0, "wins": 0, "pnl": 0.0}
        self.strat_stats[strat]["opens"] += 1

        log.info(f"OPEN  {sym:10s} {sig['side']:5s} {strat:12s} "
                 f"entry={entry:.4f} stop={stop:.4f} tp={tp:.4f} "
                 f"size=${size:.2f} lev={leverage:.1f}x")
        return True

    def check_exits(self, prices: Dict[str, float], candle_num: int) -> List[dict]:
        closed = []
        for sym in list(self.positions.keys()):
            pos   = self.positions[sym]
            price = prices.get(sym, 0)
            if price <= 0:
                continue

            hold   = candle_num - pos["open_candle"]
            reason = None

            if pos["side"] == "long":
                if price <= pos["stop"]: reason = "STOP"
                elif price >= pos["tp"]: reason = "TP"
            else:
                if price >= pos["stop"]: reason = "STOP"
                elif price <= pos["tp"]: reason = "TP"

            if reason is None and hold >= pos["max_hold"]:
                reason = "MAX_HOLD"

            if reason:
                c = self._close(sym, price, reason, candle_num)
                if c:
                    closed.append(c)

        # Kill switch: 10% drawdown
        if self.capital < self.initial * 0.90:
            log.warning("KILL SWITCH: drawdown > 10%")
            for sym in list(self.positions.keys()):
                p = prices.get(sym, 0)
                if p:
                    self._close(sym, p, "KILL", candle_num)
        return closed

    def _close(self, sym: str, price: float, reason: str, candle_num: int) -> Optional[dict]:
        if sym not in self.positions:
            return None
        pos = self.positions[sym]
        actual = price * (1 - SLIPPAGE) if pos["side"] == "long" else price * (1 + SLIPPAGE)
        notional = pos["size"] * pos["lev"]

        if pos["side"] == "long":
            raw = (actual - pos["entry"]) / pos["entry"] * notional
        else:
            raw = (pos["entry"] - actual) / pos["entry"] * notional
        fee  = notional * ROUNDTRIP_FEES
        pnl  = raw - fee

        self.capital   += pnl
        self.available += pos["size"] + pnl
        if self.capital > self.peak:
            self.peak = self.capital

        hold = candle_num - pos["open_candle"]
        trade = {
            "sym": sym, "side": pos["side"], "strategy": pos["strategy"],
            "entry": pos["entry"], "exit": actual, "reason": reason,
            "size": pos["size"], "lev": pos["lev"],
            "pnl": round(pnl, 4), "pnl_pct": round(pnl / pos["size"] * 100, 2),
            "hold_candles": hold, "fee": round(fee, 4),
            "ts": datetime.now(timezone.utc).isoformat(),
            "meta": pos.get("meta", {}),
        }
        self.closed.append(trade)
        self.trade_log.append(trade)
        self.equity_curve.append({"equity": round(self.capital, 2), "ts": trade["ts"]})

        strat = pos["strategy"]
        if strat in self.strat_stats:
            self.strat_stats[strat]["trades"] += 1
            self.strat_stats[strat]["pnl"] += pnl
            if pnl > 0:
                self.strat_stats[strat]["wins"] += 1

        self.cooldowns[sym] = candle_num + (2 if pnl < 0 else 1)
        del self.positions[sym]

        log.info(f"CLOSE {sym:10s} {pos['side']:5s} {reason:8s} "
                 f"pnl=${pnl:+.4f} ({pnl/pos['size']*100:+.2f}%) "
                 f"hold={hold}c capital=${self.capital:.2f}")
        return trade

    def metrics(self) -> dict:
        n  = len(self.closed)
        if n == 0:
            return {"trades": 0, "capital": round(self.capital, 2)}
        pnls = [t["pnl"] for t in self.closed]
        wins = [p for p in pnls if p > 0]
        return {
            "trades":        n,
            "wins":          len(wins),
            "win_rate":      round(len(wins) / n * 100, 1),
            "total_pnl":     round(sum(pnls), 2),
            "total_pnl_pct": round(sum(pnls) / self.initial * 100, 2),
            "capital":       round(self.capital, 2),
            "available":     round(self.available, 2),
            "open_pos":      len(self.positions),
            "by_strategy":   {k: {**v, "win_rate": round(v["wins"]/v["trades"]*100,1)
                                        if v["trades"] > 0 else 0}
                              for k, v in self.strat_stats.items() if v["trades"] > 0},
        }

    def save(self, path: str = "v6_state.json"):
        state = {
            "capital": self.capital, "available": self.available,
            "peak": self.peak, "trade_log": self.trade_log[-1000:],
            "equity_curve": self.equity_curve[-2000:],
            "strat_stats": self.strat_stats,
        }
        with open(path, "w") as f:
            json.dump(state, f, default=str)


# ── Main trading loop ────────────────────────────────────────────────────────

class ArtemiasEngine:
    def __init__(self, config_path: str = "g5_config.json"):
        # Load config
        cfg_p = pathlib.Path(config_path)
        self.cfg: dict = {}
        if cfg_p.exists():
            with open(cfg_p) as f:
                self.cfg = json.load(f)
        log.info(f"Loaded config from {config_path}")

        self.initial_cap = self.cfg.get("initial_capital", 500.0)
        self.max_pos     = self.cfg.get("max_simultaneous", 8)

        # Determine which strategies are validated
        validated = set(self.cfg.get("validated_strategies",
                                     list(ALL_STRATEGIES.keys())))
        self.enabled: Dict[str, type] = {k: v for k, v in ALL_STRATEGIES.items()
                                          if k in validated}
        if not self.enabled:
            self.enabled = {k: v for k, v in ALL_STRATEGIES.items()
                            if k not in ("PAIRS_MR",)}
        log.info(f"Active strategies: {list(self.enabled.keys())}")

        self.pm      = PositionManager(self.initial_cap, self.max_pos)
        self.regime: Dict[str, str] = {}   # symbol -> VolRegime state

    def get_symbols(self) -> List[str]:
        """Get liquid symbols from Hyperliquid."""
        try:
            r = requests.post(HL_URL, json={"type": "metaAndAssetCtxs"},
                              headers={"Content-Type": "application/json"}, timeout=15)
            r.raise_for_status()
            data = r.json()
            symbols = []
            for meta, ctx in zip(data[0]["universe"], data[1]):
                sym = meta["name"]
                vol = float(ctx.get("dayNtlVlm", 0) or 0)
                if vol >= 2_000_000:
                    symbols.append(sym)
            log.info(f"Universe: {len(symbols)} symbols")
            return symbols[:100]  # cap at 100 for performance
        except Exception as e:
            log.warning(f"Universe fetch failed: {e}")
            return ["ETH", "SOL", "DOGE", "XRP", "SUI", "AVAX", "LINK", "HYPE", "FARTCOIN"]

    def strategy_params(self, strat_name: str) -> dict:
        """Get best params for a strategy from config."""
        prefix = strat_name.lower().replace("_", "") + "_"
        p = {}
        for k, v in self.cfg.items():
            if k.startswith(prefix):
                p[k[len(prefix):]] = v
        # Fill missing with defaults
        cls = self.enabled.get(strat_name)
        if cls and hasattr(cls, "DEFAULT_PARAMS"):
            for k, v in cls.DEFAULT_PARAMS.items():
                p.setdefault(k, v)
        return p

    def run(self, poll_interval: float = 2.0):
        log.info("Starting Artemisia Glacialis v6")
        symbols  = self.get_symbols()
        feed     = LiveFeed(symbols, max_candles=500)
        leverage = self.cfg.get("max_leverage", 5.0)

        log.info(f"Warming up ({30} candles = 30 min)...")
        while not feed.is_warm(30):
            feed.poll()
            self._print_status(feed, warmup=True)
            time.sleep(poll_interval)

        log.info("Warmup complete. Trading active.")
        last_save = time.time()
        last_log  = time.time()

        while True:
            t_start = time.time()

            # Poll prices
            feed.poll()
            prices = feed.get_prices()
            cn     = feed.candle_num

            # Check exits
            self.pm.check_exits(prices, cn)

            # Generate signals from each strategy
            for strat_name, cls in self.enabled.items():
                params = self.strategy_params(strat_name)
                idx_max = min(len(feed.candles[s]) for s in symbols if s in feed.candles) - 1
                if idx_max < 10:
                    continue

                for sym in symbols:
                    if sym not in feed.candles or len(feed.candles[sym]) < 10:
                        continue
                    if not self.pm.can_open(sym, cn):
                        continue

                    clist = feed.get_candles(sym)
                    pre   = cls.precompute(clist)
                    if pre is None:
                        continue

                    idx = len(clist) - 1
                    if cls.NAME == "VOL_REGIME":
                        result = cls.signal(idx, pre, params, self.regime.get(sym, "NORMAL"))
                        if result and "_regime" in result:
                            self.regime[sym] = result["_regime"]
                        sig = result if result and "side" in result else None
                    else:
                        sig = cls.signal(idx, pre, params)

                    if sig and "side" in sig:
                        price = prices.get(sym, 0)
                        if price > 0:
                            self.pm.open(sym, sig, price, cn, leverage=leverage)

            # Periodic save
            if time.time() - last_save >= 60:
                self.pm.save()
                last_save = time.time()

            # Periodic log
            if time.time() - last_log >= 300:
                m = self.pm.metrics()
                log.info(
                    f"METRICS: capital=${m['capital']:.2f} "
                    f"trades={m['trades']} WR={m.get('win_rate',0):.1f}% "
                    f"PnL={m.get('total_pnl_pct',0):+.2f}% "
                    f"open={m.get('open_pos',0)}"
                )
                last_log = time.time()

            elapsed = time.time() - t_start
            sleep = max(0, poll_interval - elapsed)
            time.sleep(sleep)

    def _print_status(self, feed, warmup=False):
        """Print brief status to console."""
        n = min((len(v) for v in feed.candles.values() if v), default=0)
        m = self.pm.metrics()
        status = "WARMUP" if warmup else "LIVE"
        print(
            f"\r[{status}] candles={n:3d} "
            f"capital=${m['capital']:.2f} "
            f"trades={m['trades']} "
            f"open={len(self.pm.positions)}  ",
            end="", flush=True,
        )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="g5_config.json")
    ap.add_argument("--paper", action="store_true", default=True)
    args = ap.parse_args()

    engine = ArtemiasEngine(args.config)
    try:
        engine.run()
    except KeyboardInterrupt:
        engine.pm.save()
        m = engine.pm.metrics()
        print(f"\n\nFinal: trades={m['trades']} WR={m.get('win_rate',0):.1f}% "
              f"PnL={m.get('total_pnl_pct',0):+.2f}%")
