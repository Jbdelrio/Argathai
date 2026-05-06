"""
Artemisia Glacialis v5 — Engine
Candle-based exits: 0.8% stop, 0.5% TP (not 0.05% tick-based stops!).
"""
import json, logging, numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from config import Config, Signal, Position

logger = logging.getLogger("g5.engine")


class Metrics:
    @staticmethod
    def compute(eq, closed, cap):
        n = len(closed)
        if n == 0:
            return {k: 0 for k in [
                "total_pnl","total_pnl_pct","trades","wins","losses","win_rate",
                "avg_win","avg_loss","profit_factor","expectancy","sharpe","sortino",
                "calmar","max_dd","max_dd_pct","total_fees","avg_hold_sec",
                "avg_leverage"]} | {"by_strategy": {}}

        pnls = [p.pnl for p in closed]
        wins = [x for x in pnls if x > 0]; losses = [x for x in pnls if x <= 0]
        tp = sum(pnls); gp = sum(wins) if wins else 0; gl = abs(sum(losses)) if losses else 0

        wr = len(wins)/n*100; pf = gp/gl if gl > 0 else 0

        eqv = [cap]+[e["equity"] for e in eq]
        pk = cap; mdd = 0
        for v in eqv[1:]:
            if v > pk: pk = v
            dd = (pk-v)/pk if pk > 0 else 0
            if dd > mdd: mdd = dd

        rets = np.diff(eqv)/np.maximum(eqv[:-1],0.01) if len(eqv)>1 else np.array([])
        rets = rets[np.isfinite(rets)]
        sharpe = sortino = 0
        if len(rets)>5 and np.std(rets)>0:
            ann = max(len(rets)*365/max(1,len(rets)),1)
            mu=np.mean(rets); sig=np.std(rets,ddof=1)
            sharpe=mu/sig*np.sqrt(ann)
            dn=rets[rets<0]
            if len(dn)>0: sortino=mu/np.std(dn,ddof=1)*np.sqrt(ann)

        fees = sum(p.fees_paid for p in closed)
        hold_secs = [p.hold_candles * 60 for p in closed]  # 60s per candle

        by_s = {}
        for st in ("EMA","RSI","VWAP"):
            sc = [p for p in closed if p.strategy == st]
            if sc:
                sw = [p.pnl for p in sc if p.pnl > 0]
                by_s[st] = {"trades":len(sc), "win_rate":round(len(sw)/len(sc)*100,1),
                    "pnl":round(sum(p.pnl for p in sc),4),
                    "avg_lev":round(np.mean([p.leverage for p in sc]),1)}

        return {
            "total_pnl":round(tp,4),"total_pnl_pct":round(tp/cap*100,2),
            "trades":n,"wins":len(wins),"losses":len(losses),"win_rate":round(wr,1),
            "avg_win":round(np.mean(wins),4) if wins else 0,
            "avg_loss":round(np.mean(losses),4) if losses else 0,
            "profit_factor":round(pf,2),"expectancy":round(tp/n,4),
            "sharpe":round(sharpe,2),"sortino":round(sortino,2),
            "calmar":round((tp/cap)/mdd,2) if mdd>0 else 0,
            "max_dd":round(pk*mdd,4) if mdd>0 else 0,"max_dd_pct":round(mdd*100,2),
            "total_fees":round(fees,4),
            "avg_hold_sec":round(np.mean(hold_secs),0) if hold_secs else 0,
            "avg_leverage":round(np.mean([p.leverage for p in closed]),1),
            "by_strategy":by_s,
        }


class PaperTrader:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.capital = cfg.initial_capital
        self.available = cfg.initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed: List[Position] = []
        self.trade_log: List[dict] = []
        self.equity_curve: List[dict] = []
        self.peak = cfg.initial_capital
        self.killed = False; self.kill_reason = ""
        self._cooldowns: Dict[str, int] = {}
        self._trade_counts: Dict[str, List[int]] = {}

    def _cooled(self, sym, candle_num) -> Tuple[bool, str]:
        cd = self._cooldowns.get(sym, 0)
        if candle_num < cd: return False, f"CD→{cd}"
        counts = self._trade_counts.get(sym, [])
        recent = [t for t in counts if candle_num - t < 60]  # 60 candles = 1h
        self._trade_counts[sym] = recent
        if len(recent) >= self.cfg.max_trades_per_symbol_hour:
            return False, "Rate"
        return True, "OK"

    def can_open(self, sig: Signal) -> Tuple[bool, str]:
        if self.killed: return False, "KILLED"
        if sig.symbol in self.positions: return False, "Open"
        if len(self.positions) >= self.cfg.max_simultaneous: return False, "MaxPos"
        if self.available < 5: return False, "NoCap"
        return self._cooled(sig.symbol, sig.candle_num)

    def open(self, sig: Signal, price: float, exposure: float = 1.0) -> Optional[Position]:
        ok, reason = self.can_open(sig)
        if not ok: return None

        cfg = self.cfg
        edge_factor = min(sig.edge_bps / 20, 2.0)
        size = self.available * cfg.base_position_pct * edge_factor * exposure
        size = np.clip(size, 5.0, self.available * 0.30)

        lev = sig.leverage
        notional = size * lev

        slip = cfg.slippage_bps / 10000
        entry = price * (1+slip) if sig.side == "long" else price * (1-slip)
        entry_fee = notional * cfg.fee_rate

        # ── % STOPS (not vol-scaled!) ──
        if sig.side == "long":
            stop = entry * (1 - cfg.stop_pct)
            tp = entry * (1 + cfg.tp_pct)
        else:
            stop = entry * (1 + cfg.stop_pct)
            tp = entry * (1 - cfg.tp_pct)

        now = datetime.now(timezone.utc)
        pos = Position(
            symbol=sig.symbol, side=sig.side, strategy=sig.strategy,
            entry_price=entry, entry_time=now, entry_candle=sig.candle_num,
            size_usd=round(size,2), leverage=lev,
            stop_loss=stop, take_profit=tp,
            pnl=-entry_fee, fees_paid=entry_fee,
        )
        self.positions[sig.symbol] = pos
        self.available -= size

        if sig.symbol not in self._trade_counts:
            self._trade_counts[sig.symbol] = []
        self._trade_counts[sig.symbol].append(sig.candle_num)
        self._log("OPEN", pos, sig)
        return pos

    def check_exits(self, prices, candle_num) -> List[Position]:
        closed = []
        cfg = self.cfg

        for sym, pos in list(self.positions.items()):
            price = prices.get(sym)
            if price is None or price <= 0: continue

            pos.hold_candles = candle_num - pos.entry_candle
            reason = None

            # Stop
            if pos.side == "long" and price <= pos.stop_loss: reason = "STOP"
            elif pos.side == "short" and price >= pos.stop_loss: reason = "STOP"

            # TP
            if reason is None:
                if pos.side == "long" and price >= pos.take_profit: reason = "TP"
                elif pos.side == "short" and price <= pos.take_profit: reason = "TP"

            # Breakeven stop after N candles if in profit
            if reason is None and pos.hold_candles >= cfg.breakeven_after_candles:
                if not pos.breakeven_activated:
                    notional = pos.size_usd * pos.leverage
                    fee_buffer = notional * cfg.fee_rate * 2 / notional
                    if pos.side == "long" and price > pos.entry_price * (1 + fee_buffer):
                        pos.stop_loss = pos.entry_price * (1 + fee_buffer)
                        pos.breakeven_activated = True
                    elif pos.side == "short" and price < pos.entry_price * (1 - fee_buffer):
                        pos.stop_loss = pos.entry_price * (1 - fee_buffer)
                        pos.breakeven_activated = True

            # Max hold
            max_hold = {"EMA": cfg.ema_hold_candles, "RSI": cfg.rsi_hold_candles,
                        "VWAP": cfg.vwap_hold_candles}.get(pos.strategy, 5)
            if reason is None and pos.hold_candles >= max_hold:
                reason = "MAX_HOLD"

            if reason:
                c = self._close(sym, price, reason, candle_num)
                if c: closed.append(c)

        # Kill switch
        if not self.killed and self.capital < cfg.initial_capital * 0.90:
            self.killed = True; self.kill_reason = "DD 10%"
            for sym in list(self.positions.keys()):
                p = prices.get(sym)
                if p:
                    c = self._close(sym, p, "KILL", candle_num)
                    if c: closed.append(c)
        return closed

    def _close(self, symbol, exit_price, reason, candle_num=0):
        if symbol not in self.positions: return None
        pos = self.positions[symbol]
        cfg = self.cfg

        slip = cfg.slippage_bps / 10000
        actual = exit_price*(1-slip) if pos.side=="long" else exit_price*(1+slip)

        notional = pos.size_usd * pos.leverage
        if pos.side == "long":
            ppnl = (actual - pos.entry_price) / pos.entry_price * notional
        else:
            ppnl = (pos.entry_price - actual) / pos.entry_price * notional

        exit_fee = notional * cfg.fee_rate
        total_pnl = ppnl - exit_fee - pos.fees_paid

        now = datetime.now(timezone.utc)
        pos.pnl = total_pnl
        pos.pnl_pct = total_pnl / pos.size_usd * 100 if pos.size_usd > 0 else 0
        pos.fees_paid += exit_fee
        pos.exit_price = actual; pos.exit_time = now
        pos.exit_candle = candle_num; pos.exit_reason = reason
        pos.hold_candles = candle_num - pos.entry_candle; pos.status = "closed"

        self.capital += total_pnl
        self.available += pos.size_usd + total_pnl
        if self.capital > self.peak: self.peak = self.capital

        self.closed.append(pos)
        del self.positions[symbol]

        # Cooldown
        cd = cfg.cooldown_after_loss if total_pnl < 0 else cfg.cooldown_candles
        self._cooldowns[symbol] = candle_num + cd

        self.equity_curve.append({
            "timestamp": now, "equity": round(self.capital, 4),
            "pnl": round(total_pnl, 6), "symbol": symbol,
            "strategy": pos.strategy, "reason": reason,
            "hold_min": round(pos.hold_candles, 1),
            "leverage": pos.leverage,
        })
        self._log("CLOSE", pos)
        return pos

    def update_unrealized(self, prices):
        for sym, pos in self.positions.items():
            p = prices.get(sym)
            if p is None: continue
            n = pos.size_usd * pos.leverage
            if pos.side == "long":
                pos.pnl = (p - pos.entry_price) / pos.entry_price * n - pos.fees_paid
            else:
                pos.pnl = (pos.entry_price - p) / pos.entry_price * n - pos.fees_paid
            pos.pnl_pct = pos.pnl / pos.size_usd * 100 if pos.size_usd > 0 else 0

    def _log(self, action, pos, sig=None):
        now = datetime.now(timezone.utc)
        e = {
            "ts": now.strftime("%H:%M:%S"),
            "ts_full": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "action": action, "symbol": pos.symbol,
            "side": pos.side.upper(), "strat": pos.strategy,
            "price": round(pos.entry_price if action=="OPEN" else pos.exit_price, 4),
            "size": round(pos.size_usd, 2), "lev": pos.leverage,
            "pnl": round(pos.pnl, 4), "pnl_pct": round(pos.pnl_pct, 2),
            "fees": round(pos.fees_paid, 4),
        }
        if action == "OPEN" and sig:
            e["edge_bps"] = sig.edge_bps
            e["reason"] = sig.reason
        if action == "CLOSE":
            e["exit_reason"] = pos.exit_reason
            e["hold_candles"] = pos.hold_candles
            e["hold_min"] = pos.hold_candles  # 1 candle = 1 min
        self.trade_log.append(e)

    def get_metrics(self):
        return Metrics.compute(self.equity_curve, self.closed, self.cfg.initial_capital)

    def reset(self): self.__init__(self.cfg)

    def save(self, path="g5_state.json"):
        state = {"capital":self.capital,"available":self.available,"peak":self.peak,
            "killed":self.killed,"trade_log":self.trade_log[-500:],
            "equity_curve":[{**e,"timestamp":e["timestamp"].isoformat()
                if isinstance(e["timestamp"],datetime) else e["timestamp"]}
                for e in self.equity_curve[-1000:]]}
        with open(path,"w") as f: json.dump(state, f, default=str)

    def load(self, path="g5_state.json"):
        p = Path(path)
        if not p.exists(): return
        with open(path) as f: state = json.load(f)
        self.capital=state.get("capital",self.cfg.initial_capital)
        self.available=state.get("available",self.capital)
        self.peak=state.get("peak",self.capital)
        self.killed=state.get("killed",False)
        self.trade_log=state.get("trade_log",[])
