"""
Artemisia Glacialis — Backtester + Parameter Optimizer
Exactly mirrors engine.py exit logic on historical 1-min candle data.
Maximizes Sharpe ratio via random parameter search.
"""
import json, time, random, pathlib, sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Indicator helpers (vectorized via pandas — fast C-level) ─────────────────

def ema_series(closes: np.ndarray, period: int) -> np.ndarray:
    """Full EMA series using pandas ewm (C-level, ~500x faster than Python loop)."""
    return pd.Series(closes).ewm(span=period, adjust=False).mean().values


def rsi_series(closes: np.ndarray, period: int) -> np.ndarray:
    """Wilder RSI series via pandas ewm."""
    s = pd.Series(closes)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)
    return rsi.values


def atr_series(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR series via pandas rolling mean."""
    n = len(closes)
    if n < 2:
        return np.zeros(n)
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])),
    )
    tr_full = np.concatenate([[tr[0]], tr])
    atr = pd.Series(tr_full).rolling(window=period, min_periods=1).mean().values
    return atr


def vwap_series(closes: np.ndarray, period: int) -> np.ndarray:
    """Rolling mean of closes as VWAP proxy via pandas rolling."""
    return pd.Series(closes).rolling(window=period, min_periods=1).mean().values


# ── Precomputed indicator cache (per symbol per param set) ───────────────────

def precompute(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    p: dict,
) -> dict:
    """Vectorized indicator computation for one symbol."""
    ef = ema_series(closes, p["ema_fast"])
    es = ema_series(closes, p["ema_slow"])
    rs = rsi_series(closes, p["rsi_period"])
    at = atr_series(highs, lows, closes, 14)
    atr_pct = np.where(closes > 0, at / closes, 0.0)
    vw = vwap_series(closes, p["vwap_period"])
    return {
        "ema_fast": ef,
        "ema_slow": es,
        "rsi": rs,
        "atr_pct": atr_pct,
        "vwap": vw,
        "closes": closes,
    }


# ── Position dataclass ───────────────────────────────────────────────────────

@dataclass
class Pos:
    sym: str
    side: str
    strategy: str
    entry: float
    idx: int
    size: float
    lev: float
    stop: float
    tp: float
    be_activated: bool = False


# ── Single backtest run ──────────────────────────────────────────────────────

def run_backtest(
    candles_by_sym: Dict[str, List[dict]],
    p: dict,
    skip_symbols: List[str] = None,
) -> dict:
    """
    Full vectorized-indicator, event-driven backtest.
    p: parameter dict (same keys as Config fields).
    Returns metrics dict.
    """
    if skip_symbols is None:
        skip_symbols = ["BTC"]

    # Unpack params
    initial_cap = p.get("initial_capital", 500.0)
    base_pct = p.get("base_position_pct", 0.06)
    max_lev = p.get("max_leverage", 8.0)
    min_lev = p.get("min_leverage", 3.0)
    max_pos = p.get("max_simultaneous", 3)
    stop_pct = p.get("stop_pct", 0.003)
    tp_pct = p.get("tp_pct", 0.002)
    be_after = p.get("breakeven_after_candles", 3)
    fee_rate = p.get("maker_fee", 0.00016)
    slip = p.get("slippage_bps", 0.8) / 10_000
    min_edge = p.get("min_edge_bps", 5.0)
    cd_n = p.get("cooldown_candles", 1)
    cd_loss = p.get("cooldown_after_loss", 2)
    max_hour = p.get("max_trades_per_symbol_hour", 8)
    hold = {
        "EMA": p.get("ema_hold_candles", 5),
        "RSI": p.get("rsi_hold_candles", 3),
        "VWAP": p.get("vwap_hold_candles", 4),
    }
    fee_bps = (fee_rate * 2 + slip * 2) * 10_000

    enable_ema = p.get("enable_ema", True)
    enable_rsi = p.get("enable_rsi", True)
    enable_vwap = p.get("enable_vwap", True)
    ema_atr_mult = p.get("ema_atr_mult", 0.8)
    rsi_os = p.get("rsi_oversold", 28.0)
    rsi_ob = p.get("rsi_overbought", 72.0)
    vwap_dist = p.get("vwap_max_dist", 0.003)
    vwap_atr = p.get("vwap_atr_mult", 0.7)

    # Align symbols and precompute indicators
    trading_syms = [s for s in candles_by_sym if s not in skip_symbols and candles_by_sym[s]]
    if not trading_syms:
        return _empty_metrics()

    # Find common length (min across symbols)
    lengths = {s: len(candles_by_sym[s]) for s in trading_syms}
    min_len = min(lengths.values())
    if min_len < 50:
        return _empty_metrics()

    # Build numpy arrays per symbol (truncate to min_len)
    arrays: Dict[str, dict] = {}
    for sym in trading_syms:
        sl = candles_by_sym[sym][:min_len]
        cl = np.array([c["c"] for c in sl], dtype=float)
        hi = np.array([c["h"] for c in sl], dtype=float)
        lo = np.array([c["l"] for c in sl], dtype=float)
        arrays[sym] = precompute(cl, hi, lo, p)

    warmup = max(
        p.get("ema_slow", 15) + 2,
        p.get("rsi_period", 14) + 2,
        p.get("vwap_period", 20) + 2,
    )

    # Backtest state
    capital = initial_cap
    available = initial_cap
    peak = initial_cap
    max_dd = 0.0

    positions: Dict[str, Pos] = {}
    cooldowns: Dict[str, int] = {}
    trade_hist: Dict[str, List[int]] = {s: [] for s in trading_syms}

    pnls: List[float] = []
    fees_paid: List[float] = []
    equity_curve: List[float] = [initial_cap]
    by_strat: Dict[str, dict] = {s: {"t": 0, "w": 0, "pnl": 0.0} for s in ["EMA", "RSI", "VWAP"]}

    # EMA cross state
    ema_state: Dict[str, str] = {}

    for idx in range(warmup, min_len):
        # ── Check exits ──────────────────────────────────────────────────
        for sym in list(positions.keys()):
            pos = positions[sym]
            price = arrays[sym]["closes"][idx]
            hold_n = idx - pos.idx
            reason = None

            if pos.side == "long":
                if price <= pos.stop:
                    reason = "STOP"
                elif price >= pos.tp:
                    reason = "TP"
            else:
                if price >= pos.stop:
                    reason = "STOP"
                elif price <= pos.tp:
                    reason = "TP"

            # Breakeven
            if reason is None and hold_n >= be_after and not pos.be_activated:
                buf = fee_rate * 2
                if pos.side == "long" and price > pos.entry * (1 + buf):
                    pos.stop = pos.entry * (1 + buf)
                    pos.be_activated = True
                elif pos.side == "short" and price < pos.entry * (1 - buf):
                    pos.stop = pos.entry * (1 - buf)
                    pos.be_activated = True

            # Max hold
            if reason is None and hold_n >= hold.get(pos.strategy, 5):
                reason = "MAX_HOLD"

            if reason:
                notional = pos.size * pos.lev
                actual = price * (1 - slip) if pos.side == "long" else price * (1 + slip)
                if pos.side == "long":
                    raw_pnl = (actual - pos.entry) / pos.entry * notional
                else:
                    raw_pnl = (pos.entry - actual) / pos.entry * notional
                total_fee = notional * fee_rate * 2  # entry + exit
                trade_pnl = raw_pnl - total_fee

                capital += trade_pnl
                available += pos.size + trade_pnl
                if capital > peak:
                    peak = capital
                dd = (peak - capital) / peak if peak > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd

                pnls.append(trade_pnl)
                fees_paid.append(total_fee)
                equity_curve.append(capital)

                bs = by_strat[pos.strategy]
                bs["t"] += 1
                bs["pnl"] += trade_pnl
                if trade_pnl > 0:
                    bs["w"] += 1

                cooldowns[sym] = idx + (cd_loss if trade_pnl < 0 else cd_n)
                del positions[sym]

        # Kill switch
        if capital < initial_cap * 0.90:
            break

        if len(positions) >= max_pos or available < 5.0:
            continue

        # ── Generate signals ──────────────────────────────────────────────
        signals: List[Tuple[str, str, str, float, float]] = []  # sym, side, strat, edge, lev

        for sym in trading_syms:
            if sym in positions:
                continue
            if idx < cooldowns.get(sym, 0):
                continue
            # Hourly rate limit
            th = trade_hist[sym]
            th[:] = [t for t in th if idx - t < 60]
            if len(th) >= max_hour:
                continue

            ind = arrays[sym]
            price = ind["closes"][idx]
            atr_p = ind["atr_pct"][idx]
            if atr_p <= 0 or price <= 0:
                continue

            # EMA crossover
            if enable_ema:
                ef_val = ind["ema_fast"][idx]
                es_val = ind["ema_slow"][idx]
                cur_state = "above" if ef_val > es_val else "below"
                prev_state = ema_state.get(sym)
                ema_state[sym] = cur_state

                if prev_state is not None and prev_state != cur_state:
                    side = None
                    if prev_state == "below" and price > ef_val:
                        side = "long"
                    elif prev_state == "above" and price < ef_val:
                        side = "short"
                    if side:
                        net = atr_p * ema_atr_mult * 10_000 - fee_bps
                        if net >= min_edge:
                            lev = float(np.clip(net / 10, min_lev, max_lev))
                            signals.append((sym, side, "EMA", net, lev))

            # RSI extremes
            if enable_rsi:
                rsi_v = ind["rsi"][idx]
                side = None
                ext = 0.0
                if rsi_v < rsi_os:
                    side = "long"
                    ext = (rsi_os - rsi_v) / rsi_os
                elif rsi_v > rsi_ob:
                    side = "short"
                    ext = (rsi_v - rsi_ob) / (100 - rsi_ob)
                if side:
                    net = atr_p * (0.8 + ext) * 10_000 - fee_bps
                    if net >= min_edge:
                        lev = float(np.clip(net / 10, min_lev, max_lev))
                        signals.append((sym, side, "RSI", net, lev))

            # VWAP cross
            if enable_vwap:
                vw = ind["vwap"][idx]
                if vw > 0 and idx > 0:
                    prev_p = ind["closes"][idx - 1]
                    dist = abs(price - vw) / vw
                    side = None
                    if prev_p < vw and price > vw and dist < vwap_dist:
                        side = "long"
                    elif prev_p > vw and price < vw and dist < vwap_dist:
                        side = "short"
                    if side:
                        net = atr_p * vwap_atr * 10_000 - fee_bps
                        if net >= min_edge:
                            lev = float(np.clip(net / 10, min_lev, max_lev))
                            signals.append((sym, side, "VWAP", net, lev))

        # ── Open positions (best edge, one per symbol) ────────────────────
        signals.sort(key=lambda x: x[3], reverse=True)
        seen_syms = set(positions.keys())
        slots = max_pos - len(positions)

        for sym, side, strat, net_edge, lev in signals:
            if slots <= 0 or available < 5.0:
                break
            if sym in seen_syms:
                continue
            seen_syms.add(sym)

            price = arrays[sym]["closes"][idx]
            entry = price * (1 + slip) if side == "long" else price * (1 - slip)
            ef = min(net_edge / 20.0, 2.0)
            size = float(np.clip(available * base_pct * ef, 5.0, available * 0.30))

            s_val = entry * (1 - stop_pct) if side == "long" else entry * (1 + stop_pct)
            t_val = entry * (1 + tp_pct) if side == "long" else entry * (1 - tp_pct)

            positions[sym] = Pos(
                sym=sym, side=side, strategy=strat,
                entry=entry, idx=idx,
                size=size, lev=lev,
                stop=s_val, tp=t_val,
            )
            available -= size
            trade_hist[sym].append(idx)
            slots -= 1

    # Close any open positions at last candle
    for sym, pos in positions.items():
        price = arrays[sym]["closes"][-1]
        notional = pos.size * pos.lev
        actual = price * (1 - slip) if pos.side == "long" else price * (1 + slip)
        if pos.side == "long":
            raw_pnl = (actual - pos.entry) / pos.entry * notional
        else:
            raw_pnl = (pos.entry - actual) / pos.entry * notional
        total_fee = notional * fee_rate * 2
        trade_pnl = raw_pnl - total_fee
        capital += trade_pnl
        pnls.append(trade_pnl)

    return _compute_metrics(pnls, fees_paid, equity_curve, initial_cap, max_dd, by_strat)


def _compute_metrics(pnls, fees, equity, cap, max_dd, by_strat) -> dict:
    n = len(pnls)
    if n == 0:
        return _empty_metrics()

    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x <= 0]
    total_pnl = sum(pnls)
    gp = sum(wins)
    gl = abs(sum(losses)) if losses else 0

    # Sharpe from equity curve (each step = 1 trade)
    eq = np.array(equity)
    rets = np.diff(eq) / np.maximum(eq[:-1], 0.01)
    rets = rets[np.isfinite(rets)]
    sharpe = 0.0
    if len(rets) > 5:
        mu = np.mean(rets)
        sig = np.std(rets, ddof=1)
        if sig > 0:
            trades_per_day = max(n / 7.0, 1.0)
            sharpe = float(mu / sig * np.sqrt(252.0 * trades_per_day))

    final_equity = equity[-1] if equity else cap
    by_s = {}
    for strat, d in by_strat.items():
        if d["t"] > 0:
            by_s[strat] = {
                "trades": d["t"],
                "win_rate": round(d["w"] / d["t"] * 100, 1),
                "pnl": round(d["pnl"], 2),
            }

    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / n * 100, 1),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl / cap * 100, 2),
        "total_fees": round(sum(fees), 2),
        "final_equity": round(final_equity, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "profit_factor": round(gp / gl, 3) if gl > 0 else 0,
        "expectancy": round(total_pnl / n, 4),
        "sharpe": round(sharpe, 3),
        "by_strategy": by_s,
    }


def _empty_metrics() -> dict:
    return {
        "trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
        "total_pnl": 0, "total_pnl_pct": 0, "total_fees": 0,
        "final_equity": 0, "max_dd_pct": 0, "profit_factor": 0,
        "expectancy": 0, "sharpe": -99.0, "by_strategy": {},
    }


# ── Parameter space ──────────────────────────────────────────────────────────

BASE_PARAMS = {
    "initial_capital": 500.0,
    "base_position_pct": 0.06,
    "max_leverage": 8.0,
    "min_leverage": 3.0,
    "max_simultaneous": 3,
    "maker_fee": 0.00016,
    "slippage_bps": 0.8,
    "breakeven_after_candles": 3,
    "cooldown_candles": 1,
    "cooldown_after_loss": 2,
    "max_trades_per_symbol_hour": 8,
    "min_edge_bps": 5.0,
    "enable_ema": True,
    "enable_rsi": True,
    "enable_vwap": True,
}

PARAM_RANGES = {
    # Global stops (most impactful)
    "stop_pct":        [0.0020, 0.0025, 0.0030, 0.0040, 0.0050],
    "tp_pct":          [0.0010, 0.0015, 0.0020, 0.0025, 0.0030],
    # EMA
    "ema_fast":        [3, 5, 8],
    "ema_slow":        [13, 21, 34],
    "ema_atr_mult":    [0.5, 0.8, 1.2, 1.5],
    "ema_hold_candles":[3, 5, 8],
    # RSI
    "rsi_period":      [7, 10, 14, 21],
    "rsi_oversold":    [22, 25, 28, 32],
    "rsi_overbought":  [68, 72, 75, 78],
    "rsi_hold_candles":[2, 3, 5],
    # VWAP
    "vwap_period":     [10, 15, 20, 30],
    "vwap_max_dist":   [0.002, 0.003, 0.004, 0.005],
    "vwap_atr_mult":   [0.5, 0.7, 1.0, 1.3],
    "vwap_hold_candles":[3, 4, 6],
}


def random_params(seed: Optional[int] = None) -> dict:
    rng = random.Random(seed)
    p = dict(BASE_PARAMS)
    for k, vals in PARAM_RANGES.items():
        p[k] = rng.choice(vals)
    # Ensure tp_pct < stop_pct (keep some risk:reward variety)
    # We allow tp < stop or tp > stop (mean reversion vs trend)
    # Ensure ema_fast < ema_slow
    while p["ema_fast"] >= p["ema_slow"]:
        p["ema_fast"] = rng.choice(PARAM_RANGES["ema_fast"])
        p["ema_slow"] = rng.choice(PARAM_RANGES["ema_slow"])
    return p


def grid_params() -> List[dict]:
    """
    2-level search:
    1. Global (stop/tp): 5×5 = 25
    2. Strategy params: 3×3 + 4×4 + 4×4 = 9+16+16 = 41
    Total grid points: 25 × 41 = 1025 (pruned to unique combos ~300)
    """
    params = []
    for stop in PARAM_RANGES["stop_pct"]:
        for tp in PARAM_RANGES["tp_pct"]:
            for ef, es in [(3,13),(3,21),(5,13),(5,21),(5,34),(8,21),(8,34)]:
                for ros, rob in [(22,78),(25,75),(28,72),(32,68)]:
                    p = dict(BASE_PARAMS)
                    p.update({
                        "stop_pct": stop, "tp_pct": tp,
                        "ema_fast": ef, "ema_slow": es,
                        "ema_atr_mult": 0.8, "ema_hold_candles": 5,
                        "rsi_period": 14, "rsi_oversold": ros, "rsi_overbought": rob,
                        "rsi_hold_candles": 3,
                        "vwap_period": 20, "vwap_max_dist": 0.003,
                        "vwap_atr_mult": 0.7, "vwap_hold_candles": 4,
                    })
                    params.append(p)
    return params


# ── Optimizer ────────────────────────────────────────────────────────────────

def optimize(
    candles_by_sym: Dict[str, List[dict]],
    n_random: int = 200,
    min_trades: int = 20,
    max_dd_limit: float = 25.0,
    verbose: bool = True,
) -> Tuple[dict, dict, List[dict]]:
    """
    Combined grid + random search.
    Returns (best_params, best_metrics, all_results_sorted).
    """
    candidates = grid_params()
    for i in range(n_random):
        candidates.append(random_params(seed=i))

    total = len(candidates)
    if verbose:
        print(f"\n=== Optimizer: {total} parameter sets ===")

    results = []
    t0 = time.time()

    for i, p in enumerate(candidates):
        m = run_backtest(candles_by_sym, p)

        # Filter: must have enough trades and acceptable drawdown
        valid = (
            m["trades"] >= min_trades
            and m["max_dd_pct"] <= max_dd_limit
            and m["sharpe"] > -90
        )
        score = m["sharpe"] if valid else -99.0

        results.append({"params": p, "metrics": m, "score": score, "valid": valid})

        if verbose and (i + 1) % 50 == 0:
            best = max(results, key=lambda x: x["score"])
            elapsed = time.time() - t0
            print(
                f"  [{i+1:4d}/{total}] "
                f"elapsed={elapsed:.0f}s  "
                f"best_sharpe={best['score']:.3f}  "
                f"best_wr={best['metrics']['win_rate']:.1f}%  "
                f"best_pnl={best['metrics']['total_pnl_pct']:+.1f}%",
                flush=True,
            )

    results.sort(key=lambda x: x["score"], reverse=True)

    best = results[0]
    if verbose:
        elapsed = time.time() - t0
        print(f"\nOptimization done in {elapsed:.1f}s")
        _print_result(best, rank=1)

        print("\n── Top-5 parameter sets ──")
        for rank, r in enumerate(results[:5], 1):
            _print_result(r, rank=rank, short=True)

    return best["params"], best["metrics"], results


def _print_result(r: dict, rank: int = 1, short: bool = False) -> None:
    m = r["metrics"]
    p = r["params"]
    tag = "[VALID]" if r["valid"] else "[SKIP ]"
    print(
        f"  #{rank:02d} {tag} "
        f"Sharpe={m['sharpe']:+.3f} | "
        f"PnL={m['total_pnl_pct']:+.1f}% | "
        f"WR={m['win_rate']:.1f}% | "
        f"Trades={m['trades']} | "
        f"MaxDD={m['max_dd_pct']:.1f}% | "
        f"PF={m['profit_factor']:.2f}"
    )
    if not short:
        print(
            f"       stop={p['stop_pct']*100:.2f}% "
            f"tp={p['tp_pct']*100:.2f}% "
            f"EMA({p['ema_fast']},{p['ema_slow']}) "
            f"RSI({p['rsi_period']},{p['rsi_oversold']:.0f},{p['rsi_overbought']:.0f}) "
            f"VWAP(p={p['vwap_period']},d={p['vwap_max_dist']*100:.2f}%)"
        )
        if m.get("by_strategy"):
            for strat, sd in m["by_strategy"].items():
                print(f"       {strat}: {sd['trades']} trades, WR={sd['win_rate']}%, PnL={sd['pnl']:+.2f}")


def apply_best_params(best_params: dict, config_path: str = "g5_config.json") -> None:
    """Write best parameters to JSON config file (loaded by Config.load())."""
    # Only save non-default / strategy-related keys
    keys_to_save = [
        "stop_pct", "tp_pct",
        "ema_fast", "ema_slow", "ema_atr_mult", "ema_hold_candles",
        "rsi_period", "rsi_oversold", "rsi_overbought", "rsi_hold_candles",
        "vwap_period", "vwap_max_dist", "vwap_atr_mult", "vwap_hold_candles",
        "min_edge_bps", "cooldown_candles", "cooldown_after_loss",
        "max_trades_per_symbol_hour", "breakeven_after_candles",
    ]
    # Load existing config if present
    cfg_path = pathlib.Path(config_path)
    existing = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            existing = json.load(f)

    for k in keys_to_save:
        if k in best_params:
            existing[k] = best_params[k]

    with open(cfg_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\nBest parameters written to {config_path}")
    print("The bot will load these at startup via Config.load()")


if __name__ == "__main__":
    # Quick self-test with synthetic data
    print("Self-test: generating synthetic candles...")
    np.random.seed(42)
    n = 2000
    prices = 100.0 * np.cumprod(1 + np.random.normal(0, 0.001, n))
    synthetic = [
        {
            "t": i * 60_000,
            "o": float(prices[i]),
            "h": float(prices[i] * (1 + abs(np.random.normal(0, 0.0005)))),
            "l": float(prices[i] * (1 - abs(np.random.normal(0, 0.0005)))),
            "c": float(prices[i] * (1 + np.random.normal(0, 0.001))),
            "v": 1000.0,
        }
        for i in range(n)
    ]
    # Fix h/l
    for c in synthetic:
        c["h"] = max(c["o"], c["h"], c["c"])
        c["l"] = min(c["o"], c["l"], c["c"])

    test_data = {"ETH": synthetic, "SOL": synthetic, "DOGE": synthetic}
    m = run_backtest(test_data, dict(BASE_PARAMS, **{k: v[0] for k, v in PARAM_RANGES.items()}))
    print(f"Self-test result: {m['trades']} trades, Sharpe={m['sharpe']:.3f}, PnL={m['total_pnl_pct']:+.1f}%")
    print("OK — run run_backtest.py to launch the full optimizer.")
