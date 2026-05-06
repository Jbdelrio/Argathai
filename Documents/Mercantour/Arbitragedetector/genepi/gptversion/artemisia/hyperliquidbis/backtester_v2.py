"""
Artemisia Glacialis v6 -- Universal Backtester
Works with any strategy from strategies_v2.py.
Supports: single run, grid search, forward test, portfolio combination.

Key design:
  - Data loaded once as pandas DataFrames per symbol
  - Strategy precomputes indicators once per symbol
  - Grid search reuses precomputed data across all parameter combos
  - Parallel grid search via concurrent.futures
"""
import json, time, random, pathlib, itertools
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any

from strategies_v2 import (
    SpikeReversion, FundingHarvest, LiquidationCascade,
    MomentumCascade, VolRegimeSwitch, VALIDATION_CRITERIA,
)

DATA_DIR = pathlib.Path("data")
ROUNDTRIP_FEES = 0.00048     # 0.016% × 2 + 0.8bp × 2 ≈ 0.048%
SLIPPAGE      = 0.00008      # 0.8bp


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(pnls: List[float], equity: List[float],
                    initial_cap: float, n_days: float) -> dict:
    n = len(pnls)
    if n == 0:
        return {k: 0 for k in [
            "trades", "wins", "win_rate", "total_pnl", "total_pnl_pct",
            "profit_factor", "expectancy", "sharpe", "max_dd_pct",
            "trades_per_day", "avg_edge_bps",
        ]}
    wins   = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x <= 0]
    tp = sum(pnls)
    gp = sum(wins) if wins else 0
    gl = abs(sum(losses)) if losses else 1e-9
    pf = gp / gl

    # Sharpe from equity curve
    eq = np.array(equity)
    rets = np.diff(eq) / np.maximum(eq[:-1], 0.01)
    rets = rets[np.isfinite(rets)]
    sharpe = 0.0
    if len(rets) > 5 and np.std(rets) > 0:
        tpd = max(n / max(n_days, 1), 1)
        sharpe = float(np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(252 * tpd))

    # Max drawdown
    pk, mdd = initial_cap, 0.0
    for v in equity:
        if v > pk:
            pk = v
        dd = (pk - v) / pk if pk > 0 else 0
        if dd > mdd:
            mdd = dd

    # avg_edge_bps: P&L per trade as fraction of POSITION size
    # Assume avg position = initial_cap * base_pct (5%) → approx 5% of capital
    avg_pos_size = initial_cap * 0.05
    avg_edge = (tp / n) / avg_pos_size * 10_000 if n > 0 else 0

    return {
        "trades":        n,
        "wins":          len(wins),
        "win_rate":      round(len(wins) / n * 100, 1),
        "total_pnl":     round(tp, 2),
        "total_pnl_pct": round(tp / initial_cap * 100, 2),
        "profit_factor": round(pf, 3),
        "expectancy":    round(tp / n, 4),
        "sharpe":        round(sharpe, 3),
        "max_dd_pct":    round(mdd * 100, 2),
        "trades_per_day": round(n / max(n_days, 1), 1),
        "avg_edge_bps":  round(avg_edge, 1),
    }


def is_valid(m: dict) -> bool:
    v = VALIDATION_CRITERIA
    return (
        m["sharpe"]         >= v["sharpe"]
        and m["win_rate"]   >= v["win_rate"]
        and m["profit_factor"] >= v["profit_factor"]
        and m["max_dd_pct"] <= v["max_dd_pct"]
        and m["trades_per_day"] >= v["trades_per_day"]
        and m["avg_edge_bps"] >= v["edge_bps"]
    )


# ── Single-strategy universal backtest ───────────────────────────────────────

def run_single_strategy(
    strategy_cls,
    candles_by_sym: Dict[str, List[dict]],
    params: dict,
    initial_cap: float = 500.0,
    max_pos: int = 5,
    base_pct: float = 0.05,
    max_lev: float = 8.0,
    min_lev: float = 3.0,
    leverage_override: Optional[float] = None,
) -> dict:
    """
    Runs a complete backtest for one strategy/params combination.
    Returns metrics dict.
    """
    # Precompute indicators for each symbol
    syms = list(candles_by_sym.keys())
    precomputed = {}
    for sym in syms:
        pre = strategy_cls.precompute(candles_by_sym[sym])
        if pre is not None:
            precomputed[sym] = pre

    if not precomputed:
        return compute_metrics([], [initial_cap], initial_cap, 1)

    # Find common candle length
    lengths = {s: len(candles_by_sym[s]) for s in precomputed}
    min_len = min(lengths.values())
    if min_len < 50:
        return compute_metrics([], [initial_cap], initial_cap, 1)

    n_days = min_len / 1440.0  # 1440 min/day

    # State
    capital   = initial_cap
    available = initial_cap
    peak      = initial_cap

    positions: Dict[str, dict] = {}   # sym -> {side,entry,stop,tp,max_hold,idx,size,lev,strategy}
    cooldowns: Dict[str, int]  = {}
    pnls:  List[float] = []
    equity: List[float] = [initial_cap]

    lev = leverage_override if leverage_override else (min_lev + max_lev) / 2

    # For VolRegimeSwitch we need to track regime per symbol
    prev_regimes: Dict[str, str] = {s: "NORMAL" for s in precomputed}

    warmup = max(50, params.get("spike_lookback", 0) + params.get("atr_period", 14),
                 params.get("trend_lookback", 60), params.get("spread_window", 60),
                 params.get("vol_medium", 30) + params.get("min_squeeze_candles", 5))
    warmup = min(warmup, min_len // 3)

    for idx in range(warmup, min_len):
        # ── Check exits ──────────────────────────────────────────────────
        for sym in list(positions.keys()):
            pos  = positions[sym]
            clist = candles_by_sym[sym]
            if idx >= len(clist):
                continue
            price = clist[idx]["c"]
            hold  = idx - pos["idx"]

            reason = None
            if pos["side"] == "long":
                if price <= pos["stop"]:  reason = "STOP"
                elif price >= pos["tp"]:  reason = "TP"
            else:
                if price >= pos["stop"]:  reason = "STOP"
                elif price <= pos["tp"]:  reason = "TP"

            if reason is None and hold >= pos["max_hold"]:
                reason = "MAX_HOLD"

            if reason:
                notional = pos["size"] * pos["lev"]
                actual   = price * (1 - SLIPPAGE) if pos["side"] == "long" else price * (1 + SLIPPAGE)
                if pos["side"] == "long":
                    raw_pnl = (actual - pos["entry"]) / pos["entry"] * notional
                else:
                    raw_pnl = (pos["entry"] - actual) / pos["entry"] * notional
                fee = notional * ROUNDTRIP_FEES
                pnl = raw_pnl - fee

                capital   += pnl
                available += pos["size"] + pnl
                if capital > peak:
                    peak = capital
                pnls.append(pnl)
                equity.append(capital)

                cooldowns[sym] = idx + (2 if pnl < 0 else 1)
                del positions[sym]

        # Kill at 10% drawdown
        if capital < initial_cap * 0.90:
            break

        if len(positions) >= max_pos or available < 5.0:
            continue

        # ── Generate signals ──────────────────────────────────────────────
        for sym, pre in precomputed.items():
            if sym in positions or idx < cooldowns.get(sym, 0):
                continue

            # Dispatch to correct signal method
            if strategy_cls.NAME == "VOL_REGIME":
                result = strategy_cls.signal(idx, pre, params, prev_regimes.get(sym, "NORMAL"))
                if result and "_regime" in result:
                    prev_regimes[sym] = result["_regime"]
                    if "side" not in result:
                        continue
                elif not result:
                    continue
                sig = result
            else:
                sig = strategy_cls.signal(idx, pre, params)
                if sig is None:
                    continue

            if "side" not in sig:
                continue

            # Open position
            price = candles_by_sym[sym][idx]["c"]
            entry = price * (1 + SLIPPAGE) if sig["side"] == "long" else price * (1 - SLIPPAGE)
            stop_dist = sig["stop_dist"]
            tp_dist   = sig["tp_dist"]

            if stop_dist <= 0 or tp_dist <= 0 or entry <= 0:
                continue

            ef   = min(tp_dist / (ROUNDTRIP_FEES + 1e-10) / 20, 2.0)
            size = float(np.clip(available * base_pct * ef, 5.0, available * 0.25))

            if sig["side"] == "long":
                stop = entry * (1 - stop_dist)
                tp   = entry * (1 + tp_dist)
            else:
                stop = entry * (1 + stop_dist)
                tp   = entry * (1 - tp_dist)

            positions[sym] = {
                "side": sig["side"], "entry": entry, "stop": stop, "tp": tp,
                "max_hold": sig["max_hold"], "idx": idx,
                "size": size, "lev": lev, "strategy": sig["strategy"],
            }
            available -= size

            if len(positions) >= max_pos:
                break

    # Close remaining
    for sym, pos in positions.items():
        clist = candles_by_sym[sym]
        price = clist[-1]["c"] if clist else 0
        if price > 0:
            notional = pos["size"] * pos["lev"]
            if pos["side"] == "long":
                raw = (price - pos["entry"]) / pos["entry"] * notional
            else:
                raw = (pos["entry"] - price) / pos["entry"] * notional
            fee = notional * ROUNDTRIP_FEES
            pnls.append(raw - fee)

    return compute_metrics(pnls, equity, initial_cap, n_days)


# ── Grid search ───────────────────────────────────────────────────────────────

def _grid_worker(args):
    """Top-level function for multiprocessing."""
    strat_name, candles, params, cap = args
    from strategies_v2 import ALL_STRATEGIES
    cls = ALL_STRATEGIES[strat_name]
    m = run_single_strategy(cls, candles, params, initial_cap=cap)
    return params, m


def grid_search(
    strategy_cls,
    candles_by_sym: Dict[str, List[dict]],
    param_grid: Optional[dict] = None,
    n_random: int = 300,
    initial_cap: float = 500.0,
    max_workers: int = 4,
    verbose: bool = True,
) -> List[dict]:
    """
    Combined grid + random search.
    Returns list of {params, metrics, score, valid} sorted by Sharpe desc.
    """
    if param_grid is None:
        param_grid = strategy_cls.PARAM_GRID

    # Build candidate list: full grid + random sample
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    all_combos = list(itertools.product(*values))
    all_params = [dict(zip(keys, combo)) for combo in all_combos]

    # Add random samples
    rng = random.Random(42)
    for _ in range(n_random):
        p = {}
        for k, v in param_grid.items():
            p[k] = rng.choice(v)
        all_params.append(p)

    # Deduplicate
    seen, unique = set(), []
    for p in all_params:
        key = json.dumps(p, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(p)

    total = len(unique)
    if verbose:
        print(f"  {strategy_cls.NAME}: {total} parameter combinations", flush=True)

    results = []
    t0 = time.time()

    # Run in parallel if possible
    try:
        args_list = [(strategy_cls.NAME, candles_by_sym, p, initial_cap) for p in unique]
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_grid_worker, a): i for i, a in enumerate(args_list)}
            done = 0
            for fut in as_completed(futs):
                p, m = fut.result()
                score = m["sharpe"] if m["trades"] >= 10 else -99.0
                results.append({"params": p, "metrics": m, "score": score, "valid": is_valid(m)})
                done += 1
                if verbose and done % max(1, total // 10) == 0:
                    best = max(results, key=lambda x: x["score"])
                    print(
                        f"    [{done:4d}/{total}] t={time.time()-t0:.0f}s  "
                        f"best_sharpe={best['score']:.3f}  "
                        f"best_wr={best['metrics']['win_rate']:.1f}%",
                        flush=True,
                    )
    except Exception:
        # Fallback: sequential
        for i, p in enumerate(unique):
            m = run_single_strategy(strategy_cls, candles_by_sym, p, initial_cap)
            score = m["sharpe"] if m["trades"] >= 10 else -99.0
            results.append({"params": p, "metrics": m, "score": score, "valid": is_valid(m)})
            if verbose and (i + 1) % max(1, total // 10) == 0:
                best = max(results, key=lambda x: x["score"])
                print(
                    f"    [{i+1:4d}/{total}] t={time.time()-t0:.0f}s  "
                    f"best={best['score']:.3f}",
                    flush=True,
                )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ── Forward test ──────────────────────────────────────────────────────────────

def forward_test(
    strategy_cls,
    candles_by_sym: Dict[str, List[dict]],
    best_params: dict,
    train_frac: float = 0.80,
    initial_cap: float = 500.0,
) -> Tuple[dict, dict]:
    """
    Split data into train/test. Backtest on train, forward test on test.
    Returns (train_metrics, test_metrics).
    """
    split_idx = {}
    for sym, clist in candles_by_sym.items():
        n = len(clist)
        cut = int(n * train_frac)
        split_idx[sym] = cut

    train_data = {s: v[:split_idx[s]] for s, v in candles_by_sym.items()}
    test_data  = {s: v[split_idx[s]:] for s, v in candles_by_sym.items()}

    m_train = run_single_strategy(strategy_cls, train_data, best_params, initial_cap)
    m_test  = run_single_strategy(strategy_cls, test_data,  best_params, initial_cap)
    return m_train, m_test


# ── Portfolio combination ────────────────────────────────────────────────────

def combine_strategies(
    validated: Dict[str, dict],   # strat_name -> {params, metrics}
    total_capital: float = 500.0,
) -> Dict[str, float]:
    """
    Allocate capital to validated strategies proportional to their Sharpe ratios.
    Returns {strat_name: allocation_fraction}.
    """
    if not validated:
        return {}
    sharpes = {name: max(d.get("metrics", d.get("best_metrics", {})).get("sharpe", 0.01), 0.01)
               for name, d in validated.items()}
    total_s = sum(sharpes.values())
    alloc = {name: s / total_s for name, s in sharpes.items()}
    return alloc


# ── Pretty print ──────────────────────────────────────────────────────────────

def print_result(r: dict, rank: int = 1, label: str = "", short: bool = False):
    m = r["metrics"]
    tag = "[OK]" if r.get("valid") else "[--]"
    print(
        f"  #{rank:02d} {tag} {label:12s} "
        f"Sharpe={m['sharpe']:+.3f} | PnL={m['total_pnl_pct']:+.1f}% | "
        f"WR={m['win_rate']:.1f}% | DD={m['max_dd_pct']:.1f}% | "
        f"T/day={m['trades_per_day']:.1f} | PF={m['profit_factor']:.2f}",
        flush=True,
    )
    if not short:
        print(
            f"       params: "
            + " ".join(f"{k}={v}" for k, v in list(r["params"].items())[:6]),
            flush=True,
        )


# ── Convenience: run all 5 backtestable strategies ───────────────────────────

def run_all_strategies(
    candles_by_sym: Dict[str, List[dict]],
    n_random: int = 200,
    initial_cap: float = 500.0,
    verbose: bool = True,
) -> Dict[str, dict]:
    """
    Runs grid search for each backtestable strategy.
    Returns dict: strat_name -> {best_params, best_metrics, all_results, valid}.
    """
    from strategies_v2 import ALL_STRATEGIES
    backtestable = [
        SpikeReversion, FundingHarvest, LiquidationCascade,
        MomentumCascade, VolRegimeSwitch,
    ]
    # Note: PairsMeanReversion requires paired data handling - handled separately

    output = {}
    for cls in backtestable:
        if verbose:
            print(f"\n--- {cls.NAME} ---", flush=True)
        results = grid_search(
            cls, candles_by_sym, n_random=n_random,
            initial_cap=initial_cap, verbose=verbose,
        )
        best = results[0] if results else None
        if verbose and best:
            print_result(best, rank=1, label=cls.NAME)

        m_train, m_test = {}, {}
        if best and best["params"]:
            m_train, m_test = forward_test(cls, candles_by_sym, best["params"], initial_cap=initial_cap)
            if verbose:
                print(f"  Forward test: Sharpe={m_test['sharpe']:+.3f} PnL={m_test['total_pnl_pct']:+.1f}%")

        output[cls.NAME] = {
            "best_params": best["params"] if best else {},
            "best_metrics": best["metrics"] if best else {},
            "forward_metrics": m_test,
            "all_results": results[:20],  # keep top 20
            "valid": best["valid"] if best else False,
        }

    return output
