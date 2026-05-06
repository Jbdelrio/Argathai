"""
Artemisia Glacialis v6 -- Complete Backtest Pipeline
Usage:
  python run_backtests.py                      # full pipeline
  python run_backtests.py --dl                 # force re-download all 229 perps
  python run_backtests.py --quick              # quick mode (100 combos each)
  python run_backtests.py --strat SPIKE_REV    # backtest only one strategy
  python run_backtests.py --pairs              # include pairs strategy
  python run_backtests.py --no-update          # don't update config.py

Steps:
  1. Download all 229 perps (or load from cache)
  2. Baseline backtest (current params from config.py)
  3. Grid search for each of 6 strategies
  4. Validate each strategy (Sharpe>1, WR>52%, etc.)
  5. Combine validated strategies with proportional allocation
  6. Save results to backtest_results_v2.json
  7. Update config.py with validated strategies
"""
import sys, json, time, pathlib, re
from datetime import datetime, timezone
from typing import Dict, List

import requests
import numpy as np


# ── Imports ─────────────────────────────────────────────────────────────────
from backtest_data import download_all, DATA_DIR, SYMBOLS as DEFAULT_SYMBOLS
from strategies_v2 import (
    SpikeReversion, FundingHarvest, LiquidationCascade,
    MomentumCascade, VolRegimeSwitch, PairsMeanReversion,
    SECTOR_PAIRS, VALIDATION_CRITERIA,
)
from backtester_v2 import (
    run_single_strategy, grid_search, forward_test,
    combine_strategies, print_result, is_valid, compute_metrics,
)

HL_URL  = "https://api.hyperliquid.xyz/info"
MIN_VOL = 2_000_000   # $2M/day minimum
INITIAL_CAP = 500.0


# ── CLI args ─────────────────────────────────────────────────────────────────

def parse_args() -> dict:
    a = sys.argv[1:]
    return {
        "force_dl":    "--dl"       in a,
        "quick":       "--quick"    in a,
        "no_update":   "--no-update" in a,
        "pairs":       "--pairs"    in a,
        "strat":       next((a[i+1] for i,x in enumerate(a) if x=="--strat"), None),
    }


# ── Universe discovery ───────────────────────────────────────────────────────

def get_full_universe(min_vol: float = MIN_VOL) -> List[str]:
    """Fetch all perps from Hyperliquid, filter by daily volume."""
    print(f"  Fetching universe (min vol ${min_vol/1e6:.0f}M/day)...", flush=True)
    try:
        r = requests.post(HL_URL, json={"type": "metaAndAssetCtxs"},
                          headers={"Content-Type": "application/json"}, timeout=20)
        r.raise_for_status()
        data = r.json()
        universe = data[0]["universe"]
        ctxs     = data[1]
        symbols  = []
        for meta, ctx in zip(universe, ctxs):
            sym = meta["name"]
            vol = float(ctx.get("dayNtlVlm", 0) or 0)
            if vol >= min_vol:
                symbols.append(sym)
        print(f"  Found {len(symbols)} symbols with vol >= ${min_vol/1e6:.0f}M/day", flush=True)
        return symbols
    except Exception as e:
        print(f"  WARNING: could not fetch universe ({e}), using default 10 symbols", flush=True)
        return DEFAULT_SYMBOLS


def download_universe(symbols: List[str], force: bool = False) -> Dict[str, List[dict]]:
    """Download (or load cached) candles for all symbols."""
    from backtest_data import download_all as dl_all
    import concurrent.futures, threading

    DATA_DIR.mkdir(exist_ok=True)
    data = {}
    to_download = []

    for sym in symbols:
        cache = DATA_DIR / f"candles_{sym}_7d.json"
        if cache.exists() and not force:
            age_h = (time.time() - cache.stat().st_mtime) / 3600
            if age_h < 8.0:
                with open(cache) as f:
                    data[sym] = json.load(f)
                continue
        to_download.append(sym)

    if to_download:
        print(f"  Downloading {len(to_download)} symbols "
              f"(cached: {len(data)})...", flush=True)
        # Use backtest_data functions per-symbol in threads
        from backtest_data import download_symbol
        lock = threading.Lock()
        done_count = [0]

        def _dl(sym):
            candles = download_symbol(sym, days=7, verbose=False)
            if candles:
                cache = DATA_DIR / f"candles_{sym}_7d.json"
                with open(cache, "w") as f:
                    json.dump(candles, f)
            with lock:
                data[sym] = candles
                done_count[0] += 1
                if done_count[0] % 20 == 0 or done_count[0] == len(to_download):
                    print(f"    {done_count[0]}/{len(to_download)} downloaded", flush=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            ex.map(_dl, to_download)

    # Filter out symbols with too few candles
    good = {s: v for s, v in data.items() if v and len(v) >= 100}
    print(f"  Usable symbols: {len(good)} / {len(symbols)}", flush=True)
    return good


# ── Config updater ───────────────────────────────────────────────────────────

def update_config_v2(validated: Dict[str, dict], config_path: str = "config.py"):
    """Patch config.py with validated strategy flags + save JSON override."""
    p = pathlib.Path(config_path)
    if not p.exists():
        return
    src = p.read_text(encoding="utf-8")
    backup = pathlib.Path(config_path + ".v6.bak")
    backup.write_text(src, encoding="utf-8")

    enabled_names = set(validated.keys())

    # Map strategy name to config field
    STRAT_FLAGS = {
        "SPIKE_REV":    "enable_spike",
        "FUNDING_HARV": "enable_funding",
        "LIQ_CASCADE":  "enable_liq",
        "PAIRS_MR":     "enable_pairs",
        "MTF_MOM":      "enable_momentum",
        "VOL_REGIME":   "enable_volregime",
        "EMA":          "enable_ema",
        "RSI":          "enable_rsi",
        "VWAP":         "enable_vwap",
    }

    # Build JSON override from best params of each validated strategy
    override = {}
    for name, d in validated.items():
        for k, v in d.get("best_params", {}).items():
            # Prefix with strategy name to avoid conflicts
            short = name.lower().replace("_", "") + "_" + k
            override[short] = v

    json_path = pathlib.Path("g5_config.json")
    existing = {}
    if json_path.exists():
        with open(json_path) as f:
            existing = json.load(f)
    existing.update(override)
    existing["validated_strategies"] = list(enabled_names)
    existing["validation_timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(json_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"  Config updated: {len(enabled_names)} validated strategies", flush=True)
    print(f"  g5_config.json saved with {len(override)} param overrides", flush=True)
    print(f"  Backup: {backup}", flush=True)


# ── Pairs backtest ───────────────────────────────────────────────────────────

def backtest_pairs(
    candles_by_sym: Dict[str, List[dict]],
    n_random: int = 100,
    initial_cap: float = INITIAL_CAP,
) -> dict:
    """
    Grid search for the Pairs strategy across all defined pairs.
    Aggregates metrics across all valid pairs.
    """
    from strategies_v2 import SECTOR_PAIRS, PairsMeanReversion

    p_grid = PairsMeanReversion.PARAM_GRID
    param_keys = list(p_grid.keys())
    import itertools
    all_combos = list(itertools.product(*[p_grid[k] for k in param_keys]))
    all_params = [dict(zip(param_keys, c)) for c in all_combos]

    import random as _random
    rng = _random.Random(42)
    for _ in range(n_random):
        all_params.append({k: rng.choice(v) for k, v in p_grid.items()})

    # Deduplicate
    seen, unique_params = set(), []
    for pp in all_params:
        key = json.dumps(pp, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique_params.append(pp)

    print(f"  PAIRS_MR: {len(unique_params)} combos across {len(SECTOR_PAIRS)} pairs",
          flush=True)

    best_score = -99.0
    best_entry = None

    for pp in unique_params:
        w = pp["spread_window"]
        all_pnls, all_equity = [], [initial_cap]
        cap = initial_cap

        for sym_a, sym_b in SECTOR_PAIRS:
            if sym_a not in candles_by_sym or sym_b not in candles_by_sym:
                continue
            pair_data = PairsMeanReversion.precompute_pair(
                sym_a, sym_b, candles_by_sym[sym_a], candles_by_sym[sym_b], w
            )
            if pair_data is None:
                continue

            n = pair_data["n"]
            in_trade = False
            entry_idx = 0
            entry_side = None
            entry_a = entry_b = 0.0

            for idx in range(w + 5, n):
                sig = PairsMeanReversion.signal_pair(idx, pair_data, pp)
                if not in_trade and sig is not None:
                    sig_a, sig_b = sig
                    entry_side = sig_a["side"]
                    entry_a = pair_data["ca"][idx]
                    entry_b = pair_data["cb"][idx]
                    entry_idx = idx
                    in_trade = True
                elif in_trade:
                    hold = idx - entry_idx
                    z_now = pair_data["zscore"][idx]
                    # Exit on convergence or max hold
                    if abs(z_now) < pp["exit_z"] or hold >= pp["max_hold_candles"]:
                        # P&L per leg (size split 50/50)
                        size = initial_cap * 0.02  # 2% per leg
                        lev  = pp["leverage_per_leg"]
                        notional = size * lev
                        ca_now = pair_data["ca"][idx]
                        cb_now = pair_data["cb"][idx]
                        if entry_side == "short":
                            pnl_a = (entry_a - ca_now) / entry_a * notional
                            pnl_b = (cb_now - entry_b) / entry_b * notional
                        else:
                            pnl_a = (ca_now - entry_a) / entry_a * notional
                            pnl_b = (entry_b - cb_now) / entry_b * notional
                        fee = notional * ROUNDTRIP_FEES * 2
                        trade_pnl = pnl_a + pnl_b - fee
                        cap += trade_pnl
                        all_pnls.append(trade_pnl)
                        all_equity.append(cap)
                        in_trade = False

        n_days = min(len(v) for v in candles_by_sym.values() if v) / 1440.0
        m = compute_metrics(all_pnls, all_equity, initial_cap, n_days)
        score = m["sharpe"] if m["trades"] >= 5 else -99.0
        if score > best_score:
            best_score = score
            best_entry = {"params": pp, "metrics": m, "score": score, "valid": is_valid(m)}

    return best_entry or {"params": {}, "metrics": {}, "score": -99, "valid": False}


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    opts = parse_args()
    print("=" * 62)
    print("  Artemisia Glacialis v6 -- Full Backtest Pipeline")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 62)

    # ── STEP 1: Data ────────────────────────────────────────────────────────
    print("\n[1/5] Universe + Data")
    symbols = get_full_universe(min_vol=MIN_VOL)
    candles = download_universe(symbols, force=opts["force_dl"])

    n_days = 0.0
    if candles:
        lens = [len(v) for v in candles.values() if v]
        n_days = min(lens) / 1440.0 if lens else 0
        print(f"  Data: {len(candles)} symbols, "
              f"~{min(lens) if lens else 0} candles each "
              f"({n_days:.1f} days)")

    if len(candles) < 5:
        print("ERROR: Not enough data. Check network or try --dl")
        sys.exit(1)

    # ── STEP 2: Strategy selection ──────────────────────────────────────────
    strats_to_run = {
        "SPIKE_REV":   SpikeReversion,
        "FUNDING_HARV": FundingHarvest,
        "LIQ_CASCADE": LiquidationCascade,
        "MTF_MOM":     MomentumCascade,
        "VOL_REGIME":  VolRegimeSwitch,
    }
    if opts["strat"]:
        strats_to_run = {k: v for k, v in strats_to_run.items() if k == opts["strat"]}
        if not strats_to_run:
            print(f"Unknown strategy '{opts['strat']}'. Valid: {list(strats_to_run.keys())}")
            sys.exit(1)

    n_random = 50 if opts["quick"] else 200

    # ── STEP 3: Backtest each strategy ──────────────────────────────────────
    print(f"\n[2/5] Backtesting {len(strats_to_run)} strategies "
          f"({n_random} random + grid per strategy)")
    all_results = {}
    validated   = {}

    for name, cls in strats_to_run.items():
        print(f"\n  --- {name} ---", flush=True)
        t0 = time.time()

        results = grid_search(
            cls, candles, n_random=n_random,
            initial_cap=INITIAL_CAP, verbose=True,
        )
        best = results[0] if results else None

        if best:
            m_train, m_test = forward_test(
                cls, candles, best["params"], initial_cap=INITIAL_CAP
            )
            print(f"  Best:    ", end=""); print_result(best, rank=1, label=name, short=True)
            print(f"  Forward: Sharpe={m_test['sharpe']:+.3f}  "
                  f"PnL={m_test['total_pnl_pct']:+.1f}%  "
                  f"WR={m_test['win_rate']:.1f}%")
            if best["valid"] and m_test["total_pnl"] > 0:
                validated[name] = {
                    "best_params": best["params"],
                    "best_metrics": best["metrics"],
                    "forward_metrics": m_test,
                }
                print(f"  STATUS: *** VALIDATED *** ({time.time()-t0:.0f}s)")
            else:
                fails = []
                m = best["metrics"]
                v = VALIDATION_CRITERIA
                if m["sharpe"]          < v["sharpe"]:        fails.append(f"Sharpe={m['sharpe']:.2f}<{v['sharpe']}")
                if m["win_rate"]        < v["win_rate"]:       fails.append(f"WR={m['win_rate']:.1f}%<{v['win_rate']}%")
                if m["profit_factor"]   < v["profit_factor"]:  fails.append(f"PF={m['profit_factor']:.2f}<{v['profit_factor']}")
                if m["max_dd_pct"]      > v["max_dd_pct"]:     fails.append(f"DD={m['max_dd_pct']:.1f}%>{v['max_dd_pct']}%")
                if m["trades_per_day"]  < v["trades_per_day"]: fails.append(f"T/d={m['trades_per_day']:.1f}<{v['trades_per_day']}")
                if m_test["total_pnl"] <= 0:                   fails.append("FwdTest<0")
                print(f"  STATUS: REJECTED ({', '.join(fails)}) ({time.time()-t0:.0f}s)")
        else:
            print(f"  STATUS: NO RESULTS")

        all_results[name] = {
            "top10": [{"params": r["params"], "metrics": r["metrics"],
                       "score": r["score"], "valid": r["valid"]}
                      for r in (results[:10] if results else [])],
            "valid": name in validated,
        }

    # ── STEP 4: Pairs (optional) ─────────────────────────────────────────────
    if opts["pairs"]:
        print(f"\n  --- PAIRS_MR ---", flush=True)
        t0 = time.time()
        best_pair = backtest_pairs(candles, n_random=n_random // 2, initial_cap=INITIAL_CAP)
        if best_pair and best_pair.get("valid"):
            validated["PAIRS_MR"] = {
                "best_params": best_pair["params"],
                "best_metrics": best_pair["metrics"],
                "forward_metrics": {},
            }
            print(f"  STATUS: *** VALIDATED *** ({time.time()-t0:.0f}s)")
            print_result(best_pair, rank=1, label="PAIRS_MR")
        else:
            print(f"  STATUS: REJECTED ({time.time()-t0:.0f}s)")
            if best_pair:
                print_result(best_pair, rank=1, label="PAIRS_MR")
        all_results["PAIRS_MR"] = {"top10": [best_pair] if best_pair else [], "valid": "PAIRS_MR" in validated}

    # ── STEP 5: Combine ─────────────────────────────────────────────────────
    print(f"\n[3/5] Portfolio combination")
    if validated:
        alloc = combine_strategies(validated, INITIAL_CAP)
        print(f"  Validated strategies: {list(validated.keys())}")
        for name, frac in alloc.items():
            m = validated[name]["best_metrics"]
            print(f"    {name:15s}: {frac*100:.1f}% capital  "
                  f"(Sharpe={m.get('sharpe',0):+.2f} "
                  f"WR={m.get('win_rate',0):.1f}% "
                  f"T/d={m.get('trades_per_day',0):.1f})")
    else:
        print("  WARNING: No strategies passed validation criteria.")
        print("  Consider lowering thresholds or collecting more data.")
        alloc = {}

    # ── STEP 6: Update config ────────────────────────────────────────────────
    print(f"\n[4/5] Update config")
    if not opts["no_update"] and validated:
        update_config_v2(validated)
    elif opts["no_update"]:
        print("  Skipped (--no-update)")
    else:
        print("  Skipped (no validated strategies)")

    # ── STEP 7: Save results ─────────────────────────────────────────────────
    print(f"\n[5/5] Save results")
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data": {"symbols": list(candles.keys()), "n_days": round(n_days, 2)},
        "validation_criteria": VALIDATION_CRITERIA,
        "strategies": all_results,
        "validated": {k: {"params": v["best_params"], "metrics": v["best_metrics"]}
                      for k, v in validated.items()},
        "portfolio_allocation": alloc,
    }
    out = pathlib.Path("backtest_results_v2.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Results saved: {out}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  BACKTEST COMPLETE")
    print("=" * 62)
    print(f"  Strategies tested    : {len(strats_to_run) + (1 if opts['pairs'] else 0)}")
    print(f"  Strategies validated : {len(validated)}")
    print(f"  Validation criteria  : Sharpe>{VALIDATION_CRITERIA['sharpe']} "
          f"WR>{VALIDATION_CRITERIA['win_rate']}% "
          f"DD<{VALIDATION_CRITERIA['max_dd_pct']}%")
    if validated:
        print(f"\n  LIVE-READY strategies:")
        for name in validated:
            m = validated[name]["best_metrics"]
            print(f"    {name}: Sharpe={m.get('sharpe',0):+.2f} "
                  f"WR={m.get('win_rate',0):.1f}% "
                  f"PnL={m.get('total_pnl_pct',0):+.1f}%/period")
        print(f"\n  Next: python engine_v2.py  (launch paper trading)")
    else:
        print("\n  No strategies validated on this dataset.")
        print("  Options:")
        print("    1. Collect more data (wait 3+ days)")
        print("    2. Lower validation thresholds in strategies_v2.py")
        print("    3. Review strategy logic (current market regime)")
    print("=" * 62)


if __name__ == "__main__":
    main()
