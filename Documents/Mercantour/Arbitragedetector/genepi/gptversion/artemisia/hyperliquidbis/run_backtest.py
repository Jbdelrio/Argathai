"""
Artemisia Glacialis ? Full Backtest + Optimization Pipeline
Usage:
  python run_backtest.py              # download + optimize + update config
  python run_backtest.py --no-dl      # skip download (use cached data)
  python run_backtest.py --force-dl   # force re-download all data
  python run_backtest.py --dry-run    # optimize but don't update config.py
  python run_backtest.py --quick      # fast mode: 100 random params instead of 400
"""
import sys, json, pathlib, time
from datetime import datetime, timezone

from backtest_data import download_all, summarize, SYMBOLS
from backtester import optimize, apply_best_params, run_backtest, BASE_PARAMS, PARAM_RANGES


def parse_args():
    args = sys.argv[1:]
    return {
        "no_dl":    "--no-dl"    in args,
        "force_dl": "--force-dl" in args,
        "dry_run":  "--dry-run"  in args,
        "quick":    "--quick"    in args,
    }


def update_config_py(best: dict, config_py: str = "config.py") -> None:
    """Directly patch the default values in config.py."""
    import re

    path = pathlib.Path(config_py)
    if not path.exists():
        print(f"  {config_py} not found ? skipping")
        return

    src = path.read_text(encoding="utf-8")
    original = src

    replacements = {
        r"(ema_fast:\s*int\s*=\s*)\d+":         f"\\g<1>{best['ema_fast']}",
        r"(ema_slow:\s*int\s*=\s*)\d+":          f"\\g<1>{best['ema_slow']}",
        r"(ema_atr_mult:\s*float\s*=\s*)[\d.]+": f"\\g<1>{best['ema_atr_mult']}",
        r"(ema_hold_candles:\s*int\s*=\s*)\d+":  f"\\g<1>{best['ema_hold_candles']}",
        r"(rsi_period:\s*int\s*=\s*)\d+":         f"\\g<1>{best['rsi_period']}",
        r"(rsi_oversold:\s*float\s*=\s*)[\d.]+":  f"\\g<1>{best['rsi_oversold']}",
        r"(rsi_overbought:\s*float\s*=\s*)[\d.]+":f"\\g<1>{best['rsi_overbought']}",
        r"(rsi_hold_candles:\s*int\s*=\s*)\d+":   f"\\g<1>{best['rsi_hold_candles']}",
        r"(vwap_period:\s*int\s*=\s*)\d+":         f"\\g<1>{best['vwap_period']}",
        r"(vwap_max_dist:\s*float\s*=\s*)[\d.]+":  f"\\g<1>{best['vwap_max_dist']}",
        r"(vwap_atr_mult:\s*float\s*=\s*)[\d.]+":  f"\\g<1>{best['vwap_atr_mult']}",
        r"(vwap_hold_candles:\s*int\s*=\s*)\d+":   f"\\g<1>{best['vwap_hold_candles']}",
        r"(stop_pct:\s*float\s*=\s*)[\d.]+":       f"\\g<1>{best['stop_pct']}",
        r"(tp_pct:\s*float\s*=\s*)[\d.]+":         f"\\g<1>{best['tp_pct']}",
    }

    for pattern, replacement in replacements.items():
        src = re.sub(pattern, replacement, src)

    if src == original:
        print("  config.py: no changes needed (values already match)")
        return

    # Backup original
    backup = pathlib.Path(config_py + ".bak")
    backup.write_text(original, encoding="utf-8")
    path.write_text(src, encoding="utf-8")
    print(f"  config.py updated  (backup ? config.py.bak)")


def print_comparison(old: dict, new: dict) -> None:
    """Show before/after parameter comparison."""
    keys = [
        "stop_pct", "tp_pct",
        "ema_fast", "ema_slow", "ema_atr_mult", "ema_hold_candles",
        "rsi_period", "rsi_oversold", "rsi_overbought", "rsi_hold_candles",
        "vwap_period", "vwap_max_dist", "vwap_atr_mult", "vwap_hold_candles",
    ]
    print("\n--- Parameter Changes ------------------------------------------")
    print(f"  {'Parameter':<25}  {'Before':>10}  {'After':>10}  {'?'}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*10}  {'-'*10}")
    for k in keys:
        ov = old.get(k, "?")
        nv = new.get(k, "?")
        changed = " ?" if ov != nv else ""
        print(f"  {k:<25}  {str(ov):>10}  {str(nv):>10}{changed}")


def main():
    opts = parse_args()
    print("+==========================================================+")
    print("|   Artemisia Glacialis ? Backtest & Optimizer v1          |")
    print(f"|   {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'):<54}|")
    print("+==========================================================+\n")

    # -- Step 1: Load data ---------------------------------------------------
    print("? Step 1/4 ? Load historical data")
    trading_symbols = [s for s in SYMBOLS if s != "BTC"]  # BTC excluded from trading

    if opts["no_dl"]:
        print("  Skipping download (--no-dl flag)")
        # Load from cache only
        from backtest_data import DATA_DIR
        data = {}
        for sym in SYMBOLS:
            cache = DATA_DIR / f"candles_{sym}_7d.json"
            if cache.exists():
                with open(cache) as f:
                    data[sym] = json.load(f)
                print(f"  {sym:10s}: {len(data[sym]):6d} candles [cache]")
            else:
                print(f"  {sym:10s}: NOT FOUND (run without --no-dl)")
    else:
        print(f"  Downloading {len(SYMBOLS)} symbols ? 7 days ? 1-min candles...")
        data = download_all(
            symbols=SYMBOLS,
            days=7,
            force=opts["force_dl"],
            cache_max_age_h=6.0 if not opts["force_dl"] else 0,
        )

    summarize(data)

    # Only trade with symbols that have data
    tradeable = {s: data[s] for s in trading_symbols if data.get(s) and len(data[s]) > 100}
    print(f"\n  Trading universe: {list(tradeable.keys())}")
    if len(tradeable) < 3:
        print("  ERROR: Not enough data. Check network or use --force-dl")
        sys.exit(1)

    # -- Step 2: Baseline backtest with current config -----------------------
    print("\n? Step 2/4 ? Baseline backtest (current config.py params)")
    # Read current defaults from config.py
    try:
        from config import Config
        cfg = Config()
        current_params = {
            "initial_capital": cfg.initial_capital,
            "base_position_pct": cfg.base_position_pct,
            "max_leverage": cfg.max_leverage,
            "min_leverage": cfg.min_leverage,
            "max_simultaneous": cfg.max_simultaneous,
            "maker_fee": cfg.maker_fee,
            "slippage_bps": cfg.slippage_bps,
            "stop_pct": cfg.stop_pct,
            "tp_pct": cfg.tp_pct,
            "ema_fast": cfg.ema_fast,
            "ema_slow": cfg.ema_slow,
            "ema_atr_mult": cfg.ema_atr_mult,
            "ema_hold_candles": cfg.ema_hold_candles,
            "rsi_period": cfg.rsi_period,
            "rsi_oversold": cfg.rsi_oversold,
            "rsi_overbought": cfg.rsi_overbought,
            "rsi_hold_candles": cfg.rsi_hold_candles,
            "vwap_period": cfg.vwap_period,
            "vwap_max_dist": cfg.vwap_max_dist,
            "vwap_atr_mult": cfg.vwap_atr_mult,
            "vwap_hold_candles": cfg.vwap_hold_candles,
            "breakeven_after_candles": cfg.breakeven_after_candles,
            "cooldown_candles": cfg.cooldown_candles,
            "cooldown_after_loss": cfg.cooldown_after_loss,
            "max_trades_per_symbol_hour": cfg.max_trades_per_symbol_hour,
            "min_edge_bps": cfg.min_edge_bps,
            "enable_ema": cfg.enable_ema,
            "enable_rsi": cfg.enable_rsi,
            "enable_vwap": cfg.enable_vwap,
        }
    except Exception as e:
        print(f"  Warning: could not import Config ({e}), using built-in defaults")
        current_params = dict(BASE_PARAMS)
        current_params.update({k: v[len(v)//2] for k, v in PARAM_RANGES.items()})

    baseline = run_backtest(tradeable, current_params)
    print(
        f"  Baseline ? Sharpe={baseline['sharpe']:+.3f}  "
        f"PnL={baseline['total_pnl_pct']:+.1f}%  "
        f"WR={baseline['win_rate']:.1f}%  "
        f"Trades={baseline['trades']}  "
        f"MaxDD={baseline['max_dd_pct']:.1f}%"
    )
    if baseline.get("by_strategy"):
        for s, d in baseline["by_strategy"].items():
            print(f"    {s}: {d['trades']} trades, WR={d['win_rate']}%, PnL={d['pnl']:+.2f}")

    # -- Step 3: Optimize ----------------------------------------------------
    n_random = 100 if opts["quick"] else 300
    print(f"\n? Step 3/4 ? Optimization ({n_random} random + grid search)")
    best_params, best_metrics, all_results = optimize(
        tradeable,
        n_random=n_random,
        min_trades=15,
        max_dd_limit=30.0,
        verbose=True,
    )

    print(f"\n  Improvement over baseline:")
    print(f"    Sharpe:   {baseline['sharpe']:+.3f} ? {best_metrics['sharpe']:+.3f}")
    print(f"    PnL%:     {baseline['total_pnl_pct']:+.1f}% ? {best_metrics['total_pnl_pct']:+.1f}%")
    print(f"    Win rate: {baseline['win_rate']:.1f}% ? {best_metrics['win_rate']:.1f}%")
    print(f"    MaxDD:    {baseline['max_dd_pct']:.1f}% ? {best_metrics['max_dd_pct']:.1f}%")
    print(f"    Trades:   {baseline['trades']} ? {best_metrics['trades']}")

    print_comparison(current_params, best_params)

    # -- Step 4: Apply best params -------------------------------------------
    print("\n? Step 4/4 ? Apply best parameters")

    if opts["dry_run"]:
        print("  --dry-run flag set: skipping config update")
    else:
        # Check: only update if optimizer found something better
        if best_metrics["sharpe"] <= baseline["sharpe"] and best_metrics["sharpe"] < 0:
            print(
                "  WARNING: Optimizer found no improvement over baseline.\n"
                "  Possible causes: only 7 days data, low volatility period, or params already optimal.\n"
                "  Keeping current config.py unchanged."
            )
        else:
            print("  Updating g5_config.json (JSON override for Config.load())...")
            apply_best_params(best_params)

            print("  Patching config.py default values...")
            update_config_py(best_params)

    # -- Save full results ---------------------------------------------------
    results_path = pathlib.Path("backtest_results.json")
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_symbols": list(tradeable.keys()),
        "data_candles": {s: len(v) for s, v in tradeable.items()},
        "baseline": {"metrics": baseline, "params": current_params},
        "best": {"metrics": best_metrics, "params": best_params},
        "top10": [
            {"rank": i+1, "score": r["score"], "metrics": r["metrics"],
             "params": {k: r["params"][k] for k in PARAM_RANGES}}
            for i, r in enumerate(all_results[:10])
        ],
    }
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Full results saved to {results_path}")

    # -- Final summary -------------------------------------------------------
    print("\n+==========================================================+")
    print("|   OPTIMIZATION COMPLETE                                  |")
    print("|==========================================================|")
    print(f"|  Sharpe:     {best_metrics['sharpe']:+.3f}                                    |")
    print(f"|  PnL (7d):  {best_metrics['total_pnl_pct']:+.1f}% on $500                        |")
    print(f"|  Win rate:  {best_metrics['win_rate']:.1f}%                                    |")
    print(f"|  Max DD:    {best_metrics['max_dd_pct']:.1f}%                                    |")
    print(f"|  Trades:    {best_metrics['trades']}                                      |")
    print("|==========================================================|")
    print(f"|  stop={best_params['stop_pct']*100:.2f}%  tp={best_params['tp_pct']*100:.2f}%                              |")
    print(f"|  EMA({best_params['ema_fast']},{best_params['ema_slow']})  RSI({best_params['rsi_period']},{best_params['rsi_oversold']:.0f},{best_params['rsi_overbought']:.0f})  VWAP(p={best_params['vwap_period']}) |")
    print("+==========================================================+")

    if not opts["dry_run"]:
        print("\nNext steps:")
        print("  1. Review config.py.bak vs config.py to verify changes")
        print("  2. Run paper trading: python engine.py (or your main script)")
        print("  3. Monitor live performance vs backtest predictions")


if __name__ == "__main__":
    main()
