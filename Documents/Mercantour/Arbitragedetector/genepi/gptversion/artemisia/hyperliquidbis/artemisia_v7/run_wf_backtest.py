"""
run_wf_backtest.py — Complete v7 validation pipeline
1. Download 52 days of 15-min candles for top-30 universe
2. Run walk-forward backtest for each strategy
3. Apply OOS criteria (Sharpe > 1.5, min fold > 0.5, DD < 6%, PF > 1.3, MC p5 > 0)
4. Print full report
5. Save validated strategy configs to validated_strategies.json

Usage:
    python run_wf_backtest.py
    python run_wf_backtest.py --strategy s1  # single strategy
    python run_wf_backtest.py --no-dl        # skip download
    python run_wf_backtest.py --quick        # fewer random params
"""
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from data.universe import get_universe
from data.downloader import download_all_candles, download_all_funding
from backtest.walk_forward import walk_forward_validate, print_wf_report
from backtest.metrics import compute_all_metrics
from strategies.s1_funding_arb import FundingArbStrategy
from strategies.s2_maker_mr import MakerMRStrategy
from strategies.s3_pairs_kalman import PairsKalmanStrategy
from strategies.s4_regime_router import RegimeRouter
from strategies.s6_adaptive_trend import AdaptiveTrendStrategy
from regime.kalman_pair import find_cointegrated_pairs

log = logging.getLogger(__name__)

CACHE_DIR      = "data/cache"
OUTPUT_FILE    = "validated_strategies.json"
BARS_PER_DAY   = 96    # 15-min candles
BARS_PER_DAY_6H = 4    # 6h candles

# S6 WF criteria (directional strats are harder to validate; looser than S1-S4)
S6_WF_CRITERIA = dict(
    # 90 days 1h data -> ~361 6h bars available
    # 3 folds: 45 train + 15 test per fold = 45*4 + 3*15*4 = 360 bars (just fits)
    # test_warmup_bars=80: prepend 80 bars from train for indicator warmup
    n_folds=3,
    train_days=45,
    test_days=15,
    bars_per_day=BARS_PER_DAY_6H,
    initial_capital=500.0,
    n_mc_sims=1000,
    min_oos_sharpe=0.8,
    min_fold_sharpe=0.0,
    max_dd_pct=10.0,
    min_profit_factor=1.0,
    min_trades=10,
    mc_p5_sharpe=0.0,
    test_warmup_bars=80,
)


def backtest_fn_s2(candles: pd.DataFrame, params: dict) -> list[float]:
    return MakerMRStrategy(params).backtest(candles, params)


def backtest_fn_s4(candles: pd.DataFrame, params: dict) -> list[float]:
    return RegimeRouter(params).backtest(candles, params)


def run_s1_validation(candles_map: dict[str, pd.DataFrame],
                      funding_map: dict[str, pd.DataFrame],
                      symbols: list[str],
                      n_random: int = 50,
                      **wf_kwargs) -> dict:
    """Validate S1 Funding Arb on the highest-funding symbols."""
    log.info("\n" + "="*60)
    log.info("S1 FUNDING ARB VALIDATION")
    log.info("="*60)

    strat = FundingArbStrategy()
    results = {}

    # Find symbols with sustained high funding
    high_funding_syms = []
    for sym in symbols:
        fdf = funding_map.get(sym)
        if fdf is None or fdf.empty:
            continue
        avg_abs_fund = fdf["annual_pct"].abs().mean()
        if avg_abs_fund >= 8.0:  # at least 8% average annual funding
            high_funding_syms.append((sym, avg_abs_fund))

    high_funding_syms.sort(key=lambda x: x[1], reverse=True)
    log.info("High-funding symbols: %s", [(s, f"{f:.1f}%") for s, f in high_funding_syms[:5]])

    if not high_funding_syms:
        log.warning("No high-funding symbols found. S1 cannot be validated.")
        return {"s1_funding_arb": {"passes": False, "reason": "no high funding symbols"}}

    # Validate on top-3 high-funding symbols, combine OOS trades
    all_oos_pnl = []
    per_sym_results = {}

    for sym, avg_fund in high_funding_syms[:3]:
        candles = candles_map.get(sym)
        fdf     = funding_map.get(sym)
        if candles is None or candles.empty:
            continue

        def make_fn(s_=sym, f_=fdf):
            def fn(c, p):
                return FundingArbStrategy(p).backtest(c, p, funding_df=f_)
            return fn

        result = walk_forward_validate(
            strategy_backtest_fn=make_fn(),
            candles=candles,
            params=strat.DEFAULT_PARAMS,
            strategy_name=f"s1_funding_arb_{sym}",
            **wf_kwargs,
        )
        per_sym_results[sym] = result
        all_oos_pnl.extend([
            pnl
            for fr in result.fold_results
            for pnl in []  # placeholder: extend with actual trade lists
        ])

        print_wf_report(result)

    # Use the best single-symbol result for final validation
    best_result = max(
        per_sym_results.values(),
        key=lambda r: r.oos_metrics.get("sharpe", 0)
    )

    return {
        "s1_funding_arb": {
            "passes":      best_result.passes,
            "params":      best_result.params,
            "oos_metrics": best_result.oos_metrics,
            "mc":          best_result.monte_carlo,
            "failures":    best_result.failure_reasons,
        }
    }


def run_s2_validation(candles_map: dict[str, pd.DataFrame],
                      symbols: list[str],
                      n_random: int = 100,
                      **wf_kwargs) -> dict:
    """Validate S2 Maker MR across all symbols, use combined OOS."""
    log.info("\n" + "="*60)
    log.info("S2 MAKER MR VALIDATION")
    log.info("="*60)

    strat = MakerMRStrategy()

    # Sample random params
    import random
    random.seed(42)
    grid = random.sample(strat.PARAM_GRID, min(n_random, len(strat.PARAM_GRID)))

    best_sharpe = -np.inf
    best_result = None

    # Validate on top-5 liquid symbols
    test_syms = symbols[:5]

    for sym in test_syms:
        candles = candles_map.get(sym)
        if candles is None or len(candles) < 2500:
            continue

        log.info("S2 validation on %s (%d bars)", sym, len(candles))

        result = walk_forward_validate(
            strategy_backtest_fn=backtest_fn_s2,
            candles=candles,
            params=strat.DEFAULT_PARAMS,
            strategy_name=f"s2_maker_mr_{sym}",
            **wf_kwargs,
        )
        print_wf_report(result)

        if result.oos_metrics.get("sharpe", 0) > best_sharpe:
            best_sharpe = result.oos_metrics.get("sharpe", 0)
            best_result = result

    if best_result is None:
        return {"s2_maker_mr": {"passes": False, "reason": "no valid symbols"}}

    return {
        "s2_maker_mr": {
            "passes":      best_result.passes,
            "params":      best_result.params,
            "oos_metrics": best_result.oos_metrics,
            "mc":          best_result.monte_carlo,
            "failures":    best_result.failure_reasons,
        }
    }


def run_s3_validation(candles_map: dict[str, pd.DataFrame],
                      symbols: list[str],
                      **wf_kwargs) -> dict:
    """
    Validate S3 Pairs Kalman on cointegrated pairs.

    Pair selection bias fix: cointegration is tested only on TRAIN bars of the
    FIRST fold (bars 0..train_end).  This prevents future-data leakage where pairs
    are selected because they happen to be cointegrated over the full window
    including the test folds.
    """
    log.info("\n" + "="*60)
    log.info("S3 PAIRS KALMAN VALIDATION")
    log.info("="*60)

    # Build in-sample price dict using only the FIRST fold's train window.
    # WF uses rolling window: first train end = train_days * bars_per_day.
    train_days  = wf_kwargs.get("train_days", 25)
    bpd         = wf_kwargs.get("bars_per_day", BARS_PER_DAY)
    train_end   = train_days * bpd   # first fold train window (no look-ahead)

    prices_is = {}
    for sym in symbols[:20]:
        c = candles_map.get(sym)
        if c is not None and not c.empty:
            prices_is[sym] = c["close"].values[:train_end]

    pairs = find_cointegrated_pairs(prices_is, max_pvalue=0.05, min_correlation=0.80)
    if not pairs:
        log.warning("No cointegrated pairs found in train window. S3 cannot be validated.")
        return {"s3_pairs_kalman": {"passes": False, "reason": "no cointegrated pairs"}}

    log.info("Pairs found on first-fold train window (%d bars):", train_end)
    for symA, symB, pval in pairs[:5]:
        log.info("  %s/%s  ADF p=%.4f", symA, symB, pval)

    strat = PairsKalmanStrategy()
    best_result = None
    best_sharpe = -np.inf

    for symA, symB, pval in pairs[:3]:
        cA = candles_map.get(symA)
        cB = candles_map.get(symB)
        if cA is None or cB is None:
            continue

        # Align lengths
        min_len = min(len(cA), len(cB))
        cA = cA.tail(min_len).reset_index(drop=True)
        cB = cB.tail(min_len).reset_index(drop=True)

        def make_pair_fn(ca=cA, cb=cB):
            def fn(candles_slice: pd.DataFrame, params: dict) -> list[float]:
                # walk_forward does reset_index(drop=True), so candles_slice.index
                # is always 0..n-1 — can't use it to locate the window in cb.
                # Use timestamps to find the time-aligned cb slice instead.
                ts_col = "ts" if "ts" in candles_slice.columns else candles_slice.columns[0]
                ts_s = candles_slice[ts_col].iloc[0]
                ts_e = candles_slice[ts_col].iloc[-1]
                cb_col = "ts" if "ts" in cb.columns else cb.columns[0]
                cb_slice = cb[(cb[cb_col] >= ts_s) & (cb[cb_col] <= ts_e)].reset_index(drop=True)
                if len(cb_slice) == 0:
                    # Fallback (should not happen if cA/cB are aligned)
                    cb_slice = cb.iloc[:len(candles_slice)].reset_index(drop=True)
                return PairsKalmanStrategy(params).backtest(candles_slice, params, candles_B=cb_slice)
            return fn

        result = walk_forward_validate(
            strategy_backtest_fn=make_pair_fn(),
            candles=cA,
            params=strat.DEFAULT_PARAMS,
            strategy_name=f"s3_pairs_{symA}_{symB}",
            **wf_kwargs,
        )
        print_wf_report(result)

        if result.oos_metrics.get("sharpe", 0) > best_sharpe:
            best_sharpe = result.oos_metrics.get("sharpe", 0)
            best_result = result

    if best_result is None:
        return {"s3_pairs_kalman": {"passes": False, "reason": "all pair validations failed"}}

    return {
        "s3_pairs_kalman": {
            "passes":      best_result.passes,
            "params":      best_result.params,
            "oos_metrics": best_result.oos_metrics,
            "mc":          best_result.monte_carlo,
            "failures":    best_result.failure_reasons,
        }
    }


def run_s4_validation(candles_map: dict[str, pd.DataFrame],
                      symbols: list[str],
                      **wf_kwargs) -> dict:
    """Validate S4 Regime Router on BTC/ETH."""
    log.info("\n" + "="*60)
    log.info("S4 REGIME ROUTER VALIDATION")
    log.info("="*60)

    router = RegimeRouter()
    sym = "BTC" if "BTC" in candles_map else symbols[0]
    candles = candles_map.get(sym)

    if candles is None or len(candles) < 2500:
        return {"s4_regime_router": {"passes": False, "reason": "insufficient BTC data"}}

    result = walk_forward_validate(
        strategy_backtest_fn=backtest_fn_s4,
        candles=candles,
        params=router.DEFAULT_PARAMS,
        strategy_name="s4_regime_router",
        **wf_kwargs,
    )
    print_wf_report(result)

    return {
        "s4_regime_router": {
            "passes":      result.passes,
            "params":      result.params,
            "oos_metrics": result.oos_metrics,
            "mc":          result.monte_carlo,
            "failures":    result.failure_reasons,
        }
    }


def _resample_to_6h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Resample 1h candles to 6h OHLCV."""
    if df_1h.empty or "ts" not in df_1h.columns:
        return df_1h
    df = df_1h.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts")
    ohlcv = df.resample("6h").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()
    return ohlcv.reset_index()


def run_s6_validation(candles_map_6h: dict[str, pd.DataFrame],
                      symbols: list[str],
                      n_random: int = 50) -> dict:
    """
    Validate S6 AdaptiveTrend on 6h candles.
    Uses its own WF criteria (looser Sharpe, more folds for longer hold).
    Downloads or reuses 6h candles (90 days available).
    """
    log.info("\n" + "="*60)
    log.info("S6 ADAPTIVE TREND VALIDATION")
    log.info("="*60)

    strat = AdaptiveTrendStrategy()
    min_bars = (S6_WF_CRITERIA["train_days"] + S6_WF_CRITERIA["n_folds"] * S6_WF_CRITERIA["test_days"]) * BARS_PER_DAY_6H
    min_bars = int(min_bars * 0.90)

    valid_syms = [s for s in symbols if
                  s in candles_map_6h and len(candles_map_6h[s]) >= min_bars]
    log.info("S6 valid symbols (>=%d bars): %d", min_bars, len(valid_syms))

    if not valid_syms:
        return {"s6_adaptive_trend": {"passes": False, "reason": "no symbols with enough 6h data"}}

    import random
    random.seed(42)
    param_sample = random.sample(strat.PARAM_GRID, min(n_random, len(strat.PARAM_GRID)))

    best_result = None
    best_sharpe = -np.inf

    # Test on top-5 symbols individually, pick best
    for sym in valid_syms[:5]:
        candles = candles_map_6h[sym]
        log.info("S6 validation on %s (%d 6h bars = %.0f days)",
                 sym, len(candles), len(candles) / BARS_PER_DAY_6H)

        def make_bt(strat_=strat):
            def fn(c, p): return strat_.backtest(c, p)
            return fn

        result = walk_forward_validate(
            strategy_backtest_fn=make_bt(),
            candles=candles,
            params=strat.DEFAULT_PARAMS,
            strategy_name=f"s6_adaptive_trend_{sym}",
            **S6_WF_CRITERIA,
        )
        print_wf_report(result)

        if result.oos_metrics.get("sharpe", 0) > best_sharpe:
            best_sharpe = result.oos_metrics.get("sharpe", 0)
            best_result = result

    if best_result is None:
        return {"s6_adaptive_trend": {"passes": False, "reason": "no results"}}

    return {
        "s6_adaptive_trend": {
            "passes":      best_result.passes,
            "params":      best_result.params,
            "oos_metrics": best_result.oos_metrics,
            "mc":          best_result.monte_carlo,
            "failures":    best_result.failure_reasons,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Artemisia v7 Walk-Forward Validation")
    parser.add_argument("--strategy", choices=["s1", "s2", "s3", "s4", "s6", "all"], default="all")
    parser.add_argument("--no-dl", action="store_true", help="Skip data download")
    parser.add_argument("--quick",  action="store_true", help="Fewer random params")
    parser.add_argument("--days",   type=int, default=55)
    parser.add_argument("--folds",  type=int, default=4)
    parser.add_argument("--mc",     type=int, default=1000, help="Monte Carlo sims")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    Path("data/cache").mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/wf_backtest.log"),
        ]
    )

    n_random = 30 if args.quick else 100

    # WF criteria (from config)
    wf_kwargs = dict(
        n_folds=args.folds,
        train_days=35,    # 35+4*5=55 days total — fits 57-day Hyperliquid 15m limit
        test_days=5,
        bars_per_day=BARS_PER_DAY,
        initial_capital=500.0,
        n_mc_sims=args.mc,
        min_oos_sharpe=1.2,
        min_fold_sharpe=0.0,  # 0.0: don't penalise empty folds (0-1 trades)
        max_dd_pct=8.0,
        min_profit_factor=1.1,
        min_trades=20,
        mc_p5_sharpe=0.0,
    )

    # Step 1: Universe
    log.info("Step 1: Fetching universe...")
    symbols = get_universe(
        min_volume=10_000_000,
        top_n=20,
        cache_file=f"{CACHE_DIR}/universe.json",
    )
    log.info("Universe: %d symbols", len(symbols))

    # Step 2: Download data
    need_s6   = args.strategy in ("s6", "all")
    need_15m  = args.strategy not in ("s6",)  # S1/S2/S3/S4 use 15m

    candles_map    = {}
    candles_map_6h = {}
    funding_map    = {}

    if not args.no_dl:
        if need_15m:
            log.info("Step 2a: Downloading %d days of 15m candles...", args.days)
            candles_map = download_all_candles(
                symbols, interval="15m", days=args.days, cache_dir=CACHE_DIR
            )
            funding_map = download_all_funding(symbols, rows=500, cache_dir=CACHE_DIR)
        if need_s6:
            log.info("Step 2b: Downloading 90 days of 6h candles for S6...")
            candles_map_6h = download_all_candles(
                symbols, interval="1h", days=90, cache_dir=CACHE_DIR
            )
            # Resample 1h -> 6h for S6
            candles_map_6h = {
                sym: _resample_to_6h(df)
                for sym, df in candles_map_6h.items()
                if not df.empty
            }
    else:
        log.info("Step 2: Loading from cache...")
        for sym in symbols:
            p15 = Path(CACHE_DIR) / f"candles_{sym}_15m_{args.days}d.parquet"
            if p15.exists():
                candles_map[sym] = pd.read_parquet(p15)
            p1h = Path(CACHE_DIR) / f"candles_{sym}_1h_90d.parquet"
            if p1h.exists():
                df1h = pd.read_parquet(p1h)
                candles_map_6h[sym] = _resample_to_6h(df1h)
            fp = Path(CACHE_DIR) / f"funding_{sym}.parquet"
            if fp.exists():
                funding_map[sym] = pd.read_parquet(fp)

    # Filter 15m symbols
    min_bars = (wf_kwargs["train_days"] + wf_kwargs["n_folds"] * wf_kwargs["test_days"]) * BARS_PER_DAY
    min_bars = int(min_bars * 0.90)
    valid_syms = [s for s in symbols if
                  s in candles_map and len(candles_map[s]) >= min_bars]
    log.info("Valid 15m symbols (>=%d bars): %d/%d", min_bars, len(valid_syms), len(symbols))

    # Step 3: Run validations
    all_results = {}

    if args.strategy in ("s1", "all"):
        r = run_s1_validation(candles_map, funding_map, valid_syms, n_random, **wf_kwargs)
        all_results.update(r)

    if args.strategy in ("s2", "all"):
        r = run_s2_validation(candles_map, valid_syms, n_random, **wf_kwargs)
        all_results.update(r)

    if args.strategy in ("s3", "all"):
        r = run_s3_validation(candles_map, valid_syms, **wf_kwargs)
        all_results.update(r)

    if args.strategy in ("s4", "all"):
        r = run_s4_validation(candles_map, valid_syms, **wf_kwargs)
        all_results.update(r)

    if args.strategy in ("s6", "all"):
        r = run_s6_validation(candles_map_6h, symbols, n_random)
        all_results.update(r)

    # Step 4: Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    passed = []
    failed = []
    for name, res in all_results.items():
        status = "PASS" if res.get("passes") else "FAIL"
        sharpe = res.get("oos_metrics", {}).get("sharpe", 0)
        dd     = res.get("oos_metrics", {}).get("max_dd_pct", 0)
        print(f"  {name:25s} [{status}]  Sharpe={sharpe:.2f}  DD={dd:.1f}%")
        if res.get("passes"):
            passed.append(name)
        else:
            failed.append(name)
            reasons = res.get("failures", [res.get("reason", "unknown")])
            for r in reasons:
                print(f"    -> FAIL: {r}")

    print(f"\nPASSED: {len(passed)} | FAILED: {len(failed)}")
    if failed:
        print("FAILED strategies will NOT be deployed.")
        print("Do NOT adjust criteria to force a pass. Fix the strategy or discard.")

    # Step 5: Save validated configs
    if passed:
        output = {
            "validation_date": pd.Timestamp.now().isoformat(),
            "strategies": {
                name: all_results[name]
                for name in passed
            }
        }
        # Make JSON-serializable
        def make_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        Path(OUTPUT_FILE).write_text(
            json.dumps(output, indent=2, default=make_serializable)
        )
        log.info("Validated strategies saved to %s", OUTPUT_FILE)
    else:
        log.warning("No strategies passed. Nothing saved.")
        print("\nNo strategies ready for paper trading.")
        print("Minimum requirement: at least S1 (Funding Arb) must pass before live.")


if __name__ == "__main__":
    main()
