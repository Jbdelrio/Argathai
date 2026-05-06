"""
walk_forward.py — Walk-forward cross-validation for strategy validation
6 folds of 25 train / 5 test days on 15-min candles (52 days total = 1 buffer)
OOS criteria: Sharpe > 1.5, min fold > 0.5, DD < 6%, PF > 1.3
"""
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Optional

from .metrics import (
    compute_all_metrics, monte_carlo_sharpe,
    passes_oos_criteria, _empty_metrics
)

log = logging.getLogger(__name__)

# Bars per day for 15-min candles
BARS_PER_DAY_15M = 96


@dataclass
class FoldResult:
    fold_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_metrics: dict
    test_metrics: dict
    params: dict


@dataclass
class WalkForwardResult:
    strategy_name: str
    params: dict
    fold_results: list[FoldResult]
    oos_metrics: dict          # Aggregated OOS (all test folds combined)
    monte_carlo: dict
    passes: bool
    failure_reasons: list[str]


def _build_folds(n_bars: int,
                 n_folds: int = 6,
                 train_days: int = 25,
                 test_days: int = 5,
                 bars_per_day: int = BARS_PER_DAY_15M) -> list[tuple[int, int, int, int]]:
    """
    Build fold indices for walk-forward CV.
    Returns list of (train_start, train_end, test_start, test_end).

    Strategy: expanding or rolling window.
    We use ROLLING window (fixed train size) to prevent look-ahead bias from
    regime changes in distant history.
    """
    train_bars = train_days * bars_per_day
    test_bars = test_days * bars_per_day
    step = test_bars

    folds = []
    # Start so that all n_folds fit
    min_required = train_bars + n_folds * test_bars
    if n_bars < min_required:
        log.warning("Not enough bars (%d) for %d folds. Need %d.", n_bars, n_folds, min_required)
        # Reduce folds to what's available
        n_folds = max(1, (n_bars - train_bars) // test_bars)

    for i in range(n_folds):
        test_end = n_bars - (n_folds - 1 - i) * step
        test_start = test_end - test_bars
        train_end = test_start
        train_start = train_end - train_bars

        if train_start < 0:
            continue

        folds.append((train_start, train_end, test_start, test_end))

    return folds


def walk_forward_validate(
    strategy_backtest_fn: Callable[[pd.DataFrame, dict], list[float]],
    candles: pd.DataFrame,
    params: dict,
    strategy_name: str = "unknown",
    n_folds: int = 6,
    train_days: int = 25,
    test_days: int = 5,
    bars_per_day: int = BARS_PER_DAY_15M,
    initial_capital: float = 500.0,
    n_mc_sims: int = 1000,
    # OOS pass criteria
    min_oos_sharpe: float = 1.5,
    min_fold_sharpe: float = 0.5,
    max_dd_pct: float = 6.0,
    min_profit_factor: float = 1.3,
    min_trades: int = 30,
    mc_p5_sharpe: float = 0.0,
    # Warmup buffer: prepend N trailing train bars before each test slice
    # so strategies with long indicator warmup can still generate test-period trades.
    # Trades are collected from the full (warmup+test) slice; the strategy's own
    # internal warmup naturally suppresses entries in the prepended region.
    test_warmup_bars: int = 0,
) -> WalkForwardResult:
    """
    Run walk-forward cross-validation.

    Args:
        strategy_backtest_fn: fn(candle_slice: DataFrame, params: dict) -> list[float pnl]
            Called with train slice first (for any warm-up/fitting the strategy needs),
            then test slice for OOS evaluation.
            The function should return a list of per-trade P&L in USD.
        candles: Full candle DataFrame (columns: ts, open, high, low, close, volume)
        params: Strategy parameters (fixed, no re-optimization per fold)
        test_warmup_bars: Extra bars prepended before each test slice so that
            long-lookback strategies can compute indicators before test window starts.
        ...

    Returns:
        WalkForwardResult with per-fold breakdown and aggregated OOS metrics
    """
    n_bars = len(candles)
    folds = _build_folds(n_bars, n_folds, train_days, test_days, bars_per_day)

    if not folds:
        log.error("No valid folds for %s with %d bars", strategy_name, n_bars)
        return WalkForwardResult(
            strategy_name=strategy_name,
            params=params,
            fold_results=[],
            oos_metrics=_empty_metrics(),
            monte_carlo={},
            passes=False,
            failure_reasons=["insufficient data for walk-forward"],
        )

    log.info("WF [%s]: %d folds, %d train days, %d test days, %d total bars",
             strategy_name, len(folds), train_days, test_days, n_bars)

    fold_results = []
    all_oos_pnl = []

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        train_slice = candles.iloc[tr_s:tr_e].reset_index(drop=True)
        test_slice  = candles.iloc[te_s:te_e].reset_index(drop=True)

        # Optionally prepend warmup bars before test slice
        if test_warmup_bars > 0:
            warmup_start = max(tr_s, te_s - test_warmup_bars)
            warmup_buf   = candles.iloc[warmup_start:te_s].reset_index(drop=True)
            test_input   = pd.concat([warmup_buf, test_slice], ignore_index=True)
        else:
            test_input = test_slice

        log.debug("Fold %d: train [%d:%d] test [%d:%d] warmup=%d",
                  i, tr_s, tr_e, te_s, te_e, test_warmup_bars)

        try:
            # Train pass (strategy may fit internal models on train data)
            train_pnl = strategy_backtest_fn(train_slice, params)
            train_metrics = compute_all_metrics(
                np.array(train_pnl), len(train_slice), initial_capital, bars_per_day
            )

            # Test pass (OOS evaluation) — uses warmup+test slice if configured
            test_pnl = strategy_backtest_fn(test_input, params)
            test_metrics = compute_all_metrics(
                np.array(test_pnl), len(test_slice), initial_capital, bars_per_day
            )

            all_oos_pnl.extend(test_pnl)

        except Exception as e:
            log.error("Fold %d error: %s", i, e, exc_info=True)
            train_metrics = _empty_metrics()
            test_metrics = _empty_metrics()

        fold_results.append(FoldResult(
            fold_idx=i,
            train_start=tr_s, train_end=tr_e,
            test_start=te_s,  test_end=te_e,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            params=params,
        ))

        log.info(
            "Fold %d | Train Sharpe=%.2f | Test Sharpe=%.2f | Test DD=%.1f%% | Test trades=%d",
            i,
            train_metrics.get("sharpe", 0),
            test_metrics.get("sharpe", 0),
            test_metrics.get("max_dd_pct", 0),
            test_metrics.get("n_trades", 0),
        )

    # Aggregate OOS metrics across all test folds
    all_oos_arr = np.array(all_oos_pnl)
    total_test_bars = len(folds) * test_days * bars_per_day
    oos_metrics = compute_all_metrics(
        all_oos_arr, total_test_bars, initial_capital, bars_per_day
    )

    # Check min fold sharpe
    fold_sharpes = [fr.test_metrics.get("sharpe", 0) for fr in fold_results]
    min_fold = min(fold_sharpes) if fold_sharpes else 0.0
    oos_metrics["min_fold_sharpe"] = round(min_fold, 3)
    oos_metrics["fold_sharpes"] = [round(s, 3) for s in fold_sharpes]

    # Monte Carlo on OOS trades
    mc = monte_carlo_sharpe(
        all_oos_arr, n_sims=n_mc_sims,
        n_bars=total_test_bars, initial_capital=initial_capital,
        bars_per_day=bars_per_day,
    )
    oos_metrics["mc_p5_sharpe"] = mc.get("p5", 0.0)

    # Pass/fail evaluation
    passes, failures = passes_oos_criteria(
        oos_metrics,
        min_sharpe=min_oos_sharpe,
        min_fold_sharpe=min_fold_sharpe,
        max_dd_pct=max_dd_pct,
        min_profit_factor=min_profit_factor,
        min_trades=min_trades,
        mc_p5_sharpe=mc_p5_sharpe,
    )

    # Additional: min fold sharpe check
    if min_fold < min_fold_sharpe:
        reason = f"worst fold sharpe {min_fold:.2f} < {min_fold_sharpe}"
        if reason not in failures:
            failures.append(reason)
            passes = False

    if passes:
        log.info("WF [%s]: PASS | OOS Sharpe=%.2f | DD=%.1f%% | PF=%.2f | MC_p5=%.2f",
                 strategy_name,
                 oos_metrics["sharpe"],
                 oos_metrics["max_dd_pct"],
                 oos_metrics["profit_factor"],
                 mc.get("p5", 0))
    else:
        log.warning("WF [%s]: FAIL | %s", strategy_name, " | ".join(failures))

    return WalkForwardResult(
        strategy_name=strategy_name,
        params=params,
        fold_results=fold_results,
        oos_metrics=oos_metrics,
        monte_carlo=mc,
        passes=passes,
        failure_reasons=failures,
    )


def parameter_search_wf(
    strategy_backtest_fn: Callable,
    candles: pd.DataFrame,
    param_grid: list[dict],
    strategy_name: str = "unknown",
    n_folds: int = 6,
    train_days: int = 25,
    test_days: int = 5,
    bars_per_day: int = BARS_PER_DAY_15M,
    initial_capital: float = 500.0,
    # Optimise on train folds only, then validate on test
    n_random: Optional[int] = 100,
) -> tuple[dict, WalkForwardResult]:
    """
    Two-stage parameter search with walk-forward validation.

    Stage 1: Evaluate params on TRAIN folds only (in-sample optimization)
    Stage 2: Run WF validation with best params on TEST folds (OOS check)

    This prevents the circular bias of optimizing on the same data used to validate.

    Args:
        param_grid: List of param dicts to evaluate
        n_random: If set, subsample param_grid randomly to n_random combos

    Returns:
        (best_params, WalkForwardResult with OOS metrics)
    """
    import random

    if n_random and len(param_grid) > n_random:
        grid = random.sample(param_grid, n_random)
        log.info("Random search: %d / %d param combos", n_random, len(param_grid))
    else:
        grid = param_grid

    n_bars = len(candles)
    folds = _build_folds(n_bars, n_folds, train_days, test_days, bars_per_day)
    if not folds:
        raise ValueError("Not enough data for walk-forward parameter search")

    log.info("WF param search [%s]: %d combos x %d folds", strategy_name, len(grid), len(folds))

    # Stage 1: In-sample optimization (train folds only)
    best_params = grid[0]
    best_train_sharpe = -np.inf

    for p_idx, params in enumerate(grid):
        fold_sharpes = []
        for tr_s, tr_e, te_s, te_e in folds:
            train_slice = candles.iloc[tr_s:tr_e].reset_index(drop=True)
            try:
                pnl = strategy_backtest_fn(train_slice, params)
                m = compute_all_metrics(np.array(pnl), len(train_slice), initial_capital, bars_per_day)
                fold_sharpes.append(m.get("sharpe", 0))
            except Exception:
                fold_sharpes.append(-99)

        avg_train_sharpe = float(np.mean(fold_sharpes)) if fold_sharpes else -99
        if avg_train_sharpe > best_train_sharpe:
            best_train_sharpe = avg_train_sharpe
            best_params = params
            log.debug("New best params (train Sharpe=%.2f): %s", avg_train_sharpe, params)

        if (p_idx + 1) % 20 == 0:
            log.info("Param search progress: %d/%d (best train Sharpe=%.2f)",
                     p_idx + 1, len(grid), best_train_sharpe)

    log.info("Best train Sharpe=%.2f | params=%s", best_train_sharpe, best_params)

    # Stage 2: Full WF validation with best params (OOS test folds)
    result = walk_forward_validate(
        strategy_backtest_fn=strategy_backtest_fn,
        candles=candles,
        params=best_params,
        strategy_name=strategy_name,
        n_folds=n_folds,
        train_days=train_days,
        test_days=test_days,
        bars_per_day=bars_per_day,
        initial_capital=initial_capital,
    )

    return best_params, result


def print_wf_report(result: WalkForwardResult):
    """Print a readable walk-forward report."""
    print(f"\n{'='*60}")
    print(f"Walk-Forward Report: {result.strategy_name}")
    print(f"{'='*60}")
    print(f"Overall OOS: {'PASS' if result.passes else 'FAIL'}")
    if result.failure_reasons:
        for r in result.failure_reasons:
            print(f"  FAIL: {r}")

    print(f"\nOOS Metrics:")
    m = result.oos_metrics
    print(f"  Sharpe:       {m.get('sharpe', 0):.2f}")
    print(f"  Sortino:      {m.get('sortino', 0):.2f}")
    print(f"  Max DD:       {m.get('max_dd_pct', 0):.1f}%")
    print(f"  Profit Factor:{m.get('profit_factor', 0):.2f}")
    print(f"  Win Rate:     {m.get('win_rate', 0):.1f}%")
    print(f"  Trades/day:   {m.get('trades_per_day', 0):.1f}")
    print(f"  Total return: {m.get('total_return_pct', 0):.1f}%")
    print(f"  MC p5 Sharpe: {m.get('mc_p5_sharpe', 0):.2f}")

    print(f"\nPer-Fold OOS Sharpe: {m.get('fold_sharpes', [])}")

    print(f"\nMonte Carlo:")
    mc = result.monte_carlo
    print(f"  p5={mc.get('p5', 0):.2f}  p50={mc.get('p50', 0):.2f}  "
          f"p95={mc.get('p95', 0):.2f}  std={mc.get('std', 0):.2f}")
    print(f"{'='*60}\n")
