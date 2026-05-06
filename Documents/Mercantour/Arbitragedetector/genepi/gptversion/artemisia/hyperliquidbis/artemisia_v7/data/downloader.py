"""
downloader.py — Hyperliquid historical data downloader
- 15-min candles: ~52 days (walk-forward backtest)
- 1h candles: ~90 days (regime training)
- Funding history: 500+ hourly rows
- Storage: parquet (pyarrow)
"""
import json
import time
import logging
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

log = logging.getLogger(__name__)

HL_API = "https://api.hyperliquid.xyz/info"

# Candle interval -> approx minutes per candle
INTERVAL_MINUTES = {
    "1m":  1,
    "3m":  3,
    "5m":  5,
    "15m": 15,
    "30m": 30,
    "1h":  60,
    "4h":  240,
}

# API max candles per request
BATCH_CANDLES = 500


def _post(payload: dict, retries: int = 3) -> dict | list:
    for attempt in range(retries):
        try:
            r = requests.post(HL_API, json=payload, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            log.warning("API error (attempt %d/%d): %s", attempt + 1, retries, e)
            time.sleep(2 ** attempt)


def _fetch_candles_batch(coin: str, interval: str, start_ms: int, end_ms: int) -> list[dict]:
    """Fetch one batch of candles from the API."""
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
        }
    }
    data = _post(payload)
    return data if isinstance(data, list) else []


def fetch_candles(coin: str,
                  interval: str = "15m",
                  days: int = 52,
                  end_time_ms: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch `days` of candles for `coin` at `interval` resolution.
    Returns DataFrame with columns: [ts, open, high, low, close, volume]
    ts is UTC datetime index.
    """
    if interval not in INTERVAL_MINUTES:
        raise ValueError(f"Unknown interval: {interval}. Valid: {list(INTERVAL_MINUTES)}")

    interval_ms = INTERVAL_MINUTES[interval] * 60 * 1000
    now_ms = end_time_ms or int(time.time() * 1000)
    start_ms = now_ms - int(days * 24 * 3600 * 1000)

    all_candles = []
    cursor = start_ms

    while cursor < now_ms:
        batch_end = min(cursor + BATCH_CANDLES * interval_ms, now_ms)
        batch = _fetch_candles_batch(coin, interval, cursor, batch_end)
        if not batch:
            break
        all_candles.extend(batch)
        last_t = batch[-1].get("t", batch[-1].get("T", cursor))
        cursor = last_t + interval_ms
        if len(batch) < 2:
            break

    if not all_candles:
        log.warning("%s: no candles returned for interval=%s days=%d", coin, interval, days)
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    # Normalize: HL candles have keys T (open time), o, h, l, c, v
    rows = []
    for c in all_candles:
        try:
            rows.append({
                "ts":     pd.Timestamp(c.get("T", c.get("t")), unit="ms", tz="UTC"),
                "open":   float(c["o"]),
                "high":   float(c["h"]),
                "low":    float(c["l"]),
                "close":  float(c["c"]),
                "volume": float(c["v"]),
            })
        except (KeyError, TypeError, ValueError):
            continue

    df = pd.DataFrame(rows).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    log.info("%s [%s]: %d candles (%.1f days)", coin, interval, len(df), len(df) * INTERVAL_MINUTES[interval] / 1440)
    return df


def fetch_funding_history(coin: str, rows: int = 500) -> pd.DataFrame:
    """
    Fetch funding rate history for a coin.
    Returns DataFrame: [ts, funding_rate, premium, annual_pct]
    funding_rate is per-hour (Hyperliquid native), annual_pct = rate*24*365*100
    """
    now_ms = int(time.time() * 1000)
    # Funding is hourly; 500 rows = ~20 days
    start_ms = now_ms - int(rows * 3600 * 1000 * 1.1)

    payload = {
        "type": "fundingHistory",
        "coin": coin,
        "startTime": start_ms,
    }
    data = _post(payload)
    if not isinstance(data, list) or not data:
        log.warning("%s: no funding history", coin)
        return pd.DataFrame(columns=["ts", "funding_rate", "premium", "annual_pct"])

    rows_list = []
    for row in data:
        try:
            rate = float(row["fundingRate"])
            rows_list.append({
                "ts":           pd.Timestamp(row["time"], unit="ms", tz="UTC"),
                "funding_rate": rate,
                "premium":      float(row.get("premium", 0)),
                "annual_pct":   rate * 24 * 365 * 100,
            })
        except (KeyError, TypeError, ValueError):
            continue

    df = pd.DataFrame(rows_list).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    log.info("%s: %d funding rows (%.1f days)", coin, len(df), len(df) / 24)
    return df


def _download_symbol(coin: str,
                     interval: str,
                     days: int,
                     cache_dir: Path,
                     force: bool,
                     cache_max_age_h: float) -> tuple[str, pd.DataFrame]:
    """Worker for parallel downloads. Returns (coin, df)."""
    parquet_path = cache_dir / f"candles_{coin}_{interval}_{days}d.parquet"

    if not force and parquet_path.exists():
        age_h = (time.time() - parquet_path.stat().st_mtime) / 3600
        if age_h < cache_max_age_h:
            df = pd.read_parquet(parquet_path)
            log.info("Cache hit: %s [%s] (%d rows, %.1fh old)", coin, interval, len(df), age_h)
            return coin, df

    df = fetch_candles(coin, interval, days)
    if not df.empty:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
    return coin, df


def _download_funding(coin: str,
                      rows: int,
                      cache_dir: Path,
                      force: bool,
                      cache_max_age_h: float) -> tuple[str, pd.DataFrame]:
    """Worker for funding history downloads."""
    parquet_path = cache_dir / f"funding_{coin}.parquet"

    if not force and parquet_path.exists():
        age_h = (time.time() - parquet_path.stat().st_mtime) / 3600
        if age_h < cache_max_age_h:
            df = pd.read_parquet(parquet_path)
            return coin, df

    df = fetch_funding_history(coin, rows)
    if not df.empty:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
    return coin, df


def download_all_candles(symbols: list[str],
                         interval: str = "15m",
                         days: int = 52,
                         cache_dir: str = "data/cache",
                         force: bool = False,
                         cache_max_age_h: float = 6.0,
                         max_workers: int = 8) -> dict[str, pd.DataFrame]:
    """
    Download candles for all symbols in parallel.
    Returns {coin: DataFrame}.
    """
    cache_path = Path(cache_dir)
    results: dict[str, pd.DataFrame] = {}

    log.info("Downloading %s candles for %d symbols (%d days)...",
             interval, len(symbols), days)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_download_symbol, sym, interval, days, cache_path, force, cache_max_age_h): sym
            for sym in symbols
        }
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                _, df = fut.result()
                results[sym] = df
            except Exception as e:
                log.error("%s: download failed: %s", sym, e)
                results[sym] = pd.DataFrame()

    ok = sum(1 for df in results.values() if not df.empty)
    log.info("Downloaded %d/%d symbols successfully", ok, len(symbols))
    return results


def download_all_funding(symbols: list[str],
                         rows: int = 500,
                         cache_dir: str = "data/cache",
                         force: bool = False,
                         cache_max_age_h: float = 1.0,
                         max_workers: int = 8) -> dict[str, pd.DataFrame]:
    """
    Download funding history for all symbols in parallel.
    Returns {coin: DataFrame}.
    """
    cache_path = Path(cache_dir)
    results: dict[str, pd.DataFrame] = {}

    log.info("Downloading funding history for %d symbols...", len(symbols))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_download_funding, sym, rows, cache_path, force, cache_max_age_h): sym
            for sym in symbols
        }
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                _, df = fut.result()
                results[sym] = df
            except Exception as e:
                log.error("%s: funding download failed: %s", sym, e)
                results[sym] = pd.DataFrame()

    ok = sum(1 for df in results.values() if not df.empty)
    log.info("Funding: %d/%d symbols", ok, len(symbols))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    test_syms = ["BTC", "ETH", "SOL"]
    candles = download_all_candles(test_syms, interval="15m", days=52, cache_dir="cache")
    for sym, df in candles.items():
        if not df.empty:
            print(f"{sym}: {len(df)} candles, {df['ts'].iloc[0]} -> {df['ts'].iloc[-1]}")

    funding = download_all_funding(test_syms, cache_dir="cache")
    for sym, df in funding.items():
        if not df.empty:
            print(f"{sym} funding: {len(df)} rows, last={df['annual_pct'].iloc[-1]:.2f}% ann")
