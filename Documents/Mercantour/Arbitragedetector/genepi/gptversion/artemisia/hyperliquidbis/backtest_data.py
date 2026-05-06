"""
Artemisia Glacialis — Historical Data Downloader
Fetches 7 days of 1-min candles from Hyperliquid candleSnapshot API.
Caches to data/ directory (valid 6h).
"""
import json, time, pathlib, requests, sys
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

SYMBOLS = ["BTC", "ETH", "SOL", "DOGE", "XRP", "SUI", "AVAX", "LINK", "HYPE", "FARTCOIN"]
HL_URL = "https://api.hyperliquid.xyz/info"
DATA_DIR = pathlib.Path("data")
DAYS = 7
BATCH_MINUTES = 480   # 8h per API call (safe limit)
RATE_SLEEP = 0.20     # 200ms between requests


def _fetch_batch(symbol: str, start_ms: int, end_ms: int) -> List[dict]:
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": symbol,
            "interval": "1m",
            "startTime": start_ms,
            "endTime": end_ms,
        },
    }
    r = requests.post(
        HL_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()

    candles = []
    if not isinstance(data, list):
        return candles

    for c in data:
        try:
            if isinstance(c, dict):
                # Standard HL format: {"t":..., "o":..., "h":..., "l":..., "c":..., "v":...}
                candles.append({
                    "t": int(c["t"]),
                    "o": float(c["o"]),
                    "h": float(c["h"]),
                    "l": float(c["l"]),
                    "c": float(c["c"]),
                    "v": float(c.get("v", 0)),
                })
            elif isinstance(c, (list, tuple)) and len(c) >= 5:
                # Array format fallback
                candles.append({
                    "t": int(c[0]),
                    "o": float(c[1]),
                    "h": float(c[2]),
                    "l": float(c[3]),
                    "c": float(c[4]),
                    "v": float(c[5]) if len(c) > 5 else 0,
                })
        except (KeyError, TypeError, ValueError):
            continue

    return candles


def download_symbol(symbol: str, days: int = DAYS, verbose: bool = True) -> List[dict]:
    """Download N days of 1-min candles. Returns sorted list of candle dicts."""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - days * 24 * 60 * 60 * 1000
    batch_ms = BATCH_MINUTES * 60 * 1000

    all_candles: List[dict] = []
    current = start_ms

    while current < now_ms:
        end = min(current + batch_ms, now_ms)
        t0 = datetime.fromtimestamp(current / 1000, tz=timezone.utc).strftime("%m-%d %H:%M")
        t1 = datetime.fromtimestamp(end / 1000, tz=timezone.utc).strftime("%m-%d %H:%M")

        for attempt in range(3):
            try:
                batch = _fetch_batch(symbol, current, end)
                all_candles.extend(batch)
                if verbose:
                    print(f"  {symbol:10s} [{t0} -> {t1}] +{len(batch)} candles", flush=True)
                break
            except Exception as e:
                if attempt == 2:
                    if verbose:
                        print(f"  {symbol}: FAILED [{t0} -> {t1}]: {e}", flush=True)
                else:
                    time.sleep(1.0)

        current = end + 60_000  # next minute
        time.sleep(RATE_SLEEP)

    # Deduplicate & sort
    seen: set = set()
    unique: List[dict] = []
    for c in all_candles:
        if c["t"] not in seen and c["c"] > 0 and c["h"] >= c["l"]:
            seen.add(c["t"])
            unique.append(c)
    unique.sort(key=lambda x: x["t"])
    return unique


def download_all(
    symbols: List[str] = SYMBOLS,
    days: int = DAYS,
    force: bool = False,
    cache_max_age_h: float = 6.0,
) -> Dict[str, List[dict]]:
    """
    Download (or load from cache) candles for all symbols.
    Returns dict: symbol -> list of candle dicts.
    """
    DATA_DIR.mkdir(exist_ok=True)
    result: Dict[str, List[dict]] = {}

    for sym in symbols:
        cache_file = DATA_DIR / f"candles_{sym}_{days}d.json"

        if cache_file.exists() and not force:
            age_h = (time.time() - cache_file.stat().st_mtime) / 3600
            if age_h < cache_max_age_h:
                with open(cache_file) as f:
                    data = json.load(f)
                print(f"  {sym:10s}: {len(data):6d} candles [cache {age_h:.1f}h old]", flush=True)
                result[sym] = data
                continue

        print(f"\nDownloading {sym} ({days}d × 1m)...", flush=True)
        candles = download_symbol(sym, days)

        with open(cache_file, "w") as f:
            json.dump(candles, f)

        result[sym] = candles
        if candles:
            t0 = datetime.fromtimestamp(candles[0]["t"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            t1 = datetime.fromtimestamp(candles[-1]["t"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            print(f"  -> {len(candles)} candles saved ({t0} -> {t1})", flush=True)
        else:
            print(f"  -> WARNING: 0 candles for {sym}", flush=True)

    return result


def summarize(data: Dict[str, List[dict]]) -> None:
    print("\n=== Data Summary ===")
    for sym, candles in data.items():
        if not candles:
            print(f"  {sym:10s}: NO DATA")
            continue
        t0 = datetime.fromtimestamp(candles[0]["t"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        t1 = datetime.fromtimestamp(candles[-1]["t"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        prices = [c["c"] for c in candles]
        ret = (prices[-1] - prices[0]) / prices[0] * 100 if prices[0] > 0 else 0
        print(f"  {sym:10s}: {len(candles):6d} candles | {t0} -> {t1} | 7d ret: {ret:+.1f}%")


if __name__ == "__main__":
    force = "--force" in sys.argv
    print("=== Artemisia Glacialis — Data Downloader ===")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Period : {DAYS} days × 1-min candles")
    print(f"Force  : {force}\n")
    data = download_all(force=force)
    summarize(data)
