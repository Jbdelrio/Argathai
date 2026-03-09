"""Data loader — finds and loads 1s CSV files with portable path discovery.

Priority order (first found + enough rows wins):
  1. config/settings.yaml → data.<symbol>_path  (explicit override)
  2. config/settings.yaml → data.search_paths   (custom dirs)
  3. AGARTHAI_DATA_DIRS env var                  (runtime override)
  4. SEARCH_DIRS hardcoded list                  (known data locations)
  5. data/binance_spot/                          (local fallback / test files)

A file with fewer than MIN_ROWS rows is treated as a stub and skipped.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

# ── Known data locations (searched in order) ─────────────────────────────────
SEARCH_DIRS: list[Path] = [
    Path(r'C:\Users\jeanb\Documents\hyperstat-arb-bot\analysis_ops\data_binance\spot'),
    Path(r'C:\Users\jeanb\Documents\Agarthai\data\binance_spot'),
    Path('data/binance_spot'),
]

DEFAULT_DATA_DIR = Path('data/binance_spot')

DEFAULT_FILES = {
    'btc': 'btc_1s.csv.gz',
    'eth': 'eth_1s.csv.gz',
}

# Files with fewer rows than this are treated as stubs / dev fixtures and skipped.
MIN_ROWS = 1_000


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_settings(settings_path: Path = Path('config/settings.yaml')) -> dict:
    if settings_path.exists():
        with open(settings_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


def _candidate_paths(symbol: str) -> list[Path]:
    """Build ordered candidate file paths from config + env + SEARCH_DIRS + local."""
    symbol = symbol.lower()
    filename = DEFAULT_FILES.get(symbol, f'{symbol}_1s.csv.gz')
    settings = _load_settings()
    data_cfg = settings.get('data', {}) if isinstance(settings, dict) else {}

    candidates: list[Path] = []

    # 1. Explicit file from settings
    cfg_file = data_cfg.get(f'{symbol}_path')
    if cfg_file:
        candidates.append(Path(cfg_file))

    # 2. Custom search_paths from settings
    for entry in data_cfg.get('search_paths', []) or []:
        candidates.append(Path(entry) / filename)

    # 3. AGARTHAI_DATA_DIRS env var
    env_dirs = os.getenv('AGARTHAI_DATA_DIRS', '')
    if env_dirs:
        sep = ';' if ';' in env_dirs else os.pathsep
        for d in [p for p in env_dirs.split(sep) if p.strip()]:
            candidates.append(Path(d.strip()) / filename)

    # 4. Hardcoded known locations (includes external data repo)
    for sd in SEARCH_DIRS:
        candidates.append(sd / filename)

    # Deduplicate while preserving order
    uniq: list[Path] = []
    seen: set[str] = set()
    for c in candidates:
        key = str(c.resolve()) if c.exists() else str(c)
        if key not in seen:
            uniq.append(c)
            seen.add(key)
    return uniq


def _count_rows_fast(path: Path) -> int:
    """Count rows in a gzipped CSV quickly (without loading into memory)."""
    try:
        import gzip
        with gzip.open(path, 'rt', errors='ignore') as f:
            n = sum(1 for _ in f)
        return max(0, n - 1)   # subtract header
    except Exception:
        return 0


def _maybe_fetch_from_exchange(symbol: str, out_path: Path) -> Optional[pd.DataFrame]:
    settings = _load_settings()
    fetch_cfg = settings.get('history_fetch', {}) if isinstance(settings, dict) else {}
    if not fetch_cfg or not fetch_cfg.get('enabled', False):
        return None

    start = fetch_cfg.get('start')
    end   = fetch_cfg.get('end')
    if not start or not end:
        return None

    exchange    = fetch_cfg.get('exchange', 'bitget')
    market_map  = fetch_cfg.get('symbols', {}) or {}
    market_sym  = market_map.get(symbol.lower(), symbol.upper() + 'USDT')
    limit       = int(fetch_cfg.get('limit', 200))

    from data.exchange_fetcher import fetch_exchange_1s_to_file, to_utc_timestamp

    bars = fetch_exchange_1s_to_file(
        exchange=exchange,
        symbol=market_sym,
        start_ts=to_utc_timestamp(start),
        end_ts=to_utc_timestamp(end),
        out_path=out_path,
        limit=limit,
    )
    return prepare(bars) if not bars.empty else None


# ── Public API ────────────────────────────────────────────────────────────────

def load_1s(symbol: str = 'btc') -> pd.DataFrame:
    """Load 1-second market data.

    Iterates candidate paths in priority order, skipping files that exist but
    have fewer than MIN_ROWS rows (stub / test fixtures).
    """
    symbol     = symbol.lower()
    candidates = _candidate_paths(symbol)
    tried: list[str] = []

    for p in candidates:
        if not p.exists():
            tried.append(f"MISSING  {p}")
            continue

        n_rows = _count_rows_fast(p)
        if n_rows < MIN_ROWS:
            tried.append(f"STUB     {p}  ({n_rows} rows < {MIN_ROWS} minimum)")
            print(f"[loader] Skipping stub file ({n_rows} rows): {p}")
            continue

        print(f"[loader] Loading {symbol} from {p}  ({n_rows:,} rows)")
        df = pd.read_csv(p, compression='gzip')
        return prepare(df)

    # All paths exhausted — try exchange fetch
    fallback_out = DEFAULT_DATA_DIR / DEFAULT_FILES.get(symbol, f'{symbol}_1s.csv.gz')
    fetched = _maybe_fetch_from_exchange(symbol=symbol, out_path=fallback_out)
    if fetched is not None:
        return fetched

    # Nothing worked — raise with full diagnostic
    tried_str = '\n  '.join(tried) if tried else '(none)'
    raise FileNotFoundError(
        f"\n[loader] Cannot find {symbol}_1s.csv.gz with >= {MIN_ROWS} rows.\n"
        f"Paths tried:\n  {tried_str}\n\n"
        "Options:\n"
        "  A) Set AGARTHAI_DATA_DIRS env var to your data directory.\n"
        "  B) Add 'data.search_paths' in config/settings.yaml.\n"
        "  C) Enable 'history_fetch' in config/settings.yaml to auto-download.\n"
        f"  D) Add your data path to SEARCH_DIRS in data/loader.py."
    )


def load_1s_data(symbol: str = 'btc') -> pd.DataFrame:
    """Backward-compatible alias used by the Streamlit backtest app."""
    return load_1s(symbol)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'logret_1s' in df.columns:
        df['ret_1s'] = df['logret_1s']
    elif 'ret_1s' not in df.columns:
        df['ret_1s'] = np.log(df['last']).diff()
    if 'log_price' not in df.columns:
        df['log_price'] = np.log(df['last'])
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['ret_1s'] = df['ret_1s'].fillna(0)
    return df
