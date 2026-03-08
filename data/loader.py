"""Data loader — finds and loads CSV files with portable path discovery."""

import os
"""Data loader — finds and loads CSV files."""
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
SEARCH_DIRS = [
    Path('data/binance_spot'),
    Path(r'C:\Users\jeanb\Documents\hyperstat-arb-bot\analysis_ops\data_binance\spot'),
    Path(r'C:\Users\jeanb\Documents\Agarthai\data\binance_spot'),
]

DEFAULT_DATA_DIR = Path('data/binance_spot')
DEFAULT_FILES = {
    'btc': 'btc_1s.csv.gz',
    'eth': 'eth_1s.csv.gz',
}


def _load_settings(settings_path: Path = Path('config/settings.yaml')) -> dict:
    if settings_path.exists():
        with open(settings_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


def _candidate_paths(symbol: str) -> list[Path]:
    """Build ordered candidate file paths from config + env + local defaults."""
    symbol = symbol.lower()
    filename = DEFAULT_FILES.get(symbol, f'{symbol}_1s.csv.gz')
    settings = _load_settings()
    data_cfg = settings.get('data', {}) if isinstance(settings, dict) else {}

    candidates: list[Path] = []

    cfg_file = data_cfg.get(f'{symbol}_path')
    if cfg_file:
        candidates.append(Path(cfg_file))

    for entry in data_cfg.get('search_paths', []) or []:
        candidates.append(Path(entry) / filename)

    env_dirs = os.getenv('AGARTHAI_DATA_DIRS', '')
    if env_dirs:
        sep = ';' if ';' in env_dirs else os.pathsep
        for d in [p for p in env_dirs.split(sep) if p.strip()]:
            candidates.append(Path(d.strip()) / filename)

    candidates.append(DEFAULT_DATA_DIR / filename)

    uniq: list[Path] = []
    seen = set()
    for c in candidates:
        key = str(c)
        if key not in seen:
            uniq.append(c)
            seen.add(key)
    return uniq


def _maybe_fetch_from_exchange(symbol: str, out_path: Path) -> Optional[pd.DataFrame]:
    settings = _load_settings()
    fetch_cfg = settings.get('history_fetch', {}) if isinstance(settings, dict) else {}
    if not fetch_cfg or not fetch_cfg.get('enabled', False):
        return None

    start = fetch_cfg.get('start')
    end = fetch_cfg.get('end')
    if not start or not end:
        return None

    exchange = fetch_cfg.get('exchange', 'bitget')
    market_map = fetch_cfg.get('symbols', {}) or {}
    market_symbol = market_map.get(symbol.lower(), symbol.upper() + 'USDT')
    limit = int(fetch_cfg.get('limit', 200))

    from data.exchange_fetcher import fetch_exchange_1s_to_file

    bars = fetch_exchange_1s_to_file(
        exchange=exchange,
        symbol=market_symbol,
        start_ts=pd.Timestamp(start, tz='UTC'),
        end_ts=pd.Timestamp(end, tz='UTC'),
        out_path=out_path,
        limit=limit,
    )
    if bars.empty:
        return None
    return prepare(bars)
def load_1s(symbol='btc'):
    """Load 1-second market data for a symbol from the first existing candidate path."""
    candidates = _candidate_paths(symbol)
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p, compression='gzip')
            return prepare(df)
    raise FileNotFoundError(
        f"{symbol.lower()}_1s.csv.gz not found. Checked: {[str(p) for p in candidates]}"
    )
def load_1s_data(symbol='btc'):
    """Backward-compatible alias used by the Streamlit backtest app."""
    return load_1s(symbol)
def prepare(df):
    df = df.copy()
    if 'logret_1s' in df.columns:
        df['ret_1s'] = df['logret_1s']
    elif 'ret_1s' not in df.columns:
        df['ret_1s'] = np.log(df['last']).diff()
    if 'log_price' not in df.columns:
        df['log_price'] = np.log(df['last'])
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['ret_1s'] = df['ret_1s'].fillna(0)
    return df
