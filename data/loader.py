"""Data loader — finds and loads CSV files with portable path discovery."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

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

    # 1) Explicit file path in settings (e.g. btc_path / eth_path)
    cfg_file = data_cfg.get(f'{symbol}_path')
    if cfg_file:
        candidates.append(Path(cfg_file))

    # 2) Optional override paths in settings (list)
    for entry in data_cfg.get('search_paths', []) or []:
        candidates.append(Path(entry) / filename)

    # 3) Optional env override (semicolon or colon separated)
    env_dirs = os.getenv('AGARTHAI_DATA_DIRS', '')
    if env_dirs:
        sep = ';' if ';' in env_dirs else os.pathsep
        for d in [p for p in env_dirs.split(sep) if p.strip()]:
            candidates.append(Path(d.strip()) / filename)

    # 4) Local project default
    candidates.append(DEFAULT_DATA_DIR / filename)

    # Deduplicate while preserving order
    uniq: list[Path] = []
    seen = set()
    for c in candidates:
        key = str(c)
        if key not in seen:
            uniq.append(c)
            seen.add(key)
    return uniq


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
