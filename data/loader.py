"""Data loader — finds and loads CSV files."""
import pandas as pd, numpy as np
from pathlib import Path

SEARCH_DIRS = [
    Path('data/binance_spot'),
    Path(r'C:\Users\jeanb\Documents\hyperstat-arb-bot\analysis_ops\data_binance\spot'),
    Path(r'C:\Users\jeanb\Documents\Agarthai\data\binance_spot'),
]

def load_1s(symbol='btc'):
    fn = f'{symbol.lower()}_1s.csv.gz'
    for d in SEARCH_DIRS:
        p = d / fn
        if p.exists():
            df = pd.read_csv(p, compression='gzip')
            return prepare(df)
    raise FileNotFoundError(f"{fn} not found in {[str(d) for d in SEARCH_DIRS]}")

def prepare(df):
    df = df.copy()
    if 'logret_1s' in df.columns: df['ret_1s'] = df['logret_1s']
    elif 'ret_1s' not in df.columns: df['ret_1s'] = np.log(df['last']).diff()
    if 'log_price' not in df.columns: df['log_price'] = np.log(df['last'])
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['ret_1s'] = df['ret_1s'].fillna(0)
    return df
