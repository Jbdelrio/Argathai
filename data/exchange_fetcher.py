"""Historical market data fetcher from exchange APIs (pagination + rate-limit aware)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class FetchConfig:
    exchange: str = 'bitget'
    max_limit: int = 200
    sleep_ms: Optional[int] = None


class ExchangeHistoricalFetcher:
    """Fetch historical trade data and aggregate into 1-second bars."""

    def __init__(self, config: FetchConfig):
        self.config = config
        self.client = self._make_client(config.exchange)
        self.sleep_ms = config.sleep_ms if config.sleep_ms is not None else int(getattr(self.client, 'rateLimit', 200))

    @staticmethod
    def _make_client(exchange: str):
        try:
            import ccxt
        except Exception as exc:
            raise RuntimeError('ccxt is required to fetch historical data from exchange') from exc

        if not hasattr(ccxt, exchange):
            raise ValueError(f'Unsupported exchange for ccxt: {exchange}')
        klass = getattr(ccxt, exchange)
        return klass({'enableRateLimit': True})

    def fetch_trades_paginated(
        self,
        symbol: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch raw trades between [start_ts, end_ts] using paginated fetch_trades."""
        if limit is None:
            limit = self.config.max_limit

        since = int(pd.Timestamp(start_ts, tz='UTC').timestamp() * 1000)
        until = int(pd.Timestamp(end_ts, tz='UTC').timestamp() * 1000)

        all_rows = []
        while since < until:
            batch = self.client.fetch_trades(symbol, since=since, limit=limit)
            if not batch:
                break

            for t in batch:
                ts = t.get('timestamp')
                if ts is None or ts > until:
                    continue
                all_rows.append({
                    'timestamp': pd.to_datetime(ts, unit='ms', utc=True),
                    'price': float(t.get('price', np.nan)),
                    'amount': float(t.get('amount', np.nan)),
                    'side': t.get('side') or 'buy',
                    'id': t.get('id'),
                })

            last_ts = batch[-1].get('timestamp')
            if last_ts is None:
                break
            next_since = int(last_ts) + 1
            if next_since <= since:
                break
            since = next_since
            time.sleep(self.sleep_ms / 1000.0)

        if not all_rows:
            return pd.DataFrame(columns=['timestamp', 'price', 'amount', 'side', 'id'])

        df = pd.DataFrame(all_rows).drop_duplicates(subset=['timestamp', 'price', 'amount', 'side', 'id'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df

    @staticmethod
    def trades_to_1s(trades_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Aggregate trade tape to 1-second bars compatible with Agarthai format."""
        if trades_df.empty:
            return pd.DataFrame(columns=[
                'timestamp', 'vwap', 'last', 'qty', 'buy_qty', 'sell_qty',
                'n_trades', 'ofi_proxy', 'symbol', 'ret_1s', 'log_price',
            ])

        d = trades_df.copy()
        d['sec'] = d['timestamp'].dt.floor('1s')
        d['notional'] = d['price'] * d['amount']
        d['buy_amt'] = np.where(d['side'].str.lower() == 'buy', d['amount'], 0.0)
        d['sell_amt'] = np.where(d['side'].str.lower() == 'sell', d['amount'], 0.0)

        g = d.groupby('sec', as_index=False).agg(
            qty=('amount', 'sum'),
            notional=('notional', 'sum'),
            last=('price', 'last'),
            buy_qty=('buy_amt', 'sum'),
            sell_qty=('sell_amt', 'sum'),
            n_trades=('price', 'count'),
        )
        g.rename(columns={'sec': 'timestamp'}, inplace=True)
        g['vwap'] = g['notional'] / g['qty'].replace(0, np.nan)
        g['vwap'] = g['vwap'].fillna(g['last'])
        g['ofi_proxy'] = g['buy_qty'] - g['sell_qty']
        g['symbol'] = symbol
        g['log_price'] = np.log(g['last'].replace(0, np.nan)).ffill()
        g['ret_1s'] = g['log_price'].diff().fillna(0)

        return g[['timestamp', 'vwap', 'last', 'qty', 'buy_qty', 'sell_qty', 'n_trades', 'ofi_proxy', 'symbol', 'ret_1s', 'log_price']]

    def fetch_and_save_1s(
        self,
        symbol: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        out_path: Path,
    ) -> pd.DataFrame:
        trades = self.fetch_trades_paginated(symbol=symbol, start_ts=start_ts, end_ts=end_ts)
        bars = self.trades_to_1s(trades, symbol=symbol)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bars.to_csv(out_path, index=False, compression='gzip')
        return bars


def fetch_exchange_1s_to_file(
    exchange: str,
    symbol: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    out_path: Path,
    limit: int = 200,
) -> pd.DataFrame:
    fetcher = ExchangeHistoricalFetcher(FetchConfig(exchange=exchange, max_limit=limit))
    return fetcher.fetch_and_save_1s(symbol=symbol, start_ts=start_ts, end_ts=end_ts, out_path=out_path)
