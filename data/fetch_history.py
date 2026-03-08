"""CLI utility: fetch historical 1s bars from exchange trades and save .csv.gz."""

import argparse
from pathlib import Path

import pandas as pd

from data.exchange_fetcher import fetch_exchange_1s_to_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exchange', default='bitget')
    parser.add_argument('--symbol', required=True, help='Exchange market symbol (e.g. BTC/USDT:USDT)')
    parser.add_argument('--start', required=True, help='ISO UTC start, e.g. 2026-02-01T00:00:00Z')
    parser.add_argument('--end', required=True, help='ISO UTC end, e.g. 2026-03-04T00:00:00Z')
    parser.add_argument('--out', required=True, help='Output .csv.gz path')
    parser.add_argument('--limit', type=int, default=200)
    args = parser.parse_args()

    out = Path(args.out)
    df = fetch_exchange_1s_to_file(
        exchange=args.exchange,
        symbol=args.symbol,
        start_ts=pd.Timestamp(args.start, tz='UTC'),
        end_ts=pd.Timestamp(args.end, tz='UTC'),
        out_path=out,
        limit=args.limit,
    )
    print(f'Rows written: {len(df):,} -> {out}')


if __name__ == '__main__':
    main()
