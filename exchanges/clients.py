"""Exchange clients: Hyperliquid + Bitget (paper + live with guarded fallbacks)."""
from abc import ABC, abstractmethod
import os
import time
from typing import Optional
class ExchangeBase(ABC):
    def __init__(self, paper: bool = True):
        self.paper = paper
        self.connected = False
    @abstractmethod
    def connect(self) -> bool: 
        ...

    @abstractmethod
    def get_price(self, symbol: str) -> float:
        ...

    
    @abstractmethod
    def place_order(self, symbol: str, side: str, qty: float, order_type: str = 'market') -> dict:
        ...
    @abstractmethod
    def get_available_coins(self) -> list: 
        ...

class _CcxtMixin:
    ccxt_client: Optional[object] = None

    def _init_ccxt(self, exchange_id: str):
        if self.paper:
            return
        try:
            import ccxt  # lazy import
        except Exception as exc:
            raise RuntimeError('ccxt is required for live mode') from exc

        api_key = os.getenv(f'{exchange_id.upper()}_API_KEY')
        secret = os.getenv(f'{exchange_id.upper()}_API_SECRET')
        if not api_key or not secret:
            raise RuntimeError(f'Missing {exchange_id.upper()}_API_KEY or {exchange_id.upper()}_API_SECRET for live mode')

        klass = getattr(ccxt, exchange_id)
        self.ccxt_client = klass({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
        })
        self.ccxt_client.load_markets()

    def _ticker_last(self, symbol: str) -> float:
        if self.ccxt_client is None:
            raise RuntimeError('Live client not initialized')
        ticker = self.ccxt_client.fetch_ticker(symbol)
        last = ticker.get('last')
        if last is None:
            raise RuntimeError(f'No last price for {symbol}')
        return float(last)



class HyperliquidClient(_CcxtMixin, ExchangeBase):
    COINS = ['BTC', 'ETH', 'SOL', 'DOGE', 'ARB', 'OP', 'AVAX', 'LINK']

    def __init__(self, paper: bool = True, **kw):
        super().__init__(paper)
        self.fees = {'maker': 0.0001, 'taker': 0.00035}
    def connect(self):
        if self.paper:
            self.connected = True
            return True
        self._init_ccxt('hyperliquid')
        self.connected = self.ccxt_client is not None
        return self.connected
    def get_price(self, symbol):
        if self.paper:
            return 73000.0
        return self._ticker_last(symbol)
    def place_order(self, symbol, side, qty, order_type='market'):
        p = self.get_price(symbol)
        if self.paper:
            return {
                'id': f'paper_{int(time.time()*1000)}',
                'price': p,
                'qty': qty,
                'fee': qty * p * self.fees['taker'],
                'paper': True,
            }

        if self.ccxt_client is None:
            raise RuntimeError('Live mode not connected')
        order = self.ccxt_client.create_order(symbol, order_type, side, qty)
        order['paper'] = False
        return order

    def get_available_coins(self):
        return self.COINS

class BitgetClient(_CcxtMixin, ExchangeBase):
    COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT']

    def __init__(self, paper: bool = True, **kw):
        super().__init__(paper)
        self.fees = {'maker': 0.0002, 'taker': 0.0006}
    def connect(self):
        if self.paper:
            self.connected = True
            return True
        self._init_ccxt('bitget')
        self.connected = self.ccxt_client is not None
        return self.connected
    def get_price(self, symbol):
        if self.paper:
            return 73000.0
        return self._ticker_last(symbol)
    def place_order(self, symbol, side, qty, order_type='market'):
        p = self.get_price(symbol)
        if self.paper:
            return {
                'id': f'paper_{int(time.time()*1000)}',
                'price': p,
                'qty': qty,
                'fee': qty * p * self.fees['taker'],
                'paper': True,
            }

        if self.ccxt_client is None:
            raise RuntimeError('Live mode not connected')
        order = self.ccxt_client.create_order(symbol, order_type, side, qty)
        order['paper'] = False
        return order

    def get_available_coins(self):
        return self.COINS

def get_client(name, paper=True, **kw):
    lname = name.lower()
    if 'hyperliquid' in lname:
        return HyperliquidClient(paper, **kw)
    if 'bitget' in lname:
        return BitgetClient(paper, **kw)
    raise ValueError(f"Unknown exchange: {name}")
