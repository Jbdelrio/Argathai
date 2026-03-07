"""Exchange clients: Hyperliquid + Bitget (paper + live)."""
from abc import ABC, abstractmethod
import time

class ExchangeBase(ABC):
    def __init__(self, paper=True):
        self.paper = paper
        self.connected = False
    @abstractmethod
    def connect(self) -> bool: ...
    @abstractmethod
    def get_price(self, symbol) -> float: ...
    @abstractmethod
    def place_order(self, symbol, side, qty, order_type='market') -> dict: ...
    @abstractmethod
    def get_available_coins(self) -> list: ...

class HyperliquidClient(ExchangeBase):
    COINS = ['BTC','ETH','SOL','DOGE','ARB','OP','AVAX','LINK']
    def __init__(self, paper=True, **kw):
        super().__init__(paper)
        self.fees = {'maker': 0.0001, 'taker': 0.00035}
    def connect(self):
        self.connected = True; return True
    def get_price(self, symbol):
        return 73000.0 if self.paper else 0
    def place_order(self, symbol, side, qty, order_type='market'):
        p = self.get_price(symbol)
        return {'id': f'paper_{int(time.time()*1000)}', 'price': p, 'qty': qty, 'fee': qty*p*self.fees['taker']}
    def get_available_coins(self):
        return self.COINS

class BitgetClient(ExchangeBase):
    COINS = ['BTCUSDT','ETHUSDT','SOLUSDT','DOGEUSDT']
    def __init__(self, paper=True, **kw):
        super().__init__(paper)
        self.fees = {'maker': 0.0002, 'taker': 0.0006}
    def connect(self):
        self.connected = True; return True
    def get_price(self, symbol):
        return 73000.0 if self.paper else 0
    def place_order(self, symbol, side, qty, order_type='market'):
        p = self.get_price(symbol)
        return {'id': f'paper_{int(time.time()*1000)}', 'price': p, 'qty': qty, 'fee': qty*p*self.fees['taker']}
    def get_available_coins(self):
        return self.COINS

def get_client(name, paper=True, **kw):
    if 'hyperliquid' in name.lower(): return HyperliquidClient(paper, **kw)
    if 'bitget' in name.lower(): return BitgetClient(paper, **kw)
    raise ValueError(f"Unknown exchange: {name}")
