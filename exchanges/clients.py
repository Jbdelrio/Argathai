"""Exchange clients: Hyperliquid + Bitget (paper + live with guarded fallbacks)."""
from abc import ABC, abstractmethod
import json
import os
import time
import urllib.request
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — real-time price fetch for paper mode
# ─────────────────────────────────────────────────────────────────────────────

# Map short symbols (HL-style) → Binance spot symbols for public REST price feed
_BINANCE_SYMBOL_MAP = {
    'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'SOL': 'SOLUSDT',
    'DOGE': 'DOGEUSDT', 'ARB': 'ARBUSDT', 'OP': 'OPUSDT',
    'AVAX': 'AVAXUSDT', 'LINK': 'LINKUSDT',
    # Bitget already has USDT suffix — pass through
    'BTCUSDT': 'BTCUSDT', 'ETHUSDT': 'ETHUSDT', 'SOLUSDT': 'SOLUSDT',
    'DOGEUSDT': 'DOGEUSDT', 'ARBUSDT': 'ARBUSDT',
}

# Static fallbacks (last-known approximate values) when REST is unreachable
_FALLBACK_PRICES = {
    'BTC': 73000.0, 'BTCUSDT': 73000.0,
    'ETH': 3500.0,  'ETHUSDT': 3500.0,
    'SOL': 180.0,   'SOLUSDT': 180.0,
    'DOGE': 0.15,   'DOGEUSDT': 0.15,
    'ARB': 1.1,     'ARBUSDT': 1.1,
    'OP': 2.5,      'OPUSDT': 2.5,
    'AVAX': 35.0,   'AVAXUSDT': 35.0,
    'LINK': 15.0,   'LINKUSDT': 15.0,
}


def _fetch_binance_price(symbol: str, timeout: float = 2.0) -> Optional[float]:
    """Fetch real-time last price from Binance public REST — no API key required."""
    binance_sym = _BINANCE_SYMBOL_MAP.get(symbol, symbol + 'USDT')
    url = f'https://api.binance.com/api/v3/ticker/price?symbol={binance_sym}'
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return float(data['price'])
    except Exception:
        return None


def _paper_price(symbol: str) -> float:
    """Real market price for paper trading; falls back to static estimate on network error."""
    price = _fetch_binance_price(symbol)
    return price if price is not None else _FALLBACK_PRICES.get(symbol, 73000.0)


def _apply_paper_slippage(price: float, side: str, qty_usd: float) -> tuple:
    """Almgren-Chriss slippage for paper orders. Returns (exec_price, slippage_usd)."""
    try:
        from core.slippage import slippage_normal
        slip_pct = slippage_normal(qty_usd, price)
    except Exception:
        # Fallback: 1 bps flat (half-spread + tiny temp impact)
        slip_pct = 0.0001
    slippage_usd = slip_pct * qty_usd
    direction = 1 if side == 'buy' else -1  # buyer pays up, seller gets less
    exec_price = price * (1.0 + direction * slip_pct)
    return exec_price, slippage_usd


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class ExchangeBase(ABC):
    def __init__(self, paper: bool = True):
        self.paper = paper
        self.connected = False

    @abstractmethod
    def connect(self) -> bool: ...

    @abstractmethod
    def get_price(self, symbol: str) -> float: ...

    @abstractmethod
    def place_order(self, symbol: str, side: str, qty: float,
                    order_type: str = 'market') -> dict: ...

    @abstractmethod
    def get_available_coins(self) -> list: ...


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
            raise RuntimeError(
                f'Missing {exchange_id.upper()}_API_KEY or '
                f'{exchange_id.upper()}_API_SECRET for live mode'
            )

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


# ─────────────────────────────────────────────────────────────────────────────
# Hyperliquid
# ─────────────────────────────────────────────────────────────────────────────

class HyperliquidClient(_CcxtMixin, ExchangeBase):
    COINS = ['BTC', 'ETH', 'SOL', 'DOGE', 'ARB', 'OP', 'AVAX', 'LINK']

    def __init__(self, paper: bool = True, **kw):
        super().__init__(paper)
        # HL: maker 0.01% / taker 0.035%
        self.fees = {'maker': 0.0001, 'taker': 0.00035}

    def connect(self) -> bool:
        if self.paper:
            self.connected = True
            return True
        self._init_ccxt('hyperliquid')
        self.connected = self.ccxt_client is not None
        return self.connected

    def get_price(self, symbol: str) -> float:
        if self.paper:
            return _paper_price(symbol)
        return self._ticker_last(symbol)

    def place_order(self, symbol: str, side: str, qty: float,
                    order_type: str = 'market') -> dict:
        p = self.get_price(symbol)
        if self.paper:
            qty_usd = qty * p
            exec_price, slippage_usd = _apply_paper_slippage(p, side, qty_usd)
            return {
                'id': f'paper_{int(time.time() * 1000)}',
                'price': exec_price,       # realistic fill price (post-slippage)
                'mid_price': p,            # fair-value reference
                'qty': qty,
                'fee': qty_usd * self.fees['taker'],
                'slippage_usd': slippage_usd,
                'paper': True,
            }
        if self.ccxt_client is None:
            raise RuntimeError('Live mode not connected')
        order = self.ccxt_client.create_order(symbol, order_type, side, qty)
        order['paper'] = False
        return order

    def get_available_coins(self) -> list:
        return self.COINS


# ─────────────────────────────────────────────────────────────────────────────
# Bitget
# ─────────────────────────────────────────────────────────────────────────────

class BitgetClient(_CcxtMixin, ExchangeBase):
    COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT']

    def __init__(self, paper: bool = True, **kw):
        super().__init__(paper)
        # Bitget futures: maker 0.02% / taker 0.06%
        self.fees = {'maker': 0.0002, 'taker': 0.0006}

    def connect(self) -> bool:
        if self.paper:
            self.connected = True
            return True
        self._init_ccxt('bitget')
        self.connected = self.ccxt_client is not None
        return self.connected

    def get_price(self, symbol: str) -> float:
        if self.paper:
            return _paper_price(symbol)
        return self._ticker_last(symbol)

    def place_order(self, symbol: str, side: str, qty: float,
                    order_type: str = 'market') -> dict:
        p = self.get_price(symbol)
        if self.paper:
            qty_usd = qty * p
            exec_price, slippage_usd = _apply_paper_slippage(p, side, qty_usd)
            return {
                'id': f'paper_{int(time.time() * 1000)}',
                'price': exec_price,
                'mid_price': p,
                'qty': qty,
                'fee': qty_usd * self.fees['taker'],
                'slippage_usd': slippage_usd,
                'paper': True,
            }
        if self.ccxt_client is None:
            raise RuntimeError('Live mode not connected')
        order = self.ccxt_client.create_order(symbol, order_type, side, qty)
        order['paper'] = False
        return order

    def get_available_coins(self) -> list:
        return self.COINS


def get_client(name: str, paper: bool = True, **kw) -> ExchangeBase:
    lname = name.lower()
    if 'hyperliquid' in lname:
        return HyperliquidClient(paper, **kw)
    if 'bitget' in lname:
        return BitgetClient(paper, **kw)
    raise ValueError(f"Unknown exchange: {name}")
