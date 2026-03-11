"""Exchange clients: Hyperliquid + Bitget (paper + live with guarded fallbacks)."""
from abc import ABC, abstractmethod
import json
import logging
import os
import threading
import time
import urllib.request
from typing import Dict, List, Optional

logger = logging.getLogger('agarthai.exchange')


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
# Rich tick data — real volume / trades / OFI from Binance aggTrades
# ─────────────────────────────────────────────────────────────────────────────

_TICK_CACHE: Dict[str, dict] = {}
_TICK_CACHE_LOCK = threading.Lock()
_TICK_CACHE_TTL_MS = 800          # cache validity: 0.8 s


def _fetch_binance_tick_data(symbol: str, timeout: float = 2.0) -> dict:
    """
    Fetch recent aggTrades from Binance and aggregate into a 1-second tick.

    Returns dict with:
      price, qty, buy_qty, sell_qty, n_trades, ofi_proxy

    Uses a short-lived thread-safe cache so multiple strategies on the
    same coin don't hammer the API.
    """
    now_ms = int(time.time() * 1000)

    # ── Cache check ──────────────────────────────────────────────────
    with _TICK_CACHE_LOCK:
        cached = _TICK_CACHE.get(symbol)
        if cached and (now_ms - cached['ts_ms']) < _TICK_CACHE_TTL_MS:
            return cached['data']

    binance_sym = _BINANCE_SYMBOL_MAP.get(symbol, symbol + 'USDT')

    # Fetch trades from the last 1.5 seconds
    start_ms = now_ms - 1500
    url = (
        f'https://api.binance.com/api/v3/aggTrades'
        f'?symbol={binance_sym}&startTime={start_ms}&endTime={now_ms}'
    )

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            trades = json.loads(resp.read())
    except Exception:
        # Fallback: price-only (legacy behaviour)
        price = _fetch_binance_price(symbol)
        if price is None:
            price = _FALLBACK_PRICES.get(symbol, 73000.0)
        return {
            'price': price, 'qty': 0.0,
            'buy_qty': 0.0, 'sell_qty': 0.0,
            'n_trades': 0, 'ofi_proxy': 0.0,
        }

    if not trades:
        price = _fetch_binance_price(symbol)
        if price is None:
            price = _FALLBACK_PRICES.get(symbol, 73000.0)
        return {
            'price': price, 'qty': 0.0,
            'buy_qty': 0.0, 'sell_qty': 0.0,
            'n_trades': 0, 'ofi_proxy': 0.0,
        }

    # ── Aggregate trades ─────────────────────────────────────────────
    last_price = float(trades[-1]['p'])
    qty_total = 0.0
    buy_qty   = 0.0
    sell_qty  = 0.0

    for t in trades:
        q = float(t['q'])
        qty_total += q
        # m = True  → buyer was maker → taker SOLD  (aggressive sell)
        # m = False → seller was maker → taker BOUGHT (aggressive buy)
        if t['m']:
            sell_qty += q
        else:
            buy_qty += q

    data = {
        'price':     last_price,
        'qty':       qty_total,
        'buy_qty':   buy_qty,
        'sell_qty':  sell_qty,
        'n_trades':  len(trades),
        'ofi_proxy': buy_qty - sell_qty,
    }

    with _TICK_CACHE_LOCK:
        _TICK_CACHE[symbol] = {'ts_ms': now_ms, 'data': data}

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Symbol normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_hl_coin(symbol: str) -> str:
    """'BTCUSDT' / 'BTC' → 'BTC' (Hyperliquid uses bare coin names)."""
    s = symbol.upper().replace('USDT', '').replace('-PERP', '')
    return s


def _to_bitget_symbol(symbol: str) -> str:
    """'BTC' / 'BTCUSDT' → 'BTCUSDT' (Bitget futures uses pair format)."""
    s = symbol.upper().replace('-PERP', '')
    if not s.endswith('USDT'):
        s += 'USDT'
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Hyperliquid public API — candles + recent trades (no API key)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_hl_candles(coin: str, minutes_back: int = 120,
                      timeout: float = 10.0) -> List[dict]:
    """
    Fetch 1m candles from Hyperliquid via candleSnapshot (POST).
    Returns list of dicts: {t, o, h, l, c, v, n}
    """
    hl_coin = _to_hl_coin(coin)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - minutes_back * 60 * 1000
    payload = json.dumps({
        "type": "candleSnapshot",
        "req": {
            "coin": hl_coin,
            "interval": "1m",
            "startTime": start_ms,
            "endTime": end_ms,
        }
    }).encode()

    req = urllib.request.Request(
        "https://api.hyperliquid.xyz/info",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            candles = json.loads(resp.read())
        logger.info(f"[HL] Fetched {len(candles)} candles for {hl_coin}")
        return candles
    except Exception as e:
        logger.warning(f"[HL] candle fetch failed: {e}")
        return []


def _fetch_hl_tick_data(symbol: str, timeout: float = 3.0) -> dict:
    """
    Fetch recent trades from Hyperliquid (recentTrades endpoint).
    Returns aggregated tick: {price, qty, buy_qty, sell_qty, n_trades, ofi_proxy}
    """
    now_ms = int(time.time() * 1000)

    with _TICK_CACHE_LOCK:
        key = f'hl_{symbol}'
        cached = _TICK_CACHE.get(key)
        if cached and (now_ms - cached['ts_ms']) < _TICK_CACHE_TTL_MS:
            return cached['data']

    hl_coin = _to_hl_coin(symbol)
    payload = json.dumps({
        "type": "recentTrades",
        "coin": hl_coin,
    }).encode()

    req = urllib.request.Request(
        "https://api.hyperliquid.xyz/info",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            trades = json.loads(resp.read())
    except Exception:
        return _fetch_binance_tick_data(symbol)  # fallback

    if not trades:
        return _fetch_binance_tick_data(symbol)

    last_price = float(trades[-1]['px'])
    qty_total = 0.0
    buy_qty = 0.0
    sell_qty = 0.0
    for t in trades:
        q = float(t['sz'])
        qty_total += q
        # side "B" = buyer was aggressor (taker buy), "A" = seller was aggressor
        if t['side'] == 'B':
            buy_qty += q
        else:
            sell_qty += q

    data = {
        'price':     last_price,
        'qty':       qty_total,
        'buy_qty':   buy_qty,
        'sell_qty':  sell_qty,
        'n_trades':  len(trades),
        'ofi_proxy': buy_qty - sell_qty,
    }

    with _TICK_CACHE_LOCK:
        _TICK_CACHE[f'hl_{symbol}'] = {'ts_ms': now_ms, 'data': data}

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Bitget public API — candles + recent fills (no API key)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_bitget_candles(coin: str, minutes_back: int = 120,
                          timeout: float = 10.0) -> List[list]:
    """
    Fetch 1m candles from Bitget futures public API (GET).
    Returns list of [ts, open, high, low, close, vol_base, vol_quote].
    """
    bg_sym = _to_bitget_symbol(coin)
    limit = min(minutes_back, 1000)
    url = (
        f"https://api.bitget.com/api/v2/mix/market/candles"
        f"?symbol={bg_sym}&granularity=1m&limit={limit}"
        f"&productType=USDT-FUTURES"
    )
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = json.loads(resp.read())
        if body.get('code') != '00000':
            logger.warning(f"[Bitget] candle error: {body.get('msg')}")
            return []
        candles = body.get('data', [])
        logger.info(f"[Bitget] Fetched {len(candles)} candles for {bg_sym}")
        return candles
    except Exception as e:
        logger.warning(f"[Bitget] candle fetch failed: {e}")
        return []


def _fetch_bitget_tick_data(symbol: str, timeout: float = 3.0) -> dict:
    """
    Fetch recent fills from Bitget (public fills endpoint).
    Returns aggregated tick: {price, qty, buy_qty, sell_qty, n_trades, ofi_proxy}
    """
    now_ms = int(time.time() * 1000)

    with _TICK_CACHE_LOCK:
        key = f'bg_{symbol}'
        cached = _TICK_CACHE.get(key)
        if cached and (now_ms - cached['ts_ms']) < _TICK_CACHE_TTL_MS:
            return cached['data']

    bg_sym = _to_bitget_symbol(symbol)
    url = (
        f"https://api.bitget.com/api/v2/mix/market/fills"
        f"?symbol={bg_sym}&productType=USDT-FUTURES&limit=100"
    )
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = json.loads(resp.read())
        if body.get('code') != '00000':
            return _fetch_binance_tick_data(symbol)
        trades = body.get('data', [])
    except Exception:
        return _fetch_binance_tick_data(symbol)

    if not trades:
        return _fetch_binance_tick_data(symbol)

    last_price = float(trades[0]['price'])  # most recent first
    qty_total = 0.0
    buy_qty = 0.0
    sell_qty = 0.0
    for t in trades:
        q = float(t['size'])
        qty_total += q
        if t['side'] == 'buy':
            buy_qty += q
        else:
            sell_qty += q

    data = {
        'price':     last_price,
        'qty':       qty_total,
        'buy_qty':   buy_qty,
        'sell_qty':  sell_qty,
        'n_trades':  len(trades),
        'ofi_proxy': buy_qty - sell_qty,
    }

    with _TICK_CACHE_LOCK:
        _TICK_CACHE[f'bg_{symbol}'] = {'ts_ms': now_ms, 'data': data}

    return data


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

    def get_tick_data(self, symbol: str) -> dict:
        """
        Fetch rich tick data: price + volume + trades + OFI.
        Override in subclasses for exchange-specific implementation.
        """
        return _fetch_binance_tick_data(symbol)

    @abstractmethod
    def fetch_bootstrap_candles(self, coin: str,
                                minutes_back: int = 120) -> List[dict]:
        """
        Download recent 1m candles from the exchange for live warmup.

        Returns list of normalised dicts:
          {ts_ms, open, high, low, close, volume, n_trades}
        sorted oldest-first.
        """
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

    def get_tick_data(self, symbol: str) -> dict:
        """Tick data from Hyperliquid recentTrades (fallback: Binance)."""
        return _fetch_hl_tick_data(symbol)

    def fetch_bootstrap_candles(self, coin: str,
                                minutes_back: int = 120) -> List[dict]:
        """Fetch 1m candles from Hyperliquid for warmup bootstrap."""
        raw = _fetch_hl_candles(coin, minutes_back)
        if not raw:
            return []
        # Normalise HL format → standard dict
        out: List[dict] = []
        for c in raw:
            out.append({
                'ts_ms':    int(c['t']),
                'open':     float(c['o']),
                'high':     float(c['h']),
                'low':      float(c['l']),
                'close':    float(c['c']),
                'volume':   float(c['v']),
                'n_trades': int(c.get('n', 0)),
            })
        # Sort oldest first
        out.sort(key=lambda x: x['ts_ms'])
        return out

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

    def get_tick_data(self, symbol: str) -> dict:
        """Tick data from Bitget recent fills (fallback: Binance)."""
        return _fetch_bitget_tick_data(symbol)

    def fetch_bootstrap_candles(self, coin: str,
                                minutes_back: int = 120) -> List[dict]:
        """Fetch 1m candles from Bitget for warmup bootstrap."""
        raw = _fetch_bitget_candles(coin, minutes_back)
        if not raw:
            return []
        # Normalise Bitget format [ts, o, h, l, c, vol, vol_quote] → dict
        out: List[dict] = []
        for c in raw:
            out.append({
                'ts_ms':    int(c[0]),
                'open':     float(c[1]),
                'high':     float(c[2]),
                'low':      float(c[3]),
                'close':    float(c[4]),
                'volume':   float(c[5]),
                'n_trades': 0,  # Bitget candles don't include trade count
            })
        # Sort oldest first
        out.sort(key=lambda x: x['ts_ms'])
        return out

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
