"""
api_connectors.py – Real API connectors for Polymarket & Kalshi
Paper trading mode: read-only data, no auth required for market data

Polymarket Gamma API: https://gamma-api.polymarket.com  (public, no auth)
Polymarket CLOB:      https://clob.polymarket.com       (public reads, auth for trades)
Kalshi REST v2:       https://api.elections.kalshi.com/trade-api/v2
Kalshi Demo:          https://demo-api.kalshi.co/trade-api/v2
"""

import httpx
import asyncio
import time
import logging
import json
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field

logger = logging.getLogger("api-connectors")

# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class MarketData:
    """Unified market data across platforms"""
    platform: str
    market_id: str
    question: str
    slug: str
    yes_price: float
    no_price: float
    volume_24h: float
    liquidity: float
    outcomes: List[str]
    close_time: Optional[str] = None
    status: str = "active"
    asset: Optional[str] = None        # BTC, ETH, etc.
    market_type: Optional[str] = None  # "5min", "daily", etc.
    token_ids: Dict[str, str] = field(default_factory=dict)
    raw: Dict = field(default_factory=dict)
    fetch_latency_ms: float = 0
    fetched_at: float = field(default_factory=time.time)


@dataclass
class OrderBookLevel:
    price: float
    size: float


@dataclass
class OrderBook:
    platform: str
    market_id: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    spread: float = 0
    mid_price: float = 0
    fetch_latency_ms: float = 0

    def __post_init__(self):
        if self.bids and self.asks:
            best_bid = max(b.price for b in self.bids) if self.bids else 0
            best_ask = min(a.price for a in self.asks) if self.asks else 1
            self.spread = best_ask - best_bid
            self.mid_price = (best_bid + best_ask) / 2

    @property
    def best_bid(self) -> float:
        return max((b.price for b in self.bids), default=0)

    @property
    def best_ask(self) -> float:
        return min((a.price for a in self.asks), default=1)

    @property
    def bid_depth(self) -> float:
        return sum(b.size for b in self.bids)

    @property
    def ask_depth(self) -> float:
        return sum(a.size for a in self.asks)


# ============================================================
# POLYMARKET CONNECTOR
# ============================================================

class PolymarketConnector:
    """
    Polymarket API connector (read-only for paper trading).
    
    Gamma API (public, no auth):
      - GET /markets          → list markets
      - GET /markets/{slug}   → single market
      - GET /events           → list events with nested markets
    
    CLOB API (public for reads):
      - GET /book?token_id=X  → order book
      - GET /price?token_id=X → latest price
      - GET /midpoint?token_id=X → midpoint price
    """
    
    GAMMA_URL = "https://gamma-api.polymarket.com"
    CLOB_URL = "https://clob.polymarket.com"
    
    def __init__(self, timeout: float = 10.0):
        self.client = httpx.AsyncClient(timeout=timeout)
        self.name = "polymarket"
        self._connected = False
        self._last_latency_ms = 0.0
        self._request_count = 0
        self._error_count = 0
    
    async def connect(self) -> bool:
        """Test connectivity"""
        try:
            t0 = time.time()
            resp = await self.client.get(f"{self.GAMMA_URL}/markets", params={"limit": 1})
            self._last_latency_ms = (time.time() - t0) * 1000
            self._connected = resp.status_code == 200
            if self._connected:
                logger.info(f"Polymarket connected ({self._last_latency_ms:.0f}ms)")
            return self._connected
        except Exception as e:
            logger.error(f"Polymarket connection failed: {e}")
            self._connected = False
            return False
    
    async def get_markets(
        self,
        active: bool = True,
        limit: int = 50,
        order: str = "volume24hr",
        tag: Optional[str] = None,
    ) -> List[MarketData]:
        """Fetch markets from Gamma API"""
        params = {
            "active": str(active).lower(),
            "closed": "false",
            "limit": limit,
            "order": order,
            "ascending": "false",
        }
        if tag:
            params["tag_slug"] = tag
        
        try:
            t0 = time.time()
            resp = await self.client.get(f"{self.GAMMA_URL}/markets", params=params)
            latency = (time.time() - t0) * 1000
            self._last_latency_ms = latency
            self._request_count += 1
            
            if resp.status_code != 200:
                self._error_count += 1
                logger.warning(f"Polymarket markets error: {resp.status_code}")
                return []
            
            markets = []
            for m in resp.json():
                # Parse outcome prices
                outcome_prices = m.get("outcomePrices", "[]")
                if isinstance(outcome_prices, str):
                    try:
                        prices = json.loads(outcome_prices)
                    except:
                        prices = [0.5, 0.5]
                else:
                    prices = outcome_prices
                
                yes_price = float(prices[0]) if len(prices) > 0 else 0.5
                no_price = float(prices[1]) if len(prices) > 1 else 1 - yes_price
                
                # Parse token IDs
                token_ids = {}
                clob_tokens = m.get("clobTokenIds", "[]")
                if isinstance(clob_tokens, str):
                    try:
                        tokens = json.loads(clob_tokens)
                    except:
                        tokens = []
                else:
                    tokens = clob_tokens
                
                outcomes = m.get("outcomes", '["Yes","No"]')
                if isinstance(outcomes, str):
                    try:
                        outcomes = json.loads(outcomes)
                    except:
                        outcomes = ["Yes", "No"]
                
                if len(tokens) >= 2:
                    token_ids = {"yes": tokens[0], "no": tokens[1]}
                
                # Detect asset and market type
                question = m.get("question", "")
                asset = self._detect_asset(question)
                market_type = self._detect_market_type(question)
                
                markets.append(MarketData(
                    platform="polymarket",
                    market_id=m.get("conditionId", m.get("id", "")),
                    question=question,
                    slug=m.get("slug", ""),
                    yes_price=yes_price,
                    no_price=no_price,
                    volume_24h=float(m.get("volume24hr", 0) or 0),
                    liquidity=float(m.get("liquidity", 0) or 0),
                    outcomes=outcomes,
                    close_time=m.get("endDate"),
                    status="active" if m.get("active") else "closed",
                    asset=asset,
                    market_type=market_type,
                    token_ids=token_ids,
                    raw=m,
                    fetch_latency_ms=latency,
                ))
            
            return markets
        
        except Exception as e:
            self._error_count += 1
            logger.error(f"Polymarket get_markets error: {e}")
            return []
    
    async def get_events(
        self,
        limit: int = 50,
        order: str = "volume24hr",
    ) -> List[Dict]:
        """Fetch events (groups of related markets)"""
        try:
            t0 = time.time()
            resp = await self.client.get(f"{self.GAMMA_URL}/events", params={
                "closed": "false",
                "limit": limit,
                "order": order,
                "ascending": "false",
            })
            self._last_latency_ms = (time.time() - t0) * 1000
            self._request_count += 1
            
            if resp.status_code == 200:
                return resp.json()
            return []
        except Exception as e:
            self._error_count += 1
            logger.error(f"Polymarket get_events error: {e}")
            return []
    
    async def get_crypto_5min_markets(self) -> List[MarketData]:
        """
        Fetch specifically the 5-minute crypto Up/Down markets.
        These are the markets like "Ethereum Up or Down - April 2, 7:55AM-8:00AM ET"
        """
        all_markets = await self.get_markets(limit=100, tag="crypto-prices")
        
        # Filter for 5min Up/Down markets
        five_min = [
            m for m in all_markets
            if "up or down" in m.question.lower()
            or ("up" in m.question.lower() and "down" in m.question.lower())
        ]
        
        # If tag doesn't work, try broader search
        if not five_min:
            all_markets = await self.get_markets(limit=200)
            five_min = [
                m for m in all_markets
                if "up or down" in m.question.lower()
            ]
        
        return five_min
    
    async def get_order_book(self, token_id: str) -> Optional[OrderBook]:
        """Fetch order book from CLOB API (public endpoint)"""
        try:
            t0 = time.time()
            resp = await self.client.get(
                f"{self.CLOB_URL}/book",
                params={"token_id": token_id}
            )
            latency = (time.time() - t0) * 1000
            self._request_count += 1
            
            if resp.status_code != 200:
                self._error_count += 1
                return None
            
            data = resp.json()
            bids = [
                OrderBookLevel(float(b["price"]), float(b["size"]))
                for b in data.get("bids", [])
            ]
            asks = [
                OrderBookLevel(float(a["price"]), float(a["size"]))
                for a in data.get("asks", [])
            ]
            
            return OrderBook(
                platform="polymarket",
                market_id=token_id,
                bids=sorted(bids, key=lambda x: x.price, reverse=True),
                asks=sorted(asks, key=lambda x: x.price),
                fetch_latency_ms=latency,
            )
        except Exception as e:
            self._error_count += 1
            logger.error(f"Polymarket order book error: {e}")
            return None
    
    async def get_midpoint(self, token_id: str) -> Optional[float]:
        """Get midpoint price for a token"""
        try:
            resp = await self.client.get(
                f"{self.CLOB_URL}/midpoint",
                params={"token_id": token_id}
            )
            self._request_count += 1
            if resp.status_code == 200:
                data = resp.json()
                return float(data.get("mid", 0))
            return None
        except Exception as e:
            self._error_count += 1
            return None
    
    def _detect_asset(self, question: str) -> Optional[str]:
        """Detect crypto asset from market question"""
        q = question.lower()
        assets = {
            "bitcoin": "BTC", "btc": "BTC",
            "ethereum": "ETH", "eth ": "ETH",
            "solana": "SOL", "sol ": "SOL",
            "xrp": "XRP", "ripple": "XRP",
            "dogecoin": "DOGE", "doge": "DOGE",
            "cardano": "ADA", "ada ": "ADA",
            "polygon": "MATIC", "matic": "MATIC",
            "avalanche": "AVAX", "avax": "AVAX",
        }
        for keyword, ticker in assets.items():
            if keyword in q:
                return ticker
        return None
    
    def _detect_market_type(self, question: str) -> Optional[str]:
        """Detect market type from question"""
        q = question.lower()
        if any(t in q for t in ["5min", "5 min", "am-", "pm-", ":00am", ":00pm"]):
            return "5min"
        if "daily" in q or "end of day" in q:
            return "daily"
        if "weekly" in q or "end of week" in q:
            return "weekly"
        if "up or down" in q:
            return "5min"  # Most Up/Down are 5min
        return None
    
    @property
    def status(self) -> Dict:
        return {
            "platform": self.name,
            "connected": self._connected,
            "latency_ms": self._last_latency_ms,
            "requests": self._request_count,
            "errors": self._error_count,
        }
    
    async def close(self):
        await self.client.aclose()


# ============================================================
# KALSHI CONNECTOR
# ============================================================

class KalshiConnector:
    """
    Kalshi API connector.
    
    Production: https://api.elections.kalshi.com/trade-api/v2
    Demo:       https://demo-api.kalshi.co/trade-api/v2
    
    Market data endpoints are public (no auth needed for GET /markets).
    Trading requires RSA-PSS authentication.
    
    Note: Prices are in cents (integers). 65 = $0.65
    """
    
    PROD_URL = "https://api.elections.kalshi.com/trade-api/v2"
    DEMO_URL = "https://demo-api.kalshi.co/trade-api/v2"
    
    def __init__(
        self,
        use_demo: bool = True,
        api_key_id: Optional[str] = None,
        private_key_path: Optional[str] = None,
        timeout: float = 10.0,
    ):
        self.base_url = self.DEMO_URL if use_demo else self.PROD_URL
        self.use_demo = use_demo
        self.api_key_id = api_key_id
        self.private_key_path = private_key_path
        self.client = httpx.AsyncClient(timeout=timeout)
        self.name = "kalshi"
        self._connected = False
        self._last_latency_ms = 0.0
        self._request_count = 0
        self._error_count = 0
    
    async def connect(self) -> bool:
        """Test connectivity"""
        try:
            t0 = time.time()
            resp = await self.client.get(
                f"{self.base_url}/markets",
                params={"limit": 1, "status": "open"}
            )
            self._last_latency_ms = (time.time() - t0) * 1000
            self._connected = resp.status_code == 200
            if self._connected:
                env = "DEMO" if self.use_demo else "PROD"
                logger.info(f"Kalshi [{env}] connected ({self._last_latency_ms:.0f}ms)")
            return self._connected
        except Exception as e:
            logger.error(f"Kalshi connection failed: {e}")
            self._connected = False
            return False
    
    async def get_markets(
        self,
        limit: int = 100,
        status: str = "open",
        series_ticker: Optional[str] = None,
        cursor: Optional[str] = None,
    ) -> List[MarketData]:
        """Fetch markets from Kalshi"""
        params: Dict[str, Any] = {
            "limit": limit,
            "status": status,
        }
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
        
        try:
            t0 = time.time()
            resp = await self.client.get(f"{self.base_url}/markets", params=params)
            latency = (time.time() - t0) * 1000
            self._last_latency_ms = latency
            self._request_count += 1
            
            if resp.status_code != 200:
                self._error_count += 1
                logger.warning(f"Kalshi markets error: {resp.status_code} - {resp.text[:200]}")
                return []
            
            data = resp.json()
            markets = []
            
            for m in data.get("markets", []):
                # Kalshi prices are in dollars as strings like "0.5600"
                yes_bid = float(m.get("yes_bid_dollars", "0") or "0")
                yes_ask = float(m.get("yes_ask_dollars", "0") or "0")
                no_bid = float(m.get("no_bid_dollars", "0") or "0")
                no_ask = float(m.get("no_ask_dollars", "0") or "0")
                last_price = float(m.get("last_price_dollars", "0") or "0")
                
                yes_price = (yes_bid + yes_ask) / 2 if (yes_bid and yes_ask) else last_price
                no_price = 1 - yes_price if yes_price > 0 else 0.5
                
                ticker = m.get("ticker", "")
                question = m.get("title", m.get("yes_sub_title", ticker))
                asset = self._detect_asset(question + " " + ticker)
                
                markets.append(MarketData(
                    platform="kalshi",
                    market_id=ticker,
                    question=question,
                    slug=ticker,
                    yes_price=yes_price,
                    no_price=no_price,
                    volume_24h=float(m.get("volume_24h_fp", "0") or "0"),
                    liquidity=float(m.get("liquidity_dollars", "0") or "0"),
                    outcomes=["Yes", "No"],
                    close_time=m.get("close_time"),
                    status=m.get("status", "open"),
                    asset=asset,
                    market_type=self._detect_market_type(question + " " + ticker),
                    raw=m,
                    fetch_latency_ms=latency,
                ))
            
            return markets
        
        except Exception as e:
            self._error_count += 1
            logger.error(f"Kalshi get_markets error: {e}")
            return []
    
    async def get_market(self, ticker: str) -> Optional[MarketData]:
        """Get a single market by ticker"""
        try:
            t0 = time.time()
            resp = await self.client.get(f"{self.base_url}/markets/{ticker}")
            latency = (time.time() - t0) * 1000
            self._request_count += 1
            
            if resp.status_code != 200:
                self._error_count += 1
                return None
            
            m = resp.json().get("market", {})
            yes_price = float(m.get("last_price_dollars", "0.5") or "0.5")
            
            return MarketData(
                platform="kalshi",
                market_id=m.get("ticker", ticker),
                question=m.get("title", ticker),
                slug=ticker,
                yes_price=yes_price,
                no_price=1 - yes_price,
                volume_24h=float(m.get("volume_24h_fp", "0") or "0"),
                liquidity=float(m.get("liquidity_dollars", "0") or "0"),
                outcomes=["Yes", "No"],
                close_time=m.get("close_time"),
                status=m.get("status", "open"),
                raw=m,
                fetch_latency_ms=latency,
            )
        except Exception as e:
            self._error_count += 1
            return None
    
    async def get_order_book(self, ticker: str) -> Optional[OrderBook]:
        """Fetch order book for a market"""
        try:
            t0 = time.time()
            resp = await self.client.get(f"{self.base_url}/markets/{ticker}/orderbook")
            latency = (time.time() - t0) * 1000
            self._request_count += 1
            
            if resp.status_code != 200:
                self._error_count += 1
                return None
            
            data = resp.json().get("orderbook", {})
            
            bids = []
            for price, size in zip(
                data.get("yes", {}).get("price", []),
                data.get("yes", {}).get("size", [])
            ):
                bids.append(OrderBookLevel(price / 100, size))  # cents → dollars
            
            asks = []
            for price, size in zip(
                data.get("no", {}).get("price", []),
                data.get("no", {}).get("size", [])
            ):
                asks.append(OrderBookLevel(price / 100, size))
            
            return OrderBook(
                platform="kalshi",
                market_id=ticker,
                bids=sorted(bids, key=lambda x: x.price, reverse=True),
                asks=sorted(asks, key=lambda x: x.price),
                fetch_latency_ms=latency,
            )
        except Exception as e:
            self._error_count += 1
            return None
    
    def _detect_asset(self, text: str) -> Optional[str]:
        t = text.upper()
        for asset in ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX"]:
            if asset in t:
                return asset
        if "BITCOIN" in t:
            return "BTC"
        if "ETHEREUM" in t:
            return "ETH"
        return None
    
    def _detect_market_type(self, text: str) -> Optional[str]:
        t = text.lower()
        if any(x in t for x in ["5min", "5 min", "inx"]):
            return "5min"
        if "daily" in t:
            return "daily"
        return None
    
    @property
    def status(self) -> Dict:
        env = "demo" if self.use_demo else "prod"
        return {
            "platform": f"{self.name} ({env})",
            "connected": self._connected,
            "latency_ms": self._last_latency_ms,
            "requests": self._request_count,
            "errors": self._error_count,
        }
    
    async def close(self):
        await self.client.aclose()


# ============================================================
# UNIFIED MARKET SCANNER
# ============================================================

class MarketScanner:
    """
    Scans all connected platforms for markets and detects cross-platform matches.
    """
    
    def __init__(self):
        self.polymarket = PolymarketConnector()
        self.kalshi = KalshiConnector(use_demo=True)
        self.connectors = {
            "polymarket": self.polymarket,
            "kalshi": self.kalshi,
        }
        self._all_markets: Dict[str, List[MarketData]] = {}
        self._scan_count = 0
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all platforms"""
        results = {}
        for name, conn in self.connectors.items():
            results[name] = await conn.connect()
        return results
    
    async def scan_all(self) -> Dict[str, List[MarketData]]:
        """Fetch markets from all connected platforms"""
        self._all_markets = {}
        
        tasks = {}
        if self.polymarket._connected:
            tasks["polymarket"] = self.polymarket.get_markets(limit=100)
        if self.kalshi._connected:
            tasks["kalshi"] = self.kalshi.get_markets(limit=200)
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        for name, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Scan error for {name}: {result}")
                self._all_markets[name] = []
            else:
                self._all_markets[name] = result
        
        self._scan_count += 1
        return self._all_markets
    
    async def scan_5min_crypto(self) -> List[MarketData]:
        """Fetch specifically 5min crypto markets from Polymarket"""
        if self.polymarket._connected:
            return await self.polymarket.get_crypto_5min_markets()
        return []
    
    def find_cross_platform_matches(self) -> List[Dict]:
        """
        Find markets that exist on multiple platforms (for arbitrage).
        Matches by asset + close_time proximity.
        """
        matches = []
        pm_markets = self._all_markets.get("polymarket", [])
        ks_markets = self._all_markets.get("kalshi", [])
        
        for pm in pm_markets:
            if not pm.asset:
                continue
            for ks in ks_markets:
                if not ks.asset:
                    continue
                if pm.asset == ks.asset and pm.market_type == ks.market_type:
                    # Calculate potential edge
                    edge_yes_pm_no_ks = 1 - pm.yes_price - ks.no_price
                    edge_yes_ks_no_pm = 1 - ks.yes_price - pm.no_price
                    
                    best_edge = max(edge_yes_pm_no_ks, edge_yes_ks_no_pm)
                    
                    if best_edge > 0:
                        matches.append({
                            "asset": pm.asset,
                            "pm_market": pm,
                            "ks_market": ks,
                            "edge": best_edge,
                            "direction": "yes_pm_no_ks" if edge_yes_pm_no_ks > edge_yes_ks_no_pm else "yes_ks_no_pm",
                        })
        
        return sorted(matches, key=lambda x: x["edge"], reverse=True)
    
    def get_platform_status(self) -> List[Dict]:
        """Get status of all platforms"""
        return [conn.status for conn in self.connectors.values()]
    
    async def close_all(self):
        for conn in self.connectors.values():
            await conn.close()


# ============================================================
# TEST / DEMO
# ============================================================

async def main():
    """Test connectors with real API calls"""
    print("\n" + "="*60)
    print("  API CONNECTORS TEST – Paper Trading Mode")
    print("="*60)
    
    scanner = MarketScanner()
    
    # Connect
    print("\n[1] Connecting to platforms...")
    status = await scanner.connect_all()
    for name, ok in status.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon} {name}")
    
    # Scan Polymarket
    if scanner.polymarket._connected:
        print("\n[2] Fetching Polymarket markets...")
        pm_markets = await scanner.polymarket.get_markets(limit=20)
        print(f"  Got {len(pm_markets)} markets")
        
        for m in pm_markets[:5]:
            print(f"  • {m.question[:60]}")
            print(f"    Yes: {m.yes_price:.2%} | No: {m.no_price:.2%} | "
                  f"Vol24h: ${m.volume_24h:,.0f} | Asset: {m.asset or '-'}")
        
        # Try 5min crypto markets
        print("\n[3] Fetching 5min crypto markets...")
        crypto_5min = await scanner.polymarket.get_crypto_5min_markets()
        print(f"  Found {len(crypto_5min)} 5-minute markets")
        for m in crypto_5min[:10]:
            print(f"  • {m.question[:70]}")
            print(f"    Yes(Up): {m.yes_price:.0%} | Asset: {m.asset}")
        
        # Order book for first market with token IDs
        markets_with_tokens = [m for m in pm_markets if m.token_ids.get("yes")]
        if markets_with_tokens:
            m = markets_with_tokens[0]
            print(f"\n[4] Order book for: {m.question[:50]}...")
            book = await scanner.polymarket.get_order_book(m.token_ids["yes"])
            if book:
                print(f"  Bids: {len(book.bids)} levels | "
                      f"Asks: {len(book.asks)} levels | "
                      f"Spread: {book.spread:.4f} | "
                      f"Mid: {book.mid_price:.4f}")
                if book.bids:
                    print(f"  Best bid: {book.best_bid:.4f} ({book.bid_depth:.0f} depth)")
                if book.asks:
                    print(f"  Best ask: {book.best_ask:.4f} ({book.ask_depth:.0f} depth)")
    
    # Scan Kalshi
    if scanner.kalshi._connected:
        print("\n[5] Fetching Kalshi markets...")
        ks_markets = await scanner.kalshi.get_markets(limit=20)
        print(f"  Got {len(ks_markets)} markets")
        for m in ks_markets[:5]:
            print(f"  • {m.market_id}: {m.question[:50]}")
            print(f"    Yes: {m.yes_price:.2%} | Vol24h: ${m.volume_24h:,.0f}")
    
    # Cross-platform matches
    print("\n[6] Scanning for cross-platform matches...")
    all_data = await scanner.scan_all()
    matches = scanner.find_cross_platform_matches()
    print(f"  Found {len(matches)} potential arbitrage matches")
    for match in matches[:5]:
        print(f"  • {match['asset']} | Edge: {match['edge']:.2%} | "
              f"Direction: {match['direction']}")
    
    # Platform status
    print("\n[7] Platform status:")
    for s in scanner.get_platform_status():
        print(f"  {s['platform']}: latency={s['latency_ms']:.0f}ms | "
              f"requests={s['requests']} | errors={s['errors']}")
    
    await scanner.close_all()
    print("\n" + "="*60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    asyncio.run(main())
