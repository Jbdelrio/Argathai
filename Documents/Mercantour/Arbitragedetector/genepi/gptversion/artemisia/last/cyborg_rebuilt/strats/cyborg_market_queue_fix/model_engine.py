"""
Model Engine – Coin5min & BetMiss
Core trading logic and signal generation

Dependencies:
    pip install arch pykalman numpy scipy pandas requests websockets
"""

import numpy as np
import pandas as pd
import time
import asyncio
import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from scipy.stats import norm

logger = logging.getLogger("arb-engine")

# ============================================================
# COIN5MIN: EGARCH + Kalman Hybrid Model
# ============================================================

@dataclass
class Signal:
    """Trading signal for Coin5min"""
    asset: str
    direction: str          # "up" or "down"
    confidence: float       # 0-1
    sigma_5min: float       # EGARCH conditional volatility
    kalman_trend: float     # Kalman filtered trend value
    current_price: float
    market_yes_price: float # Polymarket "Up" price
    edge: float             # confidence - market price
    timestamp: float = field(default_factory=time.time)
    
    @property
    def actionable(self) -> bool:
        """Is the signal worth acting on?"""
        return self.edge > 0.02  # 2% minimum edge over market


class Coin5minPredictor:
    """
    Hybrid EGARCH + Kalman model for 5-minute crypto direction prediction.
    
    Flow:
    1. EGARCH(1,1) estimates conditional volatility σ_t
    2. Kalman filter extracts latent trend μ_t using σ_t as observation noise
    3. Direction = sign(μ_t - p_current)
    4. Confidence = Φ(|μ_t - p_current| / σ_t)
    """
    
    def __init__(
        self,
        lookback: int = 500,
        kalman_q_ratio: float = 0.05,   # Q/R ratio
        min_confidence: float = 0.60,
        vol_filter_multiplier: float = 2.0,
    ):
        self.lookback = lookback
        self.kalman_q_ratio = kalman_q_ratio
        self.min_confidence = min_confidence
        self.vol_filter_multiplier = vol_filter_multiplier
        self._vol_history: Dict[str, list] = {}
    
    def fit_egarch(self, returns: np.ndarray) -> Tuple[float, object]:
        """
        Fit EGARCH(1,1) on 5-minute returns.
        Returns (conditional_volatility, fitted_model)
        
        Model: log(σ²_t) = ω + α·[|z_{t-1}| - E|z_{t-1}|] + γ·z_{t-1} + β·log(σ²_{t-1})
        
        Recommended initial params for BTC 5min:
            ω ≈ -0.02, α ≈ 0.15, β ≈ 0.85, γ ≈ -0.08
        """
        from arch import arch_model
        
        # Scale returns to percentage for numerical stability
        returns_pct = pd.Series(returns * 100)
        
        model = arch_model(
            returns_pct,
            vol='EGARCH',
            p=1,           # GARCH lag
            q=1,           # ARCH lag
            mean='AR',     # AR(1) mean model
            lags=1,
            dist='skewt',  # Skewed Student-t for fat tails
        )
        
        try:
            result = model.fit(
                disp='off',
                options={'maxiter': 200},
                show_warning=False
            )
            # Get last conditional volatility (convert back from %)
            cv = result.conditional_volatility
            sigma = (cv.iloc[-1] if hasattr(cv, 'iloc') else cv[-1]) / 100
            return sigma, result
        except Exception as e:
            logger.warning(f"EGARCH fit failed: {e}. Using empirical vol.")
            sigma = np.std(returns[-50:])
            return sigma, None
    
    def kalman_trend(
        self, 
        prices: np.ndarray, 
        obs_variance: float
    ) -> float:
        """
        Adaptive Kalman filter for trend extraction.
        
        State equation:   μ_t = μ_{t-1} + η_t,   η_t ~ N(0, Q)
        Observation eq:   p_t = μ_t + ε_t,        ε_t ~ N(0, R)
        
        R = obs_variance (from EGARCH)
        Q = R × kalman_q_ratio (lower = smoother)
        """
        from pykalman import KalmanFilter
        
        Q = obs_variance * self.kalman_q_ratio
        
        kf = KalmanFilter(
            transition_matrices=np.array([[1]]),
            observation_matrices=np.array([[1]]),
            initial_state_mean=np.array([prices[0]]),
            initial_state_covariance=np.array([[obs_variance]]),
            observation_covariance=np.array([[obs_variance]]),
            transition_covariance=np.array([[Q]]),
        )
        
        state_means, state_covariances = kf.filter(prices.reshape(-1, 1))
        return float(state_means[-1, 0])
    
    def generate_signal(
        self,
        asset: str,
        prices: np.ndarray,
        market_yes_price: float,  # Polymarket "Up" price
    ) -> Optional[Signal]:
        """
        Generate a trading signal for a given asset.
        
        Args:
            asset: Asset ticker (BTC, ETH, etc.)
            prices: Array of recent 5min close prices (min 100 points)
            market_yes_price: Current Polymarket "Up" price (0-1)
        
        Returns:
            Signal if conditions met, None otherwise
        """
        if len(prices) < 100:
            logger.warning(f"{asset}: insufficient data ({len(prices)} < 100)")
            return None
        
        # 1. Compute returns
        returns = np.diff(np.log(prices))
        
        # 2. EGARCH volatility
        sigma, _ = self.fit_egarch(returns[-self.lookback:])
        
        # 3. Volatility filter: skip if extreme regime
        if asset not in self._vol_history:
            self._vol_history[asset] = []
        self._vol_history[asset].append(sigma)
        if len(self._vol_history[asset]) > 200:
            self._vol_history[asset] = self._vol_history[asset][-200:]
        
        median_vol = np.median(self._vol_history[asset])
        if sigma > self.vol_filter_multiplier * median_vol:
            logger.info(f"{asset}: vol filter triggered "
                       f"(σ={sigma:.6f} > {self.vol_filter_multiplier}×median={median_vol:.6f})")
            return None
        
        # 4. Kalman trend
        mu = self.kalman_trend(prices[-100:], sigma**2)
        
        # 5. Direction and confidence
        direction = "up" if mu > prices[-1] else "down"
        z_score = abs(mu - prices[-1]) / sigma if sigma > 0 else 0
        confidence = float(norm.cdf(z_score))
        
        # 6. Edge calculation
        if direction == "up":
            edge = confidence - market_yes_price
        else:
            edge = confidence - (1 - market_yes_price)
        
        signal = Signal(
            asset=asset,
            direction=direction,
            confidence=round(confidence, 4),
            sigma_5min=round(sigma, 8),
            kalman_trend=round(mu, 4),
            current_price=float(prices[-1]),
            market_yes_price=market_yes_price,
            edge=round(edge, 4),
        )
        
        if confidence < self.min_confidence:
            logger.debug(f"{asset}: confidence {confidence:.2%} < threshold {self.min_confidence:.2%}")
            return None
        
        return signal


# ============================================================
# BETMISS: Cross-platform Mispricing Detector
# ============================================================

@dataclass
class Opportunity:
    """Detected arbitrage opportunity"""
    event_key: str
    buy_yes_on: str
    buy_no_on: str
    yes_price: float
    no_price: float
    edge: float
    max_size: float
    expected_profit: float
    total_latency_ms: float
    timestamp: float = field(default_factory=time.time)
    slippage_estimate: float = 0.0
    
    @property
    def net_edge(self) -> float:
        return self.edge - self.slippage_estimate
    
    @property
    def priority_score(self) -> float:
        """Score composite for ranking opportunities"""
        latency_penalty = 1 - min(self.total_latency_ms / 2000, 0.9)
        return self.net_edge * self.max_size * latency_penalty


class MispricingDetector:
    """
    Detects arbitrage opportunities between prediction market platforms.
    
    Two types of arbitrage:
    1. Cross-platform: Yes(A) + No(B) < 1 for the same event
    2. Intra-platform: Σ P_i < 1 for all outcomes of a multi-outcome event
    """
    
    def __init__(
        self,
        min_edge: float = 0.02,
        max_latency_ms: float = 500,
        max_position_pct: float = 0.20,  # Max 20% of visible depth
        fee_schedule: Optional[Dict[str, float]] = None,
    ):
        self.min_edge = min_edge
        self.max_latency_ms = max_latency_ms
        self.max_position_pct = max_position_pct
        self.fee_schedule = fee_schedule or {
            "polymarket": 0.00,   # 0% taker fee
            "kalshi": 0.00,       # 0% since 2026
        }
        self._slippage_history: Dict[str, list] = {}
    
    def estimate_slippage(self, platform: str, size: float) -> float:
        """Estimate slippage based on historical data"""
        history = self._slippage_history.get(platform, [])
        if not history:
            return 0.005  # 0.5% default
        return np.mean(history[-50:]) * (size / 50)  # Scale with size
    
    def scan_cross_platform(
        self,
        event_prices: Dict[str, Dict[str, Dict]],
    ) -> List[Opportunity]:
        """
        Scan for cross-platform arbitrage.
        
        Args:
            event_prices: {event_key: {platform: {
                "yes_price": float, "no_price": float,
                "yes_depth": float, "no_depth": float,
                "latency_ms": float
            }}}
        
        Returns:
            Sorted list of opportunities (best first)
        """
        opportunities = []
        
        for event_key, platform_data in event_prices.items():
            platforms = list(platform_data.keys())
            
            for i, pA in enumerate(platforms):
                for pB in platforms[i+1:]:
                    dA = platform_data[pA]
                    dB = platform_data[pB]
                    
                    # Try: buy Yes on A, buy No on B
                    self._check_pair(
                        event_key, pA, pB, dA, dB,
                        opportunities, "yes_A_no_B"
                    )
                    
                    # Try: buy Yes on B, buy No on A
                    self._check_pair(
                        event_key, pB, pA, dB, dA,
                        opportunities, "yes_B_no_A"
                    )
        
        # Sort by priority score
        opportunities.sort(key=lambda x: x.priority_score, reverse=True)
        return opportunities
    
    def _check_pair(
        self,
        event_key: str,
        platform_yes: str,
        platform_no: str,
        data_yes: Dict,
        data_no: Dict,
        opportunities: List,
        label: str,
    ):
        """Check a specific pair for arbitrage"""
        total_cost = data_yes["yes_price"] + data_no["no_price"]
        fees = self.fee_schedule.get(platform_yes, 0) + self.fee_schedule.get(platform_no, 0)
        edge = 1 - total_cost - fees
        
        if edge < self.min_edge:
            return
        
        total_latency = data_yes.get("latency_ms", 0) + data_no.get("latency_ms", 0)
        
        if total_latency > self.max_latency_ms:
            logger.debug(f"Opportunity rejected: latency {total_latency}ms > {self.max_latency_ms}ms")
            return
        
        max_size = min(
            data_yes.get("yes_depth", 100) * self.max_position_pct,
            data_no.get("no_depth", 100) * self.max_position_pct,
        )
        
        slippage = max(
            self.estimate_slippage(platform_yes, max_size),
            self.estimate_slippage(platform_no, max_size),
        )
        
        opp = Opportunity(
            event_key=event_key,
            buy_yes_on=platform_yes,
            buy_no_on=platform_no,
            yes_price=data_yes["yes_price"],
            no_price=data_no["no_price"],
            edge=round(edge, 4),
            max_size=round(max_size, 2),
            expected_profit=round(edge * max_size, 2),
            total_latency_ms=total_latency,
            slippage_estimate=slippage,
        )
        
        if opp.net_edge > 0:
            opportunities.append(opp)
    
    def scan_intra_platform(
        self,
        platform: str,
        event_markets: List[Dict],
    ) -> Optional[Opportunity]:
        """
        Scan for intra-platform arbitrage (Σ outcomes < 1).
        
        Args:
            platform: Platform name
            event_markets: List of {
                "market_id": str, "outcome": str, "yes_price": float,
                "depth": float
            }
        """
        total = sum(m["yes_price"] for m in event_markets)
        fees = self.fee_schedule.get(platform, 0) * len(event_markets)
        
        if total < (1 - fees - self.min_edge):
            edge = 1 - total - fees
            min_depth = min(m.get("depth", 100) for m in event_markets)
            
            return Opportunity(
                event_key=f"intra_{platform}_{event_markets[0].get('event_id', 'unknown')}",
                buy_yes_on=platform,
                buy_no_on=platform,
                yes_price=total,
                no_price=0,
                edge=round(edge, 4),
                max_size=round(min_depth * self.max_position_pct, 2),
                expected_profit=round(edge * min_depth * self.max_position_pct, 2),
                total_latency_ms=0,
            )
        return None


# ============================================================
# RISK MANAGEMENT
# ============================================================

class RiskManager:
    """Centralized risk management for both strategies"""
    
    def __init__(
        self,
        total_capital: float = 10000,
        max_daily_loss_pct: float = 0.05,
        max_platform_exposure_pct: float = 0.40,
        max_asset_exposure_pct: float = 0.30,
        cooldown_after_n_fails: int = 3,
        cooldown_duration_s: int = 1800,  # 30 min
    ):
        self.total_capital = total_capital
        self.max_daily_loss = total_capital * max_daily_loss_pct
        self.max_platform_exposure = total_capital * max_platform_exposure_pct
        self.max_asset_exposure = total_capital * max_asset_exposure_pct
        self.cooldown_after_n_fails = cooldown_after_n_fails
        self.cooldown_duration_s = cooldown_duration_s
        
        self._daily_pnl = 0.0
        self._platform_exposure: Dict[str, float] = {}
        self._asset_exposure: Dict[str, float] = {}
        self._consecutive_fails: Dict[str, int] = {}
        self._cooldowns: Dict[str, float] = {}
    
    def check_trade(
        self,
        platform: str,
        asset: str,
        size: float,
        strategy: str,
    ) -> Tuple[bool, str]:
        """
        Pre-trade risk check.
        Returns (allowed, reason)
        """
        # Kill switch check
        if self._daily_pnl < -self.max_daily_loss:
            return False, f"Daily loss limit reached (${self._daily_pnl:.2f})"
        
        # Platform exposure
        current_platform_exp = self._platform_exposure.get(platform, 0)
        if current_platform_exp + size > self.max_platform_exposure:
            return False, f"Platform exposure limit: {platform} " \
                         f"(${current_platform_exp:.0f} + ${size:.0f} > ${self.max_platform_exposure:.0f})"
        
        # Asset exposure
        current_asset_exp = self._asset_exposure.get(asset, 0)
        if current_asset_exp + size > self.max_asset_exposure:
            return False, f"Asset exposure limit: {asset} " \
                         f"(${current_asset_exp:.0f} + ${size:.0f} > ${self.max_asset_exposure:.0f})"
        
        # Cooldown check
        key = f"{strategy}_{asset}"
        if key in self._cooldowns and time.time() < self._cooldowns[key]:
            remaining = self._cooldowns[key] - time.time()
            return False, f"Cooldown active for {asset} ({remaining:.0f}s remaining)"
        
        return True, "OK"
    
    def record_result(self, strategy: str, asset: str, pnl: float):
        """Record trade result and update risk state"""
        self._daily_pnl += pnl
        
        key = f"{strategy}_{asset}"
        if pnl < 0:
            self._consecutive_fails[key] = self._consecutive_fails.get(key, 0) + 1
            if self._consecutive_fails[key] >= self.cooldown_after_n_fails:
                self._cooldowns[key] = time.time() + self.cooldown_duration_s
                logger.warning(f"Cooldown activated for {key} "
                             f"({self.cooldown_after_n_fails} consecutive fails)")
                self._consecutive_fails[key] = 0
        else:
            self._consecutive_fails[key] = 0
    
    def update_exposure(self, platform: str, asset: str, delta: float):
        """Update exposure tracking"""
        self._platform_exposure[platform] = self._platform_exposure.get(platform, 0) + delta
        self._asset_exposure[asset] = self._asset_exposure.get(asset, 0) + delta
    
    def get_status(self) -> Dict:
        """Get current risk status"""
        return {
            "daily_pnl": self._daily_pnl,
            "daily_loss_limit": self.max_daily_loss,
            "pnl_remaining": self.max_daily_loss + self._daily_pnl,
            "platform_exposure": dict(self._platform_exposure),
            "asset_exposure": dict(self._asset_exposure),
            "active_cooldowns": {
                k: datetime.fromtimestamp(v).isoformat()
                for k, v in self._cooldowns.items()
                if v > time.time()
            },
        }


# ============================================================
# PLATFORM CONNECTORS (Interface)
# ============================================================

class PlatformConnector:
    """Base class for platform API connectors"""
    
    def __init__(self, name: str, api_key: str = "", api_secret: str = ""):
        self.name = name
        self.api_key = api_key
        self.api_secret = api_secret
        self.connected = False
        self.last_latency_ms = 0
    
    async def connect(self):
        raise NotImplementedError
    
    async def disconnect(self):
        raise NotImplementedError
    
    async def get_market_price(self, market_id: str) -> Dict:
        raise NotImplementedError
    
    async def place_order(self, market_id: str, side: str, 
                          price: float, size: float) -> Dict:
        raise NotImplementedError
    
    async def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError
    
    async def cancel_all_orders(self) -> int:
        raise NotImplementedError
    
    async def ping(self) -> float:
        """Health check, returns latency in ms"""
        raise NotImplementedError


class PolymarketConnector(PlatformConnector):
    """
    Polymarket API connector.
    
    Architecture:
    - Gamma API: Market metadata and discovery
    - CLOB API: Central Limit Order Book for trading
    - Data API: User positions and trade history
    - WebSocket: Real-time price updates
    
    Key endpoints:
    - GET /markets?closed=false : List active markets
    - GET /book?token_id=X : Order book
    - POST /order : Place order
    - WS wss://ws-subscriptions-clob.polymarket.com/ws/market : Price stream
    """
    
    def __init__(self, api_key: str = "", api_secret: str = ""):
        super().__init__("polymarket", api_key, api_secret)
        self.base_url = "https://clob.polymarket.com"
        self.gamma_url = "https://gamma-api.polymarket.com"
    
    async def connect(self):
        # In production: initialize py_clob_client or polymarket_apis
        # from polymarket_apis import PolymarketClient
        # self.client = PolymarketClient(...)
        self.connected = True
        logger.info("Polymarket connected")
    
    async def get_5min_markets(self) -> List[Dict]:
        """
        Fetch active 5-minute crypto Up/Down markets.
        These are the markets visible in the screenshot:
        - "Ethereum Up or Down - April 2, 7:55AM-8:00AM ET"
        - "Solana Up or Down - April 2, 7:55AM-8:00AM ET"
        etc.
        """
        # In production:
        # response = await self.client.get_markets(
        #     tag="crypto-prices", 
        #     closed=False,
        #     limit=50
        # )
        # return [m for m in response if "Up or Down" in m["question"]]
        pass
    
    async def ping(self) -> float:
        t0 = time.time()
        # await self.client.get_server_time()
        latency = (time.time() - t0) * 1000
        self.last_latency_ms = latency
        return latency


class KalshiConnector(PlatformConnector):
    """
    Kalshi API connector.
    
    Architecture:
    - REST API v2: Markets, orders, account
    - WebSocket: Real-time orderbook deltas, trades, fills
    - FIX 4.4: Low-latency institutional trading
    
    Key endpoints:
    - GET /markets : List markets (paginated, cursor-based)
    - GET /markets/{ticker} : Single market details
    - GET /markets/{ticker}/orderbook : Order book
    - POST /portfolio/orders : Place order
    
    Auth: RSA-PSS signed requests (private key PEM)
    Prices in cents (integer): 65 = $0.65
    """
    
    def __init__(self, api_key_id: str = "", private_key_path: str = ""):
        super().__init__("kalshi", api_key_id)
        self.base_url = "https://api.elections.kalshi.com/trade-api/v2"
        self.demo_url = "https://demo-api.kalshi.co/trade-api/v2"
    
    async def connect(self):
        # from kalshi_python import Configuration, KalshiClient
        # config = Configuration(host=self.base_url)
        # config.api_key_id = self.api_key
        # config.private_key_pem = open(self.private_key_path).read()
        # self.client = KalshiClient(config)
        self.connected = True
        logger.info("Kalshi connected")


# ============================================================
# MAIN TRADING LOOP
# ============================================================

class TradingEngine:
    """
    Main engine orchestrating both strategies.
    
    Architecture:
    [Market Data] → [Signal Generation] → [Risk Check] → [Execution] → [DB Log]
                         ↑                      ↑
                    [Models]              [RiskManager]
    """
    
    def __init__(self, db_path: str = "arb_trading.db"):
        self.db_path = db_path
        self.risk_manager = RiskManager()
        self.coin5min = Coin5minPredictor()
        self.betmiss = MispricingDetector()
        self.connectors: Dict[str, PlatformConnector] = {}
        self._running = False
    
    def add_platform(self, connector: PlatformConnector):
        self.connectors[connector.name] = connector
    
    async def run_coin5min_loop(self, assets: List[str], interval_s: int = 300):
        """Main loop for Coin5min strategy"""
        while self._running:
            for asset in assets:
                try:
                    # 1. Get price data
                    # prices = await self.get_price_history(asset, n=500)
                    # market_price = await self.get_polymarket_5min_price(asset)
                    
                    # 2. Generate signal
                    # signal = self.coin5min.generate_signal(asset, prices, market_price)
                    
                    # 3. Check risk
                    # if signal and signal.actionable:
                    #     allowed, reason = self.risk_manager.check_trade(
                    #         "polymarket", asset, trade_size, "coin5min"
                    #     )
                    #     if allowed:
                    #         await self.execute_coin5min(signal, trade_size)
                    pass
                    
                except Exception as e:
                    logger.error(f"Coin5min error for {asset}: {e}")
            
            await asyncio.sleep(interval_s)
    
    async def run_betmiss_loop(self, scan_interval_s: int = 10):
        """Main loop for BetMiss strategy"""
        while self._running:
            try:
                # 1. Collect prices from all platforms
                # event_prices = await self.collect_cross_platform_prices()
                
                # 2. Detect opportunities
                # opportunities = self.betmiss.scan_cross_platform(event_prices)
                
                # 3. Execute best opportunities
                # for opp in opportunities[:3]:  # Top 3
                #     allowed, reason = self.risk_manager.check_trade(
                #         opp.buy_yes_on, "multi", opp.max_size, "betmiss"
                #     )
                #     if allowed:
                #         await self.execute_betmiss(opp)
                pass
                
            except Exception as e:
                logger.error(f"BetMiss error: {e}")
            
            await asyncio.sleep(scan_interval_s)
    
    async def start(self):
        """Start all trading loops"""
        self._running = True
        
        # Connect to platforms
        for connector in self.connectors.values():
            await connector.connect()
        
        # Run strategies concurrently
        await asyncio.gather(
            self.run_coin5min_loop(["BTC", "ETH", "SOL", "XRP", "DOGE"]),
            self.run_betmiss_loop(),
        )
    
    def stop(self):
        """Stop all trading"""
        self._running = False
        logger.info("Trading engine stopped")


# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    # Demo: generate signals with mock data
    predictor = Coin5minPredictor(lookback=200, min_confidence=0.55)
    
    # Simulate BTC 5min prices
    np.random.seed(42)
    n = 300
    returns = np.random.normal(0.0001, 0.002, n)  # ~0.2% vol per 5min
    prices = 1800 * np.exp(np.cumsum(returns))  # ETH-like
    
    # Generate signal
    signal = predictor.generate_signal(
        asset="ETH",
        prices=prices,
        market_yes_price=0.20  # From the Polymarket screenshot
    )
    
    if signal:
        print(f"\n{'='*50}")
        print(f"  Signal: {signal.asset} → {signal.direction.upper()}")
        print(f"  Confiance: {signal.confidence:.2%}")
        print(f"  Volatilité 5min (σ): {signal.sigma_5min:.6f}")
        print(f"  Tendance Kalman: ${signal.kalman_trend:.2f}")
        print(f"  Prix actuel: ${signal.current_price:.2f}")
        print(f"  Prix marché (Up): {signal.market_yes_price:.0%}")
        print(f"  Edge: {signal.edge:+.2%}")
        print(f"  Actionnable: {'✅' if signal.actionable else '❌'}")
        print(f"{'='*50}\n")
    else:
        print("Pas de signal (confiance insuffisante ou filtre de volatilité)")
    
    # Demo: mispricing detection
    detector = MispricingDetector(min_edge=0.01)
    
    event_prices = {
        "ETH_Up_Down_5min": {
            "polymarket": {
                "yes_price": 0.20,  # 20% Up (from screenshot)
                "no_price": 0.78,   # Hypothetical
                "yes_depth": 500,
                "no_depth": 500,
                "latency_ms": 50,
            },
            "kalshi": {
                "yes_price": 0.23,  # Hypothetical higher price
                "no_price": 0.74,
                "yes_depth": 200,
                "no_depth": 200,
                "latency_ms": 120,
            }
        }
    }
    
    opportunities = detector.scan_cross_platform(event_prices)
    
    print(f"Opportunités détectées: {len(opportunities)}")
    for opp in opportunities:
        print(f"  {opp.event_key}: Buy Yes on {opp.buy_yes_on} ({opp.yes_price:.2f}) "
              f"+ Buy No on {opp.buy_no_on} ({opp.no_price:.2f}) "
              f"→ Edge: {opp.edge:.2%} | Profit: ${opp.expected_profit:.2f}")
