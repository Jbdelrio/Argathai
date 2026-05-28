"""
paper_trading.py – Paper Trading & Backtesting Engine
Simulates order execution without real money.
Tracks virtual portfolio, PnL, and all metrics.
"""

import sqlite3
import time
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple
from enum import Enum

logger = logging.getLogger("paper-trading")

DB_PATH = "arb_paper.db"

# ============================================================
# ENUMS & DATA MODELS
# ============================================================

class TradeStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    RESOLVED_WIN = "resolved_win"
    RESOLVED_LOSS = "resolved_loss"
    CANCELLED = "cancelled"
    FAILED = "failed"

class Strategy(Enum):
    BETMISS = "betmiss"
    COIN5MIN = "coin5min"

@dataclass
class PaperOrder:
    """Virtual order in paper trading"""
    id: str
    strategy: str
    platform: str
    market_id: str
    question: str
    asset: Optional[str]
    side: str              # "yes" or "no"
    price: float           # fill price
    size: float            # $ amount
    direction: Optional[str] = None  # "up"/"down" for coin5min
    confidence: Optional[float] = None
    edge: Optional[float] = None
    status: str = "pending"
    pnl: float = 0.0
    created_at: float = field(default_factory=time.time)
    filled_at: Optional[float] = None
    resolved_at: Optional[float] = None
    resolve_outcome: Optional[str] = None  # "yes"/"no"
    latency_ms: float = 0.0
    slippage: float = 0.0
    notes: str = ""


@dataclass
class Portfolio:
    """Virtual portfolio state"""
    initial_capital: float = 10000.0
    cash: float = 10000.0
    positions_value: float = 0.0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    _peak_value: float = 10000.0
    _daily_returns: List[float] = field(default_factory=list)
    
    @property
    def total_value(self) -> float:
        return self.cash + self.positions_value
    
    @property
    def win_rate(self) -> float:
        total = self.n_wins + self.n_losses
        return self.n_wins / total if total > 0 else 0
    
    @property
    def roi(self) -> float:
        return (self.total_value - self.initial_capital) / self.initial_capital


# ============================================================
# DATABASE
# ============================================================

def init_paper_db(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute("""CREATE TABLE IF NOT EXISTS paper_trades (
        id TEXT PRIMARY KEY,
        strategy TEXT NOT NULL,
        platform TEXT NOT NULL,
        market_id TEXT NOT NULL,
        question TEXT,
        asset TEXT,
        side TEXT NOT NULL,
        price REAL NOT NULL,
        size REAL NOT NULL,
        direction TEXT,
        confidence REAL,
        edge REAL,
        status TEXT DEFAULT 'pending',
        pnl REAL DEFAULT 0,
        created_at REAL NOT NULL,
        filled_at REAL,
        resolved_at REAL,
        resolve_outcome TEXT,
        latency_ms REAL DEFAULT 0,
        slippage REAL DEFAULT 0,
        notes TEXT DEFAULT ''
    )""")
    
    c.execute("""CREATE TABLE IF NOT EXISTS portfolio_snapshots (
        timestamp REAL PRIMARY KEY,
        cash REAL,
        positions_value REAL,
        total_value REAL,
        total_pnl REAL,
        daily_pnl REAL,
        n_trades INTEGER,
        win_rate REAL,
        max_drawdown REAL
    )""")
    
    c.execute("""CREATE TABLE IF NOT EXISTS market_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL NOT NULL,
        platform TEXT NOT NULL,
        market_id TEXT NOT NULL,
        question TEXT,
        asset TEXT,
        yes_price REAL,
        no_price REAL,
        volume_24h REAL,
        liquidity REAL
    )""")
    
    c.execute("""CREATE TABLE IF NOT EXISTS backtest_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        strategy TEXT NOT NULL,
        start_time TEXT,
        end_time TEXT,
        params TEXT,
        initial_capital REAL,
        final_value REAL,
        total_pnl REAL,
        n_trades INTEGER,
        win_rate REAL,
        sharpe REAL,
        max_drawdown REAL,
        created_at REAL NOT NULL
    )""")
    
    c.execute("CREATE INDEX IF NOT EXISTS idx_pt_strategy ON paper_trades(strategy)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_pt_status ON paper_trades(status)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_mh_asset ON market_history(asset, timestamp)")
    
    conn.commit()


# ============================================================
# PAPER TRADING ENGINE
# ============================================================

class PaperTradingEngine:
    """
    Simulates trade execution and portfolio management.
    No real money involved – all virtual.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        slippage_model: str = "fixed",  # "fixed", "proportional", "orderbook"
        slippage_bps: float = 10,       # 10 basis points = 0.1%
        db_path: str = DB_PATH,
    ):
        self.db_path = db_path
        self._in_memory = (db_path == ":memory:")
        if self._in_memory:
            self._mem_conn = sqlite3.connect(":memory:")
            self._init_tables(self._mem_conn)
        else:
            init_paper_db(db_path)
            self._mem_conn = None
        
        self.portfolio = Portfolio(initial_capital=initial_capital, cash=initial_capital)
        self.slippage_model = slippage_model
        self.slippage_bps = slippage_bps
        
        self.open_positions: Dict[str, PaperOrder] = {}
        self._trade_counter = 0
    
    def _init_tables(self, conn):
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS paper_trades (
            id TEXT PRIMARY KEY, strategy TEXT, platform TEXT, market_id TEXT,
            question TEXT, asset TEXT, side TEXT, price REAL, size REAL,
            direction TEXT, confidence REAL, edge REAL, status TEXT,
            pnl REAL, created_at REAL, filled_at REAL, resolved_at REAL,
            resolve_outcome TEXT, latency_ms REAL, slippage REAL, notes TEXT
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            timestamp REAL PRIMARY KEY, cash REAL, positions_value REAL,
            total_value REAL, total_pnl REAL, daily_pnl REAL,
            n_trades INTEGER, win_rate REAL, max_drawdown REAL
        )""")
        conn.commit()
    
    def _get_conn(self):
        if self._in_memory:
            return self._mem_conn
        return sqlite3.connect(self.db_path)
    
    def _gen_id(self) -> str:
        self._trade_counter += 1
        return f"PT-{int(time.time())}-{self._trade_counter:04d}"
    
    def _estimate_slippage(self, price: float, size: float) -> float:
        """Estimate slippage based on model"""
        if self.slippage_model == "fixed":
            return self.slippage_bps / 10000
        elif self.slippage_model == "proportional":
            # Larger orders = more slippage
            return (self.slippage_bps / 10000) * (1 + size / 1000)
        return 0
    
    def place_order(
        self,
        strategy: str,
        platform: str,
        market_id: str,
        question: str,
        side: str,          # "yes" or "no"
        market_price: float,
        size: float,
        asset: Optional[str] = None,
        direction: Optional[str] = None,
        confidence: Optional[float] = None,
        edge: Optional[float] = None,
        latency_ms: float = 0,
    ) -> Optional[PaperOrder]:
        """
        Place a paper trade. Immediately fills at market_price + slippage.
        """
        # Check cash
        if size > self.portfolio.cash:
            logger.warning(f"Insufficient cash: ${size:.2f} > ${self.portfolio.cash:.2f}")
            return None
        
        # Apply slippage
        slippage = self._estimate_slippage(market_price, size)
        fill_price = market_price + slippage  # Buying = worse price
        
        # Ensure valid price
        fill_price = max(0.01, min(0.99, fill_price))
        
        # Create order
        order = PaperOrder(
            id=self._gen_id(),
            strategy=strategy,
            platform=platform,
            market_id=market_id,
            question=question,
            asset=asset,
            side=side,
            price=fill_price,
            size=size,
            direction=direction,
            confidence=confidence,
            edge=edge,
            status="filled",
            created_at=time.time(),
            filled_at=time.time(),
            latency_ms=latency_ms,
            slippage=slippage,
        )
        
        # Update portfolio
        self.portfolio.cash -= size
        self.portfolio.positions_value += size
        self.portfolio.n_trades += 1
        
        # Track position
        self.open_positions[order.id] = order
        
        # Save to DB
        self._save_trade(order)
        
        logger.info(f"📝 PAPER FILL: {order.id} | {strategy} | "
                    f"{side.upper()} @ {fill_price:.4f} | ${size:.2f} | "
                    f"Asset: {asset} | Slip: {slippage:.4f}")
        
        return order
    
    def resolve_trade(
        self,
        trade_id: str,
        outcome: str,  # "yes" or "no"
    ) -> Optional[PaperOrder]:
        """
        Resolve a trade based on market outcome.
        Winning side pays $1 per share, losing pays $0.
        """
        if trade_id not in self.open_positions:
            logger.warning(f"Trade {trade_id} not found in open positions")
            return None
        
        order = self.open_positions.pop(trade_id)
        
        # Calculate PnL
        # If we bought "yes" and outcome is "yes" → we get $1 per share
        # Number of shares = size / price
        n_shares = order.size / order.price
        
        if order.side == outcome:
            # WIN: payout = n_shares * $1, cost was size
            payout = n_shares * 1.0
            order.pnl = payout - order.size
            order.status = "resolved_win"
            self.portfolio.n_wins += 1
        else:
            # LOSS: payout = $0
            order.pnl = -order.size
            order.status = "resolved_loss"
            self.portfolio.n_losses += 1
        
        order.resolved_at = time.time()
        order.resolve_outcome = outcome
        
        # Update portfolio
        self.portfolio.positions_value -= order.size
        self.portfolio.cash += max(0, order.size + order.pnl)
        self.portfolio.total_pnl += order.pnl
        self.portfolio.daily_pnl += order.pnl
        
        # Track drawdown
        current_value = self.portfolio.total_value
        if current_value > self.portfolio._peak_value:
            self.portfolio._peak_value = current_value
        dd = (self.portfolio._peak_value - current_value) / self.portfolio._peak_value
        self.portfolio.max_drawdown = max(self.portfolio.max_drawdown, dd)
        
        # Save to DB
        self._save_trade(order)
        
        icon = "✅" if order.pnl > 0 else "❌"
        logger.info(f"{icon} RESOLVED: {order.id} | {order.strategy} | "
                    f"PnL: ${order.pnl:+.2f} | Outcome: {outcome}")
        
        return order
    
    def snapshot_portfolio(self):
        """Save portfolio snapshot to DB"""
        conn = self._get_conn()
        conn.execute("""INSERT OR REPLACE INTO portfolio_snapshots
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
            time.time(),
            self.portfolio.cash,
            self.portfolio.positions_value,
            self.portfolio.total_value,
            self.portfolio.total_pnl,
            self.portfolio.daily_pnl,
            self.portfolio.n_trades,
            self.portfolio.win_rate,
            self.portfolio.max_drawdown,
        ))
        conn.commit()
    
    def get_trade_history(self, strategy: Optional[str] = None, limit: int = 100) -> pd.DataFrame:
        """Get trade history from DB"""
        conn = self._get_conn()
        query = "SELECT * FROM paper_trades"
        params = []
        if strategy:
            query += " WHERE strategy = ?"
            params.append(strategy)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        df = pd.read_sql(query, conn, params=params)
        if not self._in_memory: conn.close()
        return df
    
    def get_portfolio_history(self) -> pd.DataFrame:
        conn = self._get_conn()
        df = pd.read_sql("SELECT * FROM portfolio_snapshots ORDER BY timestamp", conn)
        if not self._in_memory: conn.close()
        return df
    
    def get_metrics(self) -> Dict:
        """Get current performance metrics"""
        return {
            "total_value": self.portfolio.total_value,
            "cash": self.portfolio.cash,
            "positions_value": self.portfolio.positions_value,
            "total_pnl": self.portfolio.total_pnl,
            "daily_pnl": self.portfolio.daily_pnl,
            "roi": self.portfolio.roi,
            "n_trades": self.portfolio.n_trades,
            "win_rate": self.portfolio.win_rate,
            "max_drawdown": self.portfolio.max_drawdown,
            "open_positions": len(self.open_positions),
        }
    
    def _save_trade(self, order: PaperOrder):
        conn = self._get_conn()
        conn.execute("""INSERT OR REPLACE INTO paper_trades
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            order.id, order.strategy, order.platform, order.market_id,
            order.question, order.asset, order.side, order.price, order.size,
            order.direction, order.confidence, order.edge, order.status,
            order.pnl, order.created_at, order.filled_at, order.resolved_at,
            order.resolve_outcome, order.latency_ms, order.slippage, order.notes,
        ))
        conn.commit()
    
    def save_market_snapshot(self, markets):
        """Save market data for backtesting"""
        conn = self._get_conn()
        for m in markets:
            conn.execute("""INSERT INTO market_history
                (timestamp, platform, market_id, question, asset,
                 yes_price, no_price, volume_24h, liquidity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                time.time(), m.platform, m.market_id, m.question,
                m.asset, m.yes_price, m.no_price, m.volume_24h, m.liquidity,
            ))
        conn.commit()


# ============================================================
# BACKTESTING ENGINE
# ============================================================

class BacktestEngine:
    """
    Backtests strategies on historical market data.
    Uses stored market_history or generates synthetic data.
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.results: List[Dict] = []
    
    def run_coin5min_backtest(
        self,
        prices: np.ndarray,
        market_prices: np.ndarray,
        asset: str = "ETH",
        confidence_threshold: float = 0.60,
        trade_size_pct: float = 0.01,
        kalman_q_ratio: float = 0.05,
        lookback: int = 200,
    ) -> Dict:
        """
        Backtest Coin5min strategy on price data.
        
        Args:
            prices: array of 5min close prices
            market_prices: array of Polymarket "Up" prices at each interval
            asset: asset ticker
            confidence_threshold: min confidence to trade
            trade_size_pct: % of capital per trade
            kalman_q_ratio: Kalman filter Q/R ratio
            lookback: EGARCH lookback window
        
        Returns:
            Backtest results dict
        """
        from model_engine import Coin5minPredictor
        
        predictor = Coin5minPredictor(
            lookback=lookback,
            kalman_q_ratio=kalman_q_ratio,
            min_confidence=confidence_threshold,
        )
        
        engine = PaperTradingEngine(
            initial_capital=self.initial_capital,
            db_path=":memory:",  # In-memory for backtest
        )
        
        trades = []
        equity_curve = [self.initial_capital]
        n_signals = 0
        
        # Need at least lookback + 100 points
        start_idx = max(lookback, 100)
        
        for i in range(start_idx, len(prices) - 1):
            price_window = prices[:i+1]
            mkt_yes = market_prices[i] if i < len(market_prices) else 0.5
            
            # Generate signal
            signal = predictor.generate_signal(asset, price_window, mkt_yes)
            
            if signal and signal.actionable:
                n_signals += 1
                size = engine.portfolio.total_value * trade_size_pct
                
                # Place paper trade
                side = "yes" if signal.direction == "up" else "no"
                order = engine.place_order(
                    strategy="coin5min",
                    platform="polymarket",
                    market_id=f"bt_{asset}_{i}",
                    question=f"{asset} Up or Down interval {i}",
                    side=side,
                    market_price=mkt_yes if side == "yes" else (1 - mkt_yes),
                    size=size,
                    asset=asset,
                    direction=signal.direction,
                    confidence=signal.confidence,
                    edge=signal.edge,
                )
                
                if order:
                    # Resolve: did price go up or down?
                    actual_up = prices[i+1] > prices[i]
                    outcome = "yes" if actual_up else "no"
                    resolved = engine.resolve_trade(order.id, outcome)
                    
                    if resolved:
                        trades.append({
                            "idx": i,
                            "direction": signal.direction,
                            "confidence": signal.confidence,
                            "actual_up": actual_up,
                            "correct": (signal.direction == "up") == actual_up,
                            "pnl": resolved.pnl,
                            "size": size,
                        })
            
            equity_curve.append(engine.portfolio.total_value)
        
        # Compute metrics
        metrics = engine.get_metrics()
        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]
        
        sharpe = 0
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 12)  # Annualized (12 intervals/hour)
        
        result = {
            "strategy": "coin5min",
            "asset": asset,
            "params": {
                "confidence_threshold": confidence_threshold,
                "trade_size_pct": trade_size_pct,
                "kalman_q_ratio": kalman_q_ratio,
                "lookback": lookback,
            },
            "initial_capital": self.initial_capital,
            "final_value": metrics["total_value"],
            "total_pnl": metrics["total_pnl"],
            "roi": metrics["roi"],
            "n_signals": n_signals,
            "n_trades": metrics["n_trades"],
            "win_rate": metrics["win_rate"],
            "sharpe": sharpe,
            "max_drawdown": metrics["max_drawdown"],
            "avg_pnl_per_trade": metrics["total_pnl"] / max(metrics["n_trades"], 1),
            "equity_curve": equity_curve,
            "trades": trades,
        }
        
        self.results.append(result)
        return result
    
    def run_betmiss_backtest(
        self,
        pm_prices: List[Dict],  # [{timestamp, yes_price, no_price}, ...]
        ks_prices: List[Dict],
        min_edge: float = 0.02,
        trade_size_pct: float = 0.02,
    ) -> Dict:
        """
        Backtest BetMiss cross-platform arbitrage.
        
        Simulates: if Yes(PM) + No(KS) < 1, buy both.
        Resolution: one side always wins → guaranteed profit = edge × size
        """
        engine = PaperTradingEngine(
            initial_capital=self.initial_capital,
            db_path=":memory:",
        )
        
        trades = []
        equity_curve = [self.initial_capital]
        opportunities = 0
        
        n = min(len(pm_prices), len(ks_prices))
        
        for i in range(n):
            pm = pm_prices[i]
            ks = ks_prices[i]
            
            # Check cross-platform arbitrage
            total_cost = pm["yes_price"] + ks["no_price"]
            edge = 1 - total_cost
            
            if edge > min_edge:
                opportunities += 1
                size = engine.portfolio.total_value * trade_size_pct
                half_size = size / 2
                
                # Buy Yes on PM + Buy No on KS
                order_yes = engine.place_order(
                    strategy="betmiss", platform="polymarket",
                    market_id=f"bt_pm_{i}", question=f"Arb interval {i}",
                    side="yes", market_price=pm["yes_price"],
                    size=half_size, edge=edge,
                )
                order_no = engine.place_order(
                    strategy="betmiss", platform="kalshi",
                    market_id=f"bt_ks_{i}", question=f"Arb interval {i}",
                    side="no", market_price=ks["no_price"],
                    size=half_size, edge=edge,
                )
                
                if order_yes and order_no:
                    # Resolution: one wins, one loses. 
                    # Guaranteed profit ≈ edge × size (minus slippage)
                    outcome = np.random.choice(["yes", "no"])
                    engine.resolve_trade(order_yes.id, outcome)
                    engine.resolve_trade(order_no.id, outcome)
                    
                    trades.append({
                        "idx": i,
                        "edge": edge,
                        "pm_yes": pm["yes_price"],
                        "ks_no": ks["no_price"],
                        "pnl_yes": order_yes.pnl if order_yes else 0,
                        "pnl_no": order_no.pnl if order_no else 0,
                    })
            
            equity_curve.append(engine.portfolio.total_value)
        
        metrics = engine.get_metrics()
        
        result = {
            "strategy": "betmiss",
            "params": {"min_edge": min_edge, "trade_size_pct": trade_size_pct},
            "initial_capital": self.initial_capital,
            "final_value": metrics["total_value"],
            "total_pnl": metrics["total_pnl"],
            "roi": metrics["roi"],
            "opportunities": opportunities,
            "n_trades": metrics["n_trades"],
            "win_rate": metrics["win_rate"],
            "max_drawdown": metrics["max_drawdown"],
            "equity_curve": equity_curve,
            "trades": trades,
        }
        
        self.results.append(result)
        return result
    
    def generate_synthetic_data(
        self,
        n_intervals: int = 1000,
        base_price: float = 1800,
        volatility: float = 0.002,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic 5min crypto prices + market prices.
        Returns (prices, market_yes_prices)
        """
        np.random.seed(seed)
        
        # GBM with mean-reverting volatility
        returns = np.random.normal(0.0001, volatility, n_intervals)
        # Add volatility clustering (GARCH-like)
        vol = np.zeros(n_intervals)
        vol[0] = volatility
        for i in range(1, n_intervals):
            vol[i] = 0.85 * vol[i-1] + 0.15 * abs(returns[i-1]) + 0.0001
            returns[i] = np.random.normal(0, vol[i])
        
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Synthetic market prices: noisy reflection of true probability
        true_probs = np.array([
            0.5 + 0.3 * np.tanh(5 * (prices[i+1] - prices[i]) / prices[i])
            if i < len(prices) - 1 else 0.5
            for i in range(len(prices))
        ])
        noise = np.random.normal(0, 0.08, len(prices))
        market_yes = np.clip(true_probs + noise, 0.05, 0.95)
        
        return prices, market_yes
    
    def generate_synthetic_cross_platform(
        self,
        n_intervals: int = 500,
        base_edge_rate: float = 0.05,  # 5% of intervals have arbitrage
        seed: int = 42,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Generate synthetic cross-platform price data"""
        np.random.seed(seed)
        
        pm_prices = []
        ks_prices = []
        
        for i in range(n_intervals):
            base_yes = np.random.uniform(0.2, 0.8)
            
            # PM prices: efficient
            pm_yes = base_yes + np.random.normal(0, 0.02)
            pm_no = 1 - pm_yes + np.random.normal(0, 0.01)
            
            # KS prices: occasionally mispriced
            ks_yes = base_yes + np.random.normal(0, 0.03)
            ks_no = 1 - ks_yes + np.random.normal(0, 0.02)
            
            # Occasionally inject real arbitrage
            if np.random.random() < base_edge_rate:
                ks_no -= np.random.uniform(0.02, 0.06)
            
            pm_prices.append({"yes_price": np.clip(pm_yes, 0.05, 0.95),
                             "no_price": np.clip(pm_no, 0.05, 0.95)})
            ks_prices.append({"yes_price": np.clip(ks_yes, 0.05, 0.95),
                             "no_price": np.clip(ks_no, 0.05, 0.95)})
        
        return pm_prices, ks_prices


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    print("\n" + "="*60)
    print("  BACKTESTING ENGINE – Synthetic Data")
    print("="*60)
    
    bt = BacktestEngine(initial_capital=10000)
    
    # Coin5min backtest
    print("\n[1] Coin5min Backtest (ETH, 1000 intervals)...")
    prices, mkt_prices = bt.generate_synthetic_data(n_intervals=1000)
    
    result = bt.run_coin5min_backtest(
        prices=prices,
        market_prices=mkt_prices,
        asset="ETH",
        confidence_threshold=0.58,
        trade_size_pct=0.01,
    )
    
    print(f"  Final value:  ${result['final_value']:,.2f}")
    print(f"  Total PnL:    ${result['total_pnl']:+,.2f}")
    print(f"  ROI:          {result['roi']:+.2%}")
    print(f"  Signals:      {result['n_signals']}")
    print(f"  Trades:       {result['n_trades']}")
    print(f"  Win rate:     {result['win_rate']:.1%}")
    print(f"  Sharpe:       {result['sharpe']:.2f}")
    print(f"  Max DD:       {result['max_drawdown']:.2%}")
    
    # BetMiss backtest
    print("\n[2] BetMiss Backtest (500 intervals)...")
    pm_prices, ks_prices = bt.generate_synthetic_cross_platform(
        n_intervals=500, base_edge_rate=0.08
    )
    
    result_bm = bt.run_betmiss_backtest(
        pm_prices=pm_prices,
        ks_prices=ks_prices,
        min_edge=0.02,
        trade_size_pct=0.02,
    )
    
    print(f"  Final value:  ${result_bm['final_value']:,.2f}")
    print(f"  Total PnL:    ${result_bm['total_pnl']:+,.2f}")
    print(f"  Opportunities: {result_bm['opportunities']}")
    print(f"  Trades:       {result_bm['n_trades']}")
    print(f"  Win rate:     {result_bm['win_rate']:.1%}")
    print(f"  Max DD:       {result_bm['max_drawdown']:.2%}")
    
    print("\n" + "="*60)
