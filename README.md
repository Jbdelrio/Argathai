# Agarthai — Multi-Strategy Crypto Trading System

## Quick Start
```bash
cd C:\Users\jeanb\Documents\Agarthai
pip install -r requirements.txt

# Copy your data
copy "...\btc_1s.csv.gz" data\binance_spot\
copy "...\eth_1s.csv.gz" data\binance_spot\

# Backtest (Streamlit) → http://localhost:8501
run_backtest.bat

# Live Trading (Dash) → http://localhost:8050
run_live.bat
```

## Strategies
| Code | Type | Allocation | Description |
|------|------|-----------|-------------|
| baudouin4 | Microstructure mean-reversion | 60% (900$) | TIER-Q6h-D: fade impulse after absorption |
| innocent3 | Stat arb pairs | 40% (600$) | Dynamic cointegration BTC-ETH + OFI filter |

## Adding a New Strategy
1. Create `strategies/my_strategy/strategy.py`:
```python
from strategies.common.base_strategy import BaseStrategy, Signal

class MyStrategy(BaseStrategy):
    def compute_features(self, df): ...
    def generate_signal(self, df): ...
```
2. Create `strategies/my_strategy/params.yaml`
3. Add to `config/strategies.yaml`:
```yaml
active_strategies:
  my_strategy: 0.20  # 20% allocation
```
4. Validate: `python validate_strategy.py my_strategy`
5. It appears automatically in both GUIs.

## Architecture
- `strategies/common/base_strategy.py` → ABC for all strategies
- `backtest/runner.py` → loads strategies dynamically from YAML
- `live/engine.py` → inherits BacktestEngine, adds streaming
- **Streamlit** (backtest) + **Dash** (live) share the same strategy classes
- Capital: 1500$ | Exchanges: Hyperliquid, Bitget Futures

## Claude Integration
Show `.claude/CLAUDE.md` at each session start for full project context."# Argathai" 
