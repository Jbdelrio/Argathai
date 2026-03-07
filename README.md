# Agarthai — Multi-Strategy Crypto Trading System

Agarthai reste basé sur la même idée: **backtest + live trading** autour de stratégies modulaires (baudouin4 / innocent3), avec un focus execution/risk management.

## Quick Start
```bash
cd C:\Users\jeanb\Documents\Agarthai
pip install -r requirements.txt

# Copy your data
copy "...\btc_1s.csv.gz" data\binance_spot\
copy "...\eth_1s.csv.gz" data\binance_spot\

# Backtest GUI (Streamlit) → http://localhost:8501
run_backtest.bat

# Live GUI officiel (Streamlit) → http://localhost:8502
run_live.bat
```

> Note: `live/dashboard.py` (Dash) est conservé en legacy/debug, mais l'interface live officielle est maintenant `gui/live_app.py`.

## Architecture actuelle (arborescence)
```text
Agarthai/
├── README.md
├── requirements.txt
├── run_backtest.bat
├── run_live.bat
├── config/
│   ├── settings.yaml
│   ├── strategies.yaml
│   └── exchanges.yaml
├── core/
│   ├── position_sizing.py
│   └── slippage.py
├── backtest/
│   ├── runner.py
│   └── metrics.py
├── exchanges/
│   └── clients.py
├── data/
│   ├── loader.py
│   └── binance_spot/
│       ├── btc_1s.csv.gz
│       └── eth_1s.csv.gz
├── gui/
│   ├── backtest_app.py
│   └── live_app.py
├── live/
│   ├── engine.py
│   └── dashboard.py
├── strategies/
│   ├── common/base_strategy.py
│   ├── baudouin4/
│   │   ├── strategy.py
│   │   └── params.yaml
│   └── innocent3/
│       ├── strategy.py
│       └── params.yaml
└── tests/
    ├── test_config_files.py
    ├── test_loader.py
    └── test_metrics.py
```

## Strategies
| Code | Type | Allocation | Description |
|------|------|-----------|-------------|
| baudouin4 | Microstructure mean-reversion | 60% (900$) | TIER-Q6h-D: fade impulse after absorption |
| innocent3 | Stat arb pairs | 40% (600$) | Dynamic cointegration BTC-ETH + OFI filter |

## Configuration
- `config/strategies.yaml`: stratégies actives + allocation (évite les runs silencieux sans stratégie).
- `config/exchanges.yaml`: paramètres de base exchange (fees/default market).
- `config/settings.yaml`: capital/risk + chemins data portables.

## Data portability
Le loader ne hardcode plus des chemins machine Windows.

Ordre de recherche des données:
1. chemins explicites `btc_path` / `eth_path` dans `config/settings.yaml`
2. `search_paths` dans `config/settings.yaml`
3. variable d'environnement `AGARTHAI_DATA_DIRS`
4. `data/binance_spot/`

## Live mode safety
- **Paper mode**: fonctionnement immédiat sans clé API.
- **Live mode**: nécessite des clés via variables d'environnement (`HYPERLIQUID_API_KEY`, `HYPERLIQUID_API_SECRET`, etc.) et passe par `ccxt`.

## Minimal tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```
