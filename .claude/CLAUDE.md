# CLAUDE.md — Agarthai Project Context

> Montre ce fichier à Claude en début de chaque session.

## Projet
**Agarthai** — Système de trading quantitatif multi-stratégie crypto.
Owner: Jean-Baptiste | Capital: 1500$ USD | Target: 100$/jour
Exchanges: Hyperliquid (primary), Bitget Futures (secondary)

## Stratégies
| Nom | Type | Alloc | Statut |
|-----|------|-------|--------|
| **baudouin4** | Microstructure mean-reversion (TIER-Q6h-D) | 60% | v3 |
| **innocent3** | Stat arb pairs (cointegration dynamique) | 40% | v0 |

## Architecture
- `strategies/common/base_strategy.py` → ABC dont héritent toutes les strats
- `backtest/runner.py` → charge dynamiquement via `config/strategies.yaml`
- `live/engine.py` → hérite de BacktestEngine, ajoute le streaming
- GUI Backtest: **Streamlit** (port 8501) | GUI Live: **Dash** (port 8050)
- `allocator/` supprimé → allocation dans `config/strategies.yaml`

## Données
- `btc_1s.csv.gz` / `eth_1s.csv.gz` : 13 colonnes, 1s, Binance spot
- Chemin local: `C:\Users\jeanb\Documents\hyperstat-arb-bot\analysis_ops\data_binance\spot\`

## Contraintes
Capital 1500$ | Leverage max 3x | Risk/trade 2% | Fees HL maker 0.01%
Almgren-Chriss slippage | GT-Score anti-overfit | Walk-forward purged

## Conventions
Python 3.10+ | Type hints | Dataclasses | YAML configs | Plotly charts
Streamlit backtest | Dash live | Logging 1 fichier/strat/jour
