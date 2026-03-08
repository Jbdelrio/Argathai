# Agarthai — Multi-Strategy Crypto Trading System

Agarthai garde la même vision : **backtest + live trading** avec des stratégies complémentaires (`baudouin4`, `innocent3`, `urbain2`) et une contrainte réaliste de coûts/exécution.

## Quick Start
```bash
cd C:\Users\jeanb\Documents\Agarthai
pip install -r requirements.txt

# GUI Backtest (Streamlit) → http://localhost:8501
run_backtest.bat

# GUI Live officiel (Dash) → http://localhost:8050
run_live.bat
```

## Données réelles d’exchange (historique 1s)

Tu peux continuer à utiliser tes gros fichiers locaux (`btc_1s.csv.gz`, `eth_1s.csv.gz`).

Nouveau comportement dans `data/loader.py` :
1. cherche d’abord local (`btc_path`/`eth_path`, `search_paths`, `data/binance_spot/`),
2. **si absent**, peut lancer un auto-fetch historique depuis exchange (option `history_fetch.enabled=true` dans `config/settings.yaml`),
3. le fetch est fait en mode **pagination + rate-limit** via `ccxt` et reconstruit des barres 1s depuis les trades.

> Important : selon l’exchange, l’historique tick/trades est limité. Le système gère la pagination mais reste borné par les limites API réelles.

### Fetch manuel d'historique 1s (exchange -> fichier)
```bash
python -m data.fetch_history --exchange bitget --symbol BTC/USDT:USDT \
  --start 2026-02-01T00:00:00Z --end 2026-03-04T00:00:00Z \
  --out data/binance_spot/btc_1s.csv.gz --limit 200
```

Fais la même commande pour ETH en changeant `--symbol` et `--out`.

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
├── data/
│   ├── loader.py
│   ├── exchange_fetcher.py
│   └── binance_spot/
├── backtest/
│   ├── runner.py
│   └── metrics.py
├── exchanges/
│   └── clients.py
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
│   ├── innocent3/
│   │   ├── strategy.py
│   │   └── params.yaml
│   └── urbain2/
│       ├── strategy.py
│       └── params.yaml
└── tests/
    ├── test_config_files.py
    ├── test_loader.py
    ├── test_metrics.py
    └── test_urbain2.py
```

## Allocation active
| Code | Type | Allocation cible | Rôle |
|------|------|------------------|------|
| `baudouin4` | Microstructure mean-reversion | 45% | Fade impulsion après absorption |
| `innocent3` | Stat-arb BTC/ETH | 35% | Réversion spread cointegré |
| `urbain2` | Rotation résiduelle altcoins | 20% | Alpha idiosyncratique filtré coûts/régime |

---

## Détail mathématique des stratégies

### 1) `baudouin4` — TIER-Q6h-D


a) Impulsion :
\[
I_t = \frac{|R_{w_s}(t)|}{\sigma_{w_l}(t)+\varepsilon}
      \cdot \frac{V_{w_s}(t)}{\text{med}_{w_l}(V_{w_s})+\varepsilon}
      \cdot \frac{N_{w_s}(t)}{\text{med}_{w_l}(N_{w_s})+\varepsilon}
\]

b) Stabilisation :
\[
S_t = \frac{\sigma_{w_s}(t)}{\sigma_{w_m}(t)+\varepsilon}
\]

c) Score d’épuisement :
\[
E_t^* = z_r(I_t) + z_r(C_t) + z_r(A_t) + z_r(-S_t)
\]

Entrée contrariante via machine d’état `IDLE → IMPULSE → STAB`.

### 2) `innocent3` — Cointegration BTC/ETH + OFI

a) Hedge ratio dynamique :
\[
\beta_t \approx \frac{\mathrm{Cov}(\log P_{BTC}, \log P_{ETH})}{\mathrm{Var}(\log P_{BTC})}
\]

b) Spread cointegré :
\[
s_t = \log P_{ETH,t} - \beta_t\log P_{BTC,t}
\]

c) Z-score de spread (OU local) :
\[
z_t = \frac{s_t - \mu_t}{\sigma_t + \varepsilon}
\]

Entrée de réversion si \(|z_t|\) élevé + filtre OFI + demi-vie valide.

### 3) `urbain2` — Rotation cross-sectionnelle résiduelle

a) Neutralisation factorielle :
\[
r_{i,t} = \alpha_i + \beta_{i,B}r^{BTC}_t + \beta_{i,E}r^{ETH}_t + \sum_k \beta_{i,k}PC_{k,t} + u_{i,t}
\]
On trade le résidu \(u_{i,t}\), pas le bêta brut.

b) Momentum résiduel :
\[
m_{i,t} = \frac{\sum_{h=1}^{H_m} u_{i,t-h}}{\hat\sigma_{u,i,t}\sqrt{H_m}+\varepsilon}
\]

c) OI confirmation :
\[
o_{i,t}=z_r(\Delta\log OI_{i,t})
\]

d) Funding crowding :
\[
\phi_{i,t}=\text{sign}(m_{i,t})\,z_r(F_{i,t})-\eta z_r(F_{i,t})^2
\]

e) Score liquidité :
\[
\ell_{i,t}=z_r(\log ADV_{i,t})-z_r(spread_{i,t})-z_r(impact_{i,t})
\]

f) Score final cost-aware / uncertainty-aware :
\[
s_{i,t}=G_t(\theta_1 m_{i,t}+\theta_2 o_{i,t}+\theta_3 \phi_{i,t}+\theta_4 \ell_{i,t})
\]
\[
s^*_{i,t}=s_{i,t}-\lambda q_{i,t}
\]
Trade si :
\[
|s^*_{i,t}|>\tau+\kappa c_{i,t}
\]

---

## Configuration
- `config/strategies.yaml` : stratégies actives + allocation.
- `config/settings.yaml` : chemins data, risk, et auto-fetch historique.
- `config/exchanges.yaml` : paramètres exchange de base.

## Live mode
- GUI live officiel : **Dash** (`live/dashboard.py`).
- Streamlit live (`gui/live_app.py`) reste disponible, mais non officiel.

## Tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```
