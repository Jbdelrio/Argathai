# Agarthai — Multi-Strategy Crypto Trading System
Agarthai garde la même vision : **backtest + live trading** avec des stratégies complémentaires (`baudouin4`, `innocent3`, `urbain2`) et une contrainte réaliste de coûts/exécution.

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

> Note: `live/dashboard.py` (Dash) est conservé en legacy/debug, mais l'interface live officielle est maintenant `gui/live_app.py`.
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
| Code | Type | Allocation cible | Rôle portefeuille |
|------|------|------------------|-------------------|
| `baudouin4` | Microstructure mean-reversion | 45% | Capter les excès d’impulsion à court terme, Fade impulsion après absorption |
| `innocent3` | Stat-arb BTC/ETH | 35% | Exploiter la réversion du spread cointegré |
| `urbain2` | Rotation résiduelle altcoins | 20% | Capter un alpha idiosyncratique, neutre au bêta crypto, Alpha idiosyncratique filtré coûts/régime |

---

## Détail mathématique des stratégies

### 1) `baudouin4` — TIER-Q6h-D (fade impulsion après absorption)

**Intuition :** après impulsion violente (retour + volume + activité), on attend stabilisation puis signal d’épuisement, puis on fade.

Variables principales (fenêtres courte/moyenne/longue : `w_s`, `w_m`, `w_l`) :

- Retour cumulé :

  $$
  R_{w_s}(t)=\sum_{j=t-w_s+1}^{t} r_j
  $$

- Activité volume/transactions : $V_{w_s}(t),N_{w_s}(t)$
- Impulsion :

  $$
  I_t = \frac{|R_{w_s}(t)|}{\sigma_{w_l}(t)+\varepsilon}
        \cdot \frac{V_{w_s}(t)}{\text{med}_{w_l}(V_{w_s})+\varepsilon}
        \cdot \frac{N_{w_s}(t)}{\text{med}_{w_l}(N_{w_s})+\varepsilon}
  $$

- Compression de volatilité :

  $$
  S_t = \frac{\sigma_{w_s}(t)}{\sigma_{w_m}(t)+\varepsilon}
  $$
- Score d’épuisement robuste :

  $$
  E_t^* = z_r(I_t) + z_r(C_t) + z_r(A_t) + z_r(-S_t)
  $$
  où $z_r$ est un z-score robuste (médiane/MAD).

**Entrée :** machine d’état IDLE → IMPULSE → STAB, puis entrée contrariante quand $E_t^*$ dépasse son quantile calibré.

---

### 2) `innocent3` — Stat-arb dynamique BTC-ETH (cointegration + OFI)

**Intuition :** trader la réversion d’un spread cointegré BTC/ETH, filtré par divergence de flux (OFI).

- Couverture dynamique :

  $$
  \beta_t \approx \frac{\text{Cov}(\log P_{BTC},\log P_{ETH})}{\text{Var}(\log P_{BTC})}
  $$
- Spread :

  $$
  s_t = \log P_{ETH,t} - \beta_t\log P_{BTC,t}
  $$
- Modèle OU local :

  $$
  ds_t = \kappa_t(\mu_t - s_t)dt + \sigma_t dW_t
  $$
- Z-score du spread :

  $$
  z_t = \frac{s_t-\mu_t}{\sigma_t+\varepsilon}
  $$

**Entrée :** si $|z_t|$ élevé + divergence OFI cohérente + demi-vie dans une plage exploitable. Signal contrariant (reversion).

---

### 3) `urbain2` — Rotation cross-sectionnelle résiduelle des altcoins

**Intuition :** ne pas trader le bêta crypto “déguisé”, mais la continuation **idiosyncratique** après neutralisation des facteurs communs, avec confirmation OI, pénalisation funding crowding, filtre liquidité/coûts/incertitude.

#### 3.1 Neutralisation factorielle

Pour chaque actif alt $i$, on approxime :

$$
r_{i,t} = \alpha_i + \beta_{i,B}r^{BTC}_t + \beta_{i,E}r^{ETH}_t + \sum_{k=1}^{K}\beta_{i,k}PC_{k,t} + u_{i,t}
$$
Le signal travaille sur le **résidu** $u_{i,t}$.

Dans l’implémentation actuelle (single-symbol), les facteurs communs sont pris depuis colonnes dédiées si disponibles (`btc_ret_1h`, `eth_ret_1h`, `alt_pca1`) avec fallback robuste sinon.

#### 3.2 Momentum résiduel

$$
m_{i,t} = \frac{\sum_{h=1}^{H_m}u_{i,t-h}}{\hat{\sigma}_{u,i,t}\sqrt{H_m}+\varepsilon}
$$

#### 3.3 Confirmation OI

$$
o_{i,t} = z_r\big(\Delta\log OI_{i,t}\big)
$$

#### 3.4 Funding : bonus modéré + pénalité crowding

$$
\phi_{i,t}=\operatorname{sign}(m_{i,t})\,z_r(F_{i,t})-\eta\,z_r(F_{i,t})^2
$$

#### 3.5 Qualité de liquidité

$$
\ell_{i,t}=z_r(\log ADV_{i,t})-z_r(spread_{i,t})-z_r(impact_{i,t})
$$

#### 3.6 Gate de régime

$$
G_t=\mathbf{1}\left[\text{Disp}(u_t)>Q_{0.6}(\text{Disp})\;\land\;|r^{BTC}_t|<Q_{0.9}(|r^{BTC}|)\right]
$$

#### 3.7 Score final cost-aware et uncertainty-aware

$$
s_{i,t}=G_t\big(\theta_1m_{i,t}+\theta_2o_{i,t}+\theta_3\phi_{i,t}+\theta_4\ell_{i,t}\big)
$$

$$
s^*_{i,t}=s_{i,t}-\lambda q_{i,t}
$$

Entrée seulement si :

$$
|s^*_{i,t}|>\tau+\kappa c_{i,t}
$$

avec $c_{i,t}$ coût attendu (fees + spread + slippage proxy).

---

## Configuration
- `config/strategies.yaml`: stratégies actives + allocation.
- `config/exchanges.yaml`: paramètres exchange (fees/default market).
- `config/settings.yaml`: capital/risk + chemins data portables.

## Data portability
Le loader ne hardcode pas de chemins machine.

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


# Other plan version
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