# GUIDE : Passer en exécution réelle — Coin5min × Polymarket

## Vue d'ensemble

Ce guide t'accompagne pour connecter ton bot CYBORG à Polymarket en vrai, étape par étape, avec des petites sommes ($1–$3).

---

## Étape 0 — Prérequis

### Installer le SDK Polymarket

```bash
pip install py-clob-client python-dotenv
```

### Avoir de l'USDC sur Polymarket

1. Transfère de l'USDC depuis MetaMask vers ton adresse Polymarket (réseau **Polygon**)
2. Sur polymarket.com, vérifie que ton solde apparaît dans le profil en haut à droite
3. Autorise les allowances USDC si ce n'est pas déjà fait (le site te guide)

---

## Étape 1 — Récupérer tes credentials

### Private Key

- Va sur **https://reveal.polymarket.com** (connecté à ton compte)
- Copie la clé privée affichée

### Funder Address (adresse proxy)

- Sur polymarket.com, clique sur ton profil en haut à droite
- L'adresse affichée sous "Deposit" est ton **funder** (adresse proxy)
- C'est là que vit ton USDC sur Polygon

### Signature Type

| Tu te connectes avec… | `signature_type` |
|------------------------|------------------|
| MetaMask directement (EOA, pas de compte Polymarket) | `0` |
| Email / Magic link | `1` |
| MetaMask via polymarket.com (le plus courant) | `2` |

---

## Étape 2 — Configurer le fichier .env

Copie `.env.polymarket.example` → `.env.polymarket` et remplis :

```
POLYMARKET_PRIVATE_KEY=0xta_cle_privee_ici
POLYMARKET_FUNDER=0xton_adresse_proxy_ici
POLYMARKET_SIGNATURE_TYPE=2
POLYMARKET_CHAIN_ID=137
```

> **IMPORTANT** : Ne commit JAMAIS ce fichier. Ajoute `.env.polymarket` à ton `.gitignore`.

---

## Étape 3 — Tester la connexion (sans trader)

Crée un petit script `test_connection.py` :

```python
from execution_engine import build_engine_from_env

engine = build_engine_from_env(mode="paper")  # paper = pas d'ordres réels
ok = engine.connect()
print(f"Connexion: {'OK' if ok else 'ÉCHEC'}")

# Si tu veux vérifier le solde en live (sans trader) :
engine_live = build_engine_from_env(mode="live", max_order_usd=1.0)
ok = engine_live.connect()
if ok:
    balance = engine_live.get_balance()
    print(f"Solde USDC: ${balance:.2f}" if balance else "Impossible de lire le solde")
```

---

## Étape 4 — Premier test avec $1

### 4a. Modifier `data_manager.py`

Au début du fichier, après les imports, ajoute :

```python
from execution_engine import build_engine_from_env
```

Dans la classe `LiveState.__init__`, ajoute :

```python
# après self.paper = PaperTradingEngine(...)
self.execution = build_engine_from_env(mode="paper", max_order_usd=3.0)
self.execution.connect()
```

### 4b. Brancher l'exécution réelle dans le cycle Coin5min

Dans `_run_coin5min_cycle()`, là où le bot place les ordres paper (vers la ligne 1975), le code fait :

```python
yes_order = ST.paper.place_order(...)
no_order = ST.paper.place_order(...)
```

Pour brancher l'exécution réelle **en parallèle** (paper + live), ajoute juste après chaque `ST.paper.place_order(...)` :

```python
# --- exécution réelle (si mode live) ---
if ST.execution and ST.execution.mode.value == "live" and ST.execution.is_connected:
    live_resp = ST.execution.place_limit_order(
        symbol=asset,
        token_id=mkt.get("token_yes", ""),   # ou token_no selon le side
        side="BUY",
        price=yes_adj["exec_price"],          # ou no_adj pour le NO
        size=min(yes_budget, ST.execution.max_order_usd),
        tags={"strategy": "coin5min", "asset": asset, "leg": "yes"},
    )
    ST.alert("INFO", "EXEC", f"LIVE order: {live_resp.state} {live_resp.symbol} ${live_resp.size:.2f}")
```

### 4c. Tracker les ordres dans la boucle

Dans `_runtime_loop()`, ajoute un refresh des ordres live :

```python
# dans _runtime_loop(), après _run_coin5min_cycle()
if hasattr(ST, 'execution') and ST.execution and ST.execution.mode.value == "live":
    for order in ST.execution.list_orders(states=["OPEN", "PARTIAL", "POSTING"]):
        ST.execution.refresh_order(order["client_order_id"])
```

---

## Étape 5 — Passer en mode live

Quand le test paper+live fonctionne, change le mode dans `LiveState.__init__` :

```python
# Phase 1 : paper (test la connexion)
self.execution = build_engine_from_env(mode="paper", max_order_usd=3.0)

# Phase 2 : live avec $1 max
self.execution = build_engine_from_env(mode="live", max_order_usd=1.0)

# Phase 3 : live avec $3 max
self.execution = build_engine_from_env(mode="live", max_order_usd=3.0)

# Phase 4 : montée progressive
self.execution = build_engine_from_env(mode="live", max_order_usd=10.0)
```

---

## Étape 6 — Alertes dans le dashboard

Dans `cyborg_dash.py`, les alertes de l'execution engine sont déjà disponibles via :

```python
alerts = ST.execution.get_alerts()
```

Tu peux les afficher dans la section alertes existante du dashboard. Chaque alerte contient `level`, `code`, `message`, `sound_hint`.

---

## Sécurité — Checklist

- [ ] `max_order_usd` réglé à $1 au début
- [ ] Kill switch accessible : `ST.execution.kill()` coupe tout
- [ ] `.env.polymarket` dans `.gitignore`
- [ ] Jamais de clé privée dans le code
- [ ] Tester en paper avant chaque changement
- [ ] Journal des ordres dans `orders_live.db` (SQLite)
- [ ] Toujours vérifier le solde avant d'augmenter les tailles

---

## Architecture

```
data_manager.py
  ├── PaperTradingEngine     (simulation, inchangé)
  ├── ExecutionEngine         (nouveau, exécution réelle)
  │     ├── mode=paper  →  log local uniquement
  │     └── mode=live   →  py-clob-client → Polymarket CLOB
  └── _run_coin5min_cycle()
        ├── décision TRADE/WAIT/SKIP  (inchangé)
        ├── ST.paper.place_order()    (simulation, inchangé)
        └── ST.execution.place_limit_order()  (NOUVEAU)

execution_engine.py
  ├── connect()                → dérive les API creds
  ├── place_limit_order()      → signe + post l'ordre
  ├── refresh_order()          → poll le statut CLOB
  ├── cancel_order / cancel_all
  ├── get_alerts()             → pour le dashboard
  ├── get_balance()            → solde USDC
  └── kill() / unkill()        → coupe-circuit d'urgence
```

---

## Ordre de déploiement recommandé

| Phase | Mode | max_order_usd | Durée conseillée |
|-------|------|--------------|-----------------|
| 1 | `paper` | — | jusqu'à ce que la connexion soit stable |
| 2 | `live` | $1.00 | 1-2 jours, observer les fills |
| 3 | `live` | $3.00 | 3-5 jours, vérifier le PnL réel vs paper |
| 4 | `live` | $5-10 | montée progressive selon résultats |

---

## Rappel important

Le PnL paper reste une borne haute. En réel tu auras :
- du slippage supplémentaire (le carnet bouge entre décision et fill)
- des fills partiels (pas toujours la taille complète)
- de la latence réseau (quelques centaines de ms)
- des ordres rejetés (balance insuffisante, marché fermé)

C'est exactement pour ça qu'on commence à $1.
