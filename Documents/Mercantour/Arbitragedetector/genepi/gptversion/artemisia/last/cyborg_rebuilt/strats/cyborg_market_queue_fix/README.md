# CYBORG – README détaillé

CYBORG est un dashboard de **paper trading live** pour deux stratégies sur marchés binaires :

- **BetMiss** : arbitrage systématique **cross-platform** entre **Polymarket** et **Kalshi**.
- **Coin5min** : arbitrage systématique **intra-market** sur les marchés crypto **Up / Down** de **Polymarket**, principalement sur **5 min** et **15 min**.

L'objectif du projet n'est pas de passer des ordres réels, mais de :

1. se connecter aux flux live,
2. détecter les opportunités,
3. simuler les trades,
4. suivre le PnL au fil des heures,
5. comprendre visuellement ce que fait chaque stratégie.

---

## 1. Ce que fait maintenant cette version

Cette version met l'accent sur une logique de stratégie plus claire et un GUI plus exploitable.

### Changements importants

- plus de faux trades injectés au démarrage ;
- `START` déclenche vraiment la stratégie choisie ;
- le capital et les allocations restent contrôlés depuis le Dashboard ;
- **Coin5min** n'est plus présenté comme un simple modèle directionnel : le cœur de la logique est maintenant l'**arbitrage YES + NO < 1** ;
- ajout d'un **Kill Switch** global ;
- affichage explicite du **prix de référence** pour `Above/YES` et `Below/NO` dans Coin5min ;
- affichage du **temps de warmup** Coin5min ;
- support des marchés **5 min** et **15 min** côté Coin5min.

---

## 2. Lancer le projet

```bash
pip install -r requirements.txt
python cyborg_dash.py
```

Puis ouvrir :

```bash
http://localhost:8050
```

---

## 3. Architecture rapide

- `cyborg_dash.py` : interface Dash
- `data_manager.py` : état global, polling live, logique des stratégies, suivi du paper trading
- `paper_trading.py` : moteur générique d'ordres papier
- `api_connectors.py` : anciens connecteurs plus avancés conservés dans le projet
- `model_engine.py` : ancien moteur de signaux / modèles, conservé dans le projet
- `STRATEGIES.md` : synthèse stratégique

---

## 4. Workflow normal

### Étape 1 – Configurer la session
Dans l'onglet **Dashboard** :

- choisir **Capital total ($)**
- choisir **Allocation BetMiss (%)**
- choisir **Allocation Coin5min (%)**
- cliquer **APPLIQUER & RESET PAPER**

Important :

- le capital et les allocations ne changent **qu'ici** ;
- la somme des allocations doit être `<= 100%` ;
- ce bouton remet la session paper à zéro.

### Étape 2 – Lancer une stratégie
Dans l'onglet de la stratégie :

- régler les paramètres live
- cliquer **START**

### Étape 3 – Stopper ou bloquer
- **PAUSE** : suspend les nouvelles prises de position
- **STOP** : arrête la stratégie
- **KILL SWITCH** : coupe immédiatement le lancement de nouvelles stratégies et force l'arrêt global

---

## 5. Dashboard – paramètres globaux

### Capital total
Capital paper de référence.

Exemple :
- `10000` = portefeuille paper de 10 000 $

### Allocation BetMiss / Coin5min
Pourcentage du capital réservé à chaque stratégie.

Exemple :
- capital = 10 000
- BetMiss = 40%
- Coin5min = 35%

Alors :
- enveloppe BetMiss = 4 000 $
- enveloppe Coin5min = 3 500 $
- 2 500 $ restent non alloués

### Taille / trade
Chaque stratégie applique ensuite son propre pourcentage **sur son enveloppe allouée**.

Exemple Coin5min :
- capital total = 10 000
- allocation Coin5min = 40%
- enveloppe Coin5min = 4 000
- taille/trade = 2%

Budget nominal par trade :
- `4000 × 2% = 80 $`

---

## 6. Kill Switch

Le **Kill Switch** est un coupe-circuit global.

Quand il est activé :

- aucun nouveau `START` n'est autorisé ;
- les stratégies en cours sont stoppées ;
- le GUI affiche `KILL ON`.

Quand il est désactivé :

- le GUI affiche `KILL OFF` ;
- tu peux relancer BetMiss ou Coin5min.

Usage conseillé :
- test rapide ;
- doute sur la qualité des flux ;
- comportement non attendu du matching ;
- besoin de geler la session immédiatement.

---

# 7. Stratégie 1 – BetMiss

## 7.1 Idée

BetMiss cherche un **mispricing entre deux plateformes** pour le **même événement**.

Cas type :

- acheter `YES` sur une plateforme,
- acheter `NO` sur l'autre,
- si le coût total est `< 1`, le profit théorique est verrouillé.

### Formule

Si :

```text
prix(YES plateforme A) + prix(NO plateforme B) < 1
```

alors :

```text
edge = 1 - [prix(YES A) + prix(NO B)]
```

## 7.2 Exemple

```text
Polymarket YES = 0.58
Kalshi NO     = 0.39
Somme         = 0.97
Edge          = 0.03 = 3%
```

## 7.3 Univers de marchés

BetMiss n'est **pas limité à la crypto**.

Il peut en principe scanner :
- sports,
- politique / présidentiel,
- macro,
- events divers,

à condition qu'un événement existe sur les deux plateformes avec un libellé suffisamment proche pour être apparié.

## 7.4 Paramètres BetMiss

### Seuil edge min
Seuil minimum pour considérer l'opportunité comme tradable.

Exemple :
- `2%` = on ignore toute opportunité dont l'edge est inférieur à 2%

### Taille / trade
Pourcentage de l'enveloppe BetMiss utilisé par opportunité détectée.

## 7.5 Ce que montre l'onglet BetMiss

- état de la stratégie
- opportunités live détectées
- historique des trades paper
- PnL théorique BetMiss

## 7.6 Limites BetMiss

- le matching des questions entre Polymarket et Kalshi reste volontairement simple ;
- deux marchés économiquement identiques peuvent ne pas être reconnus comme identiques ;
- les frais, la latence et la profondeur réelle peuvent réduire l'edge ;
- certaines opportunités affichées peuvent être trop petites pour être robustes.

---

# 8. Stratégie 2 – Coin5min

## 8.1 Idée

Coin5min se concentre sur les marchés Polymarket **crypto Up / Down** à très court terme.

Dans cette version, la logique principale est **systématique / arbitrage intra-market** :

- on regarde le prix d'achat de `Above/YES`
- on regarde le prix d'achat de `Below/NO`
- si la somme est `< 1`, on peut acheter les deux côtés
- le payout final théorique d'un couple de parts est `1`

Donc :

```text
prix(YES) + prix(NO) < 1  => arbitrage théorique
```

Cette logique fonctionne sur :
- **5 minutes**
- **15 minutes**

## 8.2 Interprétation des côtés

Dans l'UI :

- **Above / YES** = le prix final est au-dessus du prix de référence
- **Below / NO** = le prix final est en dessous du prix de référence

## 8.3 Prix de référence

Le GUI affiche un **prix de référence** (`ref_px`) pour chaque actif.

Ce prix sert à visualiser :
- le niveau au-dessus duquel on interprète `Above / YES`
- le niveau en dessous duquel on interprète `Below / NO`

### Important
Dans cette version, ce prix de référence est une **référence opérationnelle de dashboard / paper trading**, capturée à partir du spot observé lorsque le marché devient suivi. Ce n'est pas garanti comme étant le champ de settlement officiel du marché côté exchange.

Autrement dit :
- c'est très utile pour le suivi live,
- mais ce n'est pas encore un moteur de settlement institutionnel.

## 8.4 Formule d'arbitrage Coin5min

Si :

```text
YES_ask + NO_ask < 1
```

alors :

```text
edge = 1 - (YES_ask + NO_ask)
```

L'UI affiche :
- `yes_above`
- `no_below`
- `yes+no`
- `edge`
- le carnet `yes_bid / yes_ask / no_bid / no_ask`

## 8.5 Paramètres Coin5min

### Seuil edge min (YES+NO)
Seuil minimum d'arbitrage avant d'ouvrir un pair trade.

Exemple :
- `1.0%` = le couple `YES+NO` doit offrir au moins 1% d'edge théorique

### Taille / trade
Pourcentage de l'enveloppe Coin5min utilisé pour un pair trade.

### Intervalle
Choix du type de marché :
- `5 min`
- `15 min`

### Actifs
Actifs suivis dans l'UI :
- BTC
- ETH
- SOL
- XRP
- DOGE

## 8.6 Warmup Coin5min

Le warmup de Coin5min ne sert pas à charger "des millions de données".

Il sert surtout à :
- vérifier que le feed spot Coinbase répond ;
- capturer un minimum d'historique récent ;
- stabiliser l'affichage du spot ;
- disposer d'un repère propre pour l'interface.

Cette version utilise un warmup court.

Dans l'onglet Coin5min, tu vois :
- le temps écoulé depuis le début du warmup ;
- une ETA courte ;
- une barre de progression par actif.

## 8.7 Exécution paper Coin5min

Quand l'edge dépasse le seuil :

1. la stratégie ouvre un **pair trade** sur le marché Polymarket du moment ;
2. elle simule l'achat de `YES` et de `NO` ;
3. elle garde la position ouverte jusqu'à l'échéance du marché ;
4. elle journalise le PnL théorique dans l'historique.

## 8.8 Ce que montre l'onglet Coin5min

- statut : `CONNECTING / WARMING / RUNNING / PAUSED / STOPPED`
- temps de warmup
- prix spot live
- prix de référence
- prix `Above / YES`
- prix `Below / NO`
- somme `YES + NO`
- edge théorique
- meilleur bid/ask côté YES et côté NO
- trades Coin5min simulés

---

# 9. Ce que signifie le GUI

## Quand tu n'as rien lancé
Normalement :
- aucune stratégie ne trade ;
- pas de faux PnL injecté ;
- pas de faux trades.

## Quand tu cliques START
La stratégie commence à :
- se connecter,
- scanner les flux,
- détecter des opportunités,
- simuler des trades si les conditions sont remplies.

## Modifier les paramètres en live
Les contrôles d'une stratégie sont modifiables pendant qu'elle tourne.

Exemples :
- monter le seuil d'edge Coin5min ;
- changer 5 min → 15 min ;
- réduire la taille / trade ;
- restreindre la liste d'actifs.

Remarque :
- le **capital total** et les **allocations** restent des paramètres globaux Dashboard ;
- les paramètres propres à la stratégie peuvent être changés directement dans l'onglet de la stratégie.

---

# 10. Limitations à garder en tête

## Coin5min
- le prix de référence affiché est un proxy opérationnel, pas encore un settlement officiel branché sur une source d'arbitrage finale ;
- si l'API marché ou le carnet répond mal, certains prix peuvent rester momentanément incomplets ;
- c'est un moteur de **paper trading** et non un routeur d'ordres réels.

## BetMiss
- l'appariement sémantique des marchés reste simple ;
- toutes les opportunités cross-platform ne seront pas trouvées ;
- le profit affiché reste théorique tant qu'on ne modélise pas tous les frais et la profondeur réelle.

---

# 11. Prochaine étape logique

Les prochaines améliorations les plus utiles seraient :

1. **normalisation plus robuste** des événements BetMiss entre Polymarket et Kalshi ;
2. **source de settlement plus officielle** pour Coin5min ;
3. **journal d'exécution détaillé** par pair trade ;
4. **backfill historique live** pour évaluer le PnL sur plusieurs heures / journées ;
5. **export CSV** des trades et des opportunités.


## Coin5min / 15min – legged arb

La logique n'est plus un pur pari directionnel. Le coeur du moteur est un arbitrage de paire sur le marché Up/Down Polymarket:

- on surveille en temps réel `YES ask`, `NO ask`, `YES bid`, `NO bid`
- la règle ultime de verrouillage reste `YES + NO < 1`
- on peut acheter les deux jambes simultanément si l'opportunité est déjà ouverte
- ou acheter d'abord une jambe, attendre quelques secondes, puis couvrir l'autre jambe plus tard si le cumul devient intéressant

### Paramètres UI Coin5min

- **Seuil edge min (YES+NO)**: contrainte de marge minimum en pourcentage
- **Seuil max de somme YES+NO**: plafond absolu du coût total de la paire
- **Prix max de la première jambe**: prix maximum accepté pour ouvrir la jambe 1
- **Délai max entre jambe 1 et jambe 2**: temps maximal avant abandon ou résolution d'une jambe seule
- **Jambe 1 directionnelle on/off**:
  - on: si spot >= ref -> on privilégie YES, sinon NO
  - off: on prend juste la jambe la moins chère sous le plafond de prix
- **Source spot / ref proxy**: Binance ou Coinbase pour le spot live et le bootstrap

### Prix de référence

Le `ref_px` n'est pas pris au moment où la GUI voit le marché. Il est reconstitué à partir du timestamp de début de la fenêtre du marché, afin d'être plus proche du vrai *Price to Beat* affiché par Polymarket.

### Détection des marchés Polymarket courts

Pour les marchés crypto 5m/15m, l'app essaie maintenant d'abord une recherche déterministe par slug autour de la fenêtre courante, par exemple:

- `btc-updown-5m-<window_start_ts>`
- `eth-updown-15m-<window_start_ts>`

Puis elle retombe sur la discovery paginée si nécessaire.
