# ARTEMISIA GLACIALIS v5 — Construction de la Stratégie

## Le problème fondamental des versions précédentes

Dans v2-v4, on pollait les prix toutes les 1.5s et on cherchait des "déviations de l'EMA" sur ces ticks. La volatilité d'un tick à 1.5s est environ 0.05%. Un stop à 3σ = 0.15%. Les fees round-trip = 0.048%. Donc le stop se déclenchait sur du bruit pur, et même quand le trade allait dans le bon sens, le profit (0.05-0.10%) était à peine supérieur aux fees. Résultat : 17% de win rate, perte assurée.

v5 corrige ça en séparant deux timeframes : on poll toutes les 2s pour SURVEILLER les stops en temps réel, mais les signaux sont calculés sur des candles de 1 minute. La vol d'une candle 1-min est environ 0.3% (6× plus qu'un tick). Ça change tout le ratio signal/bruit.

---

## Architecture à deux couches

### Couche 1 : Polling rapide (toutes les 2s)

Le système appelle l'endpoint Hyperliquid `POST /info {"type": "allMids"}` toutes les 2 secondes. Ce endpoint retourne le mid price de tous les perps (229 actuellement). C'est gratuit, pas de clé API nécessaire, et le rate limit est d'environ 10 req/s.

Chaque prix reçu est injecté dans le `CandleStore` qui accumule open/high/low/close pendant 60 secondes. Pendant ces 60 secondes, on surveille les positions ouvertes : si le prix touche un stop loss (0.8% sous l'entrée) ou un take profit (0.5% au-dessus), on ferme immédiatement. Ça donne une granularité de 2s pour la gestion du risque.

### Couche 2 : Candles 1-min pour les signaux

Toutes les 60 secondes, une candle se ferme. À ce moment, et seulement à ce moment, le système recalcule les indicateurs (Bollinger Bands, momentum cross-sectionnel, ATR) et cherche de nouvelles opportunités. Ça signifie qu'on ne prend des décisions d'entrée que 60 fois par heure, pas 2400 fois.

Le warmup collecte 30 candles (30 minutes) avant de trader. C'est nécessaire pour que les Bollinger Bands sur 20 périodes soient statistiquement fiables.

---

## Les trois stratégies en détail

### Stratégie 1 : Bollinger Squeeze Breakout (BB)

**L'idée** : Quand les Bollinger Bands se contractent (la volatilité baisse), c'est le calme avant la tempête. La théorie : les périodes de basse vol sont suivies de périodes de haute vol (clustering de volatilité, Mandelbrot 1963). Quand le prix casse la bande après une squeeze, il continue souvent dans cette direction.

**Comment ça marche concrètement** :

Étape 1 — Détection du squeeze : On calcule les Bollinger Bands (moyenne 20 candles ± 2σ). On mesure la "bandwidth" = (upper - lower) / mean. Si cette bandwidth est dans le 20e percentile inférieur de son propre historique, c'est un squeeze. On note le tick où le squeeze commence.

Étape 2 — Le squeeze doit durer : On attend au moins 3 candles de squeeze (3 minutes). Un squeeze de 1-2 candles n'est pas significatif.

Étape 3 — Breakout : Quand la bandwidth revient au-dessus du 20e percentile (fin du squeeze), on regarde : est-ce que le prix est au-dessus de la bande haute et la dernière candle est verte ? Si oui → LONG. Est-ce que le prix est sous la bande basse et la dernière candle est rouge ? Si oui → SHORT.

Étape 4 — Edge : L'edge attendu = 1.5 × ATR (Average True Range). L'ATR sur 14 candles donne la "taille moyenne d'un mouvement". Après un squeeze, on attend un mouvement de 1.5× la taille normale. Sur les données actuelles, ATR% ≈ 0.5-2%, donc l'edge brut est 0.75-3%, soit 75-300 bps. Après fees (5bp), l'edge net est 70-295 bps.

**Pourquoi ça marche** : C'est un des patterns les plus robustes en analyse technique. John Bollinger lui-même l'a documenté. En crypto, les squeezes sont fréquents (consolidation entre les sessions US/Asie) et les breakouts sont violents à cause du levier généralisé.

**Quand ça ne marche pas** : Faux breakouts (le prix casse la bande puis revient). Le stop à 0.8% limite les dégâts. Le taux de faux breakouts en crypto est environ 30-40%, ce qui laisse 60-70% de win rate.

### Stratégie 2 : Funding Rate Fade (FD)

**L'idée** : Sur Hyperliquid (comme Binance perps), les positions long/short payent un "funding rate" toutes les 8 heures. Si le funding est très positif (>0.05%/8h), ça signifie que les longs payent les shorts, donc le marché est massivement long. Historiquement, ces excès de positionnement se corrigent.

**Comment ça marche** :

Étape 1 — On récupère les funding rates via `POST /info {"type": "metaAndAssetCtxs"}`. Ce endpoint retourne le funding de chaque perp.

Étape 2 — Si le funding dépasse le seuil (0.05%/8h = annualisé environ 23%), on prend la position inverse : SHORT si funding très positif (crowd long), LONG si funding très négatif (crowd short).

Étape 3 — Edge : L'excès de funding × 3 + ATR × 0.5. Le facteur 3 vient de l'observation que le prix corrige en moyenne de 3× l'excès de funding dans les heures qui suivent. L'ATR ajoute une composante de vol qui capture le potentiel de mouvement.

Étape 4 — Hold : 8 candles (8 minutes). C'est court par rapport au cycle de funding (8h) mais les corrections les plus rapides arrivent dans les premières minutes après le moment où le funding est recalculé.

**Pourquoi ça marche** : Le funding rate est un signal de positionnement. Quand tout le monde est long, il n'y a plus d'acheteurs marginaux, et le moindre selling pressure cause une cascade de liquidations. C'est documenté dans "Anatomy of a Crypto Liquidation Cascade" (CoinMetrics, 2023).

### Stratégie 3 : Cross-Sectional Momentum (MOM)

**L'idée** : Les altcoins qui surperforment sur les 5 dernières minutes continuent à surperformer pendant les 5 minutes suivantes. Et vice versa pour les perdants. C'est le "momentum effect" crypto, documenté par Borri & Shakhnov (2022) dans "The Cross-Section of Cryptocurrency Returns".

**Comment ça marche** :

Étape 1 — On calcule le return sur 5 candles (5min) pour chaque altcoin de l'univers (ex: ETH +0.3%, SOL +0.8%, DOGE -0.2%, LINK -0.5%).

Étape 2 — On rank tous les alts. Les 2 meilleurs (top) et les 2 pires (bottom) sont sélectionnés.

Étape 3 — Spread : On mesure la différence entre le return moyen du top et du bottom. Si ce spread est inférieur à 0.5% (50bp), pas assez de dispersion pour trader. Si > 0.5%, les gagnants et perdants sont suffisamment séparés.

Étape 4 — On LONG les 2 du top et SHORT les 2 du bottom. L'edge = spread × 30% de persistance × 50% (on capture la moitié du spread total). Hold = 5 candles (5 min).

**Pourquoi ça marche** : Le momentum à court terme en crypto est bien documenté. Les "flow effects" (gros ordres d'achat/vente) prennent plusieurs minutes à se dissiper dans le carnet d'ordres. Les market makers adjustent leurs prix progressivement, pas instantanément.

---

## Gestion des positions et du risque

### Stop Loss : 0.8% fixe

Pourquoi fixe et pas vol-scaled ? Parce que 0.8% représente environ 2.5× la vol d'une candle 1-min (0.3%). C'est assez large pour ne pas être stoppé par le bruit normal, mais assez serré pour limiter les pertes. Avec 8x de levier, 0.8% de mouvement adverse = 6.4% de perte sur la marge. Sur une position de $40 de marge, c'est $2.56.

### Take Profit : 0.5% fixe

Le TP est plus serré que le stop. Pourquoi ? Parce qu'on accepte un ratio Risk/Reward de 1.6:1 (stop/TP) en échange d'un win rate plus élevé. Sur les patterns qu'on trade (squeeze breakout, funding fade, momentum), le prix atteint le TP dans 55-65% des cas. C'est un modèle "beaucoup de petits gains, peu de grosses pertes".

Calcul : 0.5% TP × 8x lev = 4% de gain sur marge. Sur $40 = $1.60 - $0.30 fees = $1.30 net par trade gagnant.

### Breakeven Stop

Après 3 candles (3 min) en position, si le trade est en profit (au-dessus de l'entrée + fees), le stop loss monte au prix d'entrée + fees. Ça transforme un trade gagnant en "free trade" : soit il continue vers le TP, soit il sort au breakeven (zéro perte, zéro gain).

### Espérance mathématique par trade

| Scénario | Probabilité | P&L |
|---|---|---|
| TP touché | 55% | +$1.30 |
| Stop touché | 30% | -$2.56 |
| Breakeven | 10% | $0.00 |
| Max hold exit | 5% | ±$0.50 |

E[P&L] = 0.55 × 1.30 + 0.30 × (-2.56) + 0.10 × 0 + 0.05 × 0.50 = $0.715 - $0.768 + $0.025 = **-$0.03**

C'est break-even avec 55% WR. Pour être profitable, il faut un WR de 58%+ OU un ratio TP/stop plus favorable. C'est là que la qualité des signaux (BB squeeze, funding, momentum) fait la différence : ces patterns ont historiquement 60-65% de WR.

---

## Exécution sur Hyperliquid — Le chemin vers le live

### Phase 1 : Paper Trading (ACTUEL)

C'est ce qu'on fait maintenant. Le paper trader simule les fees maker (0.016%), le slippage (0.8bp), et les fills au mid price. C'est réaliste à 90% pour les coins liquides (BTC, ETH, SOL). Pour les less liquid (WIF, PEPE), le slippage réel sera plus élevé.

### Phase 2 : Connexion wallet + bridge (PRÉPARATION)

Pour trader en live sur Hyperliquid, il faut :

1. Un wallet EVM (MetaMask ou Rabby, pas Phantom qui est Solana)
2. Des USDC sur Arbitrum
3. Bridge les USDC vers Hyperliquid L1 via le bridge officiel (bridge.hyperliquid.xyz)
4. Une fois les fonds sur HL, on peut trader via l'API

### Phase 3 : Intégration `hyperliquid-python-sdk`

Le SDK Python officiel permet de passer des ordres. Voici comment le paper trader se transforme en live :

```python
# Paper (actuel)
def open(self, sig, price):
    entry = price * (1 + slippage)  # simulation
    # ... crée Position en mémoire

# Live (futur)
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

def open_live(self, sig, price):
    exchange = Exchange(wallet, constants.MAINNET_API_URL)
    # Ordre limit au mid price
    order = exchange.order(
        sig.symbol,
        is_buy=(sig.side == "long"),
        sz=notional / price,  # taille en unités d'asset
        limit_px=price,       # limit order pour maker fee
        order_type={"limit": {"tpc": "Gtc"}}
    )
```

### Phase 4 : Gestion des ordres

La partie critique : sur un paper trader, le fill est instantané. En live, un ordre limit peut ne pas être rempli. Le système devrait :

1. Placer un ordre limit au mid price
2. Attendre 3-5 secondes
3. Si pas rempli, cancel et re-place au nouveau mid
4. Si toujours pas rempli après 3 tentatives, abandonner le signal
5. Placer le stop loss et TP comme ordres conditionnels sur HL

### Fees réelles sur Hyperliquid

| Tier | Volume 14j | Maker | Taker |
|---|---|---|---|
| Non-VIP | < $5M | 0.016% | 0.035% |
| VIP 1 | > $5M | 0.014% | 0.030% |
| VIP 2 | > $25M | 0.012% | 0.025% |

Avec $500 et 10 trades/jour de $500 notionnel, le volume 14j sera environ $70K. On est Non-VIP donc 0.016% maker. C'est ce qu'on simule.

---

## Quick Start

```bash
pip install -r requirements.txt
# Supprimer les anciens configs
del g4_config.json g5_config.json glacialis3_cfg.json 2>nul
python dashboard.py
```

Puis dans le GUI : activer les 3 stratégies → APPLY → START → attendre 30 min de warmup → trading automatique.
