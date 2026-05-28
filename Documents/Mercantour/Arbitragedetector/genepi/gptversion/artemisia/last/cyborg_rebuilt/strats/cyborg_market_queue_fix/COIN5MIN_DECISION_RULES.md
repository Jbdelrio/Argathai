# Coin5min — règles de décision

## Principe

Coin5min est une stratégie **intra-market** sur les marchés Polymarket **Up / Down** court terme (5 min / 15 min).

La règle ultime est :

```text
YES_ask + NO_ask < 1
```

Si cette condition est satisfaite, le coût total de la paire est inférieur au payout final théorique de 1.

```text
edge = 1 - (YES_ask + NO_ask)
```

## Mode réel utilisé dans le moteur

La stratégie n'attend pas forcément de prendre les deux jambes simultanément.

1. détecter un marché Polymarket court terme actif sur l'actif choisi ;
2. lire le **Price to Beat** du marché Polymarket ;
3. lire le **spot proxy** (Binance ou Coinbase) + la micro-tendance avant / pendant la fenêtre ;
4. choisir une **jambe 1** si son prix est suffisamment bon ;
5. attendre la **jambe 2** pendant quelques secondes ;
6. verrouiller la paire seulement si la somme finale passe sous le seuil.

## Sélection d'un marché tradable

Un marché devient tradable seulement si :

- marché Polymarket détecté ;
- carnet disponible ;
- `YES_ask` et `NO_ask` exploitables ;
- temps restant suffisant ;
- budget autorisé par le cap de risque.

## Choix de la jambe 1

### Mode directionnel ON

La première jambe est choisie avec un léger biais directionnel :

- `spot >= ref_px` ou tendance haussière => préférence pour `YES`
- `spot < ref_px` ou tendance baissière => préférence pour `NO`

### Mode directionnel OFF

On prend la jambe la moins chère entre `YES` et `NO`.

## Conditions d'entrée jambe 1

On n'ouvre pas jambe 1 si :

- son prix est au-dessus de `max_first_leg_price`
- le marché est trop proche de l'échéance
- le budget effectif dépasse le cap en dollars

## Condition de couverture jambe 2

Si la jambe 1 a été ouverte à `p1`, alors la jambe 2 ne peut être prise que si :

```text
p1 + p2 <= max_sum_yes_no
```

avec en pratique :

```text
max_sum_yes_no = min(seuil utilisateur, 1 - edge_min)
```

Exemple :

- achat `YES = 0.39`
- seuil max `YES+NO = 0.99`
- alors la couverture `NO` ne doit être prise que si `NO <= 0.60`

## Délai maximal entre les deux jambes

La jambe 2 n'est cherchée que pendant `max_leg_hold_sec`.

Si elle n'arrive pas à temps :

- soit la jambe 1 reste seule jusqu'à résolution ;
- soit une règle de coupe future pourra être ajoutée.

## Features utilisées pour maximiser la probabilité de trouver `YES + NO < 1`

### 1. Features directes de marché

- `YES_ask`
- `NO_ask`
- `YES_bid`
- `NO_bid`
- `sum_yes_no = YES_ask + NO_ask`
- `edge = 1 - sum_yes_no`

### 2. Features de contexte court terme

- `ref_px` = Price to Beat Polymarket
- `spot`
- `spot - ref_px`
- `trend_bps` sur le lookback court
- `trend_dir`

### 3. Features de risque

- `budget_pct`
- `budget_cap_usd`
- `budget_effective_usd`
- `trend_bias_used`

## Logique de sizing

Le moteur ne doit jamais engager tout le capital.

Budget effectif :

```text
budget_pct_usd = capital_total * allocation_coin5min * trade_size_pct
budget_effective_usd = min(budget_pct_usd, max_trade_usd)
```

Exemple :

- capital = 500 $
- allocation Coin5min = 50 %
- taille/trade = 8 %
- cap = 20 $

Alors :

```text
budget_pct_usd = 500 * 0.50 * 0.08 = 20
budget_effective_usd = 20
```

## Résumé en une phrase

La stratégie idéale n'est pas :

> je prédis simplement Up ou Down

mais :

> j'ouvre tôt une jambe bon marché dans un marché encore mal ajusté, je choisis intelligemment le côté grâce au contexte spot / ref / tendance, puis j'attends que l'autre jambe revienne à un prix qui me permet de verrouiller `YES + NO < 1` avant la fin de la fenêtre.
