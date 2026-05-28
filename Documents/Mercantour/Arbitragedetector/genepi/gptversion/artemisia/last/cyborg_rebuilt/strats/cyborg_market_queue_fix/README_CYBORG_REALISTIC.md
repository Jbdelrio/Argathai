# CYBORG – Version réaliste PnL

## Ce qui a changé

Cette version réduit le caractère trop optimiste du paper trading précédent.

### 1. Référence / Price to Beat
- le `Price to Beat` Polymarket est parsé avec un contrôle de cohérence
- si la valeur trouvée est aberrante, le moteur retombe sur un fallback cohérent

### 2. Probabilité directionnelle court terme
Pour chaque actif, le moteur calcule :
- distance spot / ref
- tendance spot court terme
- volatilité récente
- temps restant

Il en déduit :
- `p_up`
- `p_down`

La jambe 1 n’est plus prise seulement parce qu’un côté est “pas cher” :
elle est prise si l’EV implicite est suffisante.

### 3. Exécution plus réaliste
Le moteur applique :
- cap de taille par profondeur visible au meilleur niveau
- slippage additionnel
- fee drag
- spread cost

Le PnL affiché est donc plus conservateur.

### 4. Risk engine
Le moteur bloque de nouvelles entrées si :
- drawdown max atteint
- trop de pertes consécutives
- pause de risque en cours

### 5. Lecture du carnet
La stratégie distingue :
- `mark_sum_yes_no`
- `exec_sum_yes_no`

`mark_sum` = photo du marché  
`exec_sum` = coût réellement exécutable simulé

La décision se fait sur `exec_sum`, pas sur `mark_sum`.

## Frais Polymarket
Polymarket utilise un modèle de frais/maker-rebates qui peut varier selon le token et le rôle maker/taker.  
Voir la documentation officielle :
- Fees
- Fee Rate endpoint
- Maker Rebates

Dans ce moteur, on utilise une approximation conservatrice via `taker_fee_bps`.

## Interprétation des résultats
Même avec cette version plus réaliste, le paper trading reste une simulation.
Il ne modélise pas parfaitement :
- priorité temporelle dans le carnet
- latence réseau réelle
- remplissages partiels multi-niveaux complets
- annulations concurrentes

Les résultats doivent donc être considérés comme une borne haute raisonnablement pénalisée, pas comme du PnL garanti en réel.

## Fichiers à remplacer
- `data_manager.py`
- `cyborg_dash.py`
