# README — ARTEMISIA / CYBORG Coin5min

## Objectif

Cette application pilote une stratégie courte durée sur les marchés crypto Up / Down 5 min de Polymarket.

Le but n'est pas de "prédire le Bitcoin" comme un swing trader.
Le but est de :
1. lire le Price to Beat / ref du marché,
2. estimer un biais court terme sur le spot,
3. entrer une jambe 1 à un bon prix,
4. tenter de compléter avec une jambe 2 à un meilleur prix,
5. ou, si la jambe 2 ne vient pas, gérer le risque d'une jambe seule.

## Différence entre Price to Beat, mark_sum et exec_sum

### Price to Beat / ref
C'est le niveau de référence du marché.
Si le prix final du BTC à l'échéance est au-dessus, le marché résout YES. Sinon il résout NO.

### YES mark / NO mark
Ce sont les prix indicatifs de marché des deux côtés.

### YES exec / NO exec
Ce sont les prix exécutables simulés utilisés par le moteur.

### mark_sum
YES_mark + NO_mark.
Utile pour lire le marché, mais pas suffisant pour décider un trade.

### exec_sum
YES_exec + NO_exec.
C'est la vraie somme utilisée par le moteur pour décider :
- TRADE
- WAIT
- SKIP

## Logique de la stratégie

### Étape 1 — Détection du marché
Le bot cherche les marchés Polymarket crypto :
- BTC
- ETH
- SOL
- XRP
- DOGE

### Étape 2 — Warmup
Le bot charge un historique spot court avant de trader.

### Étape 3 — Calcul du biais
Le moteur calcule :
- tendance court terme
- distance spot / ref
- probabilité simple p_up / p_down

### Étape 4 — Décision
Le moteur produit :
- TRADE
- WAIT
- SKIP

en fonction de :
- exec_sum
- target_sum
- biais
- temps restant
- risque

### Étape 5 — Exécution
Le moteur peut :
- poser une jambe 1
- poser ensuite un hedge
- requoter
- annuler / remplacer
- laisser expirer une jambe si elle n'est pas hedge

## Pourquoi le PnL paper peut être trop beau

Même avec la version réaliste, le paper reste optimiste par rapport au réel.

Causes principales :
- le carnet réel peut bouger entre la décision et l'ordre,
- certains ordres restent pendants longtemps,
- les fills partiels peuvent être pires que prévu,
- le slippage réel peut dépasser le slippage simulé,
- le fee drag réel peut varier selon le marché.

Donc :
- un gros PnL paper est un bon signe,
- mais ce n'est pas encore une preuve de profit réel.

## Ce qui est déjà pris en compte dans cette version

- lecture du Price to Beat / ref
- exécution paper plus réaliste
- slippage simulé
- fee drag simulé
- multi-level fills simulés
- cancel / replace
- pending / requote journal
- stop directionnel simple
- blocage drawdown / loss streak

## Ce qui manque encore pour du vrai réel exchange-grade

- latence réseau mesurée ordre par ordre
- FIFO réel complet dans le carnet
- partial fills intra-tick plus précis
- persistance d'ordres plus robuste
- reprise propre après crash / reboot
- reconciliation réel portefeuille / ordres / positions

## Comment lire l'UI

### Bloc MARCHÉ / EXÉCUTION LIVE
Tu y lis :
- le marché détecté
- le Price to Beat / ref
- le spot live
- le temps restant
- YES mark / NO mark
- YES exec / NO exec
- mark_sum
- exec_sum
- target_sum
- decision
- reason

### Opportunity Ranking
Classe les actifs par opportunité actuelle.

### Pending / Requote Journal
Montre :
- type d'ordre
- side
- target
- dernier prix vu
- statut
- nombre de requotes
- raison de fill / annulation

## Interprétation pratique

### Cas 1 — exec_sum > 1
Pas d'arbitrage propre immédiat.
Le moteur attend ou évite.

### Cas 2 — exec_sum < target_sum
Le moteur peut entrer plus agressivement.

### Cas 3 — jambe 1 ouverte mais pas de hedge
Le moteur :
- requote,
- surveille,
- hedge si possible,
- ou laisse expirer / stoppe selon les règles de risque.

## Recommandation de déploiement

Ne pas passer directement avec taille normale.

Ordre recommandé :
1. paper trading
2. paper avec paramètres réalistes
3. réel très petite taille
4. montée progressive

## Rappel clé

Le Price to Beat / ref n'est pas le YES+NO.

- ref sert à déterminer qui gagne à la fin,
- YES/NO servent à l'exécution de la stratégie,
- exec_sum sert au déclenchement.

C'est la séparation la plus importante du système.
