# CYBORG – Documentation des stratégies

## 1. BetMiss

### Type
Arbitrage **systématique** cross-platform.

### Univers
Tous types de marchés si une correspondance existe entre **Polymarket** et **Kalshi** :
- sports
- politique
- macro
- événements divers

### Règle
On cherche un événement identique ou quasi identique sur les deux plateformes.

Si :

```text
YES_A + NO_B < 1
```

alors on a un edge théorique :

```text
edge = 1 - (YES_A + NO_B)
```

### Paramètres
- **Seuil edge min** : edge minimum pour prendre l'opportunité
- **Taille / trade** : pourcentage de l'enveloppe BetMiss utilisé par arbitrage

### Risques / limites
- matching sémantique imparfait
- liquidité
- latence
- frais non entièrement modélisés

---

## 2. Coin5min

### Type
Arbitrage **systématique intra-market** sur les marchés **Up / Down** crypto de Polymarket.

### Univers
Marchés :
- **5 min**
- **15 min**

Actifs suivis :
- BTC
- ETH
- SOL
- XRP
- DOGE

### Règle principale
On cherche :

```text
Above / YES + Below / NO < 1
```

Ce qui donne :

```text
edge = 1 - (YES + NO)
```

### Interprétation
- **Above / YES** : le prix final est au-dessus du prix de référence
- **Below / NO** : le prix final est en dessous du prix de référence

### Paramètres
- **Seuil edge min (YES+NO)**
- **Taille / trade**
- **Intervalle 5 min / 15 min**
- **Actifs suivis**

### Warmup
Warmup court pour :
- vérifier la connexion spot,
- récupérer un petit historique récent,
- stabiliser le suivi graphique.

### Affichage live
L'interface montre :
- prix spot
- prix de référence
- prix YES / NO
- somme YES+NO
- edge
- carnet YES / NO
- état du warmup
- trades simulés

### Limites
Le prix de référence affiché est un proxy opérationnel pour le paper trading GUI ; ce n'est pas encore un settlement officiel branché de bout en bout.
