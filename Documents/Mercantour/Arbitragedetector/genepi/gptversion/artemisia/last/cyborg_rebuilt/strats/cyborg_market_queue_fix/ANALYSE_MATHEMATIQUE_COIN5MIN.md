# Analyse mathématique — Stratégie Coin5min CYBORG

## 1. Principe fondamental : l'arbitrage YES + NO

Sur Polymarket, chaque marché binaire 5 minutes a deux tokens : **YES** et **NO**.

À l'expiration, exactement un des deux vaut $1, l'autre vaut $0.

Donc si tu achètes YES à `p_yes` et NO à `p_no`, ton coût total est :

```
C = p_yes + p_no
```

Et ton gain garanti à l'expiration est toujours **$1** (quel que soit le résultat).

**Le profit théorique par dollar investi est :**

```
edge = 1 - C = 1 - (p_yes + p_no)
```

**Condition nécessaire pour un trade profitable :**

```
C < 1   ⟹   p_yes + p_no < 1
```

C'est le cœur de ta stratégie : trouver des moments où `exec_sum < 1`.

---

## 2. Mark sum vs Exec sum — la distinction critique

### Mark sum (indicatif)

```
mark_sum = YES_mark + NO_mark
```

C'est ce que le marché affiche. Pas ce que tu paies réellement.

### Exec sum (exécutable)

```
exec_sum = YES_best_ask + NO_best_ask
```

C'est le vrai coût d'achat en frappant les deux asks. **Toutes les décisions se font sur exec_sum, pas mark_sum.**

### Relation typique

```
exec_sum ≥ mark_sum   (toujours, à cause du spread)
```

Le spread entre bid et ask crée un surcoût invisible si tu regardes seulement le mark.

---

## 3. Coûts réels d'exécution

Le profit brut `1 - exec_sum` est ensuite réduit par trois coûts :

### 3a. Frais taker (fee drag)

```
fee_usd = size_usd × (taker_fee_bps / 10000)
```

Avec `taker_fee_bps = 30` (valeur par défaut), sur un trade de $3 :

```
fee = 3 × 30/10000 = $0.009 par jambe → $0.018 pour les deux
```

### 3b. Slippage (impact de marché)

```
slippage_usd = (exec_price_ajusté - exec_price_brut) × (size / exec_price)
```

Le moteur simule un slippage additionnel de `extra_slippage_bps = 8` bps.

### 3c. Spread cost

```
spread_cost = spread × 0.20
```

Le moteur ne prend que 20% du spread comme coût car on utilise des ordres limit, pas des ordres market.

### Profit net réel

```
profit_net = (1 - exec_sum) × size_usd  -  fee_yes - fee_no  -  slippage_yes - slippage_no
```

Pour que le trade soit vraiment profitable :

```
exec_sum < 1 - 2×(taker_fee_bps/10000) - 2×(extra_slippage_bps/10000)

Numériquement: exec_sum < 1 - 0.006 - 0.0016 = 0.9924
```

C'est pour ça que le paramètre `target_sum_yes_no = 0.985` est conservateur — il laisse de la marge pour ces coûts.

---

## 4. Estimation de la probabilité directionnelle

Le moteur calcule `p_up` (probabilité que le spot finisse au-dessus du ref) via un score sigmoïde :

```
dist_pct = (spot - ref) / ref
trend_bps = (prix_fin / prix_début - 1) × 10000
vol_bps = écart-type des rendements tick-to-tick × 10000

z = (dist_pct × 1800 + trend_bps / max(6, vol_bps × 1.5)) × time_factor

p_up = σ(z) = 1 / (1 + e^(-z))
p_down = 1 - p_up
```

Où `time_factor` :
- `1.2` si < 60s restantes (momentum plus fiable)
- `0.85` si > 240s (trop de temps, reversion possible)
- `1.0` sinon

### Interprétation

- Si le spot est **au-dessus** du ref et que la tendance monte → `z > 0` → `p_up > 0.5`
- Le vol_bps au dénominateur **normalise** : une tendance de +10bps dans un marché qui bouge de ±20bps est moins significative que +10bps dans un marché à ±3bps

---

## 5. Score d'opportunité (decision engine)

Le moteur calcule un score composite pour chaque actif :

```
opp_score = 0.45 × sum_score + 0.15 × trend_score + 0.10 × spread_score + 0.15 × time_score + 0.15 × prob_edge
```

### Composantes

**sum_score** — qualité du prix :
```
si exec_sum ≤ target_sum → 1.0
si exec_sum ≤ wait_cap   → interpolation linéaire [1.0, 0.0]
sinon                     → 0.0
```

**trend_score** — force de la tendance :
```
trend_score = min(1.0, |trend_bps| / 20.0)
```

**spread_score** — qualité de l'exécution :
```
spread_proxy = |yes_exec - yes_mark| + |no_exec - no_mark|
spread_score = max(0, 1 - spread_proxy / 0.08)
```

**time_score** — temps restant :
```
≤ 30s (no_new_entry_last_sec)  → 0.0 (bloqué)
< 45s (min_entry_time_left)     → 0.0 (trop tard)
[60s, 180s]                     → 0.70
≥ 180s                          → 1.0
```

**prob_edge** — avantage directionnel :
```
prob_edge = max(|p_up - 0.5|, |p_down - 0.5|) × 2
```

### Décision

```
TRADE  si exec_sum ≤ target_sum ET prob_edge ≥ min_prob_edge (0.06)
WAIT   si exec_sum < 1.0 mais pas assez bon
SKIP   si exec_sum ≥ 1.0 ou temps épuisé
```

---

## 6. Legging : jambe 1 puis hedge

Si le prix n'est pas assez bon pour acheter YES + NO simultanément, le moteur peut entrer en **legging** :

### EV de chaque jambe

```
EV_yes = p_up - yes_exec_price
EV_no  = p_down - no_exec_price
```

Le moteur choisit la jambe avec le meilleur EV ajusté par le biais directionnel :

```
score_yes = EV_yes + (0.02 si bias = "above")
score_no  = EV_no  + (0.02 si bias = "below")
```

### Trigger de hedge

```
hedge_trigger = max(0.01, 1.0 - first_price - target_edge)
```

Exemple : si tu achètes YES à 0.42, avec target_edge = 0.015 :
```
hedge_trigger = 1.0 - 0.42 - 0.015 = 0.565
→ le moteur attend NO ≤ 0.565 pour compléter
→ coût total verrouillé = 0.42 + 0.565 = 0.985
```

### Risque du legging

Si le hedge ne vient pas avant `max_leg_hold_sec` (45s) ou l'expiration :
```
net_pnl_leg1 = pnl_resolution - fee - slippage - 0.15 (pénalité adverse selection)
```

La pénalité de 0.15 reflète le fait qu'une jambe seule est très risquée.

---

## 7. Risk engine

### Drawdown max
```
equity = capital + total_pnl - fee_drag - slippage_drag
peak_equity = max(peak_equity, equity)
drawdown_pct = (peak_equity - equity) / peak_equity × 100

Si drawdown_pct ≥ max_daily_drawdown_pct (8%) → pause 30min
```

### Pertes consécutives
```
Si consecutive_losses ≥ max_consecutive_losses (6) → pause 30min
```

---

## 8. Cap de taille par profondeur

```
max_usd = level_size × price × depth_fill_ratio

size_réel = min(size_demandé, max_usd)
```

Avec `depth_fill_ratio = 0.5`, le moteur ne prend jamais plus de 50% de la liquidité visible au meilleur niveau. Ça évite de vider le carnet et subir du slippage catastrophique.

---

## 9. MON AVIS CONSTRUIT

### Ce qui est bien fait ✓

**L'intuition mathématique est correcte.** Acheter YES + NO < $1 sur un marché binaire est un arbitrage structurellement valide. C'est la même logique qu'un market maker qui achète les deux côtés du spread.

**La distinction mark_sum / exec_sum** est critique et bien implémentée. Beaucoup de bots paper-trade sur les marks et découvrent en live que le vrai coût est plus élevé.

**Le risk engine** avec drawdown max + streak de pertes est une bonne pratique. Sans ça, une série de slippage kills te ruine avant de t'en rendre compte.

**Le legging directionnel** est intelligent — entrer sur la jambe qui a un EV positif attendu, puis hedge quand l'autre côté devient accessible. Ça élargit la fenêtre d'opportunité.

### Points de vigilance ⚠

**1. Le modèle de probabilité est très simple.**
La sigmoïde sur `dist_pct × 1800 + trend_bps / vol_bps` est un raccourci. En réalité, le prix BTC sur 5 minutes suit approximativement un mouvement brownien géométrique avec drift micro-structurel. La "tendance court terme" mesurée sur 20 minutes de ticks est très bruitée. Sur 5 minutes, la variance domine largement le drift.

**Mon avis :** le p_up/p_down est probablement calibré trop confiant quand la tendance est forte. En réalité, un trend_bps de +15 sur 20 minutes ne prédit que très faiblement les 5 prochaines minutes. Je recommanderais de **diviser time_factor par 2** ou d'ajouter un shrinkage vers 0.5 :

```
p_up_adjusted = 0.7 × p_up_modele + 0.3 × 0.5
```

**2. Le legging single-leg est le risque principal.**
Quand tu es en leg1 seule, tu as essentiellement un pari directionnel pur à 0.42-0.65 cents par share. La pénalité de 0.15 dans le paper ne reflète pas complètement le risque réel. En vrai :
- Si tu achètes YES à 0.42 et que le prix finit en-dessous du ref → tu perds 0.42 (100% de la mise)
- L'espérance sur un coin flip non biaisé serait -0.08 par share

**Mon avis :** commencer avec `legging_enabled = False` en live. Ne faire que des trades simultanés YES+NO au début. Activer le legging seulement après avoir validé que le bot trouve suffisamment de fenêtres `exec_sum < target_sum`.

**3. Le timing 5 minutes est difficile.**
Les marchés crypto 5 minutes ont un cycle de vie très court. Entre la détection du marché, le warmup, la lecture du carnet, et le placement : il reste souvent < 3 minutes utiles. Le réseau ajoute ~200-500ms par appel API. Sur un marché qui bouge vite, 3 requêtes = 1.5s = le carnet a déjà changé.

**Mon avis :** en live, utiliser des ordres **limit postOnly** plutôt que des ordres marketables. Ça te donne le rebate maker au lieu du fee taker, et surtout ça évite le slippage si le carnet bouge pendant l'envoi.

**4. Le fee model est approximatif.**
Le `taker_fee_bps = 30` est une approximation. Polymarket a un modèle de fees avec des maker rebates variables. En tant que **maker** (ordre limit non marketable), tu reçois un rebate. En tant que **taker** (ordre marketable), tu paies des frais. La différence peut être de 4-6 cents par dollar.

**Mon avis :** vérifier les vrais frais sur les premiers trades live et ajuster `taker_fee_bps` en conséquence.

**5. La sélection de marché par slug/regex est fragile.**
Si Polymarket change le format des slugs ou des questions, le bot ne trouvera plus les marchés. Il y a beaucoup de fallback (slug, search, pagination), mais ça ajoute de la latence.

### Recommandation globale

**Phase 1 (semaine 1-2) :**
- `exec_mode = "live"`, `max_order_usd = 1.0`
- `legging_enabled = False` (simultané uniquement)
- Observer le taux de fill, le slippage réel vs simulé, les rejets
- Comparer paper PnL vs live PnL sur les mêmes trades

**Phase 2 (semaine 3-4) :**
- Ajuster `taker_fee_bps` sur les données réelles
- Si le taux de fill simultané est > 70%, passer à `max_order_usd = 3.0`
- Activer `legging_enabled = True` avec `max_leg_hold_sec = 20` (réduit vs 45)

**Phase 3 (mois 2+) :**
- Monter progressivement si le PnL réel est positif
- Ajouter des ordres postOnly pour capturer le maker rebate
- Mesurer la latence ordre par ordre et l'intégrer au modèle

### Estimation réaliste de rendement

Avec `target_sum = 0.985` et des frais réels ~0.3-0.5% :
- Edge brut par trade: ~1.5%
- Edge net après frais/slippage: ~0.5-0.8%
- Si 20 trades/jour à $3 : profit attendu ~$0.30-$0.48/jour
- Ce n'est pas un get-rich-quick — c'est un edge micro-structurel qui nécessite du volume et de la régularité

**Le modèle est viable à petite échelle comme exercice d'apprentissage et de validation.** Pour que ça devienne économiquement intéressant, il faudrait monter en taille ($50-100 par trade) et en fréquence (50+ trades/jour), ce qui nécessite une infrastructure beaucoup plus robuste (WebSocket pour le carnet, gestion FIFO réelle, reconciliation automatique).
