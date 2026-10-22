# Lab 2 — Deep Reinforcement Learning : REINFORCE, baselines et Lunar Lander

> 🇬🇧 An English version of this document is available in [README.md](README.md).

Ce dépôt contient ma solution du deuxième laboratoire du cours **Deep Learning Applications**. L'objectif du lab est de faire fonctionner des algorithmes de Deep Reinforcement Learning de type *policy gradient*, de mesurer leur stabilité et de les améliorer. En partant d'une implémentation de base de **REINFORCE** sur l'environnement CartPole, j'ai :

1. amélioré la façon dont les performances de l'agent sont évaluées pendant l'entraînement ;
2. ajouté des **baselines** (standardisation des retours et réseau de valeur appris) pour réduire la variance du gradient, et comparé ces variantes de façon équitable sur plusieurs seeds ;
3. appliqué la meilleure version à un environnement plus difficile, **Lunar Lander**.

---

## Sommaire

- [Structure du dépôt](#structure-du-dépôt)
- [Installation](#installation)
- [Rappels : REINFORCE et baselines](#rappels--reinforce-et-baselines)
- [Exercice 1 — Améliorer l'évaluation de REINFORCE](#exercice-1--améliorer-lévaluation-de-reinforce)
- [Exercice 2 — REINFORCE avec une baseline](#exercice-2--reinforce-avec-une-baseline)
- [Exercice 3.1 — Lunar Lander avec REINFORCE](#exercice-31--lunar-lander-avec-reinforce)
- [Conclusion](#conclusion)

---

## Structure du dépôt

```
.
├── DLA-Lab2-DRL.ipynb      # Notebook avec tout le code, les expériences et les sorties
├── README.md               # Version anglaise
├── README_FR.md            # Ce fichier (français)
└── images/                 # Graphiques utilisés dans ce README
    ├── ex1_reinforce_run1.png
    ├── ex1_reinforce_run2.png
    ├── ex2_value_baseline_single_run.png
    ├── ex2_baselines_comparison.png
    └── ex3_lunarlander.png
```

Le notebook suit l'ordre des exercices :

| Section | Contenu |
|---|---|
| Exercice 1 | Exploration de CartPole, implémentation REINFORCE fournie, nouveau protocole d'évaluation (récompense moyenne et longueur des épisodes tous les N épisodes) |
| Exercice 2 | REINFORCE avec la baseline de standardisation, puis avec une baseline apprise (fonction de valeur) ; comparaison équitable des trois variantes sur 3 seeds |
| Exercice 3.1 | REINFORCE avec baseline valeur sur LunarLander, évaluation stochastique et greedy |

## Installation

```bash
pip install torch numpy matplotlib "gymnasium[box2d]"
```

Le notebook utilise `CartPole-v1` et `LunarLander-v2`. Dans les versions récentes de Gymnasium (≥ 1.0), Lunar Lander s'appelle `LunarLander-v3`.

---

## Rappels : REINFORCE et baselines

Dans les méthodes *policy gradient*, un réseau de neurones produit directement une **politique** π(a | s, θ) : une distribution de probabilité sur les actions pour chaque état. On cherche à maximiser le retour espéré J(θ). **REINFORCE** est l'algorithme le plus simple de cette famille :

1. jouer un épisode complet avec la politique actuelle, en mémorisant les états, les actions et les récompenses ;
2. calculer, pour chaque pas de temps, le retour actualisé $G_t = \sum_{k \ge t} \gamma^{k-t} r_k$ ;
3. mettre à jour la politique dans la direction qui rend les bonnes actions plus probables :

$$\theta \leftarrow \theta + \alpha \sum_t G_t \, \nabla_\theta \log \pi(A_t \mid S_t, \theta)$$

En pratique, on minimise la loss $-\sum_t G_t \log \pi(A_t \mid S_t)$.

**Le problème : la variance.** $G_t$ est estimé à partir d'un seul épisode tiré au hasard, il est donc très bruité. Deux épisodes joués avec la même politique peuvent donner des retours très différents. Le gradient a donc une très forte variance, ce qui rend l'apprentissage lent et instable.

**La solution : une baseline.** On peut soustraire à $G_t$ n'importe quelle fonction $b(s)$ qui ne dépend pas de l'action, sans biaiser le gradient :

$$\theta \leftarrow \theta + \alpha \sum_t \big(G_t - b(S_t)\big) \, \nabla_\theta \log \pi(A_t \mid S_t, \theta)$$

Intuitivement, l'agent ne se demande plus « le retour était-il élevé ? » mais « le retour était-il **meilleur que prévu** ? ». Les actions qui ont fait mieux que prévu sont renforcées, les autres sont découragées. Cela réduit fortement la variance du gradient. Dans ce lab, je compare deux baselines :

- **Standardisation** (constante sur l'épisode) : $\frac{G_t - \bar G}{\sigma_G}$, où la moyenne et l'écart-type sont calculés sur l'épisode.
- **Fonction de valeur** (dépend de l'état) : $b(s) = \hat v(s \mid w)$, un second réseau de neurones entraîné à prédire le retour espéré depuis l'état $s$.

---

## Exercice 1 — Améliorer l'évaluation de REINFORCE

### Objectif

Faire fonctionner REINFORCE sur CartPole et remplacer l'évaluation d'origine (une moyenne glissante des scores d'entraînement) par un protocole plus fiable : tous les **N** épisodes d'entraînement, exécuter l'agent pendant **M** épisodes séparés et mesurer (1) la **récompense totale moyenne** et (2) la **longueur moyenne des épisodes**.

### Environnement : CartPole-v1

Un bâton est fixé sur un chariot qui se déplace sur un rail. L'agent doit maintenir le bâton en équilibre.
- **Observation** (4 valeurs) : position du chariot, vitesse du chariot, angle du bâton, vitesse angulaire du bâton.
- **Actions** (2) : pousser le chariot à gauche ou à droite.
- **Récompense** : +1 à chaque pas de temps où le bâton reste debout. L'épisode se termine quand le bâton tombe ou que le chariot sort du rail (`terminated`), ou après **500 pas** (`truncated`). Le score maximal est donc 500.

J'ai d'abord exploré l'environnement (`reset()`, `observation_space`, `action_space`, quelques `step()` aléatoires) pour comprendre le format des observations et la différence entre `terminated` et `truncated`.

### Modèle et méthode

- **Réseau de politique** : un petit MLP, 4 → 128 (ReLU) → 2, suivi d'un softmax qui donne la probabilité de chaque action. Les actions sont **échantillonnées** selon cette distribution, ce qui assure l'exploration.
- **Entraînement** : l'implémentation REINFORCE fournie avec le lab (qui standardise déjà les retours dans chaque épisode), optimiseur Adam avec lr = 0.01, γ = 0.99, 1000 épisodes.
- **Nouvelle évaluation** : `evaluate_policy(env, policy, M)` exécute M épisodes sous `torch.no_grad()` et retourne la récompense totale moyenne et la longueur moyenne. `reinforce(...)` l'appelle tous les `eval_interval` = N épisodes et retourne trois listes : les scores d'entraînement, les récompenses d'évaluation et les longueurs d'évaluation.

**Pourquoi cette évaluation est meilleure.** Le score d'entraînement d'un seul épisode est très bruité, et une moyenne glissante mélange d'anciennes versions de la politique. Évaluer la politique *actuelle* sur M épisodes donne une image plus fidèle de ses performances réelles à ce moment-là.

### Résultats

<!-- Graphique : premier entraînement de REINFORCE sur CartPole (lr = 0.01, N = 10, M = 10) -->
![REINFORCE sur CartPole — run 1](images/ex1_reinforce_run1.png)

<!-- Graphique : second entraînement avec la même configuration (N = 50) -->
![REINFORCE sur CartPole — run 2](images/ex1_reinforce_run2.png)

### Analyse

- Sur CartPole, la récompense vaut +1 par pas, donc **la récompense moyenne et la longueur moyenne sont identiques** (les deux graphiques de droite sont les mêmes). La longueur ne devient une métrique informative que dans des environnements où la récompense varie, comme Lunar Lander.
- Les nouvelles métriques révèlent clairement l'**instabilité de REINFORCE** avec un learning rate élevé :
  - **Run 1** : l'agent atteint ~250, **s'effondre à ~10** entre les épisodes 200 et 370, atteint le maximum de 500 vers l'épisode 550, s'effondre de nouveau et termine vers 100.
  - **Run 2**, avec la même configuration : la performance oscille entre ~70 et ~240 sans jamais se stabiliser, et termine à 89.
  - Lors d'une exécution précédente du même code, l'agent terminait à 500.
- Avec **un seul run, le résultat final dépend essentiellement du hasard**. Avec lr = 0.01 et un estimateur de gradient à forte variance, quelques mauvaises mises à jour suffisent à détruire une bonne politique. Comme les méthodes policy gradient sont *on-policy*, une mauvaise politique collecte de mauvaises données, ce qui ralentit le retour à une bonne politique.

Ces observations motivent les deux améliorations suivantes : **réduire la variance du gradient avec une baseline**, et **comparer les méthodes sur plusieurs seeds** plutôt que sur un seul run.

---

## Exercice 2 — REINFORCE avec une baseline

### Objectif

Réduire la variance de REINFORCE en soustrayant une baseline au retour, et comparer trois variantes sur CartPole : sans baseline, avec la baseline de standardisation, et avec une baseline apprise (fonction de valeur).

### Modèles et méthode

Les trois variantes sont implémentées dans **une seule fonction**, `reinforce(..., baseline="none" | "std" | "value")`. Tout le reste est identique (réseaux, optimiseur, γ, nombre d'épisodes, évaluation), donc la comparaison est équitable.

- **Réseau de politique** : 4 → 128 (ReLU) → 2 + softmax (comme à l'exercice 1).
- **Réseau de valeur** $\hat v(s \mid w)$ : **la même architecture** que la politique, mais avec une seule sortie : 4 → 128 (ReLU) → 1.
- **Deux optimiseurs Adam séparés** (lr = 1e-3 chacun). La loss de valeur et la loss de politique ont des échelles très différentes : en les additionnant avec un seul optimiseur, la loss de valeur dominerait.

Calcul de la cible (l'« avantage ») pour chaque variante :

| Variante | Cible utilisée dans la loss de politique |
|---|---|
| `none` | retour brut $G_t$ |
| `std` | $(G_t - \bar G) / \sigma_G$, calculé sur l'épisode |
| `value` | $A_t = G_t - \hat v(S_t)$, puis normalisé |

**Entraînement du réseau de valeur.** Après chaque épisode, $\hat v$ est mis à jour avec une loss MSE vers les retours actualisés **bruts** : $\mathcal{L}_v = \frac{1}{T}\sum_t (\hat v(S_t) - G_t)^2$.

> **Détail d'implémentation important.** Ma première implémentation entraînait le réseau de valeur sur des retours *déjà standardisés* par épisode. C'est une erreur : après standardisation, chaque épisode a une moyenne de 0 et un écart-type de 1, donc le même état reçoit une cible différente selon la durée de l'épisode. Le réseau ne pouvait pas apprendre $v(s)$ et finissait par prédire ≈ 0, ce qui rendait la « baseline valeur » presque identique à la simple standardisation. La version corrigée entraîne $\hat v$ sur les retours bruts et normalise seulement l'avantage $G_t - \hat v(S_t)$.

La valeur de la baseline est **détachée** du graphe de calcul (`values.detach()`) au moment de calculer l'avantage, pour que le gradient de la politique ne remonte pas dans le réseau de valeur.

### Résultats

**Un seul run avec la baseline valeur** (1000 épisodes, évaluation tous les 50 épisodes sur 10 épisodes) :

<!-- Graphique : REINFORCE avec baseline valeur, un seul run -->
![CartPole — baseline valeur, un seul run](images/ex2_value_baseline_single_run.png)

L'agent progresse régulièrement jusqu'à ~400, connaît deux chutes (vers les épisodes 550 et 650), puis atteint 460-500 à partir de l'épisode 750. La moyenne glissante des scores d'entraînement reste autour de 450-470 à la fin.

**Comparaison équitable des trois variantes** (3 seeds : 0, 1, 2 ; 1000 épisodes ; lr = 1e-3 ; évaluation tous les 50 épisodes sur 10 épisodes ; les courbes montrent la moyenne ± l'écart-type entre les seeds) :

<!-- Graphique : comparaison des trois baselines sur 3 seeds -->
![CartPole — comparaison des baselines](images/ex2_baselines_comparison.png)

| Méthode | Récompense finale d'évaluation (moyenne ± écart-type sur 3 seeds) |
|---|---|
| Sans baseline | 442,8 ± 35,2 |
| Standardisation | **496,5 ± 4,9** |
| Baseline valeur $\hat v(s)$ | 451,3 ± 68,9 |

### Analyse

- **Les trois variantes résolvent CartPole** (≥ 440 en moyenne après 1000 épisodes). Avec lr = 1e-3, même la version sans baseline apprend. Le learning rate de 0.01 utilisé à l'exercice 1 était donc une cause majeure des effondrements observés.
- **Les baselines accélèrent l'apprentissage.** Entre les épisodes 450 et 800, les deux variantes avec baseline sont nettement au-dessus de la version sans baseline. Vers l'épisode 600, la version sans baseline est autour de 300, contre ~400-450 avec une baseline. C'est l'effet attendu : soustraire une baseline réduit la variance du gradient sans le biaiser.
- **La standardisation donne le meilleur résultat final et le plus stable** (496,5 ± 4,9).
- **La baseline valeur ne fait pas mieux que la standardisation ici.** Plusieurs raisons l'expliquent :
  1. avec seulement 3 seeds, les zones d'écart-type se chevauchent largement, donc les différences **ne sont pas statistiquement significatives** ;
  2. CartPole est un environnement simple, avec une récompense constante à chaque pas : une baseline constante par épisode capture déjà l'essentiel de ce qu'une baseline dépendant de l'état pourrait apporter ;
  3. dans mon implémentation, l'avantage $G_t - \hat v(S_t)$ est aussi normalisé, ce qui rapproche les deux méthodes ;
  4. le réseau de valeur part de poids aléatoires et doit lui-même apprendre pendant que la politique change : ses estimations sont donc mauvaises au début de l'entraînement.
- On s'attend à ce qu'une baseline dépendant de l'état apporte davantage dans des environnements où la valeur des états varie beaucoup, comme Lunar Lander. C'est pourquoi je l'utilise à l'exercice 3.1.

---

## Exercice 3.1 — Lunar Lander avec REINFORCE

### Objectif

Appliquer REINFORCE à un environnement plus difficile, **LunarLander**, et analyser les performances de l'agent avec les mêmes métriques qu'avant.

### Environnement : LunarLander-v2

L'agent contrôle un module lunaire et doit le poser en douceur sur une plateforme d'atterrissage, entre deux drapeaux.
- **Observation** (8 valeurs) : position (x, y), vitesses, angle, vitesse angulaire, et deux booléens qui indiquent si chaque pied touche le sol.
- **Actions** (4) : ne rien faire, allumer le moteur gauche, allumer le moteur principal, allumer le moteur droit.
- **Récompense** : récompense « façonnée ». L'agent gagne des points quand il se rapproche de la plateforme et ralentit, +10 par pied en contact avec le sol, **+100 pour un atterrissage** et **−100 pour un crash**. Allumer les moteurs coûte un peu à chaque pas (carburant). L'épisode se termine après un atterrissage, un crash, ou **1000 pas**.
- L'environnement est considéré comme **résolu** à partir d'un score moyen de **200**.

### Modèle et méthode

J'ai réutilisé l'implémentation REINFORCE + **baseline valeur** de l'exercice 2, sans modification à part les dimensions d'entrée et de sortie (8 → 128 → 4 pour la politique, 8 → 128 → 1 pour le réseau de valeur) :
- 3000 épisodes d'entraînement, γ = 0.99, lr = 1e-3 pour les deux réseaux ;
- évaluation tous les 100 épisodes sur 10 épisodes (récompense moyenne et longueur moyenne) ;
- **évaluation finale** sur 50 épisodes, avec deux politiques :
  - **stochastique** : actions échantillonnées selon π, comme pendant l'entraînement ;
  - **greedy** : toujours l'action la plus probable (argmax).

J'ai choisi la baseline valeur parce que, contrairement à CartPole, les états de Lunar Lander ont des valeurs très différentes : un module proche de la plateforme vaut beaucoup plus qu'un module penché à haute altitude. Une baseline qui dépend de l'état est donc bien adaptée.

### Résultats

<!-- Graphique : LunarLander — scores d'entraînement, récompense et longueur d'évaluation -->
![LunarLander — REINFORCE avec baseline valeur](images/ex3_lunarlander.png)

| Épisode | Récompense moyenne (éval.) | Longueur moyenne (éval.) |
|---|---|---|
| 100 | −121,9 | 110,8 |
| 800 | −20,8 | 448,7 |
| 2000 | 105,1 | 612,7 |
| 2200 | **149,6** (meilleur) | 763,6 |
| 3000 | 121,2 | 847,8 |

**Évaluation finale (50 épisodes) :**

| Politique | Récompense moyenne | Longueur moyenne |
|---|---|---|
| Stochastique | **138,3** | 1000,0 |
| Greedy (argmax) | 81,9 | 996,1 |

### Analyse

- **L'agent apprend clairement.** La récompense d'évaluation passe d'environ −130 (crashs fréquents) à 120-150, et la moyenne glissante des scores d'entraînement monte de −150 à ~130. L'essentiel de la progression a lieu entre les épisodes 500 et 1600, suivi d'un plateau.
- **L'environnement n'est pas résolu** (seuil de 200). La performance plafonne autour de 120-150 à partir de l'épisode ~1600.
- **La longueur des épisodes explique pourquoi.** Elle passe de ~110 pas (le module s'écrase rapidement) à 850-1000 pas, et lors de l'évaluation finale stochastique, **tous les épisodes durent exactement 1000 pas**, c'est-à-dire la limite de temps. L'agent a appris à **planer** : il évite la pénalité de crash (−100) et récolte la récompense de façonnage en restant près de la plateforme, mais il n'atterrit jamais et n'éteint jamais ses moteurs. Il ne touche donc jamais le bonus d'atterrissage (+100), et continue de payer le coût du carburant. C'est un **optimum local** classique de REINFORCE sur Lunar Lander. Planer est une stratégie « sûre » et facile à trouver, alors qu'apprendre à atterrir demande d'explorer des comportements risqués près du sol, où une erreur est fortement pénalisée.
- **La politique greedy est moins bonne que la politique stochastique** (81,9 contre 138,3). La politique apprise n'est pas assez « tranchée » pour être utilisée de façon déterministe : choisir toujours l'action la plus probable mène à des trajectoires répétitives, alors que l'aléa de la politique stochastique l'aide à corriger sa trajectoire. Cela montre aussi que REINFORCE a appris une politique qui s'appuie sur son propre bruit.

**Pistes d'amélioration :** entraîner plus longtemps ; ajouter un **bonus d'entropie** pour continuer à explorer ; ajuster γ et le learning rate ; faire les mises à jour sur des lots de plusieurs épisodes pour réduire encore la variance ; ou utiliser une méthode *off-policy* avec replay buffer, comme le **Deep Q-Learning** (exercice 3.2), qui réutilise l'expérience passée et utilise en général les données plus efficacement.

---

## Conclusion

Ce lab a montré concrètement pourquoi le Deep Reinforcement Learning est réputé **difficile à entraîner de façon stable** :

- **L'évaluation compte.** Évaluer la politique actuelle tous les N épisodes sur M épisodes (exercice 1) a révélé que REINFORCE avec un learning rate élevé peut atteindre le score maximal puis s'effondrer complètement. Un seul run ne suffit pas pour conclure : il faut comparer les méthodes **sur plusieurs seeds**.
- **Les baselines réduisent la variance.** Soustraire une baseline au retour (exercice 2) rend l'apprentissage plus rapide et plus stable sur CartPole. La standardisation a donné le meilleur résultat et le plus stable, et la baseline valeur apprise était comparable. Sur un environnement aussi simple, la différence entre les deux n'est pas significative sur 3 seeds. Les détails d'implémentation comptent beaucoup : le réseau de valeur doit être entraîné sur les retours **bruts**, avec son propre optimiseur, et sa sortie doit être **détachée** pour calculer l'avantage.
- **REINFORCE a ses limites sur des tâches plus difficiles.** Sur Lunar Lander (exercice 3.1), REINFORCE avec baseline valeur a appris une politique raisonnable (≈ 140 de récompense moyenne), mais s'est retrouvé bloqué dans un optimum local où il **plane** sans atterrir, et n'a pas atteint le seuil de 200. Analyser la **longueur des épisodes**, et pas seulement la récompense, a été essentiel pour comprendre ce comportement.

Au final, REINFORCE est simple et fonctionne sur des problèmes faciles. Mais sa forte variance et sa nature *on-policy* le rendent sensible aux hyperparamètres et sujet aux optima locaux. Cela motive des méthodes plus avancées : méthodes actor-critic, PPO, ou algorithmes off-policy avec replay buffer et réseau cible comme DQN.
