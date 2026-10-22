# Lab 3 — Transformers dans l'écosystème HuggingFace : analyse de sentiment avec DistilBERT

> 🇬🇧 An English version of this document is available in [README.md](README.md).

Ce dépôt contient ma solution du troisième laboratoire du cours **Deep Learning Applications**. L'objectif du lab est d'apprendre à adapter un Transformer pré-entraîné à une nouvelle tâche avec l'écosystème HuggingFace (`transformers`, `datasets`, `peft`). La tâche est l'**analyse de sentiment binaire** de critiques de films. Je compare trois façons d'utiliser un **DistilBERT** pré-entraîné, de la moins coûteuse à la plus coûteuse :

1. comme **extracteur de features figé**, suivi d'un classifieur classique (SVM) : la *baseline stable* ;
2. le **fine-tuning complet** de tout le réseau ;
3. le **fine-tuning efficace** avec **LoRA**, qui n'entraîne qu'environ 1 % des paramètres.

---

## Sommaire

- [Structure du dépôt](#structure-du-dépôt)
- [Installation](#installation)
- [Rappels : Transformers, token [CLS] et fine-tuning](#rappels--transformers-token-cls-et-fine-tuning)
- [Exercice 1 — Dataset, modèle pré-entraîné et baseline stable](#exercice-1--dataset-modèle-pré-entraîné-et-baseline-stable)
- [Exercice 2 — Fine-tuning complet de DistilBERT](#exercice-2--fine-tuning-complet-de-distilbert)
- [Exercice 3.1 — Fine-tuning efficace avec LoRA](#exercice-31--fine-tuning-efficace-avec-lora)
- [Récapitulatif des résultats](#récapitulatif-des-résultats)
- [Conclusion](#conclusion)

---

## Structure du dépôt

```
.
├── Lab3-Transformer.ipynb   # Notebook avec tout le code, les expériences et les sorties
├── README.md                # Version anglaise
└── README_FR.md             # Ce fichier (français)
```

Le notebook suit l'ordre des exercices :

| Section | Contenu |
|---|---|
| Exercice 1.1 | Chargement et exploration du dataset Rotten Tomatoes |
| Exercice 1.2 | Chargement de DistilBERT et de son tokenizer, inspection de leurs sorties |
| Exercice 1.3 | Baseline stable : features `[CLS]` + SVM linéaire |
| Exercice 2.1 | Tokenisation du dataset avec `Dataset.map` |
| Exercice 2.2 | Construction du modèle `AutoModelForSequenceClassification` |
| Exercice 2.3 | Fine-tuning complet avec le `Trainer` HuggingFace |
| Exercice 3.1 | Fine-tuning efficace avec LoRA (PEFT) et comparaison finale |

Ce lab ne contient pas de graphiques : tous les résultats sont présentés sous forme de **tableaux**.

## Installation

```bash
pip install torch transformers datasets peft scikit-learn numpy
# optionnel, pour le suivi des expériences :
pip install wandb
```

Un GPU est fortement recommandé pour les exercices 2 et 3 (les temps d'entraînement ci-dessous ont été mesurés sur GPU CUDA avec `fp16=True`).

---

## Rappels : Transformers, token [CLS] et fine-tuning

**DistilBERT** est une version compressée de BERT. Il a été entraîné par *distillation de connaissances*, avec le modèle BERT-base complet comme professeur, sur le même corpus de texte. Il garde 6 couches Transformer au lieu de 12 (dimension cachée 768, ~66 M de paramètres) : il est environ 40 % plus petit et 60 % plus rapide que BERT, tout en conservant l'essentiel de sa compréhension du langage.

**Tokenisation et [CLS].** Le tokenizer découpe une phrase en sous-mots (*tokens*), les convertit en identifiants, et ajoute deux tokens spéciaux : `[CLS]` au **début** et `[SEP]` à la **fin**. Grâce à la self-attention, chaque vecteur de sortie dépend de toute la phrase. Par convention, la sortie du token `[CLS]` dans la dernière couche sert de **résumé de la phrase entière** pour les tâches de classification.

**Trois façons d'adapter un modèle pré-entraîné :**

| Approche | Ce qui est entraîné | Coût |
|---|---|---|
| Extraction de features | Rien dans le Transformer ; seulement un classifieur externe sur ses features figées | Très faible |
| Fine-tuning complet | Tous les poids du Transformer + une nouvelle tête de classification | Élevé |
| LoRA (fine-tuning efficace) | De petites matrices de rang faible ajoutées à certaines couches + la tête de classification ; les poids d'origine restent gelés | Faible |

**LoRA (Low-Rank Adaptation).** Au lieu de mettre à jour une grande matrice de poids $W \in \mathbb{R}^{d \times d}$, LoRA gèle $W$ et apprend une mise à jour de rang faible :

$$W' = W + \frac{\alpha}{r} \, B A, \qquad A \in \mathbb{R}^{r \times d},\; B \in \mathbb{R}^{d \times r},\; r \ll d$$

Avec $d = 768$ et $r = 8$, une matrice 768×768 (589 824 paramètres) est adaptée avec seulement 2 × 8 × 768 = 12 288 paramètres. $B$ est initialisée à zéro, donc l'entraînement part exactement du modèle pré-entraîné.

---

## Exercice 1 — Dataset, modèle pré-entraîné et baseline stable

### 1.1 Objectif : explorer le dataset

**Dataset : Rotten Tomatoes** (Cornell Movie Review Data). Des phrases courtes extraites de critiques de films, étiquetées **positif (1)** ou **négatif (0)**. Il contient 5 331 phrases positives et 5 331 négatives : il est parfaitement équilibré.

J'ai listé les splits disponibles avec `get_dataset_split_names`, puis je les ai chargés avec `load_dataset("rotten_tomatoes")` :

| Split | Nombre de phrases | Colonnes |
|---|---|---|
| train | 8 530 | `text`, `label` |
| validation | 1 066 | `text`, `label` |
| test | 1 066 | `text`, `label` |

Exemple (train[0]) : *« the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger, jean-claud van damme or steven segal. »*, label 1 (positif).

> **Observation importante.** Les splits sont **triés par label** : toutes les phrases positives d'abord, puis toutes les négatives. Un sous-ensemble formé des N *premiers* exemples ne contient donc qu'une seule classe. Dans ma première version de la baseline, je n'utilisais que les 100 premiers exemples de validation, qui étaient tous positifs. L'« accuracy » obtenue mesurait seulement la proportion de prédictions « positif » du SVM. La version finale utilise les **splits complets**.

### 1.2 Objectif : charger DistilBERT et son tokenizer

J'ai chargé `distilbert-base-uncased` avec `AutoTokenizer` et `AutoModel`, tokenisé une phrase et l'ai passée dans le modèle :

| Élément | Résultat |
|---|---|
| Tokens (début … fin) | `['[CLS]', 'the', 'rock', 'is', 'destined']` … `['.', '[SEP]']` |
| Clés de la sortie | `last_hidden_state` |
| Forme de `last_hidden_state` | `(1, 47, 768)` : (batch, nombre de tokens, dimension cachée) |
| Vecteur `[CLS]` | `last_hidden_state[:, 0, :]`, forme `(1, 768)` |

`AutoModel` ne renvoie que les états cachés de la dernière couche : un vecteur de 768 dimensions **par token**. Comme le tokenizer place toujours `[CLS]` en position 0, la représentation de la phrase est le vecteur d'indice 0 sur la dimension des tokens. Cela répond à la question du lab : *« comment savoir quelle sortie correspond au token [CLS] ? »*.

### 1.3 Objectif : construire une baseline stable

**Méthode.**
1. **Extraction des features** avec un `pipeline("feature-extraction")` HuggingFace qui exécute DistilBERT sur le GPU. Pour chaque phrase, le pipeline renvoie un tableau de forme `(1, nb_tokens, 768)` : le vecteur `[CLS]` de la dernière couche est donc `output[0][0]`. Les phrases trop longues sont tronquées (`truncation=True`).
2. Les features des splits train, validation et test **complets** sont **standardisées** (`StandardScaler`, ajusté uniquement sur le train).
3. Un **SVM linéaire** (`LinearSVC`) est entraîné sur les features du train. Le paramètre de régularisation C est choisi sur la **validation**, puis le modèle final est évalué une seule fois sur le **test**.

**Pourquoi cette approche.** Le Transformer n'est pas du tout entraîné : on teste seulement si ses représentations pré-entraînées contiennent déjà l'information de sentiment. C'est peu coûteux, déterministe et facile à reproduire, d'où le nom de *baseline stable*. Un classifieur linéaire suffit pour mesurer à quel point les phrases positives et négatives sont *linéairement séparables* dans l'espace des vecteurs `[CLS]`.

**Résultats : choix de C (accuracy de validation)**

| C | 0.001 | 0.01 | 0.1 | 1.0 |
|---|---|---|---|---|
| Accuracy val | **0.8236** | 0.8208 | 0.8189 | 0.8180 |

**Baseline finale (C = 0.001)**

| Split | Accuracy |
|---|---|
| Validation | **82,36 %** |
| Test | **80,11 %** |

**Analyse.**
- Sans aucun entraînement du Transformer, un simple classifieur linéaire sur le vecteur `[CLS]` atteint déjà **~80 % d'accuracy**. Les représentations apprises par DistilBERT pendant son pré-entraînement (modélisation du langage masqué) encodent déjà une grande partie du sentiment d'une phrase.
- Le meilleur C est la **plus petite** valeur testée : une forte régularisation fonctionne mieux, ce qui est cohérent avec 768 dimensions pour seulement 8 530 exemples. Un C plus grand surapprend légèrement. Comme 0.001 est au bord de la grille, on pourrait tester un C encore plus petit, mais la courbe est très plate (moins de 0,6 point d'écart entre toutes les valeurs).
- L'écart entre validation (82,4 %) et test (80,1 %) vient en partie du fait que C a été choisi sur la validation, et en partie de la petite taille des splits (1 066 phrases, erreur standard ≈ ±1,2 point).
- La limite de cette approche : les features sont **génériques**, elles n'ont pas été optimisées pour le sentiment. C'est ce que le fine-tuning va changer.

---

## Exercice 2 — Fine-tuning complet de DistilBERT

### 2.1 Objectif : tokeniser le dataset

J'ai utilisé `Dataset.map(tokenize_function, batched=True)` pour tokeniser efficacement tous les splits. Le tokenizer est appelé avec `truncation=True, max_length=256` et **sans padding** : le padding est fait plus tard, batch par batch, par le data collator. Cela évite de padder chaque phrase à la longueur de la plus longue phrase du dataset.

| Étape | Colonnes de chaque exemple |
|---|---|
| Après `map` | `text`, `label`, `input_ids`, `attention_mask` (comme demandé par le lab) |
| Après nettoyage | `labels`, `input_ids`, `attention_mask` (`text` supprimée, `label` renommée en `labels` comme l'attend le `Trainer`, format PyTorch) |

### 2.2 Objectif : préparer le modèle pour la classification

`AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)` charge DistilBERT pré-entraîné et ajoute une **nouvelle tête de classification initialisée aléatoirement**, branchée sur le token `[CLS]` :

```
vecteur [CLS] (768) → pre_classifier (768 → 768) → ReLU → dropout → classifier (768 → 2) → logits
```

Le warning *« Some weights … are newly initialized: classifier.*, pre_classifier.* »* confirme que seule la tête est nouvelle. Tout le reste vient du checkpoint pré-entraîné.

### 2.3 Objectif : fine-tuner avec le `Trainer` HuggingFace

**Configuration.**

| Composant | Choix |
|---|---|
| Data collator | `DataCollatorWithPadding` : padde chaque batch à la longueur de sa plus longue phrase |
| Métriques (`compute_metrics`) | accuracy et F1-score, calculés avec scikit-learn à partir de `(logits, labels)` |
| Optimiseur | AdamW (par défaut dans le Trainer), learning rate **2e-5**, weight decay 0.01 |
| Taille de batch | 16 (train et éval) |
| Époques | 3 |
| Précision | `fp16=True` (précision mixte) |
| Évaluation / sauvegarde | à la fin de chaque époque ; le **meilleur checkpoint selon l'accuracy de validation** est rechargé à la fin |
| Paramètres entraînés | **tous** : 66 955 010 |

Un petit learning rate (2e-5) est la norme pour fine-tuner des modèles de type BERT. Les poids pré-entraînés sont déjà bons, et de grandes mises à jour détruiraient ce qui a été appris pendant le pré-entraînement (« oubli catastrophique »).

**Résultats**

| Méthode | Paramètres entraînés | Temps d'entraînement | Accuracy val | Accuracy test |
|---|---|---|---|---|
| SVM sur `[CLS]` (baseline) | — | — | 82,36 % | 80,11 % |
| **Fine-tuning complet** | 66 955 010 | 5,6 min | **86,21 %** | **84,05 %** |

**Analyse.**
- Le fine-tuning améliore l'accuracy de test de **+3,9 points** (80,1 % → 84,1 %) et celle de validation de +3,9 points. Au lieu d'utiliser des features génériques, toutes les couches du Transformer sont adaptées à la tâche de sentiment : la représentation `[CLS]` devient spécialisée.
- Le prix à payer est que **les 67 M de paramètres** sont mis à jour. Il faut stocker les gradients et les états de l'optimiseur pour chaque poids, et sauvegarder une copie complète du modèle pour chaque nouvelle tâche.
- L'accuracy de validation est plus élevée que celle du test parce que le meilleur checkpoint a été **sélectionné** sur la validation : le score de validation est donc légèrement optimiste. Le score de test est l'estimation non biaisée.

---

## Exercice 3.1 — Fine-tuning efficace avec LoRA

### Objectif

Fine-tuner DistilBERT sur Rotten Tomatoes de façon plus *efficace* que le fine-tuning complet, et comparer le compromis coût/performance.

### Méthode

J'ai utilisé **LoRA** avec la bibliothèque **PEFT** de HuggingFace :

| Hyperparamètre | Valeur | Rôle |
|---|---|---|
| `r` | 8 | rang des matrices de mise à jour $A$ et $B$ |
| `lora_alpha` | 16 | facteur d'échelle (la mise à jour est multipliée par α / r = 2) |
| `target_modules` | `q_lin`, `v_lin` | LoRA est appliqué aux projections **query** et **value** de l'attention, dans chacune des 6 couches |
| `lora_dropout` | 0.1 | dropout sur la branche LoRA |
| `modules_to_save` | `pre_classifier`, `classifier` | la nouvelle tête de classification est entièrement entraînée |
| Learning rate | **5e-4** | plus élevé que pour le fine-tuning complet, car beaucoup moins de paramètres sont entraînés |

Le pipeline d'entraînement (`Trainer`, data collator, `compute_metrics`, 3 époques, batch 16, fp16, meilleur checkpoint selon l'accuracy de validation) est **identique** à celui de l'exercice 2 : la comparaison est donc équitable.

**Détails d'implémentation importants :**
- LoRA est appliqué à un modèle `distilbert-base-uncased` **neuf**. Dans ma première version, il était appliqué au modèle déjà entièrement fine-tuné à l'exercice 2, ce qui rendait la comparaison sans valeur.
- Par défaut, avec `task_type="SEQ_CLS"`, PEFT n'entraîne que le module nommé `classifier`. Or la tête de DistilBERT contient aussi une couche `pre_classifier` initialisée aléatoirement, qui serait restée **gelée avec des valeurs aléatoires**. L'ajouter à `modules_to_save` corrige ce problème.
- `label_names=["labels"]` est indiqué explicitement, car PEFT enveloppe le modèle et masque les arguments de sa méthode `forward`.

### Où se trouvent les paramètres entraînés

| Composant | Paramètres entraînés |
|---|---|
| Adaptateurs LoRA (6 couches × 2 matrices (`q_lin`, `v_lin`) × (8×768 + 768×8)) | 147 456 |
| `pre_classifier` (768×768 + 768) | 590 592 |
| `classifier` (768×2 + 2) | 1 538 |
| **Total** | **739 586** (1,09 % du modèle) |

PEFT affiche `all params: 67 694 596` au lieu de 66 955 010, parce qu'il garde une copie gelée de la tête d'origine à côté de la copie entraînable (`modules_to_save`), et qu'il ajoute les matrices LoRA.

### Résultats

| Méthode | Paramètres entraînés | % du modèle complet | Temps d'entraînement | Accuracy val | Accuracy test |
|---|---|---|---|---|---|
| SVM sur `[CLS]` (baseline) | — | — | — | 82,36 % | 80,11 % |
| Fine-tuning complet | 66 955 010 | 100 % | 5,6 min | **86,21 %** | 84,05 % |
| **LoRA (r = 8)** | 739 586 | **1,10 %** | **1,9 min** | 83,77 % | **84,43 %** |

### Analyse

- **Les mêmes performances pour une fraction du coût.** LoRA atteint la même accuracy de test que le fine-tuning complet (84,4 % contre 84,1 %), en n'entraînant que **1,1 %** des paramètres, et l'entraînement est **~3× plus rapide** (1,9 min contre 5,6 min). Il y a moins de gradients et d'états d'optimiseur à calculer et à stocker, ce qui réduit aussi la mémoire GPU utilisée.
- **Les adaptateurs eux-mêmes sont minuscules.** La majorité des paramètres entraînés par LoRA (80 %) appartient à la tête de classification, qui doit de toute façon être entraînée. Les adaptateurs LoRA ne représentent que **147 456 paramètres (0,22 % du modèle)**. Pour adapter DistilBERT à une nouvelle tâche, il suffit de stocker ces adaptateurs et la tête (quelques Mo), au lieu d'une copie complète de 67 M de paramètres.
- **Pourquoi ça marche.** Les modifications de poids nécessaires pour adapter un modèle pré-entraîné à une tâche précise ont un « rang intrinsèque » faible : un petit nombre de directions dans l'espace des paramètres suffit. Adapter les projections query et value permet au modèle de changer *ce à quoi il fait attention* pour la tâche de sentiment, sans modifier le reste du réseau.
- **Les différences entre LoRA et le fine-tuning complet ne sont pas significatives.** LoRA fait 2,4 points de moins en validation, mais 0,4 point de plus en test. Avec 1 066 phrases par split, l'erreur standard est d'environ ±1,1 point, et chaque méthode n'a été exécutée qu'avec une seule seed. La conclusion honnête est que les deux méthodes sont **aussi performantes** l'une que l'autre sur cette tâche.
- **Une autre approche efficace,** suggérée par le lab, qui ne demande pas de changer le pipeline d'entraînement : **geler une partie du réseau**, par exemple les embeddings et les premières couches du Transformer, et ne fine-tuner que les dernières couches et la tête. Cela réduit le nombre de paramètres entraînés et le coût de la rétropropagation, au prix d'une moindre flexibilité.

---

## Récapitulatif des résultats

| Méthode | Ce qui est entraîné | Paramètres entraînés | Temps | Acc val | Acc test |
|---|---|---|---|---|---|
| Features `[CLS]` + SVM linéaire | classifieur externe uniquement | — | — | 82,36 % | 80,11 % |
| Fine-tuning complet | tout le réseau | 66 955 010 (100 %) | 5,6 min | 86,21 % | 84,05 % |
| LoRA (r = 8) | adaptateurs + tête | 739 586 (1,1 %) | 1,9 min | 83,77 % | 84,43 % |

---

## Conclusion

- **Les représentations pré-entraînées sont déjà solides.** Un SVM linéaire sur les vecteurs `[CLS]` figés de DistilBERT atteint 80 % d'accuracy en test sans entraîner le Transformer. C'est une *baseline stable* solide et peu coûteuse. Bien comprendre les sorties du modèle a été essentiel : le token `[CLS]` est en position 0 du dernier état caché. L'organisation du dataset (trié par label) peut complètement fausser une évaluation si l'on n'en prend que les premiers exemples.
- **Le fine-tuning apporte un gain net.** Adapter tout le réseau à la tâche améliore l'accuracy de test d'environ 4 points (84 %). Le `Trainer` HuggingFace, avec `Dataset.map` pour la tokenisation et `DataCollatorWithPadding` pour le padding dynamique, rend ce pipeline court et réutilisable.
- **LoRA offre le meilleur compromis.** En entraînant seulement 1,1 % des paramètres (0,22 % pour les adaptateurs eux-mêmes) et avec un entraînement 3 fois plus court, LoRA égale l'accuracy du fine-tuning complet. Les méthodes de fine-tuning efficaces permettent d'adapter de grands modèles pré-entraînés avec des ressources limitées, et de stocker un petit adaptateur par tâche au lieu d'un modèle complet par tâche.

**Limites et pistes :** chaque configuration a été exécutée avec une seule seed, et les splits sont petits : des différences d'environ 1 point ne sont pas significatives. Plusieurs seeds, une recherche d'hyperparamètres plus large (rang `r`, modules ciblés, learning rate) ou une comparaison avec un gel partiel des couches rendraient la comparaison plus robuste. Les exercices 3.2 (CLIP) et 3.3 n'ont pas été traités.
