# 🌋 INGV — Volcanic Eruption Prediction

**Prédiction du temps restant avant une éruption volcanique à partir de signaux sismiques**  
*Projet universitaire de Machine Learning — Python · Pandas · Scikit-learn · XGBoost*

## Présentation

Ce projet s'appuie sur les données de la compétition Kaggle **INGV — Volcanic Eruption Prediction**, proposée en lien avec l'Istituto Nazionale di Geofisica e Vulcanologia (INGV).

L'objectif est de construire un modèle de **régression supervisée** capable d'estimer le **temps restant avant une éruption** (`time_to_eruption`, en secondes), à partir de mesures issues de **10 capteurs sismiques**.

Le principal défi consiste à transformer des signaux bruts et volumineux en variables exploitables par un algorithme de Machine Learning. Le jeu de données utilisé représente **plus de 30 Go** de fichiers, avec des contraintes de chargement et de mémoire.

## Données

Chaque segment sismique est enregistré dans un fichier CSV distinct, identifié par un `segment_id` :

- **`train.csv`** : identifiants des segments d'entraînement et valeur cible `time_to_eruption` ;
- **`train/`** : mesures des 10 capteurs pour chaque segment d'entraînement ;
- **`test/`** : segments à prédire ;
- **`sample_submission.csv`** : structure de soumission fournie par la compétition.

Le notebook exploite **4 431 segments d'entraînement** et génère des prédictions pour **4 520 segments de test**.

Les données brutes ne sont pas incluses dans ce dépôt, compte tenu de leur volume. Elles sont accessibles depuis la [compétition Kaggle](https://www.kaggle.com/c/predict-volcanic-eruptions-ingv-oe/data).

## Méthodologie

### 1. Préparation des données et feature engineering

Les fichiers de chaque segment sont lus avec **Pandas**, puis résumés à l'aide de statistiques descriptives calculées sur chaque capteur :

- somme, moyenne et écart-type ;
- minimum et maximum ;
- quantiles **10 %, 20 %, 50 %, 70 % et 90 %**.

Cette approche permet de construire **100 caractéristiques statistiques par segment** (10 statistiques × 10 capteurs). Le code prévoit également la gestion des erreurs de lecture et de certaines valeurs manquantes.

### 2. Modélisation et optimisation

- Séparation des données en **80 % d'entraînement et 20 % de test** ;
- Entraînement d'un **`XGBRegressor`** avec l'objectif `reg:squarederror` ;
- Recherche d'hyperparamètres avec **`RandomizedSearchCV`** (**20 configurations**) ;
- **Validation croisée à 5 plis** (`KFold`) pour sélectionner la configuration du modèle.

La dernière exécution enregistrée dans le notebook retient notamment `n_estimators=300`, `max_depth=7`, `learning_rate=0.1` et `subsample=0.6`.

### 3. Évaluation

Les performances sont étudiées à l'aide de trois métriques : **R²**, **MAE** (erreur absolue moyenne) et **RMSE** (racine de l'erreur quadratique moyenne). Les erreurs temporelles sont converties en jours, heures, minutes et secondes afin de faciliter leur interprétation.

## Résultats

Résultats de la **dernière exécution enregistrée dans le notebook**, sur un ensemble de test interne (20 % des données) :

| Métrique | Résultat |
| --- | ---: |
| Meilleur R² en validation croisée | **0,781** |
| R² sur l'ensemble de test | **0,792** |
| MAE sur l'ensemble de test | **48 j 18 h 49 min** |
| RMSE sur l'ensemble de test | **69 j 13 h 40 min** |

Le R² obtenu est encourageant, mais **les erreurs en unités de temps restent importantes** : le modèle ne peut donc pas être considéré comme un outil fiable de prévision opérationnelle d'éruptions. Les valeurs prédites négatives sont ramenées à zéro dans l'étape d'analyse des résultats du notebook.

> **Note :** le compte rendu PDF indique un résultat différent (R² ≈ 0,83), sans que l'écart avec le notebook soit expliqué. Les valeurs ci-dessus proviennent du notebook fourni et ne correspondent **pas à un score officiel Kaggle**.

## Technologies

**Python**, **Pandas**, **NumPy**, **Scikit-learn**, **XGBoost**, **tqdm**.  
Le travail comprend également une exploration et une visualisation des données sismiques.

## Organisation du dépôt

```text
.
├── Projet_ML.ipynb   # Préparation des données, entraînement et prédictions
├── Projet_ML.pdf     # Compte rendu universitaire et discussion des résultats
└── README.md
```

## Reproduire le projet

1. Télécharger les données depuis Kaggle et installer les bibliothèques nécessaires :

   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn tqdm jupyter
   ```

2. Organiser les fichiers dans un dossier local, par exemple :

   ```text
   data/
   ├── train.csv
   ├── sample_submission.csv
   ├── train/
   │   └── <segment_id>.csv
   └── test/
       └── <segment_id>.csv
   ```

3. **Adapter les chemins dans le notebook** : la version fournie contient des chemins absolus propres à l'ordinateur utilisé lors du projet. Par exemple :

   ```python
   folder_path = "./data/train/"
   folder_path2 = "./data/test/"
   train = pd.read_csv("./data/train.csv")
   test = pd.read_csv("./data/sample_submission.csv")
   ```

4. Exécuter les cellules dans l'ordre. Prévoir des ressources mémoire suffisantes pour le traitement des données brutes.

**Attention à la soumission Kaggle :** dans la version enregistrée, l'export final convertit les prédictions en durées et utilise des noms de colonnes différents de ceux attendus. Pour soumettre un fichier à Kaggle, conserver les prédictions **numériques en secondes** et respecter exactement les colonnes **`segment_id`** et **`time_to_eruption`**.

## Limites et pistes d'amélioration

- **Traitement des volumes :** réduire les temps de lecture et la consommation mémoire, notamment en évitant de recharger systématiquement tous les CSV.
- **Signaux temporels :** enrichir les statistiques descriptives avec des caractéristiques fréquentielles ou temporelles, actuellement peu exploitées.
- **Évaluation :** fixer la graine de la séparation entraînement/test et tester une validation mieux adaptée à la structure des segments sismiques.
- **Modélisation :** comparer XGBoost à d'autres méthodes et approfondir la gestion des prédictions négatives.

## Contexte

Projet réalisé dans le cadre d'un **Master Mathématiques et Applications**, parcours **Probabilités et Statistiques**, à l'**Université Gustave Eiffel**.
