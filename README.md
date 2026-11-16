# INGV — Estimer le temps avant une éruption volcanique à partir de signaux sismiques

**Projet personnel de Data Science** · Traitement du signal · Feature engineering · Régression · XGBoost / LightGBM

> Un volcan ne dispose pas d'un compte à rebours. Ses signaux sismiques contiennent-ils néanmoins assez d'informations pour estimer le temps restant avant une éruption ?

## Pourquoi ce projet ?

L'anticipation de l'activité volcanique représente un enjeu important pour la surveillance des territoires exposés. Les capteurs sismiques enregistrent des vibrations susceptibles de refléter certains changements d'activité, mais leur interprétation reste complexe.

Ce projet cherche à **évaluer le potentiel prédictif de ces signaux** : peut-on transformer des enregistrements sismiques en indicateurs exploitables par un modèle de régression ? L'objectif est de relier une démarche de Machine Learning à une problématique de surveillance, **sans prétendre construire un système opérationnel d'alerte ou d'évacuation**.

## Données et démarche

Le jeu de données INGV contient des segments d'enregistrements provenant de **10 capteurs sismiques**. Pour chaque segment étiqueté, la variable cible `time_to_eruption` représente le temps restant avant la prochaine éruption, dans l'unité fournie par le jeu de données.

Le notebook suit un pipeline reproductible :

1. **Contrôler les données** : identifiants, correspondance des fichiers, valeurs manquantes et signaux constants.
2. **Extraire 143 caractéristiques** par segment afin de résumer les signaux sans traiter directement toutes les mesures brutes.
3. **Préparer les données** : associer chaque segment à sa cible et réserver 20 % des observations pour l'évaluation finale (*holdout*).
4. **Comparer les modèles** : baseline médiane, XGBoost et LightGBM, avec validation croisée à 5 folds et recherche de 10 configurations d'hyperparamètres par algorithme.
5. **Analyser les erreurs** : MAE, RMSE, R², prédictions réelles versus prédites, résidus et biais par quartile de `time_to_eruption`.

### Les 143 caractéristiques

| Famille | Description | Nombre |
| --- | --- | ---: |
| Statistiques | Moyenne, écart-type, RMS, MAD, kurtosis, 95e percentile de l'amplitude absolue | 60 |
| Temporelles | RMS sur quatre fenêtres, autocorrélation au retard 100 | 50 |
| Fréquentielles | Centroïde spectral et proportion de puissance basse fréquence (méthode de Welch) | 20 |
| Qualité des mesures | Proportion de valeurs manquantes par capteur | 10 |
| Globales | Nombre de capteurs exploitables, moyenne et dispersion des RMS | 3 |
| **Total** | | **143** |

Les fréquences calculées avec `fs=1` sont exprimées en **cycles par échantillon**, et non en hertz physiques.

## Résultats

Les hyperparamètres sont sélectionnés à l'aide de la **MAE en validation croisée**, même si les deux modèles utilisent une perte quadratique pendant leur entraînement.

| Modèle | MAE en validation croisée | Écart-type entre folds |
| --- | ---: | ---: |
| **XGBoost** | **3,38 M** | **0,065 M** |
| LightGBM | 3,76 M | 0,098 M |

**XGBoost a été retenu** sur la base de sa MAE en validation croisée.

Sur le *holdout* de **887 segments** :

| Indicateur | XGBoost |
| --- | ---: |
| MAE | **3,14 M** |
| RMSE | **4,31 M** |
| R² | **0,8999** |
| MAE de la baseline médiane | 11,83 M |
| Réduction de MAE par rapport à la baseline | **73,46 %** |

*Les valeurs en millions (M) sont exprimées dans l'unité originale de `time_to_eruption`. Résultats issus de l'exécution étudiée ; ils peuvent varier si les données ou le protocole changent.*

### Ce que révèlent les erreurs

L'analyse par quartile du temps **réel** restant révèle une concentration des prédictions vers les valeurs intermédiaires :

- **Éruptions les plus proches (Q1)** : biais moyen de **+2,53 M** ; le modèle **surestime** le temps restant.
- **Éruptions les plus éloignées (Q4)** : biais moyen de **−3,07 M** ; le modèle **sous-estime** le temps restant.

Ce constat est essentiel du point de vue métier : une faible erreur moyenne globale ne garantit pas une précision homogène, en particulier pour les situations où le temps restant est court. Ces biais sont descriptifs ; ils ne prouvent pas à eux seuls un défaut de calibration.

## Reproduire l'analyse

Voir les instructions de lancement ci-dessous et le [descriptif des données](data/raw/README.md).

## Limites et pistes d'amélioration

La validation est **aléatoire**, sans regroupement fiable des observations par épisode volcanique. Des segments apparentés peuvent donc rendre les scores plus optimistes qu'ils ne le seraient sur de nouveaux événements indépendants. Par ailleurs, les 143 caractéristiques résument les signaux sans en conserver toute l'information, et le modèle fournit une estimation ponctuelle sans quantifier directement son incertitude.

Des prolongements pertinents seraient une validation **par épisodes indépendants** (si les métadonnées nécessaires sont disponibles), une étude de calibration sur des prédictions hors échantillon et une estimation de l'incertitude.

**Ce travail est une étude exploratoire de Data Science : ses prédictions ne doivent pas être utilisées pour prendre des décisions réelles de sécurité ou d'évacuation.**

## Rapport du projet

Le compte rendu détaillé présente le **contexte métier**, les principes théoriques des caractéristiques et des modèles, les résultats expérimentaux et une **discussion critique des limites** : [reports/Compte_rendu_INGV.pdf](reports/Compte_rendu_INGV.pdf).

## Organisation du dépôt

```text
.
├── README.md
├── .gitignore
├── requirements.txt
├── notebooks/     # Analyses exécutables
├── reports/       # Rapport PDF existant
└── data/
    └── raw/       # Données locales non versionnées
```

### Livrables

- [INGV_143_Notebook_Projet.ipynb](notebooks/INGV_143_Notebook_Projet.ipynb)
- [Compte_rendu_INGV.pdf](reports/Compte_rendu_INGV.pdf)

## Exécution locale

Les données sources ne sont pas incluses. Voir [data/raw/README.md](data/raw/README.md) pour les fichiers attendus et la provenance disponible.

Depuis la racine du dépôt :

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell : .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m notebook
```

Ouvrir le notebook dans `notebooks/`, sélectionner le noyau de cet environnement puis exécuter les cellules dans l'ordre. Les chemins fonctionnent depuis la racine du dépôt et depuis `notebooks/`. Les données sont lues dans `data/raw/` par défaut ; la variable d'environnement `PROJECT_DATA_DIR` permet de choisir un autre dossier (chemin absolu recommandé).

Les dépendances reflètent les imports du notebook. Le dépôt initial ne fournit aucun environnement verrouillé ; la reproduction numérique exacte des résultats historiques n'est donc pas garantie.

Les résultats, graphiques, modèles et caches sont écrits dans `outputs/` (ignoré par Git). `INGV_OUTPUT_DIR` permet de changer cette destination. L’extraction et les recherches d’hyperparamètres peuvent être longues.
