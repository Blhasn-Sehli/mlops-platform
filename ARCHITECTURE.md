# MLOps Platform — Architecture & Flux Complet

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Arborescence du projet](#2-arborescence-du-projet)
3. [Référence des fichiers](#3-référence-des-fichiers)
   - 3.1 Configuration centralisée — `src/config.py`
   - 3.2 Logging structuré — `src/logging_config.py`
   - 3.3 Couche données — `src/data/`
   - 3.4 Entraînement — `src/training/train.py`
   - 3.5 API d'inférence — `src/api/main.py`
   - 3.6 Monitoring & Drift — `src/monitoring/drift_detector.py`
   - 3.7 Pipeline Kubeflow — `kubeflow/pipeline.py`
   - 3.8 Tests
   - 3.9 Infrastructure Docker
   - 3.10 CI/CD — GitHub Actions
   - 3.11 Alerting — Prometheus + Alertmanager
   - 3.12 Kubernetes — `k8s/`
4. [Flux complet end-to-end](#4-flux-complet-end-to-end)
5. [Variables d'environnement](#5-variables-denvironnement)
6. [Commandes de démarrage rapide](#6-commandes-de-démarrage-rapide)

---

## 1. Vue d'ensemble

```
Développeur
    │  git push → main
    ▼
GitHub Actions (3 jobs)
    ├── test       : train → pytest --cov → upload coverage XML
    ├── push-image : build → push GHCR (sha + latest)
    └── security-scan : Trivy → GitHub Security tab
    ▼
MLFlow Model Registry
    │  Pipeline(StandardScaler + RandomForest) enregistré
    │  alias "champion" assigné si accuracy ≥ 0.90
    ▼
FastAPI (mlops_api)  ←── Docker Compose / Kubernetes
    │  charge Pipeline depuis Registry (ou .pkl fallback)
    │  charge challenger (alias "challenger") si présent → A/B routing
    │  expose /health  /predict  /predict/batch  /explain  /metrics  /stats  /ab/status
    │  authentification X-API-Key (optionnelle en dev)
    ▼
Client HTTP (curl / app / Kubeflow)
    │
    ├── POST /predict       → PredictionResponse (+ variant + latency_ms)
    │       ├── champion (1 - AB_TRAFFIC_SPLIT % du trafic)
    │       └── challenger (AB_TRAFFIC_SPLIT % du trafic)
    ├── POST /predict/batch → BatchPredictionResponse (vectorisé, toujours champion)
    └── POST /explain       → ExplainResponse (SHAP values par feature)
    ▼
PredictionLogger (SQLite)
    │  persiste chaque prédiction : features, output, latency_ms, timestamp
    ▼
Prometheus (scrape /metrics toutes les 15s)
    │  + gauge mlops_model_registry_active
    ▼
Alertmanager  ←── alert.rules.yml (7 règles : APIDown, HighErrorRate, ModelFallbackActive, etc.)
    │  routing critical/warning → Slack / email / webhook
    ▼
Grafana (dashboard auto-provisionné, 6 panels, refresh 10s)
    ▼
DriftDetector (Evidently AI)
    │  PredictionLogger.get_recent() → build_current_data() → detect()
    │  génère rapport HTML horodaté
    ▼
Kubeflow Pipelines (re-entraînement orchestré sur cluster K8s)
    load_data → train_model → evaluate_model → [si OK] register_model
```

---

## 2. Arborescence du projet

```
mlops-platform/
├── .dvc/                          DVC config (versioning données)
├── .github/
│   └── workflows/
│       └── ci.yml                 CI/CD : test + push-image + security-scan
├── .pre-commit-config.yaml        Hooks : black, isort, ruff, hadolint
├── conftest.py                    Ajout racine à sys.path pour pytest
├── docker/
│   └── Dockerfile.api             Image FastAPI (copie tout src/)
├── docker-compose.yml             5 services : mlflow, api, prometheus, alertmanager, grafana
├── k8s/
│   ├── namespace.yml              Namespace "mlops"
│   ├── configmap.yml              Variables d'environnement K8s
│   ├── deployment.yml             Deployment 2 replicas + probes + PVC
│   ├── service.yml                ClusterIP port 80→8000 + PVC claim
│   └── hpa.yml                    HPA 2→10 pods (CPU 70%, memory 80%)
├── kubeflow/
│   └── pipeline.py                Pipeline KFP v2 : 4 composants
├── models/
│   └── random_forest.pkl          Fallback local (Pipeline sérialisé)
├── monitoring/
│   ├── alert.rules.yml            6 règles Prometheus (critical/warning)
│   ├── alertmanager.yml           Routing, receivers, inhibitions
│   ├── prometheus.yml             Scrape config + alertmanager + rules
│   ├── reports/                   Rapports HTML Evidently (drift)
│   ├── predictions.db             SQLite — log de toutes les prédictions
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources/       prometheus.yml → auto-configure datasource
│       │   └── dashboards/        dashboard.yml  → provider de dashboards
│       └── dashboards/
│           └── mlops_dashboard.json  6 panels Grafana
├── projet_complexe.pdf
├── requirements.txt               170+ dépendances + evidently, kfp, structlog, pytest-cov, shap
├── src/
│   ├── __init__.py
│   ├── config.py                  Settings Pydantic (toutes les constantes)
│   ├── logging_config.py          Configuration structlog (JSON)
│   ├── api/
│   │   └── main.py                FastAPI : 6 endpoints + auth + SHAP + batch
│   ├── data/
│   │   ├── ingestion.py           Chargement + validation des données
│   │   ├── preprocessing.py       Pipeline sklearn + split + stats
│   │   └── prediction_logger.py   SQLAlchemy — log des prédictions en base
│   ├── monitoring/
│   │   └── drift_detector.py      Evidently — détection de drift
│   └── training/
│       └── train.py               GridSearchCV sur Pipeline(Scaler+RF)
└── tests/
    ├── unit/
    │   └── test_api.py            4 tests unitaires (TestClient)
    └── integration/
        └── test_api_integration.py  13 tests d'intégration
```

---

## 3. Référence des fichiers

---

### 3.1 `src/config.py` — Configuration centralisée

**Rôle** : Source unique de vérité pour toutes les constantes du projet. Utilise `pydantic-settings` : chaque champ est d'abord lu depuis la variable d'environnement correspondante, puis depuis `.env`, sinon la valeur par défaut s'applique.

**Classe `Settings(BaseSettings)`**

| Champ | Type | Défaut | Env var |
|---|---|---|---|
| `mlflow_tracking_uri` | `str` | `file:./mlruns` | `MLFLOW_TRACKING_URI` |
| `mlflow_model_name` | `str` | `iris-random-forest` | `MLFLOW_MODEL_NAME` |
| `mlflow_model_alias` | `str` | `champion` | `MLFLOW_MODEL_ALIAS` |
| `mlflow_challenger_alias` | `str` | `challenger` | `MLFLOW_CHALLENGER_ALIAS` |
| `ab_traffic_split` | `float` | `0.0` | `AB_TRAFFIC_SPLIT` |
| `accuracy_threshold` | `float` | `0.90` | `ACCURACY_THRESHOLD` |
| `fallback_model_path` | `str` | `models/random_forest.pkl` | `FALLBACK_MODEL_PATH` |
| `test_size` | `float` | `0.20` | `TEST_SIZE` |
| `random_state` | `int` | `42` | `RANDOM_STATE` |
| `api_host` | `str` | `0.0.0.0` | `API_HOST` |
| `api_port` | `int` | `8000` | `API_PORT` |
| `log_level` | `str` | `INFO` | `LOG_LEVEL` |
| `api_key` | `str \| None` | `None` | `API_KEY` |
| `predictions_db_url` | `str` | `sqlite:///monitoring/predictions.db` | `PREDICTIONS_DB_URL` |
| `drift_reports_dir` | `str` | `monitoring/reports` | `DRIFT_REPORTS_DIR` |

**Fonction `get_settings() → Settings`**

Singleton mis en cache via `@lru_cache`. Tous les modules importent `from src.config import settings`.

---

### 3.2 `src/logging_config.py` — Logging structuré

**Rôle** : Configure `structlog` pour produire des logs JSON sur stdout. Remplace tous les `print()` du projet.

**Fonction `setup_logging(level: str) → None`**

Configure la chaîne de processors structlog : filtre par niveau → nom du logger → niveau → timestamp ISO UTC → stack info → exc info → JSON. Réduit le bruit des loggers tiers (`uvicorn.access`, `mlflow`, `httpx`) à `WARNING`.

**Fonction `get_logger(name: str) → BoundLogger`**

Appelle `setup_logging()` puis retourne un logger lié au nom du module. Usage :

```python
from src.logging_config import get_logger
log = get_logger(__name__)
log.info("prediction", prediction="setosa", confidence=0.97, latency_ms=1.2)
# → {"event": "prediction", "prediction": "setosa", "confidence": 0.97,
#    "latency_ms": 1.2, "logger": "src.api.main", "level": "info",
#    "timestamp": "2026-04-30T10:23:01.456Z"}
```

---

### 3.3 `src/data/` — Couche données

#### `src/data/ingestion.py`

**Rôle** : Charge et valide les données brutes avant tout traitement.

| Fonction | Signature | Description |
|---|---|---|
| `load_from_sklearn` | `() → pd.DataFrame` | Charge le dataset Iris depuis scikit-learn. Utilisé en dev, CI et Kubeflow. |
| `load_from_csv` | `(path: str\|Path) → pd.DataFrame` | Charge depuis un CSV, appelle `validate()` automatiquement. Lève `FileNotFoundError` si le fichier est absent. |
| `validate` | `(df: pd.DataFrame) → None` | Orchestrateur des 5 checks. Lève `ValueError` au premier échec. |
| `_check_not_empty` | privée | Vérifie `len(df) > 0`. |
| `_check_required_columns` | privée | Vérifie que les 4 colonnes features sont présentes. |
| `_check_numeric_types` | privée | Vérifie que chaque feature est de type numérique. |
| `_check_no_nulls` | privée | Vérifie qu'aucune valeur nulle n'est présente dans les features. |
| `_check_value_bounds` | privée | Vérifie que chaque feature est dans `]0, 50]` cm (bornes Iris). |

#### `src/data/preprocessing.py`

**Rôle** : Utilitaires de preprocessing réutilisables en entraînement, en Kubeflow et en inférence.

| Fonction | Signature | Description |
|---|---|---|
| `build_preprocessing_pipeline` | `() → Pipeline` | Retourne un `Pipeline([("scaler", StandardScaler())])` non fitté. |
| `split` | `(df, test_size, random_state) → (X_train, X_test, y_train, y_test)` | Split stratifié selon les paramètres de `settings`. |
| `features_to_array` | `(row: dict) → np.ndarray` | Convertit un dict de features en tableau `(1, 4)` pour inférence. |
| `describe_features` | `(df: pd.DataFrame) → dict` | Retourne mean, std, min, q25, median, q75, max par feature. |
| `class_distribution` | `(df: pd.DataFrame) → dict` | Retourne `{class_id: count}` depuis la colonne `TARGET_NAME`. |

#### `src/data/prediction_logger.py`

**Rôle** : Persiste chaque appel à `/predict` ou `/predict/batch` dans SQLite via SQLAlchemy. Fournit les données brutes pour le drift detector.

**Modèle SQLAlchemy `PredictionRecord`**

| Colonne | Type SQL | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `timestamp` | DATETIME | UTC, généré automatiquement |
| `sepal_length/width` | REAL | Features d'entrée |
| `petal_length/width` | REAL | Features d'entrée |
| `prediction` | TEXT | Nom de classe (`"setosa"`, …) |
| `class_id` | INTEGER | Index de classe (0, 1, 2) |
| `confidence` | REAL | Probabilité de la classe prédite |
| `model_source` | TEXT | URI registry ou `"local"` |
| `latency_ms` | REAL | Temps de traitement en ms (NULL pour batch) |

**Classe `PredictionLogger`**

| Méthode | Signature | Description |
|---|---|---|
| `__init__` | `(db_url: str)` | Crée le dossier parent SQLite si absent. Crée la table si elle n'existe pas (`create_all`). |
| `log` | `(features, prediction, class_id, confidence, model_source, latency_ms) → None` | Insère une ligne. Les exceptions sont silencieuses — ne crashe jamais l'API. |
| `get_recent` | `(limit: int) → list[dict]` | Retourne les N dernières prédictions triées par timestamp desc. Format compatible avec `DriftDetector.build_current_data()`. |
| `stats` | `() → dict` | Retourne `{total_predictions, avg_confidence, avg_latency_ms, class_distribution}` via requêtes SQL agrégées. |
| `count` | `() → int` | Compte rapide du total de prédictions. Affiché dans `/health`. |

---

### 3.4 `src/training/train.py` — Entraînement

**Rôle** : Entraîne un `Pipeline(StandardScaler → RandomForestClassifier)` avec grid search, logue dans MLFlow, l'enregistre dans le Model Registry et le promeut en alias `champion` si le seuil est atteint.

**Flux d'exécution (script top-level)**

| Étape | Ce qui se passe |
|---|---|
| **Config** | Lit tout depuis `settings` (MLFlow URI, seuil, test_size, random_state) |
| **Données** | `load_iris()` → `train_test_split()` (80/20, seed=42) |
| **Pipeline** | `Pipeline([("scaler", StandardScaler()), ("classifier", RandomForest())])` |
| **Grid Search** | `GridSearchCV` sur `classifier__n_estimators ∈ {50,100,200}` et `classifier__max_depth ∈ {3,5,10}`, cv=5, n_jobs=-1 (45 fits) |
| **Évaluation** | `accuracy_score()` + `f1_score(average="weighted")` sur le test set |
| **MLFlow logging** | `log_params()` (clés `classifier__*`), `log_metric()`, `set_tag(model_type, dataset)` |
| **Registry** | `mlflow.sklearn.log_model(..., registered_model_name=MODEL_NAME)` — le Pipeline complet (scaler + classifieur) est enregistré |
| **Fallback** | `joblib.dump(best_pipeline, fallback_model_path)` — le même Pipeline en `.pkl` |
| **Promotion** | `MlflowClient.search_model_versions()` → `set_registered_model_alias("champion", version)` si accuracy ≥ seuil. Log structuré du résultat. |

> **Pourquoi le Pipeline ?** Le `StandardScaler` est fitté sur les données d'entraînement et bundlé avec le classifieur. À l'inférence, `model.predict()` applique automatiquement la même transformation — il est impossible d'oublier le scaler ou d'utiliser des statistiques différentes.

---

### 3.5 `src/api/main.py` — API d'inférence

**Rôle** : Serveur FastAPI. Charge le Pipeline au démarrage, expose 7 endpoints, authentifie via API Key optionnelle, logue chaque prédiction en base, publie un gauge Prometheus pour alerter sur le fallback, et route le trafic entre champion et challenger pour A/B testing.

#### Fonctions de démarrage (exécutées une fois à l'import)

| Fonction | Retour | Description |
|---|---|---|
| `_load_model()` | `(model, model_source)` | Tente l'URI `models:/iris-random-forest@champion`. En cas d'échec, charge le `.pkl` local. Met à jour `MODEL_REGISTRY_ACTIVE` gauge (1/0). |
| `_load_challenger()` | `(challenger_model, challenger_source)` | Tente l'URI `models:/iris-random-forest@challenger`. Retourne `(None, None)` silencieusement si l'alias n'existe pas — l'A/B testing est alors désactivé. |
| `_build_explainer(model)` | `(explainer, scaler)` | Extrait le classifieur du Pipeline (`named_steps["classifier"]`), crée `shap.TreeExplainer`. Retourne `(None, None)` si SHAP indisponible. |

#### Métriques Prometheus custom

| Métrique | Type | Description |
|---|---|---|
| `mlops_model_registry_active` | Gauge | `1` si le modèle vient du Registry MLFlow, `0` si fallback local. Déclenche l'alerte `ModelFallbackActive`. |
| `mlops_ab_requests_total{variant}` | Counter | Nombre de requêtes `/predict` par variant (`champion` ou `challenger`). Permet de calculer le débit et l'error rate par variant dans Grafana. |

#### Logique de routage A/B

`_route_to_challenger() → bool` — retourne `True` si `challenger_model` est chargé **et** `random.random() < settings.ab_traffic_split`. Activé uniquement si l'alias `challenger` est présent dans le Registry et que `AB_TRAFFIC_SPLIT > 0`.

Comportement par défaut (`ab_traffic_split = 0.0`) : 100 % du trafic vers le champion. Le batch (`/predict/batch`) et l'explication (`/explain`) utilisent toujours le champion.

#### Dépendance d'authentification

`_verify_api_key(api_key)` — injectée via `Depends()` sur les endpoints protégés. Si `settings.api_key` est `None`, l'auth est désactivée (mode dev). Sinon, le header `X-API-Key` est obligatoire → HTTP 403 si absent ou incorrect.

#### Schémas Pydantic

| Classe | Champs | Validations |
|---|---|---|
| `PredictionRequest` | `sepal_length, sepal_width, petal_length, petal_width: float` | `@field_validator` : toutes les valeurs > 0 |
| `PredictionResponse` | `prediction: str`, `class_id: int`, `confidence: float`, `model_source: str`, `variant: str`, `latency_ms: float` | — |
| `BatchPredictionRequest` | `items: list[PredictionRequest]` | Non vide, ≤ 1000 items |
| `BatchPredictionResponse` | `predictions: list[PredictionResponse]`, `total: int`, `batch_latency_ms: float` | — |
| `ExplainRequest` | `sepal_length, sepal_width, petal_length, petal_width: float` | — |
| `ExplainResponse` | `prediction, class_id, confidence, shap_values: dict[str, float], model_source` | — |

#### Endpoints

| Méthode | Route | Auth | Fonction | Description |
|---|---|---|---|---|
| `GET` | `/health` | Non | `health()` | `{status, model, model_source, total_predictions}` — utilisé par Docker + K8s probes |
| `GET` | `/metrics` | Non | `metrics()` | Métadonnées statiques du modèle (type, features, classes) |
| `GET` | `/stats` | Non | `stats()` | Agrégats sur toutes les prédictions loguées (via `PredictionLogger.stats()`) |
| `GET` | `/ab/status` | Non | `ab_status()` | État A/B : `{ab_enabled, traffic_split, champion: {loaded, source, alias}, challenger: {loaded, source, alias}}` |
| `GET` | `/metrics` (Prometheus) | Non | Auto-injecté | `http_requests_total`, `http_request_duration_seconds`, etc. |
| `POST` | `/predict` | Oui | `predict()` | Prédiction unitaire avec routage A/B. Mesure latency_ms. Logue en base. Incrémente `AB_REQUESTS`. |
| `POST` | `/predict/batch` | Oui | `predict_batch()` | Vectorise toute la batch en une seule opération numpy. Toujours sur le champion. |
| `POST` | `/explain` | Oui | `explain()` | SHAP values pour la classe prédite. Applique le scaler du Pipeline avant TreeExplainer. HTTP 503 si SHAP indisponible. |

#### Helpers internes

| Fonction | Description |
|---|---|
| `_vectorize(req)` | Convertit un `PredictionRequest` en `np.ndarray` (1, 4) |
| `_route_to_challenger()` | Décision aléatoire basée sur `ab_traffic_split`. Retourne `False` si challenger non chargé. |

---

### 3.6 `src/monitoring/drift_detector.py` — Drift Detection

**Rôle** : Compare la distribution des données de production (prédictions récentes) à la distribution de référence (dataset Iris d'entraînement). Génère des rapports HTML Evidently AI horodatés.

#### Classe `DriftDetector`

| Attribut | Type | Description |
|---|---|---|
| `reports_dir` | `str` | Dossier de sortie (`settings.drift_reports_dir`) |
| `reference_data` | `pd.DataFrame` | 150 lignes Iris — baseline immuable |
| `column_mapping` | `ColumnMapping` | Indique à Evidently : target = `species`, features numériques = 4 colonnes |

| Méthode | Signature | Description |
|---|---|---|
| `detect` | `(current_data: pd.DataFrame) → dict` | Lance `DataDriftPreset` + `DataQualityPreset`, sauvegarde HTML, retourne `{drift_detected, n_drifted_features, share_drifted, report_path, timestamp, n_samples}`. Log structuré du résultat. |
| `build_current_data` | `(predictions_log: list[dict]) → pd.DataFrame` | Convertit la sortie de `PredictionLogger.get_recent()` en DataFrame Evidently-compatible. |
| `simulate_drifted_data` | `(n_samples, noise_scale) → pd.DataFrame` | Génère des données artificiellement driftées (bruit gaussien) pour tests et démos. Utilise `settings.random_state`. |
| `_build_reference` | privée | Charge Iris scikit-learn, retourne DataFrame avec colonne `species`. |

**Connexion avec `PredictionLogger`**

```python
detector = DriftDetector()
logs = PredictionLogger().get_recent(limit=500)
current_df = detector.build_current_data(logs)
result = detector.detect(current_df)
# result["drift_detected"] → True/False
```

---

### 3.7 `kubeflow/pipeline.py` — Orchestration

**Rôle** : Pipeline Kubeflow Pipelines v2 pour le re-entraînement automatisé. Chaque étape tourne dans un conteneur Python isolé.

#### Composants `@component`

| Composant | Inputs | Outputs | Description |
|---|---|---|---|
| `load_data` | — | `train: Dataset`, `test: Dataset` | `load_iris()` → split 80/20 → CSV |
| `train_model` | `train: Dataset`, `n_estimators`, `max_depth` | `model: Model` | GridSearchCV sur Pipeline sklearn, sauvegarde best model + métadonnées KFP |
| `evaluate_model` | `model: Model`, `test: Dataset`, `threshold` | `metrics: Metrics`, `bool` | accuracy + F1 → logue KFP Metrics → retourne `True` si ≥ seuil |
| `register_model` | `model: Model`, `mlflow_uri` | — | Run MLFlow → `log_model` → Registry → alias `champion` |

#### Pipeline `iris_pipeline`

```
load_data ──► train_model ──► evaluate_model
                                   └── [True] ──► register_model
```

Compilation → `iris_mlops_pipeline.yaml` → soumis au cluster KFP.

---

### 3.8 Tests

#### `tests/unit/test_api.py`

Charge l'app FastAPI via `TestClient`. Le modèle est chargé depuis le fallback `.pkl` local (MLFlow non requis).

| Fonction | Ce qu'elle vérifie |
|---|---|
| `test_health()` | HTTP 200 + `status == "ok"` |
| `test_predict_setosa()` | `prediction == "setosa"` |
| `test_predict_virginica()` | `prediction == "virginica"` |
| `test_metrics()` | HTTP 200 + champ `"model"` présent |

#### `tests/integration/test_api_integration.py`

Nécessite que `train.py` ait été exécuté. Fixture `client` avec scope `module`.

| Fonction | Ce qu'elle vérifie |
|---|---|
| `test_health_ok` | HTTP 200 + `status == "ok"` + `model_source` présent |
| `test_health_reports_model_source` | `model_source` = `"local"` ou URI `models:/...` |
| `test_metrics_endpoint_shape` | `model`, `features == 4`, 3 classes, `model_source` |
| `test_predict_known_cases` (×3) | Setosa / versicolor / virginica correctement prédits |
| `test_predict_response_schema` | 6 champs exacts : `prediction, class_id, confidence, model_source, variant, latency_ms` |
| `test_predict_setosa_high_confidence` | confidence ≥ 0.80 pour setosa typique |
| `test_predict_class_id_matches_class_name` | `class_map[class_id] == prediction` sur les 3 cas |
| `test_predict_missing_field_returns_422` | Champ manquant → HTTP 422 |
| `test_predict_wrong_type_returns_422` | Type invalide → HTTP 422 |
| `test_predict_empty_body_returns_422` | Body vide → HTTP 422 |
| `test_predict_variant_field` | `variant ∈ {"champion", "challenger"}` |
| `test_ab_status_endpoint` | `GET /ab/status` → HTTP 200, champs `ab_enabled`, `traffic_split`, `champion`, `challenger` |
| `test_prometheus_metrics_exposed` | `GET /metrics` → HTTP 200 |

#### `conftest.py`

Ajoute la racine du projet à `sys.path` pour résoudre les imports `from src.*`.

---

### 3.9 Infrastructure Docker

#### `docker/Dockerfile.api`

| Étape | Description |
|---|---|
| `FROM python:3.11-slim` | Image minimale |
| `pip install -r requirements.txt` | Installation sans cache |
| `COPY src/ ./src/` | Tout le package `src` (config, data, api, monitoring, logging_config) |
| `COPY models/ ./models/` | Fallback `.pkl` |
| `RUN mkdir -p monitoring` | Crée le dossier pour SQLite predictions.db |
| `EXPOSE 8000` | Port déclaré |
| `CMD uvicorn src.api.main:app` | Démarrage |

#### `docker-compose.yml`

| Service | Image | Port | Dépend de | Rôle |
|---|---|---|---|---|
| `mlflow` | `ghcr.io/mlflow/mlflow:v2.11.0` | `5000` | — | Tracking + Registry (SQLite backend, volume `mlflow_data`) |
| `api` | Build local | `8000` | `mlflow` (healthy) | FastAPI + volume `predictions_data` pour SQLite log |
| `prometheus` | `prom/prometheus:latest` | `9090` | `api` (healthy) | Scrape + évaluation des règles d'alerte |
| `alertmanager` | `prom/alertmanager:latest` | `9093` | `prometheus` | Routing des alertes (Slack / email / webhook) |
| `grafana` | `grafana/grafana:latest` | `3000` | `prometheus` | Dashboard auto-provisionné |

**Volumes** : `mlflow_data`, `grafana_data`, `predictions_data` (SQLite prediction log persisté).

#### `monitoring/prometheus.yml`

| Section | Valeur | Description |
|---|---|---|
| `scrape_interval` | `15s` | Fréquence de collecte |
| `alertmanager` target | `alertmanager:9093` | Envoi des alertes déclenchées |
| `rule_files` | `/etc/prometheus/alert.rules.yml` | Fichier de règles monté en volume |
| `job: mlops-api` | `api:8000` | Metrics FastAPI |
| `job: prometheus` | `localhost:9090` | Auto-monitoring |
| `job: alertmanager` | `alertmanager:9093` | Monitoring Alertmanager |

---

### 3.10 CI/CD — `.github/workflows/ci.yml`

Pipeline en **3 jobs enchaînés**.

#### Job 1 : `test` (PR + push main)

| Étape | Description |
|---|---|
| `actions/checkout@v4` | Clone le repo |
| `actions/setup-python@v5` (3.11) | Python avec cache pip |
| `pip install -r requirements.txt` | Dépendances |
| `python src/training/train.py` | Entraîne le modèle (requis pour les tests) |
| `pytest tests/ -v --cov=src --cov-fail-under=70` | Tests + coverage. Échoue si < 70%. |
| `upload-artifact` | Publie `coverage.xml` |

#### Job 2 : `push-image` (push main seulement, après `test`)

| Étape | Description |
|---|---|
| `docker/login-action@v3` | Login GHCR avec `GITHUB_TOKEN` |
| `docker/metadata-action@v5` | Génère tags `sha-XXXXXXX` + `latest` |
| `docker/build-push-action@v5` | Build + push avec cache GitHub Actions (`cache-from/to: gha`) |

#### Job 3 : `security-scan` (après `push-image`)

| Étape | Description |
|---|---|
| `aquasecurity/trivy-action` | Scan CRITICAL + HIGH sur l'image GHCR |
| `codeql-action/upload-sarif` | Résultats dans l'onglet Security GitHub |

#### `.pre-commit-config.yaml`

| Hook | Outil | Description |
|---|---|---|
| `black` | 24.3.0 | Formatage (line-length=100) |
| `isort` | 5.13.2 | Tri des imports (profil black) |
| `ruff` | v0.4.4 | Linting + auto-fix |
| `trailing-whitespace` | pre-commit-hooks | Espaces en fin de ligne |
| `no-commit-to-branch` | pre-commit-hooks | Protège `main` des commits directs |
| `detect-private-key` | pre-commit-hooks | Empêche les commits de secrets |
| `hadolint-docker` | v2.12.0 | Lint du Dockerfile |

---

### 3.11 Alerting

#### `monitoring/alert.rules.yml`

| Alerte | Expr PromQL | Durée | Sévérité | Description |
|---|---|---|---|---|
| `APIDown` | `absent(http_requests_total)` | 5 min | critical | Aucune métrique reçue |
| `HighErrorRate` | taux 5xx > 5% | 1 min | critical | Erreurs serveur |
| `ElevatedClientErrors` | taux 4xx > 20% | 3 min | warning | Erreurs client |
| `SlowPredictions` | p99 /predict > 500 ms | 2 min | warning | Latence élevée |
| `CriticalPredictionLatency` | p95 /predict > 2 s | 1 min | critical | Service dégradé |
| `ModelFallbackActive` | `mlops_model_registry_active == 0` | 1 min | warning | Registry MLFlow inaccessible |
| `UnexpectedTrafficSpike` | > 100 req/s sur /predict | 2 min | warning | Pic de trafic anormal |

#### `monitoring/alertmanager.yml`

| Section | Description |
|---|---|
| **Route critical** | group_wait 10s, repeat 15min → receiver `critical` |
| **Route warning** | group_wait 30s, repeat 1h → receiver `default` |
| **Inhibition** | Si `APIDown` → supprime toutes les autres alertes `mlops` |
| **Inhibition** | Si `critical` → supprime le `warning` du même alertname |
| **Receivers** | Webhook (template Slack / email à décommenter) |

---

### 3.12 Kubernetes — `k8s/`

| Fichier | Ressource | Description |
|---|---|---|
| `namespace.yml` | `Namespace` | Namespace `mlops` isolé |
| `configmap.yml` | `ConfigMap` | Variables non-secrètes : MLFlow URI, model name, log level |
| `deployment.yml` | `Deployment` | 2 replicas, rolling update zero-downtime (`maxUnavailable=0`), ressources CPU/RAM, readiness + liveness probes sur `/health`, PVC pour SQLite |
| `service.yml` | `Service` + `PVC` | ClusterIP port 80→8000 + `PersistentVolumeClaim` 1Gi pour predictions.db |
| `hpa.yml` | `HorizontalPodAutoscaler` | 2→10 pods, CPU 70%, memory 80%, scaleUp stable après 60s, scaleDown après 300s (anti-flapping) |

> L'API Key est injectée depuis un `Secret` Kubernetes (`mlops-secrets/api-key`) — jamais en clair dans le ConfigMap.

---

## 4. Flux complet end-to-end

### Phase A — Développement & CI

```
1. git push → main
   └─► GitHub Actions déclenche 3 jobs

2. Job "test" :
   ├─ pip install requirements.txt
   ├─ python src/training/train.py
   │     ├─ Charge données : load_iris() → split 80/20
   │     ├─ Pipeline = StandardScaler → RandomForestClassifier
   │     ├─ GridSearchCV : 9 combinaisons × 5 folds (45 fits)
   │     ├─ MLFlow : log_params (classifier__*), log_metric, set_tag, log_model(Pipeline)
   │     ├─ joblib.dump(best_pipeline, "models/random_forest.pkl")  [fallback]
   │     └─ Si accuracy ≥ 0.90 : set_registered_model_alias("champion", version=N)
   │
   ├─ pytest tests/ -v --cov=src --cov-fail-under=70
   │     ├─ tests/unit/test_api.py (4 tests) — modèle chargé depuis .pkl fallback
   │     └─ tests/integration/ (11 tests) — schéma, 3 classes, 422, model_source
   │
   └─ Upload coverage.xml

3. Job "push-image" (seulement sur push main) :
   ├─ Login GHCR avec GITHUB_TOKEN
   ├─ Tags : sha-abc1234 + latest
   ├─ docker build -f docker/Dockerfile.api (cache GHA)
   └─ docker push → ghcr.io/ORG/mlops-platform/mlops-api

4. Job "security-scan" :
   └─ Trivy scan → SARIF → GitHub Security tab
```

### Phase B — Déploiement (docker-compose)

```
5. docker-compose up --build

6. mlflow démarre → SQLite /mlflow/mlflow.db (volume persistant)
   └─ Healthcheck GET /health → OK

7. api démarre (attend mlflow healthy)
   ├─ _load_model() :
   │     ├─ mlflow.sklearn.load_model("models:/iris-random-forest@champion")
   │     │     ✓ → model_source = "models:/iris-random-forest@champion"
   │     │     │   MODEL_REGISTRY_ACTIVE.set(1)
   │     │     ✗ → joblib.load("models/random_forest.pkl")
   │     │           model_source = "local"
   │     │           MODEL_REGISTRY_ACTIVE.set(0)  ← déclenche alerte Alertmanager
   │     └─ model est un Pipeline(StandardScaler, RandomForestClassifier)
   ├─ _load_challenger() :
   │     ├─ mlflow.sklearn.load_model("models:/iris-random-forest@challenger")
   │     │     ✓ → challenger_model chargé, A/B possible si AB_TRAFFIC_SPLIT > 0
   │     │     ✗ → challenger_model = None (silence) — A/B inactif
   ├─ _build_explainer(model) → shap.TreeExplainer(classifier)
   ├─ PredictionLogger() → crée monitoring/predictions.db si absent
   └─ Instrumentator().instrument(app) → métriques Prometheus auto

8. prometheus démarre
   ├─ Monte prometheus.yml + alert.rules.yml
   ├─ Scrape api:8000 toutes les 15s
   └─ Évalue les 7 règles d'alerte toutes les 15s

9. alertmanager démarre
   └─ Écoute Prometheus sur :9093, route les alertes selon alertmanager.yml

10. grafana démarre
    ├─ Datasource "Prometheus" → http://prometheus:9090 (auto-provisionné)
    └─ Dashboard → mlops_dashboard.json (6 panels, refresh 10s)
```

### Phase C — Inférence unitaire

```
11. POST http://localhost:8000/predict
    Header : X-API-Key: secret  (si API_KEY configuré)
    Body   : {"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}

12. _verify_api_key() → OK (ou HTTP 403)
    Pydantic valide → PredictionRequest (valeurs > 0 vérifiées)
    t0 = time.perf_counter()

    _route_to_challenger() → False (default) ou True (si alias "challenger" chargé + random < split)
      ├─ False → active_model = model (champion), variant = "champion"
      └─ True  → active_model = challenger_model,  variant = "challenger"

    features = np.array([[5.1, 3.5, 1.4, 0.2]])
    active_model.predict(features)
        ├─ scaler.transform(features) → features normalisées
        └─ classifier.predict(scaled) → [0]
    active_model.predict_proba(features)[0][0] → 0.97
    latency_ms = (perf_counter() - t0) * 1000 → ex: 1.24 ms

    AB_REQUESTS.labels(variant="champion").inc()
    PredictionLogger.log(features, "setosa", 0, 0.97, active_source, 1.24)
    log.info("prediction", prediction="setosa", confidence=0.97, latency_ms=1.24, variant="champion")

    HTTP 200 :
    {
      "prediction":   "setosa",
      "class_id":     0,
      "confidence":   0.97,
      "model_source": "models:/iris-random-forest@champion",
      "variant":      "champion",
      "latency_ms":   1.24
    }

    Prometheus auto-incrémente :
    http_requests_total{handler="/predict", status_code="200"} += 1
    http_request_duration_seconds → observe(0.00124)
```

### Phase D — Inférence batch

```
13. POST /predict/batch
    Body : {"items": [{...}, {...}, ..., {...}]}  (≤ 1000 items)

    Vectorisation numpy d'un seul coup :
    matrix = np.array([[...], [...], ...])  shape (N, 4)
    class_ids = model.predict(matrix)       shape (N,)
    probas    = model.predict_proba(matrix) shape (N, 3)

    Log individuel pour chaque item (drift tracking)
    Retourne BatchPredictionResponse :
    {
      "predictions":      [...],
      "total":            N,
      "batch_latency_ms": 12.5
    }
```

### Phase E — Explication SHAP

```
14. POST /explain
    Body : {"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}

    model.predict(features)     → class_id = 0 (setosa)
    _pipeline_scaler.transform(features) → features normalisées
    explainer.shap_values(scaled)        → liste [arr_class0, arr_class1, arr_class2]
    class_shap = shap_vals[0][0]         → valeurs pour setosa

    HTTP 200 :
    {
      "prediction":  "setosa",
      "class_id":    0,
      "confidence":  0.97,
      "shap_values": {
        "sepal_length": -0.012,
        "sepal_width":   0.003,
        "petal_length":  0.521,   ← feature la plus déterminante
        "petal_width":   0.449
      },
      "model_source": "models:/iris-random-forest@champion"
    }
    (valeurs positives = poussent vers setosa, négatives = poussent contre)
```

### Phase F — Monitoring & Alerting

```
15. Prometheus évalue alert.rules.yml toutes les 15s

    Exemple — SlowPredictions déclenche :
    p99 latence /predict = 650ms > seuil 500ms pendant 2 min
    └─► Prometheus envoie alert à Alertmanager :9093

    Alertmanager route :
    ├─ severity=warning → receiver "default"
    │     group_wait 30s → envoie webhook / Slack
    └─ Si APIDown aussi firing :
          inhibit_rules → supprime SlowPredictions (conséquence, pas cause)

    ModelFallbackActive :
    └─ Si mlops_model_registry_active == 0 (mlflow down ou alias absent)
       → alerte "API is using local fallback model"
```

### Phase G — Drift Detection

```
16. L'équipe ML exécute périodiquement :
    from src.data.prediction_logger import PredictionLogger
    from src.monitoring.drift_detector import DriftDetector

    logs = PredictionLogger().get_recent(limit=500)
    # → 500 dernières prédictions depuis monitoring/predictions.db

    detector = DriftDetector()
    current_df = detector.build_current_data(logs)
    result = detector.detect(current_df)
    # → Compare distributions vs. 150 lignes Iris de référence
    # → DataDriftPreset : test statistique par feature (Wasserstein, KS, etc.)
    # → DataQualityPreset : valeurs manquantes, outliers
    # → Sauvegarde monitoring/reports/drift_YYYYMMDD_HHMMSS.html

    result = {
      "drift_detected":     True,
      "n_drifted_features": 2,
      "share_drifted":      0.50,
      "report_path":        "monitoring/reports/drift_20260430_143022.html",
      "n_samples":          500
    }

    Si drift_detected → re-entraîner via Kubeflow (Phase H)
```

### Phase H — Re-entraînement Kubeflow

```
17. Compilation et soumission :
    python kubeflow/pipeline.py → iris_mlops_pipeline.yaml
    kfp run create --experiment-name iris-mlops \
                   --pipeline-package-path iris_mlops_pipeline.yaml

18. Kubeflow orchestre 4 pods :

    [Pod 1] load_data
    → load_iris() → CSV train/test → MinIO/GCS

    [Pod 2] train_model  (lit train.csv)
    → Pipeline(StandardScaler + RF) + GridSearchCV
    → best_model.pkl + metadata KFP (best_params, cv_accuracy)

    [Pod 3] evaluate_model  (lit best_model + test.csv)
    → accuracy + F1 → KFP Metrics
    → retourne True si accuracy ≥ 0.90

    [Condition] True :
    [Pod 4] register_model
    → MLFlow run (tag: pipeline=kubeflow)
    → log_model(Pipeline) → Registry
    → set_registered_model_alias("champion", version=N+1)

19. Prochain redémarrage de l'API :
    _load_model() charge la nouvelle version N+1
    MODEL_REGISTRY_ACTIVE.set(1)
    log.info("model_loaded", source="models:/iris-random-forest@champion")
```

---

## 5. Variables d'environnement

Toutes lues depuis `src/config.py`. Peuvent être définies dans `.env` ou passées directement.

| Variable | Défaut | Utilisé dans | Description |
|---|---|---|---|
| `MLFLOW_TRACKING_URI` | `file:./mlruns` | train, api, kubeflow | URI du serveur MLFlow |
| `MLFLOW_MODEL_NAME` | `iris-random-forest` | train, api | Nom du modèle dans le Registry |
| `MLFLOW_MODEL_ALIAS` | `champion` | api | Alias du modèle à charger |
| `MLFLOW_CHALLENGER_ALIAS` | `challenger` | api | Alias du modèle challenger pour A/B testing |
| `AB_TRAFFIC_SPLIT` | `0.0` | api | Fraction du trafic /predict routée vers le challenger (0.0 = désactivé) |
| `ACCURACY_THRESHOLD` | `0.90` | train | Seuil de promotion |
| `FALLBACK_MODEL_PATH` | `models/random_forest.pkl` | train, api | Chemin du fallback local |
| `TEST_SIZE` | `0.20` | train | Fraction du jeu de test |
| `RANDOM_STATE` | `42` | train, preprocessing | Seed global |
| `API_KEY` | `None` | api | Clé d'authentification (désactivée si absente) |
| `LOG_LEVEL` | `INFO` | api, train | Niveau de log structlog |
| `PREDICTIONS_DB_URL` | `sqlite:///monitoring/predictions.db` | api, drift | SQLite prediction log |
| `DRIFT_REPORTS_DIR` | `monitoring/reports` | drift | Dossier des rapports HTML |

---

## 6. Commandes de démarrage rapide

```bash
# ── Préparation ──────────────────────────────────────────────────────────────

# Installer les dépendances
pip install -r requirements.txt

# Installer les pre-commit hooks (première fois seulement)
pre-commit install

# ── Entraînement ─────────────────────────────────────────────────────────────

# Entraîner le modèle (Pipeline StandardScaler+RF, grid search, MLFlow Registry)
python src/training/train.py
# ou
python -m src.training.train

# ── Tests ────────────────────────────────────────────────────────────────────

# Tests unitaires + intégration
pytest tests/ -v

# Avec couverture de code (seuil 70%)
pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=70

# ── Stack locale ─────────────────────────────────────────────────────────────

# Démarrer tous les services (MLFlow + API + Prometheus + Alertmanager + Grafana)
docker-compose up --build

# ── Appels API ───────────────────────────────────────────────────────────────

# Health check
curl http://localhost:8000/health

# Prédiction unitaire (sans auth)
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'

# Prédiction unitaire (avec API Key)
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -H "X-API-Key: votre-cle-secrete" \
     -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'

# Prédiction batch
curl -X POST http://localhost:8000/predict/batch \
     -H "Content-Type: application/json" \
     -d '{"items":[
       {"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2},
       {"sepal_length":6.3,"sepal_width":3.3,"petal_length":6.0,"petal_width":2.5}
     ]}'

# Explication SHAP
curl -X POST http://localhost:8000/explain \
     -H "Content-Type: application/json" \
     -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'

# Statistiques des prédictions loguées
curl http://localhost:8000/stats

# ── A/B Testing ──────────────────────────────────────────────────────────────

# Vérifier l'état A/B (champion + challenger + split actuel)
curl http://localhost:8000/ab/status

# Activer A/B testing : 20% du trafic vers le challenger
# 1. Assigner l'alias "challenger" à une autre version du modèle dans MLFlow
python -c "
from mlflow import MlflowClient
c = MlflowClient('http://localhost:5000')
c.set_registered_model_alias('iris-random-forest', 'challenger', version=1)
"
# 2. Redémarrer l'API avec AB_TRAFFIC_SPLIT=0.2 (ou via .env)
AB_TRAFFIC_SPLIT=0.2 uvicorn src.api.main:app --reload
# → /predict route automatiquement 20% vers le challenger
# → mlops_ab_requests_total{variant="champion"} et {variant="challenger"} visibles dans Prometheus

# ── Monitoring ───────────────────────────────────────────────────────────────

# Analyse de drift manuelle sur les 500 dernières prédictions
python -c "
from src.data.prediction_logger import PredictionLogger
from src.monitoring.drift_detector import DriftDetector
logs = PredictionLogger().get_recent(500)
detector = DriftDetector()
current = detector.build_current_data(logs)
print(detector.detect(current))
"

# Simulation drift (données artificiellement driftées)
python src/monitoring/drift_detector.py

# ── Kubeflow ─────────────────────────────────────────────────────────────────

# Compiler le pipeline
python kubeflow/pipeline.py  # → iris_mlops_pipeline.yaml

# Soumettre au cluster (kfp CLI requis + cluster configuré)
kfp run create --experiment-name iris-mlops \
               --pipeline-package-path iris_mlops_pipeline.yaml

# ── Kubernetes ───────────────────────────────────────────────────────────────

# Déployer sur un cluster K8s
kubectl apply -f k8s/namespace.yml
kubectl apply -f k8s/configmap.yml
kubectl apply -f k8s/service.yml
kubectl apply -f k8s/deployment.yml
kubectl apply -f k8s/hpa.yml

# Créer le secret API Key
kubectl create secret generic mlops-secrets \
  --from-literal=api-key=votre-cle-secrete \
  -n mlops

# ── Interfaces web ───────────────────────────────────────────────────────────
# MLFlow      → http://localhost:5000
# API Swagger → http://localhost:8000/docs
# Prometheus  → http://localhost:9090
# Alertmanager→ http://localhost:9093
# Grafana     → http://localhost:3000  (admin / admin)
```
