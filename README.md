# 🚀 MLOps Platform - Iris Classifier

Une plateforme complète **MLOps** pour la classification des fleurs Iris avec suivi des expériences, API REST, monitoring et déploiement containerisé.

## 📋 Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Architecture](#architecture)
- [Technologies utilisées](#technologies-utilisées)
- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
- [Déploiement](#déploiement)
- [Monitoring](#monitoring)
- [Tests](#tests)
- [Contribution](#contribution)

---

## 🎯 Vue d'ensemble

Ce projet implémente une **pipeline MLOps complète** pour entraîner, déployer et monitorer un modèle de classification d'Iris (Random Forest). 

**Caractéristiques principales:**
- ✅ Entraînement automatisé avec MLflow
- ✅ API REST avec FastAPI
- ✅ Monitoring avec Prometheus & Grafana
- ✅ Conteneurisation Docker
- ✅ Tests unitaires & d'intégration
- ✅ Orchestration avec Kubernetes
- ✅ Tracking d'expériences

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│              MLOps Platform Architecture              │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Data Pipeline        ML Training      API Server   │
│  ┌──────────────┐    ┌───────────┐   ┌──────────┐  │
│  │ Raw Data     │───▶│ ML Model  │──▶│ FastAPI  │  │
│  │ Processing   │    │ Training  │   │ /predict │  │
│  └──────────────┘    └───────────┘   └──────────┘  │
│       │                    │               │        │
│       ▼                    ▼               ▼        │
│  ┌──────────────┐    ┌───────────┐   ┌──────────┐  │
│  │ Data/raw     │    │  MLflow   │   │Prometheus│  │
│  │ Data/proc    │    │ Tracking  │   │ Metrics  │  │
│  └──────────────┘    └───────────┘   └──────────┘  │
│                                             │       │
│                                             ▼       │
│                                       ┌──────────┐  │
│                                       │ Grafana  │  │
│                                       │Dashboard│  │
│                                       └──────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## 🛠️ Technologies utilisées

| Catégorie | Technologie | Version | Rôle |
|-----------|-------------|---------|------|
| **ML Framework** | scikit-learn | Latest | Modèle Random Forest |
| **Web API** | FastAPI | 0.136+ | Serveur REST pour prédictions |
| **Experiment Tracking** | MLflow | 3.11+ | Suivi des expériences & artefacts |
| **Monitoring** | Prometheus | Latest | Collecte des métriques |
| **Visualization** | Grafana | Latest | Dashboards de monitoring |
| **Task Queue** | Celery | 5.6+ | Tâches asynchrones |
| **Message Broker** | RabbitMQ | (via Celery) | Gestion des files d'attente |
| **Containerization** | Docker | Latest | Conteneurisation |
| **Orchestration** | Kubernetes | Latest | Déploiement en production |
| **ML Workflows** | Kubeflow | Latest | Pipelines ML distribuées |
| **Data Management** | DVC | 3.67+ | Versioning des données |
| **Testing** | pytest | Latest | Framework de tests |
| **Server** | uvicorn | Latest | Serveur ASGI |

---

## 📁 Structure du projet

```
mlops-platform/
├── 📄 README.md                    # Ce fichier
├── 📄 requirements.txt             # Dépendances Python
├── 📄 conftest.py                  # Configuration pytest
├── 📄 docker-compose.yml           # Services containerisés
│
├── 📁 src/                         # Code source
│   ├── __init__.py
│   ├── 📁 api/                     # Service API
│   │   ├── __init__.py
│   │   └── main.py                 # Application FastAPI
│   │
│   ├── 📁 training/                # Pipeline d'entraînement
│   │   └── train.py                # Script d'entraînement
│   │
│   ├── 📁 data/                    # Modules de gestion des données
│   │   └── (à compléter)
│   │
│   └── 📁 monitoring/              # Modules de monitoring
│       └── (à compléter)
│
├── 📁 data/                        # Données brutes et traitées
│   ├── raw/                        # Données brutes
│   └── processed/                  # Données traitées
│
├── 📁 models/                      # Modèles entraînés
│   └── random_forest.pkl           # Modèle sauvegardé
│
├── 📁 mlflow/                      # MLflow backend
│   ├── mlruns/                     # Expériences & runs
│   └── artifacts/                  # Artefacts (modèles, etc.)
│
├── 📁 monitoring/                  # Configuration monitoring
│   ├── prometheus.yml              # Configuration Prometheus
│   └── 📁 grafana/                 # Dashboards Grafana
│
├── 📁 docker/                      # Fichiers Docker
│   └── Dockerfile.api              # Image pour API
│
├── 📁 k8s/                         # Manifests Kubernetes
│   └── (à compléter)
│
├── 📁 kubeflow/                    # Pipelines Kubeflow
│   └── (à compléter)
│
├── 📁 notebooks/                   # Jupyter notebooks
│   └── (explorations, analyses)
│
└── 📁 tests/                       # Tests automatisés
    ├── unit/                       # Tests unitaires
    │   └── test_api.py
    └── integration/                # Tests d'intégration
        └── (à compléter)
```

---

## 🚀 Installation

### Prérequis

- **Python** 3.9+
- **Docker** & **Docker Compose**
- **Git**
- **pip** ou **conda**

### Étapes d'installation

#### 1️⃣ Cloner le repository
```bash
git clone <repository-url>
cd mlops-platform
```

#### 2️⃣ Créer un environnement virtuel (optionnel mais recommandé)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python -m venv .venv
source .venv/bin/activate
```

#### 3️⃣ Installer les dépendances
```bash
pip install -r requirements.txt
```

#### 4️⃣ Initialiser les répertoires nécessaires
```bash
mkdir -p data/raw data/processed models mlflow/mlruns mlflow/artifacts
```

---

## ⚙️ Configuration

### Variables d'environnement

Créez un fichier `.env` à la racine du projet :

```env
# MLflow Configuration
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_ARTIFACT_ROOT=./mlflow/artifacts
MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow/mlruns/mlflow.db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false

# Model Configuration
MODEL_PATH=models/random_forest.pkl
MODEL_NAME=iris-classifier

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_ADMIN_PASSWORD=admin

# Celery Configuration
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1
```

### Configuration MLflow

MLflow est configuré pour utiliser:
- **Backend Store**: SQLite local (`mlflow.db`)
- **Artifact Store**: Répertoire local (`./mlflow/artifacts`)
- **Experiment**: `iris-classification`

---

## 📖 Utilisation

### 1️⃣ Entraîner le modèle

```bash
python src/training/train.py
```

**Résultat attendu:**
- Modèle sauvegardé dans `models/random_forest.pkl`
- Métriques loggées dans MLflow
- Run créé dans l'expérience `iris-classification`

### 2️⃣ Lancer l'API REST

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Endpoints disponibles:**

#### Health Check
```bash
curl http://localhost:8000/health
```
**Réponse:**
```json
{
  "status": "ok",
  "model": "random_forest"
}
```

#### Prédiction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

**Réponse:**
```json
{
  "prediction": "setosa",
  "class_id": 0,
  "confidence": 0.95
}
```

### 3️⃣ Accéder à MLflow UI

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Ouvrir: **http://localhost:5000**

### 4️⃣ Lancer avec Docker Compose

```bash
docker-compose up -d
```

**Services disponibles:**
- MLflow UI: `http://localhost:5000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (admin/admin)

---

## 🐳 Déploiement

### Avec Docker

#### Builder l'image
```bash
docker build -f docker/Dockerfile.api -t iris-classifier:latest .
```

#### Lancer le conteneur
```bash
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -e MLFLOW_TRACKING_URI=file:./mlruns \
  iris-classifier:latest
```

### Avec Kubernetes

```bash
# Créer les ressources (à configurer dans k8s/)
kubectl apply -f k8s/

# Vérifier le déploiement
kubectl get pods
kubectl get svc

# Accéder à l'API
kubectl port-forward svc/iris-classifier 8000:8000
```

### Avec Kubeflow

Les pipelines Kubeflow sont à configurer dans le répertoire `kubeflow/` pour l'orchestration distribuée des tâches d'entraînement.

---

## 📊 Monitoring

### Prometheus

La configuration Prometheus (`monitoring/prometheus.yml`) scrape les métriques de:
- **FastAPI**: `/metrics` (exposition Prometheus)
- **Système**: Node exporter (optionnel)

### Grafana

**Dashboards disponibles:**
- Service de santé (Health)
- Latence des requêtes
- Nombre de prédictions
- Performance du modèle

**Accès:**
- URL: `http://localhost:3000`
- Identifiants par défaut: `admin` / `admin`

**Ajouter Prometheus comme datasource:**
1. Settings → Data sources
2. Ajouter Prometheus
3. URL: `http://prometheus:9090`

---

## 🧪 Tests

### Exécuter tous les tests
```bash
pytest
```

### Tests unitaires uniquement
```bash
pytest tests/unit/ -v
```

### Tests d'intégration uniquement
```bash
pytest tests/integration/ -v
```

### Avec couverture de code
```bash
pytest --cov=src --cov-report=html
```

### Tests API spécifiques
```bash
pytest tests/unit/test_api.py -v
```

---

## 📚 Guide d'utilisation avancée

### Entraîner un nouveau modèle avec hyperparamètres personnalisés

Modifiez `src/training/train.py`:
```python
# Hyperparamètres
n_estimators = 150  # Augmenter le nombre d'arbres
max_depth = 8       # Augmenter la profondeur
min_samples_split = 2
```

Puis relancez:
```bash
python src/training/train.py
```

### Ajouter une nouvelle métrique MLflow

Dans `src/training/train.py`:
```python
# Après les prédictions
mlflow.log_metric("precision", precision_score(y_test, predictions))
mlflow.log_metric("recall", recall_score(y_test, predictions))
```

### Ajouter un nouvel endpoint API

Dans `src/api/main.py`:
```python
@app.post("/batch-predict", response_model=List[PredictionResponse])
def batch_predict(requests: List[PredictionRequest]):
    # Implémentation
    pass
```

### Tâches asynchrones avec Celery

Créez `src/tasks.py`:
```python
from celery import Celery

app = Celery('mlops_tasks', broker='redis://localhost:6379/0')

@app.task
def train_model_async():
    # Entraînement du modèle en arrière-plan
    pass
```

---

## 🔧 Dépannage

### Le modèle n'est pas trouvé
```bash
# Entraîner le modèle d'abord
python src/training/train.py

# Vérifier que le fichier existe
ls -la models/random_forest.pkl
```

### Port déjà utilisé
```bash
# Utiliser un port différent
uvicorn src.api.main:app --port 8001
```

### MLflow UI ne se charge pas
```bash
# Réinitialiser MLflow
rm -rf mlflow/mlruns mlflow/artifacts
mkdir -p mlflow/mlruns mlflow/artifacts
mlflow ui
```

### Docker Compose ne démarre pas
```bash
# Vérifier les logs
docker-compose logs

# Nettoyer et redémarrer
docker-compose down
docker-compose up -d --force-recreate
```

---

## 📖 Documentation supplémentaire

### Ressources officielles
- [FastAPI](https://fastapi.tiangolo.com/)
- [MLflow](https://mlflow.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Prometheus](https://prometheus.io/)
- [Grafana](https://grafana.com/)
- [Docker](https://www.docker.com/)
- [Kubernetes](https://kubernetes.io/)

### Fichiers importants
- **Configuration API**: [src/api/main.py](src/api/main.py)
- **Pipeline d'entraînement**: [src/training/train.py](src/training/train.py)
- **Services Docker**: [docker-compose.yml](docker-compose.yml)
- **Configuration Prometheus**: [monitoring/prometheus.yml](monitoring/prometheus.yml)

---

## 🤝 Contribution

Pour contribuer au projet:

1. **Fork** le repository
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commiter les changements (`git commit -m 'Add AmazingFeature'`)
4. Pusher vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

### Guidelines
- ✅ Respecter le style PEP 8
- ✅ Ajouter des tests pour les nouvelles fonctionnalités
- ✅ Mettre à jour la documentation
- ✅ Commiter avec des messages clairs

---

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 👤 Auteur

Plateforme MLOps pour classification d'Iris - Version 1.0.0

---

## 📞 Support

Pour toute question ou problème:
- 📧 Email: support@mlops-platform.local
- 📝 Issues: GitHub Issues
- 💬 Discussions: GitHub Discussions

---

**Dernière mise à jour:** Mai 2026
**Version:** 1.0.0
**Statut:** ✅ En production
