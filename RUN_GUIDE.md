# 🚀 Run Guide — Local, Kubernetes & Full MLOps Flow

This guide explains how to run the project in three modes:

* Local execution
* Kubernetes deployment
* End-to-end MLOps pipeline (Kubeflow + MLflow)

---

## ⚠️ Important Rule

Use **only one mode at a time**:

* Kubernetes mode → API, MLflow, Kubeflow run in cluster
* Local monitoring mode → Prometheus/Grafana via Docker Compose
* ❌ Do NOT mix Kubernetes port-forwarding with local monitoring stack

---

# 1. ✅ Prerequisites

Make sure you have:

* Windows
* Python 3.11+
* Docker Desktop running
* kubectl configured
* Kubernetes cluster available
* Virtual environment already created (`venv/`)

---

# 2. 💻 Local Execution Flow

## 2.1 Activate environment

### PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

### Git Bash

```bash
source /d/ME/mlops-platform/venv/Scripts/activate
```

---

## 2.2 Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2.3 Run tests

### Unit tests

```bash
pytest tests/unit/ -v
```

### Integration tests

```bash
pytest tests/integration/ -v
```

---

## 2.4 Train model

```bash
python src/training/train.py
```

Expected:

* `models/random_forest.pkl` created/updated
* MLflow run logged locally

---

## 2.5 Start API

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 2.6 Test API

### Health check

```bash
curl http://localhost:8000/health
```

### Prediction

```bash
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"sepal_length\":5.1,\"sepal_width\":3.5,\"petal_length\":1.4,\"petal_width\":0.2}"
```

---

## 2.7 Optional: Monitoring stack

```bash
docker compose up -d --no-deps prometheus grafana alertmanager
```

Access:

* Prometheus → [http://localhost:9090](http://localhost:9090)
* Grafana → [http://localhost:3000](http://localhost:3000)
* Alertmanager → [http://localhost:9093](http://localhost:9093)

---

# 3. ☸️ Kubernetes Deployment Flow

---

## 3.1 Build API image

```bash
docker build -f docker/Dockerfile.api -t mlops-api:latest .
```

---

## 3.2 Create namespace

```bash
kubectl apply -f k8s/namespace.yml
```

---

## 3.3 Deploy core resources

```bash
kubectl apply -f k8s/configmap.yml
kubectl apply -f k8s/service.yml
kubectl apply -f k8s/deployment.yml
kubectl apply -f k8s/hpa.yml
```

---

# 3.4 Kubeflow Pipelines

## ✅ Option 1 (Recommended: simple local execution)

Use this if you do NOT need full Kubeflow cluster:

```bash
python kubeflow/pipeline.py
python run_pipeline.py
```

This runs:

* load_data
* train_model
* evaluate_model
* register_model

---

## ⚙️ Option 2 (Full Kubeflow on Kubernetes)

### Step 1: Set version

```powershell
$env:PIPELINE_VERSION="2.15.0"
```

### Step 2: Download manifests

```bash
Invoke-WebRequest -Uri "https://github.com/kubeflow/pipelines/archive/refs/tags/2.15.0.zip" -OutFile "D:\kfp.zip"
Expand-Archive "D:\kfp.zip" -DestinationPath "D:\kfp"
```

### Step 3: Install cluster resources

```bash
kubectl apply -k "D:\kfp\pipelines-2.15.0\manifests\kustomize\cluster-scoped-resources"
```

### Step 4: Install platform

```bash
kubectl apply -k "D:\kfp\pipelines-2.15.0\manifests\kustomize\env\platform-agnostic"
```

### Step 5: Check pods

```bash
kubectl get pods -n kubeflow
```

### Step 6: Port-forward UI

```bash
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

Open:
👉 [http://localhost:3000](http://localhost:3000)

---

# 3.5 Deploy MLflow in Kubernetes

```bash
kubectl apply -f k8s/mlflow-secrets.yml
kubectl apply -f k8s/mlflow-postgres-pvc.yml
kubectl apply -f k8s/mlflow-postgres.yml
kubectl apply -f k8s/mlflow-server.yml
```

Check:

```bash
kubectl get pods -n mlops
```

---

# 3.6 Run pipeline

```bash
python kubeflow/pipeline.py
```

or

```bash
kfp run create --experiment-name iris-mlops --package-file iris_mlops_pipeline.yaml --watch
```

---

# 3.7 Access MLflow

```bash
kubectl port-forward -n mlops svc/mlflow 5000:5000
```

Open:
👉 [http://localhost:5000](http://localhost:5000)

---

# 3.8 Access API

```bash
kubectl port-forward -n mlops svc/mlops-api-service 8000:80
```

Test:

```bash
curl http://localhost:8000/health
```

---

# 4. 🔁 End-to-End Flow

## Order

1. Start Docker Desktop
2. Activate venv
3. Install dependencies
4. Run tests
5. Start Kubernetes services
6. Port-forward MLflow + API
7. Run pipeline
8. Verify MLflow model registry
9. Test API

---

## Quick run

```bash
kubectl port-forward -n mlops svc/mlflow 5000:5000
kubectl port-forward -n mlops svc/mlops-api-service 8000:80

pytest tests/unit/ -v
pytest tests/integration/ -v
python run_pipeline.py
```

---

# 5. ⚠️ Troubleshooting

### API not starting

* Check `models/random_forest.pkl`

### MLflow not reachable

```bash
kubectl get pods -n mlops
kubectl logs -n mlops deploy/mlflow-server
```

### Pipeline fails

* Check MLflow port-forward
* Check dataset loading
* Check accuracy threshold (> 90%)

### Port-forward issues

* Another process may already use port 5000 or 8000

---

# ✅ Done

You now have a full MLOps workflow:

* Local training
* Kubernetes deployment
* Kubeflow pipeline
* MLflow model registry
* FastAPI inference service
