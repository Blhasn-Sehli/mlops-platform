import random
import time

import mlflow.sklearn
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from prometheus_client import Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, field_validator

import mlflow
from src.config import settings
from src.data.prediction_logger import PredictionLogger
from src.logging_config import get_logger

log = get_logger(__name__)

CLASSES = ["setosa", "versicolor", "virginica"]
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# 1 = model from MLFlow Registry, 0 = local fallback
MODEL_REGISTRY_ACTIVE = Gauge(
    "mlops_model_registry_active",
    "1 if the model was loaded from MLFlow Registry, 0 if local fallback",
)

# Counts predictions per A/B variant — enables per-variant throughput/error dashboards
AB_REQUESTS = Counter(
    "mlops_ab_requests_total",
    "Prediction requests routed per A/B variant",
    ["variant"],
)


# ── Model loading ─────────────────────────────────────────────────────────────


def _load_model():
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    uri = f"models:/{settings.mlflow_model_name}@{settings.mlflow_model_alias}"
    try:
        loaded = mlflow.sklearn.load_model(uri)
        MODEL_REGISTRY_ACTIVE.set(1)
        log.info("model_loaded", source=uri)
        return loaded, uri
    except Exception as exc:
        log.warning(
            "mlflow_registry_unavailable", error=str(exc), fallback=settings.fallback_model_path
        )

    import os

    import joblib

    if not os.path.exists(settings.fallback_model_path):
        raise FileNotFoundError(
            f"No model in registry and no fallback at {settings.fallback_model_path}. "
            "Run src/training/train.py first."
        )
    loaded = joblib.load(settings.fallback_model_path)
    MODEL_REGISTRY_ACTIVE.set(0)
    log.info("model_loaded", source="local", path=settings.fallback_model_path)
    return loaded, "local"


def _load_challenger():
    """Load the challenger model from MLFlow Registry. Returns (None, None) if unavailable."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    uri = f"models:/{settings.mlflow_model_name}@{settings.mlflow_challenger_alias}"
    try:
        loaded = mlflow.sklearn.load_model(uri)
        log.info("challenger_loaded", source=uri)
        return loaded, uri
    except Exception as exc:
        log.info("challenger_unavailable", alias=settings.mlflow_challenger_alias, error=str(exc))
        return None, None


def _build_explainer(loaded_model):
    """Build a SHAP TreeExplainer, extracting the classifier from a sklearn Pipeline if needed."""
    try:
        import shap

        if hasattr(loaded_model, "named_steps"):
            classifier = loaded_model.named_steps["classifier"]
            scaler = loaded_model.named_steps.get("scaler")
        else:
            classifier = loaded_model
            scaler = None
        expl = shap.TreeExplainer(classifier)
        log.info("shap_explainer_ready")
        return expl, scaler
    except Exception as exc:
        log.warning("shap_explainer_unavailable", error=str(exc))
        return None, None


model, model_source = _load_model()
explainer, _pipeline_scaler = _build_explainer(model)
challenger_model, challenger_source = _load_challenger()
prediction_logger = PredictionLogger()


def _route_to_challenger() -> bool:
    """Return True if this request should be served by the challenger model."""
    if challenger_model is None or settings.ab_traffic_split <= 0.0:
        return False
    return random.random() < settings.ab_traffic_split


# ── Authentication ────────────────────────────────────────────────────────────

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _verify_api_key(api_key: str | None = Security(_API_KEY_HEADER)) -> None:
    """Enforce API key when settings.api_key is configured. Pass-through in dev mode."""
    if settings.api_key and api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid or missing API key (X-API-Key header)")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MLOps Platform - Iris Classifier",
    description="REST API for Iris flower classification",
    version="1.0.0",
)

Instrumentator().instrument(app).expose(app)


# ── Schemas ───────────────────────────────────────────────────────────────────


class PredictionRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    @field_validator("sepal_length", "sepal_width", "petal_length", "petal_width")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("feature values must be positive")
        return v


class PredictionResponse(BaseModel):
    prediction: str
    class_id: int
    confidence: float
    model_source: str
    variant: str
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    items: list[PredictionRequest]

    @field_validator("items")
    @classmethod
    def items_not_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("items list must not be empty")
        if len(v) > 1000:
            raise ValueError("batch size must not exceed 1000")
        return v


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total: int
    batch_latency_ms: float


class ExplainRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class ExplainResponse(BaseModel):
    prediction: str
    class_id: int
    confidence: float
    shap_values: dict[str, float]
    model_source: str


# ── Helpers ───────────────────────────────────────────────────────────────────


def _vectorize(req: PredictionRequest) -> np.ndarray:
    return np.array([[req.sepal_length, req.sepal_width, req.petal_length, req.petal_width]])


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "random_forest",
        "model_source": model_source,
        "total_predictions": prediction_logger.count(),
    }


@app.get("/metrics")
def metrics():
    return {
        "model": "Pipeline(StandardScaler + RandomForestClassifier)",
        "features": 4,
        "classes": CLASSES,
        "model_source": model_source,
    }


@app.get("/stats")
def stats():
    """Aggregate statistics over all logged predictions."""
    return prediction_logger.stats()


@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(_verify_api_key)],
)
def predict(request: PredictionRequest):
    t0 = time.perf_counter()
    try:
        features = _vectorize(request)

        use_challenger = _route_to_challenger()
        active_model = challenger_model if use_challenger else model
        active_source = challenger_source if use_challenger else model_source
        variant = "challenger" if use_challenger else "champion"

        class_id = int(active_model.predict(features)[0])
        confidence = float(active_model.predict_proba(features)[0][class_id])
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        AB_REQUESTS.labels(variant=variant).inc()
        prediction_logger.log(
            features=request.model_dump(),
            prediction=CLASSES[class_id],
            class_id=class_id,
            confidence=confidence,
            model_source=active_source,
            latency_ms=latency_ms,
        )
        log.info(
            "prediction",
            prediction=CLASSES[class_id],
            confidence=confidence,
            latency_ms=latency_ms,
            variant=variant,
        )

        return PredictionResponse(
            prediction=CLASSES[class_id],
            class_id=class_id,
            confidence=round(confidence, 4),
            model_source=active_source,
            variant=variant,
            latency_ms=latency_ms,
        )
    except HTTPException:
        raise
    except Exception as exc:
        log.error("prediction_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    dependencies=[Depends(_verify_api_key)],
)
def predict_batch(request: BatchPredictionRequest):
    t0 = time.perf_counter()
    try:
        # Vectorize all items at once for efficiency
        matrix = np.array(
            [
                [it.sepal_length, it.sepal_width, it.petal_length, it.petal_width]
                for it in request.items
            ]
        )
        class_ids = model.predict(matrix).tolist()
        probas = model.predict_proba(matrix).tolist()
        batch_latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        responses: list[PredictionResponse] = []
        for i, item in enumerate(request.items):
            cid = int(class_ids[i])
            conf = round(float(probas[i][cid]), 4)
            prediction_logger.log(
                features=item.model_dump(),
                prediction=CLASSES[cid],
                class_id=cid,
                confidence=conf,
                model_source=model_source,
                latency_ms=None,
            )
            responses.append(
                PredictionResponse(
                    prediction=CLASSES[cid],
                    class_id=cid,
                    confidence=conf,
                    model_source=model_source,
                    variant="champion",
                    latency_ms=0.0,
                )
            )

        log.info("batch_prediction", n_items=len(request.items), batch_latency_ms=batch_latency_ms)
        return BatchPredictionResponse(
            predictions=responses,
            total=len(responses),
            batch_latency_ms=batch_latency_ms,
        )
    except HTTPException:
        raise
    except Exception as exc:
        log.error("batch_prediction_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/ab/status")
def ab_status():
    """Current A/B testing configuration and model availability."""
    return {
        "ab_enabled": challenger_model is not None and settings.ab_traffic_split > 0.0,
        "traffic_split": settings.ab_traffic_split,
        "champion": {
            "loaded": True,
            "source": model_source,
            "alias": settings.mlflow_model_alias,
        },
        "challenger": {
            "loaded": challenger_model is not None,
            "source": challenger_source,
            "alias": settings.mlflow_challenger_alias,
        },
    }


@app.post(
    "/explain",
    response_model=ExplainResponse,
    dependencies=[Depends(_verify_api_key)],
)
def explain(request: ExplainRequest):
    if explainer is None:
        raise HTTPException(status_code=503, detail="SHAP explainer not available for this model")
    try:
        features = np.array(
            [[request.sepal_length, request.sepal_width, request.petal_length, request.petal_width]]
        )
        class_id = int(model.predict(features)[0])
        confidence = float(model.predict_proba(features)[0][class_id])

        # Apply scaler before passing to TreeExplainer (operates on classifier input space)
        features_for_shap = _pipeline_scaler.transform(features) if _pipeline_scaler else features
        shap_vals = explainer.shap_values(features_for_shap)

        # For multiclass RF, shap_values is a list [class_0_array, class_1_array, class_2_array]
        class_shap = shap_vals[class_id][0] if isinstance(shap_vals, list) else shap_vals[0]

        return ExplainResponse(
            prediction=CLASSES[class_id],
            class_id=class_id,
            confidence=round(confidence, 4),
            shap_values={
                name: round(float(val), 4) for name, val in zip(FEATURE_NAMES, class_shap)
            },
            model_source=model_source,
        )
    except HTTPException:
        raise
    except Exception as exc:
        log.error("explain_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
