from functools import lru_cache
import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── MLFlow ──────────────────────────────────────────────────────────────
    # Default: use persistent Kubernetes MLflow if available, else local file backend
    mlflow_tracking_uri: str = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://mlflow.mlops:5000"  # K8s persistent MLflow
        if os.getenv("KUBERNETES_SERVICE_HOST") else "file:./mlruns"  # Local fallback
    )
    mlflow_model_name: str = "iris-random-forest"
    mlflow_model_alias: str = "champion"
    mlflow_challenger_alias: str = "challenger"

    # ── A/B testing ─────────────────────────────────────────────────────────
    # Fraction of /predict traffic routed to the challenger model (0.0 = disabled)
    ab_traffic_split: float = 0.0

    # ── Training ────────────────────────────────────────────────────────────
    accuracy_threshold: float = 0.90
    fallback_model_path: str = "models/random_forest.pkl"
    test_size: float = 0.20
    random_state: int = 42

    # ── API ─────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    api_key: str | None = None  # if None, authentication is disabled (dev mode)

    # ── Monitoring ──────────────────────────────────────────────────────────
    predictions_db_url: str = "sqlite:///monitoring/predictions.db"
    drift_reports_dir: str = "monitoring/reports"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
