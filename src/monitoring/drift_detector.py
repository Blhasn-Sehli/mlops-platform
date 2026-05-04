"""
Data drift detection using Evidently AI.

Usage:
    detector = DriftDetector()
    recent = PredictionLogger().get_recent(limit=500)
    current_df = detector.build_current_data(recent)
    result = detector.detect(current_df)
    print(result["drift_detected"], result["report_path"])

CLI:
    python src/monitoring/drift_detector.py
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report
from sklearn.datasets import load_iris

from src.config import settings
from src.logging_config import get_logger

log = get_logger(__name__)

FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_NAME = "species"


class DriftDetector:
    """Detects distribution shift between reference (training) data and production data."""

    def __init__(self, reports_dir: str = settings.drift_reports_dir):
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)
        self.reference_data = self._build_reference()
        self.column_mapping = ColumnMapping(
            target=TARGET_NAME,
            numerical_features=FEATURE_NAMES,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, current_data: pd.DataFrame) -> dict:
        """
        Run drift analysis against the Iris training distribution.

        Returns:
            drift_detected (bool), n_drifted_features (int),
            share_drifted (float), report_path (str),
            timestamp (str), n_samples (int).
        """
        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.reports_dir, f"drift_{timestamp}.html")
        report.save_html(report_path)

        summary = report.as_dict()
        drift_result = summary["metrics"][0]["result"]

        result = {
            "drift_detected": drift_result["dataset_drift"],
            "n_drifted_features": drift_result.get("number_of_drifted_columns", 0),
            "share_drifted": drift_result.get("share_of_drifted_columns", 0.0),
            "report_path": report_path,
            "timestamp": timestamp,
            "n_samples": len(current_data),
        }

        log.info("drift_check_complete", **{k: v for k, v in result.items() if k != "report_path"})
        return result

    def build_current_data(self, predictions_log: list[dict]) -> pd.DataFrame:
        """
        Convert PredictionLogger.get_recent() output into an Evidently-compatible DataFrame.
        Each dict must have: sepal_length, sepal_width, petal_length, petal_width, class_id.
        """
        rows = [
            {
                "sepal_length": e["sepal_length"],
                "sepal_width": e["sepal_width"],
                "petal_length": e["petal_length"],
                "petal_width": e["petal_width"],
                TARGET_NAME: e["class_id"],
            }
            for e in predictions_log
        ]
        return pd.DataFrame(rows)

    def simulate_drifted_data(self, n_samples: int = 100, noise_scale: float = 1.5) -> pd.DataFrame:
        """Generate artificially drifted data (demo / testing only)."""
        rng = np.random.default_rng(seed=settings.random_state)
        iris = load_iris()
        indices = rng.choice(len(iris.data), size=n_samples, replace=True)
        data = iris.data[indices].copy() + rng.normal(0, noise_scale, (n_samples, 4))
        df = pd.DataFrame(data, columns=FEATURE_NAMES)
        df[TARGET_NAME] = iris.target[indices]
        return df

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_reference(self) -> pd.DataFrame:
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=FEATURE_NAMES)
        df[TARGET_NAME] = iris.target
        return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    detector = DriftDetector()

    log.info("drift_cli_started", scenario="clean_data")
    clean = detector._build_reference().sample(50, random_state=0)
    r1 = detector.detect(clean)
    log.info("drift_result", **r1)

    log.info("drift_cli_started", scenario="drifted_data")
    drifted = detector.simulate_drifted_data(n_samples=100, noise_scale=2.0)
    r2 = detector.detect(drifted)
    log.info("drift_result", **r2)
