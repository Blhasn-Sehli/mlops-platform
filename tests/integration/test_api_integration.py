"""
Integration tests — run AFTER training (train.py must have been executed).
These tests exercise the full API stack including model loading from MLFlow or fallback.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from src.api.main import app

    return TestClient(app)


# ---------------------------------------------------------------------------
# Health & metadata
# ---------------------------------------------------------------------------


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_source" in data


def test_health_reports_model_source(client):
    data = client.get("/health").json()
    assert data["model_source"] in [
        "local",
        "models:/iris-random-forest@champion",
    ] or data[
        "model_source"
    ].startswith("models:/")


def test_metrics_endpoint_shape(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "RandomForestClassifier" in data["model"]
    assert data["features"] == 4
    assert len(data["classes"]) == 3
    assert "model_source" in data


# ---------------------------------------------------------------------------
# Prediction correctness
# ---------------------------------------------------------------------------

IRIS_CASES = [
    # (features_dict, expected_class)
    ({"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}, "setosa"),
    (
        {"sepal_length": 6.0, "sepal_width": 2.7, "petal_length": 5.1, "petal_width": 1.6},
        "versicolor",
    ),
    (
        {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5},
        "virginica",
    ),
]


@pytest.mark.parametrize("features, expected", IRIS_CASES)
def test_predict_known_cases(client, features, expected):
    response = client.post("/predict", json=features)
    assert response.status_code == 200
    data = response.json()
    assert (
        data["prediction"] == expected
    ), f"Expected {expected!r}, got {data['prediction']!r} for {features}"


def test_predict_response_schema(client):
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {
        "prediction",
        "class_id",
        "confidence",
        "model_source",
        "variant",
        "latency_ms",
    }
    assert data["prediction"] in ["setosa", "versicolor", "virginica"]
    assert isinstance(data["class_id"], int)
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["latency_ms"] >= 0


def test_predict_setosa_high_confidence(client):
    # Setosa is linearly separable — model should be very confident
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )
    assert response.json()["confidence"] >= 0.80


def test_predict_class_id_matches_class_name(client):
    class_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    for features, _ in IRIS_CASES:
        data = client.post("/predict", json=features).json()
        assert class_map[data["class_id"]] == data["prediction"]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_predict_missing_field_returns_422(client):
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            # missing petal_width
        },
    )
    assert response.status_code == 422


def test_predict_wrong_type_returns_422(client):
    response = client.post(
        "/predict",
        json={
            "sepal_length": "not_a_number",
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )
    assert response.status_code == 422


def test_predict_empty_body_returns_422(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Prometheus metrics endpoint
# ---------------------------------------------------------------------------


def test_predict_variant_field(client):
    data = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    ).json()
    # With ab_traffic_split=0.0 (default), all requests go to champion
    assert data["variant"] in ("champion", "challenger")


def test_ab_status_endpoint(client):
    response = client.get("/ab/status")
    assert response.status_code == 200
    data = response.json()
    assert "ab_enabled" in data
    assert "traffic_split" in data
    assert "champion" in data
    assert "challenger" in data
    assert data["champion"]["loaded"] is True
    # Challenger is not loaded in test environment (no registry alias set)
    assert isinstance(data["challenger"]["loaded"], bool)


def test_prometheus_metrics_exposed(client):
    # Make a request first to generate metrics
    client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )
    response = client.get("/metrics")
    # Our custom /metrics endpoint returns JSON
    assert response.status_code == 200
