from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_setosa():
    response = client.post(
        "/predict",
        json={"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    )
    assert response.status_code == 200
    assert response.json()["prediction"] == "setosa"


def test_predict_virginica():
    response = client.post(
        "/predict",
        json={"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5},
    )
    assert response.status_code == 200
    assert response.json()["prediction"] == "virginica"


def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "model" in response.json()
