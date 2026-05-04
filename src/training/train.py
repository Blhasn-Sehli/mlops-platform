import os

import joblib
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import mlflow
from mlflow import MlflowClient
from src.config import settings
from src.logging_config import get_logger

log = get_logger(__name__)

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
mlflow.set_experiment("iris-classification")

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=settings.test_size,
    random_state=settings.random_state,
)

# Pipeline: StandardScaler guarantees the same transformation at train and inference time.
# GridSearchCV keys use the "classifier__" prefix to target the second step.
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(random_state=settings.random_state)),
    ]
)

param_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [3, 5, 10],
}

with mlflow.start_run() as run:
    log.info("training_started", experiment="iris-classification", param_grid=str(param_grid))

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_

    predictions = best_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")

    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.set_tag("model_type", "Pipeline(StandardScaler+RandomForest)")
    mlflow.set_tag("dataset", "iris")

    # The full Pipeline is logged — scaler is bundled with the classifier.
    mlflow.sklearn.log_model(
        sk_model=best_pipeline,
        artifact_path="model",
        registered_model_name=settings.mlflow_model_name,
    )

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipeline, settings.fallback_model_path)

    run_id = run.info.run_id

    log.info(
        "training_complete",
        run_id=run_id,
        best_params=best_params,
        accuracy=round(accuracy, 4),
        f1_score=round(f1, 4),
    )

# Promote to 'champion' alias if accuracy meets threshold
client = MlflowClient()
versions = client.search_model_versions(
    f"name='{settings.mlflow_model_name}' and run_id='{run_id}'"
)

if versions:
    version = versions[0].version
    if accuracy >= settings.accuracy_threshold:
        client.set_registered_model_alias(settings.mlflow_model_name, "champion", version)
        log.info(
            "model_promoted",
            version=version,
            alias="champion",
            accuracy=round(accuracy, 4),
            threshold=settings.accuracy_threshold,
        )
    else:
        log.warning(
            "model_not_promoted",
            accuracy=round(accuracy, 4),
            threshold=settings.accuracy_threshold,
        )
else:
    log.warning("registered_version_not_found", run_id=run_id)
