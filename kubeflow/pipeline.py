"""
Kubeflow Pipelines v2 — Iris MLOps Pipeline.

Steps:
    1. load_data     — charge le dataset Iris et le sauvegarde
    2. train_model   — entraîne un RandomForest avec grid search
    3. evaluate      — calcule accuracy & F1, décide si on enregistre
    4. register      — enregistre dans MLFlow Model Registry + alias champion

Compile:
    python kubeflow/pipeline.py
    # → produces iris_mlops_pipeline.yaml

Submit (requires a running KFP cluster):
    kfp run create --experiment-name iris-mlops \
                   --pipeline-package-path iris_mlops_pipeline.yaml
"""

from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output, component, pipeline

BASE_IMAGE = "python:3.11-slim"
SKLEARN_PACKAGES = ["scikit-learn==1.8.0", "pandas==2.3.3", "numpy==2.4.4", "joblib==1.5.3"]
MLFLOW_PACKAGES = ["mlflow==3.11.1"]


# ---------------------------------------------------------------------------
# Component 1 — Load & validate data
# ---------------------------------------------------------------------------


@component(
    base_image=BASE_IMAGE,
    packages_to_install=SKLEARN_PACKAGES,
)
def load_data(output_train: Output[Dataset], output_test: Output[Dataset]):
    """Load Iris dataset and persist train/test splits as CSV."""
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    df = pd.DataFrame(iris.data, columns=feature_names)
    df["target"] = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_names], df["target"], test_size=0.2, random_state=42
    )

    train_df = X_train.copy()
    train_df["target"] = y_train.values
    test_df = X_test.copy()
    test_df["target"] = y_test.values

    train_df.to_csv(output_train.path, index=False)
    test_df.to_csv(output_test.path, index=False)

    print(f"Train samples : {len(train_df)}")
    print(f"Test  samples : {len(test_df)}")


# ---------------------------------------------------------------------------
# Component 2 — Train with grid search
# ---------------------------------------------------------------------------


@component(
    base_image=BASE_IMAGE,
    packages_to_install=SKLEARN_PACKAGES,
)
def train_model(
    input_train: Input[Dataset],
    output_model: Output[Model],
    n_estimators: int = 100,
    max_depth: int = 5,
):
    """Train RandomForest with grid search and save best model."""
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV

    train_df = pd.read_csv(input_train.path)
    X_train = train_df.drop("target", axis=1)
    y_train = train_df["target"]

    param_grid = {
        "n_estimators": [50, n_estimators, n_estimators * 2],
        "max_depth": [3, max_depth, max_depth * 2],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    joblib.dump(grid.best_estimator_, output_model.path)

    print(f"Best params : {grid.best_params_}")
    print(f"Best CV acc : {grid.best_score_:.4f}")

    output_model.metadata["best_params"] = str(grid.best_params_)
    output_model.metadata["cv_accuracy"] = str(round(grid.best_score_, 4))


# ---------------------------------------------------------------------------
# Component 3 — Evaluate
# ---------------------------------------------------------------------------


@component(
    base_image=BASE_IMAGE,
    packages_to_install=SKLEARN_PACKAGES,
)
def evaluate_model(
    input_model: Input[Model],
    input_test: Input[Dataset],
    metrics: Output[Metrics],
    accuracy_threshold: float = 0.90,
) -> bool:
    """Evaluate model on test set; returns True if above threshold."""
    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score

    model = joblib.load(input_model.path)

    test_df = pd.read_csv(input_test.path)
    X_test = test_df.drop("target", axis=1)
    y_test = test_df["target"]

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    metrics.log_metric("accuracy", round(accuracy, 4))
    metrics.log_metric("f1_score", round(f1, 4))
    metrics.log_metric("threshold", accuracy_threshold)
    metrics.log_metric("promoted", int(accuracy >= accuracy_threshold))

    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Promoted : {accuracy >= accuracy_threshold}")

    return accuracy >= accuracy_threshold


# ---------------------------------------------------------------------------
# Component 4 — Register in MLFlow
# ---------------------------------------------------------------------------


@component(
    base_image=BASE_IMAGE,
    packages_to_install=SKLEARN_PACKAGES + MLFLOW_PACKAGES,
)
def register_model(
    input_model: Input[Model],
    mlflow_tracking_uri: str,
    model_name: str = "iris-random-forest",
):
    """Register the model in MLFlow Model Registry and assign 'champion' alias."""
    import joblib
    import mlflow.sklearn

    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("iris-classification-kubeflow")

    model = joblib.load(input_model.path)

    with mlflow.start_run() as run:
        mlflow.set_tag("pipeline", "kubeflow")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
        )
        run_id = run.info.run_id

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}' and run_id='{run_id}'")
    if versions:
        version = versions[0].version
        client.set_registered_model_alias(model_name, "champion", version)
        print(f"✅ Model v{version} promoted to alias 'champion'")
    else:
        print("⚠️  Could not find registered version")


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------


@pipeline(
    name="iris-mlops-pipeline",
    description="End-to-end MLOps pipeline: load → train → evaluate → register",
)
def iris_pipeline(
    mlflow_tracking_uri: str = "http://mlflow:5000",
    n_estimators: int = 100,
    max_depth: int = 5,
    accuracy_threshold: float = 0.90,
):
    load_task = load_data()

    train_task = train_model(
        input_train=load_task.outputs["output_train"],
        n_estimators=n_estimators,
        max_depth=max_depth,
    )

    eval_task = evaluate_model(
        input_model=train_task.outputs["output_model"],
        input_test=load_task.outputs["output_test"],
        accuracy_threshold=accuracy_threshold,
    )

    # Conditional registration — only if evaluate_model returns True
    with dsl.If(eval_task.output, name="above-threshold"):
        register_model(
            input_model=train_task.outputs["output_model"],
            mlflow_tracking_uri=mlflow_tracking_uri,
        )


# ---------------------------------------------------------------------------
# Compile when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    output_file = "iris_mlops_pipeline.yaml"
    compiler.Compiler().compile(iris_pipeline, output_file)
    print(f"✅ Pipeline compiled → {output_file}")
