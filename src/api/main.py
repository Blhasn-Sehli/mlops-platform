from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from prometheus_fastapi_instrumentator import Instrumentator

# Load the trained model
MODEL_PATH = "models/random_forest.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# Iris class names
CLASSES = ["setosa", "versicolor", "virginica"]

# FastAPI app
app = FastAPI(
    title="MLOps Platform - Iris Classifier",
    description="REST API for Iris flower classification",
    version="1.0.0"
)

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Input schema
class PredictionRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Output schema
class PredictionResponse(BaseModel):
    prediction: str
    class_id: int
    confidence: float

# Routes
@app.get("/health")
def health():
    return {"status": "ok", "model": "random_forest"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        features = np.array([[
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width
        ]])

        class_id = int(model.predict(features)[0])
        confidence = float(model.predict_proba(features)[0][class_id])

        return PredictionResponse(
            prediction=CLASSES[class_id],
            class_id=class_id,
            confidence=round(confidence, 4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
def model_info():
    return {
        "model": "RandomForestClassifier",
        "features": 4,
        "classes": CLASSES
    }