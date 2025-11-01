# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import os
import joblib
import numpy as np
import sys
import app.srf_model as srf_model

app = FastAPI(title="Random Forest API", version="1.0.0")
MODEL_PATH = os.getenv("MODEL_PATH", "model/srf_propio_model.pkl")

try:
    sys.modules['__main__'] = srf_model
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"No se pudo cargar el modelo desde {MODEL_PATH}: {e}")

# Esquemas de entrada/salida
class PredictRequest(BaseModel):
    features: conlist(float, min_length=1) # type: ignore

class PredictResponse(BaseModel):
    prediction: str


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {
        "team": "GPT-4o mini",
        "model": type(model).__name__,
        "n_estimators": getattr(model, "n_estimators", None),
        "max_depth": getattr(model, "max_depth", None),
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        X = np.array(req.features, dtype=float).reshape(1, -1)
        pred = model.predict(X)[0]
        iris_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        if isinstance(pred, (int, np.integer)):
            pred = iris_map.get(pred, str(pred))
        return {"prediction": str(pred)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Solicitud inválida: {e}")
