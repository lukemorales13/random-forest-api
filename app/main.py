# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from typing import List, Optional
import os
import joblib
import numpy as np

from app.srf_model import SimpleRandomForest
app = FastAPI(title="Random Forest API", version="1.0.0")

# Permite override vía env var como sugiere el anexo (MODEL_PATH)
MODEL_PATH = os.getenv("MODEL_PATH", "model/model.pkl")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    # Si falla la carga, devuelve error claro
    raise RuntimeError(f"No se pudo cargar el modelo desde {MODEL_PATH}: {e}")

# ---- Esquemas de entrada/salida ----
class PredictRequest(BaseModel):
    # Ajusta el tamaño a tus features reales; para Iris suele ser 4
    features: conlist(float, min_items=1)

class PredictResponse(BaseModel):
    prediction: str

# ---- Endpoints obligatorios ----

@app.get("/health")
def health():
    # Debe devolver {"status":"ok"}
    # (requisito mínimo del elemento 3)
    return {"status": "ok"}

@app.get("/info")
def info():
    # Devuelve metadatos básicos del equipo y el modelo
    return {
        "team": "GPT-4o mini",                  # cámbialo por tu equipo
        "model": type(model).__name__,        # p.ej. RandomForestClassifier
        "n_estimators": getattr(model, "n_estimators", None),
        "max_depth": getattr(model, "max_depth", None),
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        X = np.array(req.features, dtype=float).reshape(1, -1)
        pred = model.predict(X)[0]
        # Si tu y eran strings (setosa/versicolor/virginica), saldrá como str;
        # si eran ints, convierte a str para cumplir el formato del PDF:
        return {"prediction": str(pred)}
    except Exception as e:
        # Manejo de entradas inválidas (bonus + robustez)
        raise HTTPException(status_code=400, detail=f"Solicitud inválida: {e}")
