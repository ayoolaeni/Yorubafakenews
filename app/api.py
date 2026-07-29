"""Run with: make api  (i.e. uvicorn app.api:app --reload --port 8000)"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.models.predictor import Predictor, load_best_predictor

app = FastAPI(title="Yoruba Fake News Detection API")


@lru_cache
def get_predictor() -> Predictor:
    return load_best_predictor()


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    confidence: float
    model: str
    variant: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(payload: PredictRequest) -> PredictResponse:
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    try:
        predictor = get_predictor()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="No trained model available yet; run the training pipeline first.",
        )
    result = predictor.predict_one(payload.text)
    return PredictResponse(
        label=result["label"],
        confidence=result["confidence"],
        model=predictor.model_name,
        variant=predictor.variant,
    )