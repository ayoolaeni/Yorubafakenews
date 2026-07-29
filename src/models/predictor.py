"""Single-text prediction wrapper shared by app/streamlit_app.py and app/api.py.

Loading a model and preprocessing raw input the same way training data was
preprocessed are two easy places for the app to silently drift from the
pipeline; centralising both here means both UIs go through one path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.config import settings
from src.models.inference import load_model, predict, predict_proba
from src.preprocessing.pipeline import preprocess_text_for_variant
from src.utils.io import read_json


def max_length_for(model_name: str) -> int:
    return settings.models_config.get("transformers", {}).get(model_name, {}).get("max_length", 256)


@dataclass
class Predictor:
    variant: str
    model_name: str
    model_path: str
    model: Any

    def predict_one(self, text: str) -> dict:
        processed = preprocess_text_for_variant(text, self.variant)
        series = pd.Series([processed])
        max_length = max_length_for(self.model_name)
        label = predict(self.model, series, max_length)[0]
        confidence = predict_proba(self.model, series, max_length)[0]
        return {"label": label, "confidence": confidence, "processed_text": processed}


def load_predictor(variant: str, model_name: str, model_path: str) -> Predictor:
    return Predictor(variant, model_name, model_path, load_model(model_path))


def load_best_predictor() -> Predictor:
    best_path = settings.paths.results / "metrics" / "best_model.json"
    best = read_json(best_path)
    return load_predictor(best["variant"], best["model"], best["model_path"])