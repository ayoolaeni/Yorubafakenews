"""Loads a saved model (classical .joblib or transformer directory) and
predicts string labels ("genuine"/"fake") for a batch of texts.

Used by src.evaluation.report, src.evaluation.error_analysis, and app/*, so
"how do I load whatever `make train` produced" only has one implementation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

ID2LABEL = {0: "genuine", 1: "fake"}


def is_transformer_path(model_path: str | Path) -> bool:
    return Path(model_path).is_dir()


def _resolve_model_path(model_path: str | Path) -> Path:
    """Metrics JSON stores the absolute path model_path had at training time,
    which breaks the moment the project moves to a different machine, user,
    drive letter, or a Docker container. If that path is gone, rebuild it
    from the current models dir using the filename train.py always derives
    from variant+model_name (see src/models/train_classical.py:save_model
    and train_transformer.py:save_model)."""
    path = Path(model_path)
    if path.exists():
        return path
    from src.config import settings

    # pathlib.Path(...).name only splits on the *current* OS's separator, so
    # a Windows-written path like "C:\\...\\model.joblib" doesn't split on
    # POSIX (backslash isn't a separator there) and .name returns the whole
    # string. Normalise both separators ourselves before taking the tail.
    name = str(model_path).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    for sub in ("classical", "transformers"):
        candidate = settings.paths.models / sub / name
        if candidate.exists():
            return candidate
    return path


def load_model(model_path: str | Path) -> Any:
    model_path = _resolve_model_path(model_path)
    if is_transformer_path(model_path):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer

    import joblib

    return joblib.load(model_path)


def predict(model: Any, texts: pd.Series, max_length: int = 256) -> list[str]:
    if isinstance(model, tuple):
        return _predict_transformer(*model, texts, max_length)
    return list(model.predict(texts))


def predict_proba(model: Any, texts: pd.Series, max_length: int = 256) -> list[float]:
    """Confidence (probability) of the predicted class, one per text."""
    if isinstance(model, tuple):
        return _predict_transformer_proba(*model, texts, max_length)
    probs = model.predict_proba(texts)
    return [float(row.max()) for row in probs]


def _predict_transformer(model, tokenizer, texts: pd.Series, max_length: int) -> list[str]:
    import torch

    model.eval()
    encodings = tokenizer(
        list(texts), truncation=True, max_length=max_length, padding=True, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**encodings).logits
    predicted_ids = logits.argmax(dim=-1).tolist()
    return [ID2LABEL[i] for i in predicted_ids]


def _predict_transformer_proba(model, tokenizer, texts: pd.Series, max_length: int) -> list[float]:
    import torch

    model.eval()
    encodings = tokenizer(
        list(texts), truncation=True, max_length=max_length, padding=True, return_tensors="pt"
    )
    with torch.no_grad():
        probs = torch.softmax(model(**encodings).logits, dim=-1)
    return probs.max(dim=-1).values.tolist()