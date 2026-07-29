"""Grid-searches one classical sklearn model over one preprocessed variant.

Only the training split is touched (5-fold CV within it, per
models.yaml training.cv_folds) -- val/test stay unseen until
src.evaluation.report loads the saved pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.models.registry import CLASSICAL_MODEL_CLASSES
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

PRIMARY_METRIC_TO_SCORING = {"f1": "f1_macro", "accuracy": "accuracy"}


def train_classical_model(
    train_texts: pd.Series,
    train_labels: pd.Series,
    model_cfg: dict[str, Any],
    cv_folds: int,
    scoring: str,
) -> tuple[Pipeline, dict[str, Any], float]:
    clf_class = CLASSICAL_MODEL_CLASSES[model_cfg["class"]]
    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", clf_class())])
    param_grid = {f"clf__{k}": v for k, v in model_cfg["grid"].items()}

    search = GridSearchCV(pipeline, param_grid, cv=cv_folds, scoring=scoring, n_jobs=-1)
    search.fit(train_texts, train_labels)

    best_pipeline = search.best_estimator_
    if model_cfg.get("calibrate"):
        best_clf_params = {k.split("clf__", 1)[1]: v for k, v in search.best_params_.items()}
        best_pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                ("clf", CalibratedClassifierCV(clf_class(**best_clf_params), cv=cv_folds)),
            ]
        )
        best_pipeline.fit(train_texts, train_labels)

    return best_pipeline, search.best_params_, float(search.best_score_)


def save_model(pipeline: Pipeline, out_dir: Path, variant: str, model_name: str) -> Path:
    import joblib

    out_path = Path(out_dir) / f"{variant}__{model_name}.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)
    return out_path