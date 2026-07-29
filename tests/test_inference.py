import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.models.inference import load_model, predict, predict_proba


def _fit_tiny_pipeline() -> Pipeline:
    texts = ["iroyin gidi nipa ilu", "iroyin gidi miran", "etan iro patapata", "etan omiran paapaa"]
    labels = ["genuine", "genuine", "fake", "fake"]
    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    pipeline.fit(texts, labels)
    return pipeline


def test_load_model_and_predict_classical(tmp_path):
    pipeline = _fit_tiny_pipeline()
    model_path = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_path)

    model = load_model(model_path)
    predictions = predict(model, pd.Series(["iroyin gidi nipa ilu"]))
    assert predictions[0] in {"genuine", "fake"}


def test_predict_proba_returns_value_between_0_and_1(tmp_path):
    pipeline = _fit_tiny_pipeline()
    model_path = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_path)

    model = load_model(model_path)
    probs = predict_proba(model, pd.Series(["etan iro patapata"]))
    assert 0.0 <= probs[0] <= 1.0


def test_is_transformer_path_distinguishes_file_from_dir(tmp_path):
    from src.models.inference import is_transformer_path

    file_path = tmp_path / "model.joblib"
    file_path.write_text("x")
    dir_path = tmp_path / "transformer_dir"
    dir_path.mkdir()

    assert is_transformer_path(file_path) is False
    assert is_transformer_path(dir_path) is True