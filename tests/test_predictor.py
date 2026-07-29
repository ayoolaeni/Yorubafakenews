from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.models.predictor import Predictor


def _fit_tiny_pipeline() -> Pipeline:
    texts = ["iroyin gidi nipa ilu", "iroyin miran gidi", "etan iro patapata", "etan miran paapaa"]
    labels = ["genuine", "genuine", "fake", "fake"]
    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    pipeline.fit(texts, labels)
    return pipeline


def test_predict_one_preprocesses_then_predicts():
    predictor = Predictor(
        variant="diacritic_stripped__stopwords_kept",
        model_name="logistic_regression",
        model_path="unused-for-this-test",
        model=_fit_tiny_pipeline(),
    )
    result = predictor.predict_one("Ìròyìn gidi nípa ìlú Èkó")
    assert result["label"] in {"genuine", "fake"}
    assert 0.0 <= result["confidence"] <= 1.0
    # diacritic_stripped variant: the diacritics should be gone from the processed text
    assert "ò" not in result["processed_text"] and "í" not in result["processed_text"]


def test_predict_one_preserves_diacritics_for_preserved_variant():
    predictor = Predictor(
        variant="diacritic_preserved__stopwords_kept",
        model_name="logistic_regression",
        model_path="unused-for-this-test",
        model=_fit_tiny_pipeline(),
    )
    result = predictor.predict_one("Ìròyìn gidi nípa ìlú Èkó")
    assert "ìròyìn" in result["processed_text"]