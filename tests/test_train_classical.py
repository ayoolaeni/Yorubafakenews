import pandas as pd

from src.models.train_classical import save_model, train_classical_model


def _synthetic_train_data(n_per_class: int = 20) -> tuple[pd.Series, pd.Series]:
    genuine = [f"iroyin gidi nipa ilu eko ati ijoba nomba {i}" for i in range(n_per_class)]
    fake = [f"etan iro nipa oogun idan ati aje nomba {i}" for i in range(n_per_class)]
    texts = pd.Series(genuine + fake)
    labels = pd.Series(["genuine"] * n_per_class + ["fake"] * n_per_class)
    return texts, labels


def test_train_classical_model_naive_bayes_fits_and_predicts():
    texts, labels = _synthetic_train_data()
    model_cfg = {"class": "MultinomialNB", "grid": {"alpha": [0.1, 1.0]}}
    pipeline, best_params, best_score = train_classical_model(texts, labels, model_cfg, cv_folds=3, scoring="f1_macro")

    assert "alpha" in {k.split("clf__")[-1] for k in best_params}
    assert 0.0 <= best_score <= 1.0
    predictions = pipeline.predict(["iroyin gidi nipa ilu eko"])
    assert predictions[0] in {"genuine", "fake"}


def test_train_classical_model_calibrated_svm_has_predict_proba():
    texts, labels = _synthetic_train_data()
    model_cfg = {
        "class": "LinearSVC",
        "grid": {"C": [0.1, 1.0]},
        "calibrate": True,
    }
    pipeline, _, _ = train_classical_model(texts, labels, model_cfg, cv_folds=3, scoring="f1_macro")
    probs = pipeline.predict_proba(["etan iro nipa oogun idan"])
    assert probs.shape[1] == 2
    assert abs(probs.sum() - 1.0) < 1e-6


def test_save_model_writes_joblib_file(tmp_path):
    texts, labels = _synthetic_train_data(n_per_class=5)
    model_cfg = {"class": "MultinomialNB", "grid": {"alpha": [1.0]}}
    pipeline, _, _ = train_classical_model(texts, labels, model_cfg, cv_folds=2, scoring="f1_macro")

    out_path = save_model(pipeline, tmp_path, "diacritic_preserved__stopwords_kept", "naive_bayes")
    assert out_path.exists()
    assert out_path.name == "diacritic_preserved__stopwords_kept__naive_bayes.joblib"