from src.config import settings


def test_settings_load():
    assert settings.project_name == "yoruba-fake-news-detection"
    assert settings.random_seed == 42


def test_split_ratios_sum_to_one():
    s = settings.split
    assert abs((s.train + s.val + s.test) - 1.0) < 1e-6


def test_preprocessing_both_variants_configured():
    assert set(settings.preprocessing.variants) == {
        "diacritic_stripped",
        "diacritic_preserved",
    }


def test_paths_resolve_under_project_root():
    from src.config import PROJECT_ROOT

    assert str(settings.paths.raw).startswith(str(PROJECT_ROOT))
