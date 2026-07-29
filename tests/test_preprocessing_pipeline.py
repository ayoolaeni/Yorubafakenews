import pandas as pd

from src.preprocessing.pipeline import (
    build_variant,
    clean_text,
    count_and_strip_emoji,
    make_splits,
    strip_diacritics,
    tokenize,
)


def test_strip_diacritics_removes_tone_and_dot_below():
    assert strip_diacritics("ẹ ọ ṣ") == "e o s"
    assert strip_diacritics("Ọjọ́ Àjíǹde") == "Ojo Ajinde"


def test_count_and_strip_emoji():
    text, count = count_and_strip_emoji("Ìròyìn gidi 😀🔥 nípa ìlú")
    assert count == 2
    assert "😀" not in text and "🔥" not in text


def test_clean_text_strips_urls_and_collapses_whitespace():
    text = clean_text("Wo eyi   https://example.com/x  fun alaye")
    assert "https" not in text
    assert "  " not in text


def test_tokenize_splits_on_unicode_word_chars():
    assert tokenize("ìròyìn kan, nípa ìlú!") == ["ìròyìn", "kan", "nípa", "ìlú"]


def test_build_variant_stopword_removal_shrinks_token_count():
    df = pd.DataFrame(
        {
            "id": ["corpus_000000"],
            "label": ["genuine"],
            "source": ["BBC"],
            "text": ["ti mo ni oro kan pataki nipa ilu"],
        }
    )
    base_stopwords = ["ti", "mo", "ni", "kan"]
    kept = build_variant(df, "diacritic_preserved", False, base_stopwords, retain_english_segments=True)
    removed = build_variant(df, "diacritic_preserved", True, base_stopwords, retain_english_segments=True)
    assert removed.iloc[0]["token_count"] < kept.iloc[0]["token_count"]
    assert "ti" not in removed.iloc[0]["text"].split()


def test_make_splits_respects_ratios_and_refuses_overwrite(tmp_path):
    df = pd.DataFrame(
        {
            "id": [f"corpus_{i:06d}" for i in range(20)],
            "label": ["genuine", "fake"] * 10,
        }
    )
    make_splits(df, tmp_path, seed=42, force=False)
    train = pd.read_csv(tmp_path / "train.csv")
    val = pd.read_csv(tmp_path / "val.csv")
    test = pd.read_csv(tmp_path / "test.csv")
    assert len(train) + len(val) + len(test) == 20
    assert len(train) == 14

    # second call without --force must not change the files
    make_splits(df.iloc[:5], tmp_path, seed=42, force=False)
    assert len(pd.read_csv(tmp_path / "train.csv")) == 14