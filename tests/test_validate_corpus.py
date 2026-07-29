import pandas as pd

from src.annotation.validate_corpus import (
    assign_ids,
    check_source_leakage,
    compute_balance_stats,
    drop_near_duplicates,
    filter_by_length,
    token_count,
)


def test_token_count():
    assert token_count("ọ̀rọ̀ kan meji mẹta") == 4


def test_filter_by_length_drops_outside_range():
    df = pd.DataFrame(
        {
            "text": [
                "short",
                "this is a long enough sentence with plenty of tokens in it now",
            ]
        }
    )
    filtered = filter_by_length(df, min_tokens=5, max_tokens=20)
    assert len(filtered) == 1
    assert filtered.iloc[0]["text"].startswith("this is")


def test_drop_near_duplicates_removes_identical_text():
    df = pd.DataFrame(
        {
            "text": [
                "iroyin kan nipa ilu Eko ati awon eniyan",
                "iroyin kan nipa ilu Eko ati awon eniyan",
                "oro miran patapata nipa bola aje ti orile ede",
            ]
        }
    )
    deduped, dropped = drop_near_duplicates(df, threshold=0.95)
    assert dropped == 1
    assert len(deduped) == 2


def test_compute_balance_stats():
    df = pd.DataFrame(
        {
            "label": ["genuine", "genuine", "fake"],
            "source": ["BBC", "BBC", "Dubawa"],
        }
    )
    stats = compute_balance_stats(df)
    assert stats["total"] == 3
    assert stats["label_counts"]["genuine"] == 2
    assert abs(stats["genuine_share"] - 2 / 3) < 1e-9


def test_check_source_leakage_insufficient_data_returns_not_flagged():
    df = pd.DataFrame({"label": ["genuine"], "source": ["BBC"]})
    result = check_source_leakage(df, seed=42, threshold=0.7)
    assert result["flagged"] is False
    assert result["source_only_accuracy"] is None


def test_assign_ids_are_unique_and_sequential():
    df = pd.DataFrame(
        {
            "label": ["fake", "genuine"],
            "source": ["Dubawa", "BBC"],
            "url": ["https://b.example/1", "https://a.example/1"],
        }
    )
    out = assign_ids(df)
    assert list(out["id"]) == ["corpus_000000", "corpus_000001"]
    assert out["id"].is_unique