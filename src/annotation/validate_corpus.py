"""CLI: python -m src.annotation.validate_corpus

Invoked by `make corpus` after scraping. Merges every data/raw/<class>/*.jsonl
file into a single annotated corpus, applying the three checks the
dissertation's methodology requires before any modelling happens:

1. length filter (corpus.min_tokens / max_tokens in config.yaml)
2. near-duplicate removal (dedup.near_duplicate_cosine_threshold)
3. source-leakage flag: if a classifier trained on *source name alone* can
   already predict the label above leakage.source_only_accuracy_flag_threshold,
   genuine/fake is confounded with source rather than content, which would
   invalidate any downstream model's reported accuracy.

Writes the merged corpus to paths.annotated and a JSON report (counts,
dedup stats, balance, leakage accuracy) to results/metrics/.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from src.config import settings
from src.utils.io import read_jsonl, write_csv, write_json
from src.utils.logging_config import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


def load_raw_records(raw_dir: Path) -> pd.DataFrame:
    records: list[dict] = []
    for label_dir in ("genuine", "fake"):
        for path in sorted((Path(raw_dir) / label_dir).glob("*.jsonl")):
            records.extend(read_jsonl(path))
    return pd.DataFrame.from_records(records)


def token_count(text: str) -> int:
    return len(text.split())


def filter_by_length(df: pd.DataFrame, min_tokens: int, max_tokens: int) -> pd.DataFrame:
    df = df.copy()
    df["token_count"] = df["text"].map(token_count)
    return df[(df["token_count"] >= min_tokens) & (df["token_count"] <= max_tokens)].reset_index(
        drop=True
    )


def drop_near_duplicates(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, int]:
    if len(df) < 2:
        return df, 0

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(df["text"])
    similarity = cosine_similarity(tfidf)

    keep_mask = [True] * len(df)
    kept_indices: list[int] = []
    for i in range(len(df)):
        is_duplicate = any(similarity[i, j] > threshold for j in kept_indices)
        if is_duplicate:
            keep_mask[i] = False
        else:
            kept_indices.append(i)

    deduped = df[keep_mask].reset_index(drop=True)
    return deduped, int(len(df) - len(deduped))


def compute_balance_stats(df: pd.DataFrame) -> dict:
    label_counts = df["label"].value_counts().to_dict()
    source_counts = df.groupby(["label", "source"]).size().to_dict()
    total = len(df)
    genuine_share = label_counts.get("genuine", 0) / total if total else 0.0
    return {
        "total": total,
        "label_counts": label_counts,
        "source_counts": {f"{label}/{source}": n for (label, source), n in source_counts.items()},
        "genuine_share": genuine_share,
    }


def check_source_leakage(df: pd.DataFrame, seed: int, threshold: float) -> dict:
    if df["source"].nunique() < 2 or len(df) < 20:
        return {"source_only_accuracy": None, "flagged": False, "reason": "not enough data/sources"}

    X = pd.get_dummies(df["source"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    return {"source_only_accuracy": accuracy, "flagged": accuracy > threshold}


def assign_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["label", "source", "url"]).reset_index(drop=True)
    df.insert(0, "id", [f"corpus_{i:06d}" for i in range(len(df))])
    return df


def main() -> None:
    set_global_seed(settings.random_seed)

    df = load_raw_records(settings.paths.raw)
    if df.empty:
        raise SystemExit(
            f"No raw records found under {settings.paths.raw}. Run `make corpus` "
            "scraping (src.scraping.run_scrapers) first."
        )
    raw_count = len(df)
    logger.info("loaded %d raw records", raw_count)

    df = filter_by_length(df, settings.corpus.min_tokens, settings.corpus.max_tokens)
    length_dropped = raw_count - len(df)
    logger.info("dropped %d records outside [%d, %d] tokens", length_dropped,
                settings.corpus.min_tokens, settings.corpus.max_tokens)

    df, dedup_dropped = drop_near_duplicates(df, settings.dedup_cosine_threshold)
    logger.info("dropped %d near-duplicate records (cosine > %.2f)", dedup_dropped,
                settings.dedup_cosine_threshold)

    df = assign_ids(df)

    balance = compute_balance_stats(df)
    leakage = check_source_leakage(df, settings.random_seed, settings.leakage_accuracy_threshold)

    warnings: list[str] = []
    if not (settings.corpus.target_size_min <= len(df) <= settings.corpus.target_size_max):
        warnings.append(
            f"corpus size {len(df)} outside target range "
            f"[{settings.corpus.target_size_min}, {settings.corpus.target_size_max}]"
        )
    if abs(balance["genuine_share"] - settings.corpus.target_balance) > 0.1:
        warnings.append(f"class balance {balance['genuine_share']:.2f} far from target 0.5")
    if leakage["flagged"]:
        warnings.append(
            f"source-only accuracy {leakage['source_only_accuracy']:.2f} exceeds leakage "
            f"threshold {settings.leakage_accuracy_threshold:.2f} -- label may be confounded with source"
        )
    for warning in warnings:
        logger.warning(warning)

    output_columns = ["id", "label", "source", "title", "text", "token_count", "collected_at"]
    write_csv(settings.paths.annotated, df[output_columns])
    logger.info("wrote %d annotated records to %s", len(df), settings.paths.annotated)

    report = {
        "raw_count": raw_count,
        "length_dropped": length_dropped,
        "dedup_dropped": dedup_dropped,
        "final_count": len(df),
        "balance": balance,
        "leakage": leakage,
        "warnings": warnings,
    }
    report_path = settings.paths.results / "metrics" / "corpus_validation_report.json"
    write_json(report_path, report)
    logger.info("wrote validation report to %s", report_path)


if __name__ == "__main__":
    main()