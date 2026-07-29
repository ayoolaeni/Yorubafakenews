"""CLI: python -m src.preprocessing.pipeline --build-all-variants
     CLI: python -m src.preprocessing.pipeline --make-splits

Reads the annotated corpus (paths.annotated) and, per config.yaml:

- ``--build-all-variants`` writes every combination of
  preprocessing.variants x preprocessing.stopword_removal to
  data/processed/<diacritic_mode>__<stopwords_removed|kept>.csv. Emoji are
  always stripped from the text and their count kept as a numeric feature
  (preprocessing.emoji_as_feature), and English code-switched segments are
  only stripped if preprocessing.retain_english_segments is False.

- ``--make-splits`` stratifies the annotated corpus once into train/val/test
  id lists under data/splits/, per split.*. Refuses to overwrite an existing
  split unless --force is passed, because the split must be written ONCE
  (see the comment in config.yaml) -- regenerating it after any modelling
  has started would silently leak test data into training.
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import settings
from src.utils.io import read_csv, write_csv
from src.utils.logging_config import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

STOPWORDS_PATH = Path(__file__).resolve().parent.parent.parent / "fixtures" / "stopwords_yoruba.txt"

# Function words used only when preprocessing.retain_english_segments is False.
ENGLISH_FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "and", "or", "but", "of",
    "to", "in", "on", "for", "with", "this", "that", "it", "as", "at", "by",
}

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]+"
)


def load_stopwords() -> list[str]:
    words: list[str] = []
    with open(STOPWORDS_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                words.append(line)
    return words


def strip_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def stopwords_for_variant(base_stopwords: list[str], diacritic_mode: str) -> set[str]:
    if diacritic_mode == "diacritic_stripped":
        return {strip_diacritics(w).lower() for w in base_stopwords}
    return {w.lower() for w in base_stopwords}


def count_and_strip_emoji(text: str) -> tuple[str, int]:
    emoji_count = sum(len(match) for match in _EMOJI_RE.findall(text))
    return _EMOJI_RE.sub("", text), emoji_count


def clean_text(text: str) -> str:
    text = _URL_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def preprocess_one(
    raw_text: str,
    diacritic_mode: str,
    remove_stopwords: bool,
    stopwords: set[str],
    retain_english_segments: bool,
) -> tuple[str, int, int]:
    """Returns (processed_text, emoji_count, token_count) for a single string.

    This is the one place the per-text transform is defined; build_variant
    (batch, at preprocess time) and preprocess_text_for_variant (single
    string, at inference time in app/) both call it so a served prediction
    sees text preprocessed exactly like training data did.
    """
    text, emoji_count = count_and_strip_emoji(raw_text)
    text = clean_text(text)
    if diacritic_mode == "diacritic_stripped":
        text = strip_diacritics(text)
    text = text.lower()

    tokens = tokenize(text)
    if not retain_english_segments:
        tokens = [t for t in tokens if t not in ENGLISH_FUNCTION_WORDS]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stopwords]

    return " ".join(tokens), emoji_count, len(tokens)


def parse_variant_name(variant: str) -> tuple[str, bool]:
    diacritic_mode, suffix = variant.split("__", 1)
    return diacritic_mode, suffix == "stopwords_removed"


def preprocess_text_for_variant(text: str, variant: str) -> str:
    diacritic_mode, remove_stopwords = parse_variant_name(variant)
    stopwords = (
        stopwords_for_variant(load_stopwords(), diacritic_mode) if remove_stopwords else set()
    )
    processed, _, _ = preprocess_one(
        text, diacritic_mode, remove_stopwords, stopwords, settings.preprocessing.retain_english_segments
    )
    return processed


def build_variant(
    df: pd.DataFrame,
    diacritic_mode: str,
    remove_stopwords: bool,
    base_stopwords: list[str],
    retain_english_segments: bool,
) -> pd.DataFrame:
    stopwords = stopwords_for_variant(base_stopwords, diacritic_mode) if remove_stopwords else set()

    texts: list[str] = []
    emoji_counts: list[int] = []
    token_counts: list[int] = []
    for raw_text in df["text"]:
        text, emoji_count, token_count = preprocess_one(
            raw_text, diacritic_mode, remove_stopwords, stopwords, retain_english_segments
        )
        texts.append(text)
        emoji_counts.append(emoji_count)
        token_counts.append(token_count)

    out = df[["id", "label", "source"]].copy()
    out["text"] = texts
    out["emoji_count"] = emoji_counts
    out["token_count"] = token_counts
    return out


def build_all_variants(df: pd.DataFrame, output_dir: Path) -> None:
    base_stopwords = load_stopwords()
    for diacritic_mode in settings.preprocessing.variants:
        for remove_stopwords in settings.preprocessing.stopword_removal:
            variant_df = build_variant(
                df,
                diacritic_mode,
                remove_stopwords,
                base_stopwords,
                settings.preprocessing.retain_english_segments,
            )
            suffix = "stopwords_removed" if remove_stopwords else "stopwords_kept"
            out_path = Path(output_dir) / f"{diacritic_mode}__{suffix}.csv"
            write_csv(out_path, variant_df)
            logger.info("wrote %d rows to %s", len(variant_df), out_path)


def make_splits(df: pd.DataFrame, splits_dir: Path, seed: int, force: bool) -> None:
    splits_dir = Path(splits_dir)
    existing = list(splits_dir.glob("*.csv")) if splits_dir.exists() else []
    if existing and not force:
        logger.warning(
            "splits already exist under %s (%s); refusing to regenerate. "
            "Pass --force if you really intend to redo the split.",
            splits_dir, [p.name for p in existing],
        )
        return

    split_cfg = settings.split
    stratify_col = df[split_cfg.stratify_by]
    train_df, rest_df = train_test_split(
        df, train_size=split_cfg.train, random_state=seed, stratify=stratify_col
    )
    val_share_of_rest = split_cfg.val / (split_cfg.val + split_cfg.test)
    val_df, test_df = train_test_split(
        rest_df,
        train_size=val_share_of_rest,
        random_state=seed,
        stratify=rest_df[split_cfg.stratify_by],
    )

    for name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        out_path = splits_dir / f"{name}.csv"
        write_csv(out_path, split_df[["id", split_cfg.stratify_by]])
        logger.info("wrote %d ids to %s", len(split_df), out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-all-variants", action="store_true")
    parser.add_argument("--make-splits", action="store_true")
    parser.add_argument("--force", action="store_true", help="Allow --make-splits to overwrite an existing split.")
    args = parser.parse_args()

    if not args.build_all_variants and not args.make_splits:
        parser.error("pass --build-all-variants and/or --make-splits")

    set_global_seed(settings.random_seed)
    df = read_csv(settings.paths.annotated)
    logger.info("loaded %d annotated records from %s", len(df), settings.paths.annotated)

    if args.build_all_variants:
        build_all_variants(df, settings.paths.processed)
    if args.make_splits:
        make_splits(df, settings.paths.splits, settings.random_seed, args.force)


if __name__ == "__main__":
    main()