"""Joins a preprocessed variant CSV against the frozen train/val/test id splits.

Both src.models.train and src.evaluation.report need "just the rows for
this split, for this variant" -- centralising the join here means the
test split is only ever read through one code path, which is what the
"opened exactly once" logging note in src/utils/logging_config.py is for.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import read_csv


def list_variants(processed_dir: Path) -> list[str]:
    return sorted(p.stem for p in Path(processed_dir).glob("*.csv"))


def load_split(processed_dir: Path, splits_dir: Path, variant: str, split_name: str) -> pd.DataFrame:
    variant_df = read_csv(Path(processed_dir) / f"{variant}.csv")
    split_ids = read_csv(Path(splits_dir) / f"{split_name}.csv")
    merged = variant_df.merge(split_ids[["id"]], on="id", how="inner")
    if len(merged) != len(split_ids):
        missing = len(split_ids) - len(merged)
        raise ValueError(
            f"{missing} ids in {split_name}.csv were not found in variant {variant!r}; "
            "did you regenerate the split after building variants?"
        )
    return merged