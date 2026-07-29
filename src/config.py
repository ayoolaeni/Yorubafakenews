"""Typed access to config/config.yaml, config/sources.yaml, config/models.yaml.

Import ``settings`` (a module-level singleton) rather than re-reading YAML
elsewhere, so every module agrees on paths, seed, and split ratios.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required config file missing: {path}. "
            "Copy/create it before running any pipeline stage."
        )
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass(frozen=True)
class Paths:
    raw: Path
    interim: Path
    annotated: Path
    processed: Path
    splits: Path
    models: Path
    results: Path


@dataclass(frozen=True)
class SplitConfig:
    train: float
    val: float
    test: float
    stratify_by: str


@dataclass(frozen=True)
class CorpusConfig:
    target_size_min: int
    target_size_max: int
    target_balance: float
    min_tokens: int
    max_tokens: int


@dataclass(frozen=True)
class PreprocessingConfig:
    variants: list[str]
    stopword_removal: list[bool]
    retain_english_segments: bool
    emoji_as_feature: bool


@dataclass(frozen=True)
class Settings:
    project_name: str
    random_seed: int
    paths: Paths
    split: SplitConfig
    corpus: CorpusConfig
    preprocessing: PreprocessingConfig
    dedup_cosine_threshold: float
    leakage_accuracy_threshold: float
    sources: dict[str, Any] = field(repr=False)
    models_config: dict[str, Any] = field(repr=False)
    contact_email: str = ""

    def resolve(self, relative: str | Path) -> Path:
        """Resolve a path from config.yaml relative to the project root."""
        p = Path(relative)
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_settings(
    config_path: Path | None = None,
    sources_path: Path | None = None,
    models_path: Path | None = None,
) -> Settings:
    config_path = config_path or PROJECT_ROOT / os.getenv("CONFIG_PATH", "config/config.yaml")
    sources_path = sources_path or PROJECT_ROOT / os.getenv("SOURCES_PATH", "config/sources.yaml")
    models_path = models_path or PROJECT_ROOT / os.getenv("MODELS_CONFIG_PATH", "config/models.yaml")

    raw = _load_yaml(config_path)
    sources = _load_yaml(sources_path)
    models_config = _load_yaml(models_path)

    project = raw["project"]
    paths_raw = raw["paths"]
    split_raw = raw["split"]
    corpus_raw = raw["corpus"]
    preproc_raw = raw["preprocessing"]

    split_sum = split_raw["train"] + split_raw["val"] + split_raw["test"]
    if abs(split_sum - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {split_sum}")

    return Settings(
        project_name=project["name"],
        random_seed=project["random_seed"],
        paths=Paths(
            raw=PROJECT_ROOT / paths_raw["raw"],
            interim=PROJECT_ROOT / paths_raw["interim"],
            annotated=PROJECT_ROOT / paths_raw["annotated"],
            processed=PROJECT_ROOT / paths_raw["processed"],
            splits=PROJECT_ROOT / paths_raw["splits"],
            models=PROJECT_ROOT / paths_raw["models"],
            results=PROJECT_ROOT / paths_raw["results"],
        ),
        split=SplitConfig(**split_raw),
        corpus=CorpusConfig(**corpus_raw),
        preprocessing=PreprocessingConfig(
            variants=preproc_raw["variants"],
            stopword_removal=preproc_raw["stopword_removal"],
            retain_english_segments=preproc_raw["retain_english_segments"],
            emoji_as_feature=preproc_raw["emoji_as_feature"],
        ),
        dedup_cosine_threshold=raw.get("dedup", {}).get("near_duplicate_cosine_threshold", 0.95),
        leakage_accuracy_threshold=raw.get("leakage", {}).get(
            "source_only_accuracy_flag_threshold", 0.70
        ),
        sources=sources,
        models_config=models_config,
        contact_email=os.getenv("RESEARCH_CONTACT_EMAIL", "unset@example.com"),
    )


settings = load_settings()
