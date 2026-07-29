"""CLI: python -m src.models.train --model all --features all

"--features" selects which preprocessed variant(s) in data/processed/ to
train on ("all", or a variant stem like diacritic_stripped__stopwords_kept).
"--model" selects which model(s) from config/models.yaml ("all", or a
specific classical/transformer name). Only the train split is used for
fitting; classical models are tuned with cv_folds-fold CV, transformers are
tuned against the val split with early stopping -- test stays untouched
until `make evaluate`.
"""
from __future__ import annotations

import argparse

from src.config import settings
from src.models.data import list_variants, load_split
from src.models.train_classical import PRIMARY_METRIC_TO_SCORING, save_model, train_classical_model
from src.utils.io import write_json
from src.utils.logging_config import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


def resolve_variants(features_arg: str) -> list[str]:
    available = list_variants(settings.paths.processed)
    if not available:
        raise SystemExit(
            f"No preprocessed variants found under {settings.paths.processed}. "
            "Run `make preprocess` first."
        )
    if features_arg == "all":
        return available
    if features_arg not in available:
        raise SystemExit(f"Unknown --features {features_arg!r}. Available: {available}")
    return [features_arg]


def resolve_classical_models(model_arg: str) -> dict:
    classical = settings.models_config.get("classical", {})
    if model_arg == "all":
        return classical
    if model_arg in classical:
        return {model_arg: classical[model_arg]}
    return {}


def resolve_transformer_models(model_arg: str) -> dict:
    transformers_cfg = settings.models_config.get("transformers", {})
    if model_arg == "all":
        return transformers_cfg
    if model_arg in transformers_cfg:
        return {model_arg: transformers_cfg[model_arg]}
    return {}


def run_classical(variant: str, model_name: str, model_cfg: dict) -> None:
    training_cfg = settings.models_config.get("training", {})
    cv_folds = training_cfg.get("cv_folds", 5)
    scoring = PRIMARY_METRIC_TO_SCORING.get(training_cfg.get("primary_metric", "f1"), "f1_macro")

    train_df = load_split(settings.paths.processed, settings.paths.splits, variant, "train")
    logger.info("[%s / %s] training on %d rows", variant, model_name, len(train_df))

    pipeline, best_params, best_score = train_classical_model(
        train_df["text"], train_df["label"], model_cfg, cv_folds, scoring
    )
    model_path = save_model(pipeline, settings.paths.models / "classical", variant, model_name)
    logger.info(
        "[%s / %s] best_params=%s cv_%s=%.4f -> %s",
        variant, model_name, best_params, scoring, best_score, model_path,
    )

    report_path = settings.paths.results / "metrics" / f"train__{variant}__{model_name}.json"
    write_json(
        report_path,
        {
            "variant": variant,
            "model": model_name,
            "cv_folds": cv_folds,
            "scoring": scoring,
            "best_params": best_params,
            "best_cv_score": best_score,
            "model_path": str(model_path),
        },
    )


def run_transformer(variant: str, model_name: str, model_cfg: dict) -> None:
    from src.models.train_transformer import save_model as save_transformer, train_transformer_model

    training_cfg = settings.models_config.get("training", {})
    patience = training_cfg.get("early_stopping_patience", 2)

    train_df = load_split(settings.paths.processed, settings.paths.splits, variant, "train")
    val_df = load_split(settings.paths.processed, settings.paths.splits, variant, "val")
    logger.info(
        "[%s / %s] training on %d rows, validating on %d rows",
        variant, model_name, len(train_df), len(val_df),
    )

    model, tokenizer, best_params, best_f1 = train_transformer_model(
        train_df, val_df, model_cfg, patience, settings.random_seed, settings.paths.models
    )
    model_path = save_transformer(model, tokenizer, settings.paths.models / "transformers", variant, model_name)
    logger.info("[%s / %s] best_params=%s val_f1_macro=%.4f -> %s", variant, model_name, best_params, best_f1, model_path)

    report_path = settings.paths.results / "metrics" / f"train__{variant}__{model_name}.json"
    write_json(
        report_path,
        {
            "variant": variant,
            "model": model_name,
            "best_params": best_params,
            "best_val_f1_macro": best_f1,
            "model_path": str(model_path),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="all")
    parser.add_argument("--features", default="all")
    args = parser.parse_args()

    set_global_seed(settings.random_seed)
    variants = resolve_variants(args.features)
    classical_models = resolve_classical_models(args.model)
    transformer_models = resolve_transformer_models(args.model)

    if not classical_models and not transformer_models:
        raise SystemExit(f"Unknown --model {args.model!r}: not in config/models.yaml")

    for variant in variants:
        for model_name, model_cfg in classical_models.items():
            run_classical(variant, model_name, model_cfg)
        for model_name, model_cfg in transformer_models.items():
            try:
                run_transformer(variant, model_name, model_cfg)
            except Exception as exc:
                # Transformer training depends on network/HF Hub/optional tokenizer
                # backends/hardware in ways classical sklearn training doesn't -- one
                # checkpoint failing shouldn't take down the other variants/models.
                logger.warning("skipping %s (%s): %s: %s", model_name, variant, type(exc).__name__, exc)

    logger.info("training complete: variants=%d classical=%d transformers=%d",
                len(variants), len(classical_models), len(transformer_models))


if __name__ == "__main__":
    main()