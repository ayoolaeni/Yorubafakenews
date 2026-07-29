"""CLI: python -m src.evaluation.error_analysis --model best
     CLI: python -m src.evaluation.error_analysis --model svm --features diacritic_stripped__stopwords_kept

Invoked by `make evaluate` (after src.evaluation.report has produced
results/metrics/best_model.json). Re-predicts on the TEST split for one
model/variant, isolates the misclassified rows, and writes:

- results/metrics/error_analysis__<variant>__<model>.csv (misclassified rows)
- results/metrics/error_analysis__<variant>__<model>_summary.json
  (error rate, false-positive/false-negative counts, errors broken down by
  source, and average token_count for errors vs. correct predictions --
  a quick signal for whether errors cluster around unusually short/long
  or single-source text).
"""
from __future__ import annotations

import argparse

from src.config import settings
from src.models.data import load_split
from src.models.inference import load_model, predict
from src.utils.io import read_json, write_csv, write_json
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def resolve_target(model_arg: str, features_arg: str | None) -> tuple[str, str, str]:
    metrics_dir = settings.paths.results / "metrics"
    if model_arg == "best":
        best = read_json(metrics_dir / "best_model.json")
        return best["variant"], best["model"], best["model_path"]

    if not features_arg:
        raise SystemExit("--features is required when --model is not 'best'")

    report_path = metrics_dir / f"train__{features_arg}__{model_arg}.json"
    if not report_path.exists():
        raise SystemExit(f"No trained-model report at {report_path}; run `make train` for this pair first.")
    report = read_json(report_path)
    return report["variant"], report["model"], report["model_path"]


def max_length_for(model_name: str) -> int:
    return settings.models_config.get("transformers", {}).get(model_name, {}).get("max_length", 256)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="best")
    parser.add_argument("--features", default=None, help="Variant stem; required unless --model best")
    args = parser.parse_args()

    variant, model_name, model_path = resolve_target(args.model, args.features)
    logger.info("error analysis for %s / %s", variant, model_name)

    test_df = load_split(settings.paths.processed, settings.paths.splits, variant, "test")
    model = load_model(model_path)
    test_df = test_df.copy()
    test_df["predicted"] = predict(model, test_df["text"], max_length_for(model_name))

    errors = test_df[test_df["label"] != test_df["predicted"]].copy()
    errors["error_type"] = errors.apply(
        lambda r: "false_positive" if r["label"] == "genuine" else "false_negative", axis=1
    )

    metrics_dir = settings.paths.results / "metrics"
    error_columns = ["id", "source", "label", "predicted", "error_type", "token_count", "text"]
    write_csv(metrics_dir / f"error_analysis__{variant}__{model_name}.csv", errors[error_columns])

    correct = test_df[test_df["label"] == test_df["predicted"]]
    summary = {
        "variant": variant,
        "model": model_name,
        "test_size": len(test_df),
        "error_count": len(errors),
        "error_rate": len(errors) / len(test_df) if len(test_df) else 0.0,
        "false_positive_count": int((errors["error_type"] == "false_positive").sum()),
        "false_negative_count": int((errors["error_type"] == "false_negative").sum()),
        "errors_by_source": errors["source"].value_counts().to_dict(),
        "avg_token_count_errors": float(errors["token_count"].mean()) if len(errors) else None,
        "avg_token_count_correct": float(correct["token_count"].mean()) if len(correct) else None,
    }
    write_json(metrics_dir / f"error_analysis__{variant}__{model_name}_summary.json", summary)
    logger.info(
        "%d/%d test rows misclassified (%.1f%%); wrote error_analysis__%s__%s.csv",
        len(errors), len(test_df), summary["error_rate"] * 100, variant, model_name,
    )


if __name__ == "__main__":
    main()