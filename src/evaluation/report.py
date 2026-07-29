"""CLI: python -m src.evaluation.report

Invoked by `make evaluate`. Discovers every model `make train` produced (by
reading results/metrics/train__*.json, written once per variant/model by
src.models.train), evaluates each against its variant's TEST split -- the
one time the test split is read in the whole pipeline -- and writes:

- results/metrics/eval__<variant>__<model>.json (full metrics + confusion matrix)
- results/figures/confusion_matrix__<variant>__<model>.png
- results/metrics/leaderboard.csv (every variant/model, ranked by f1_macro)
- results/metrics/best_model.json (the top row, consumed by --model best
  in src.evaluation.error_analysis)
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import settings
from src.evaluation.metrics import LABELS, compute_metrics
from src.models.data import load_split
from src.models.inference import load_model, predict
from src.utils.io import read_json, write_csv, write_json
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def discover_trained_models(metrics_dir) -> list[dict]:
    return [read_json(p) for p in sorted(metrics_dir.glob("train__*.json"))]


def max_length_for(model_name: str) -> int:
    return settings.models_config.get("transformers", {}).get(model_name, {}).get("max_length", 256)


def plot_confusion_matrix(cm: list[list[int]], labels: list[str], out_path) -> None:
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    metrics_dir = settings.paths.results / "metrics"
    figures_dir = settings.paths.results / "figures"
    trained = discover_trained_models(metrics_dir)
    if not trained:
        raise SystemExit(f"No trained models found under {metrics_dir}. Run `make train` first.")

    leaderboard_rows = []
    for entry in trained:
        variant, model_name, model_path = entry["variant"], entry["model"], entry["model_path"]
        logger.info("evaluating %s / %s on the test split", variant, model_name)

        test_df = load_split(settings.paths.processed, settings.paths.splits, variant, "test")
        model = load_model(model_path)
        predictions = predict(model, test_df["text"], max_length_for(model_name))

        result = compute_metrics(test_df["label"], predictions)
        result.update({"variant": variant, "model": model_name, "model_path": model_path})
        write_json(metrics_dir / f"eval__{variant}__{model_name}.json", result)
        plot_confusion_matrix(
            result["confusion_matrix"], LABELS, figures_dir / f"confusion_matrix__{variant}__{model_name}.png"
        )
        leaderboard_rows.append(
            {
                "variant": variant,
                "model": model_name,
                "accuracy": result["accuracy"],
                "precision_macro": result["precision_macro"],
                "recall_macro": result["recall_macro"],
                "f1_macro": result["f1_macro"],
                "model_path": model_path,
            }
        )

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    write_csv(metrics_dir / "leaderboard.csv", leaderboard)
    logger.info("wrote leaderboard (%d rows) to %s", len(leaderboard), metrics_dir / "leaderboard.csv")

    best = leaderboard.iloc[0].to_dict()
    write_json(metrics_dir / "best_model.json", best)
    logger.info("best model: %s / %s (f1_macro=%.4f)", best["variant"], best["model"], best["f1_macro"])


if __name__ == "__main__":
    main()
