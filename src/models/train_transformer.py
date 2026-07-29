"""Fine-tunes one transformer checkpoint over one preprocessed variant.

Imports torch/transformers/datasets lazily (they're the heaviest, most
optional deps in requirements.txt) so that `--model <classical name>` runs
never pay for them, and so a machine without a GPU/these packages can still
use the rest of the pipeline. Grid-searches models.yaml's
transformers.<name>.grid by full-training one run per combination and
keeping the one with the best val F1 (early stopping on
training.early_stopping_patience).
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

LABEL2ID = {"genuine": 0, "fake": 1}
ID2LABEL = {0: "genuine", 1: "fake"}


def _require_transformer_deps() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import datasets  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Transformer training needs torch, transformers, datasets, and accelerate. "
            "Install them with: pip install torch transformers datasets accelerate"
        ) from exc


def _tokenize_dataset(dataset, tokenizer, max_length: int):
    def _tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length, padding="max_length")

    return dataset.map(_tokenize, batched=True)


def _compute_metrics(eval_pred):
    import numpy as np
    from sklearn.metrics import f1_score

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"f1_macro": f1_score(labels, predictions, average="macro")}


def train_transformer_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    checkpoint_cfg: dict[str, Any],
    early_stopping_patience: int,
    seed: int,
    work_dir: Path,
) -> tuple[Any, Any, dict[str, Any], float]:
    """Returns (model, tokenizer, best_hyperparams, best_val_f1)."""
    _require_transformer_deps()

    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    checkpoint = checkpoint_cfg["checkpoint"]
    max_length = checkpoint_cfg["max_length"]
    grid = checkpoint_cfg["grid"]

    train_ds = Dataset.from_pandas(
        pd.DataFrame({"text": train_df["text"], "label": train_df["label"].map(LABEL2ID)})
    )
    val_ds = Dataset.from_pandas(
        pd.DataFrame({"text": val_df["text"], "label": val_df["label"].map(LABEL2ID)})
    )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    train_ds = _tokenize_dataset(train_ds, tokenizer, max_length)
    val_ds = _tokenize_dataset(val_ds, tokenizer, max_length)

    best_f1 = -1.0
    best_model = None
    best_params: dict[str, Any] = {}

    keys = list(grid.keys())
    for values in itertools.product(*grid.values()):
        params = dict(zip(keys, values))
        logger.info("training %s with %s", checkpoint, params)

        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
        )
        args = TrainingArguments(
            output_dir=str(Path(work_dir) / "_tmp_trainer"),
            learning_rate=params["learning_rate"],
            per_device_train_batch_size=params["batch_size"],
            per_device_eval_batch_size=params["batch_size"],
            num_train_epochs=params["epochs"],
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            seed=seed,
            report_to=[],
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=_compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
        )
        trainer.train()
        metrics = trainer.evaluate()
        f1 = metrics["eval_f1_macro"]
        logger.info("%s with %s -> val f1_macro=%.4f", checkpoint, params, f1)

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_params = params

    return best_model, tokenizer, best_params, best_f1


def save_model(model: Any, tokenizer: Any, out_dir: Path, variant: str, model_name: str) -> Path:
    out_path = Path(out_dir) / f"{variant}__{model_name}"
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    return out_path