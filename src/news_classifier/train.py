"""Training pipeline: split, tokenise, LoRA fine-tune, evaluate, save."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .config import DEFAULT_DATA_PATH, DEFAULT_QUESTION, LABELS, ModelConfig, TrainConfig
from .data import (
    DatasetSplit,
    Example,
    class_counts,
    deduplicate,
    describe,
    load_examples,
    split_examples,
)
from .metrics import classification_report, confusion_matrix, format_confusion_matrix, hf_metrics
from .modeling import (
    METRICS_FILENAME,
    build_model,
    load_tokenizer,
    merge_adapter,
    resolve_device,
    save_bundle,
)

logger = logging.getLogger(__name__)


class TokenizedDataset:
    """Minimal map-style dataset; removes the ``datasets`` dependency.

    Samples are tokenised once without padding; the collator pads per batch
    (dynamic padding), which is both faster and less wasteful than the legacy
    ``padding="max_length"``.
    """

    def __init__(self, features: dict[str, list[list[int]]], labels: Sequence[int]):
        self.features = features
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        item = {key: values[index] for key, values in self.features.items()}
        item["labels"] = self.labels[index]
        return item


class PadCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(
                [feature["labels"] for feature in features], dtype=torch.long
            ),
        }


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize_examples(
    examples: Sequence[Example],
    tokenizer,
    *,
    max_length: int,
) -> TokenizedDataset:
    from .data import build_input_text

    prompts = [build_input_text(example.title, example.question) for example in examples]
    encoded = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )
    features = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }
    return TokenizedDataset(features, [example.label for example in examples])


@dataclass(frozen=True)
class DataBundle:
    split: DatasetSplit
    report: dict[str, Any]


def prepare_data(
    data_path: str | Path,
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> DataBundle:
    # Force one canonical question: the prompt must be byte-identical between
    # training and inference, and the seed data contains three near-duplicate
    # question variants.
    raw = load_examples(data_path, question=DEFAULT_QUESTION)
    examples, dedup_report = deduplicate(raw)
    split = split_examples(
        examples,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    report = {
        "source": str(data_path),
        "raw": len(raw),
        "dedup": dedup_report.to_dict(),
        "after_dedup": describe(examples),
        "split": split.to_dict(),
        "split_class_counts": {
            name: class_counts(getattr(split, name))
            for name in ("train", "validation", "test")
        },
    }
    return DataBundle(split=split, report=report)


def run_training(
    *,
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_config: ModelConfig | None = None,
    train_config: TrainConfig | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Fine-tune, evaluate on the held-out test split and save artifacts."""

    import torch
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

    model_config = model_config or ModelConfig()
    train_config = train_config or TrainConfig()
    set_seed(train_config.seed)

    bundle = prepare_data(
        data_path,
        val_ratio=train_config.val_ratio,
        test_ratio=train_config.test_ratio,
        seed=train_config.seed,
    )
    split = bundle.split
    if not split.train:
        raise RuntimeError("training split is empty; check the dataset and ratios")

    tokenizer = load_tokenizer(model_config.model_id)
    train_dataset = tokenize_examples(
        split.train, tokenizer, max_length=model_config.max_length
    )
    eval_dataset = tokenize_examples(
        split.validation or split.test,
        tokenizer,
        max_length=model_config.max_length,
    )
    test_dataset = (
        tokenize_examples(split.test, tokenizer, max_length=model_config.max_length)
        if split.test
        else None
    )
    if len(eval_dataset) == 0:
        raise RuntimeError(
            "validation/test split is empty; set --val-ratio or --test-ratio > 0"
        )

    model, tokenizer = build_model(model_config, tokenizer=tokenizer)

    use_cuda = torch.cuda.is_available() and resolve_device(device) == "cuda"
    bf16 = bool(use_cuda and torch.cuda.is_bf16_supported())
    fp16 = bool(use_cuda and not bf16)

    steps_per_epoch = max(
        1,
        len(train_dataset)
        // (
            train_config.per_device_train_batch_size
            * train_config.gradient_accumulation_steps
        ),
    )
    warmup_steps = int(
        steps_per_epoch * train_config.num_epochs * train_config.warmup_ratio
    )

    training_args = TrainingArguments(
        output_dir=str(train_config.output_dir),
        learning_rate=train_config.learning_rate,
        num_train_epochs=train_config.num_epochs,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        per_device_eval_batch_size=train_config.per_device_eval_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        weight_decay=train_config.weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=train_config.logging_steps,
        logging_first_step=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=train_config.metric_for_best_model,
        greater_is_better=True,
        report_to="none",
        seed=train_config.seed,
        data_seed=train_config.seed,
        fp16=fp16,
        bf16=bf16,
        remove_unused_columns=False,
    )

    callbacks = []
    if train_config.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=train_config.early_stopping_patience
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=PadCollator(tokenizer.pad_token_id),
        compute_metrics=hf_metrics,
        callbacks=callbacks,
    )

    logger.info("starting fine-tuning on %d examples", len(train_dataset))
    trainer.train()

    metrics: dict[str, Any] = {"validation": trainer.evaluate()}
    if test_dataset is not None:
        test_output = trainer.predict(test_dataset)
        y_true = test_output.label_ids.tolist()
        y_pred = test_output.predictions.argmax(axis=-1).tolist()
        metrics["test"] = {
            "accuracy": float(test_output.metrics.get("test_accuracy", 0.0)),
            "macro_f1": float(test_output.metrics.get("test_macro_f1", 0.0)),
            "report": classification_report(y_true, y_pred),
            "confusion_matrix": format_confusion_matrix(
                confusion_matrix(y_true, y_pred)
            ),
            "support": len(y_true),
        }

    merged = merge_adapter(trainer.model)
    save_bundle(
        merged,
        tokenizer,
        train_config.output_dir,
        model_config=model_config,
        train_config=train_config,
        metrics=metrics,
        extra={"data": bundle.report},
    )

    report_path = Path(train_config.output_dir) / METRICS_FILENAME
    print(f"saved model and report to {train_config.output_dir}")
    if "test" in metrics:
        print(metrics["test"]["report"])
        print(metrics["test"]["confusion_matrix"])
    print(f"metrics: {report_path}")
    return metrics


def run_data_report(data_path: str | Path = DEFAULT_DATA_PATH) -> dict[str, Any]:
    """Validate + clean + split without touching a model (``--dry-run``)."""

    bundle = prepare_data(
        data_path,
        val_ratio=TrainConfig().val_ratio,
        test_ratio=TrainConfig().test_ratio,
        seed=TrainConfig().seed,
    )
    report = {**bundle.report, "labels": list(LABELS)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report
