"""Offline evaluation of a saved checkpoint on a labelled dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DEFAULT_DATA_PATH, DEFAULT_QUESTION, LABELS
from .data import deduplicate, load_examples
from .metrics import (
    accuracy,
    classification_report,
    confusion_matrix,
    format_confusion_matrix,
    macro_f1,
    per_class_scores,
)
from .predict import NewsClassifier


@dataclass(frozen=True)
class EvaluationResult:
    total: int
    accuracy: float
    macro_f1: float
    report: str
    confusion: str
    per_class: tuple[dict[str, Any], ...]
    errors: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "accuracy": round(self.accuracy, 6),
            "macro_f1": round(self.macro_f1, 6),
            "per_class": list(self.per_class),
            "errors": list(self.errors),
        }


def evaluate_model(
    model_dir: str | Path,
    *,
    data_path: str | Path = DEFAULT_DATA_PATH,
    device: str | None = None,
    max_length: int = 128,
    drop_duplicates: bool = True,
    max_errors: int = 20,
) -> EvaluationResult:
    classifier = NewsClassifier.from_pretrained(
        model_dir, device=device, max_length=max_length
    )
    examples = load_examples(data_path, question=DEFAULT_QUESTION)
    if drop_duplicates:
        examples, _ = deduplicate(examples)

    predictions = classifier.predict_batch([example.title for example in examples])
    y_true = [example.label for example in examples]
    y_pred = [prediction.label_id for prediction in predictions]

    errors = tuple(
        {
            "title": example.title,
            "expected": LABELS[example.label],
            "predicted": prediction.category,
            "confidence": round(prediction.confidence, 6),
        }
        for example, prediction in zip(examples, predictions)
        if prediction.label_id != example.label
    )[:max_errors]

    return EvaluationResult(
        total=len(examples),
        accuracy=accuracy(y_true, y_pred),
        macro_f1=macro_f1(y_true, y_pred),
        report=classification_report(y_true, y_pred),
        confusion=format_confusion_matrix(confusion_matrix(y_true, y_pred)),
        per_class=tuple(
            {
                "label": LABELS[score.label],
                "precision": round(score.precision, 6),
                "recall": round(score.recall, 6),
                "f1": round(score.f1, 6),
                "support": score.support,
            }
            for score in per_class_scores(y_true, y_pred)
        ),
        errors=errors,
    )


def write_result(result: EvaluationResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
