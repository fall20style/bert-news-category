"""Evaluation metrics implemented without scikit-learn.

The project only needs a handful of metrics, so keeping them dependency-free
makes the evaluation path easy to test on a CPU-only machine.  ``hf_metrics``
is an adapter for ``transformers.Trainer``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .config import LABELS


def softmax(logits: Sequence[float]) -> list[float]:
    """Numerically stable softmax for a single example."""

    if not logits:
        raise ValueError("logits must not be empty")
    peak = max(logits)
    exps = [math.exp(value - peak) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        return 0.0
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    return correct / len(y_true)


def confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int] | None = None,
) -> list[list[int]]:
    labels = list(labels if labels is not None else range(len(LABELS)))
    position = {label: index for index, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for truth, prediction in zip(y_true, y_pred):
        if truth in position and prediction in position:
            matrix[position[truth]][position[prediction]] += 1
    return matrix


@dataclass(frozen=True)
class ClassScore:
    label: int
    precision: float
    recall: float
    f1: float
    support: int


def per_class_scores(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int] | None = None,
) -> list[ClassScore]:
    labels = list(labels if labels is not None else range(len(LABELS)))
    scores: list[ClassScore] = []
    for label in labels:
        true_positive = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        false_positive = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        false_negative = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        scores.append(
            ClassScore(
                label=label,
                precision=precision,
                recall=recall,
                f1=f1,
                support=true_positive + false_negative,
            )
        )
    return scores


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Unweighted mean F1 over classes that actually occur in ``y_true``.

    Classes with no support are ignored so that a class missing from a small
    test split does not silently drag the score to zero.
    """

    scores = [score for score in per_class_scores(y_true, y_pred) if score.support > 0]
    if not scores:
        return 0.0
    return sum(score.f1 for score in scores) / len(scores)


def top_k_accuracy(
    probabilities: Iterable[Sequence[float]],
    y_true: Sequence[int],
    k: int = 3,
) -> float:
    probabilities = list(probabilities)
    if len(probabilities) != len(y_true):
        raise ValueError("probabilities and y_true must have the same length")
    if not probabilities:
        return 0.0
    hits = 0
    for scores, truth in zip(probabilities, y_true):
        best = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:k]
        hits += int(truth in best)
    return hits / len(probabilities)


def format_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    labels: Sequence[str] = LABELS,
) -> str:
    width = max(len(label) for label in labels) + 2
    header = " " * width + "".join(f"{label:>{width}}" for label in labels)
    lines = [header]
    for label, row in zip(labels, matrix):
        lines.append(f"{label:>{width}}" + "".join(f"{value:>{width}}" for value in row))
    return "\n".join(lines)


def classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[str] = LABELS,
) -> str:
    scores = per_class_scores(y_true, y_pred)
    width = max(len(label) for label in labels) + 2
    lines = [
        f"{'label':>{width}}{'precision':>12}{'recall':>10}{'f1':>10}{'support':>10}",
        "-" * (width + 42),
    ]
    for score in scores:
        lines.append(
            f"{labels[score.label]:>{width}}"
            f"{score.precision:>12.3f}"
            f"{score.recall:>10.3f}"
            f"{score.f1:>10.3f}"
            f"{score.support:>10}"
        )
    lines.append("-" * (width + 42))
    lines.append(
        f"{'accuracy':>{width}}{accuracy(y_true, y_pred):>12.3f}"
        f"{'':>10}{'':>10}{len(y_true):>10}"
    )
    lines.append(f"{'macro_f1':>{width}}{macro_f1(y_true, y_pred):>12.3f}")
    return "\n".join(lines)


def hf_metrics(eval_prediction: Any) -> dict[str, float]:
    """``compute_metrics`` adapter for ``transformers.Trainer``."""

    predictions, labels = eval_prediction
    # Trainer passes numpy arrays; avoid importing numpy just for argmax.
    if hasattr(predictions, "argmax"):
        y_pred = predictions.argmax(axis=-1).tolist()
    else:
        y_pred = [argmax(row) for row in predictions]
    y_true = labels.tolist() if hasattr(labels, "tolist") else list(labels)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
    }
