"""Inference service: prompt building, scoring and human-readable output.

This replaces the duplicated inference code in the legacy notebook (the
"ONNX" helper and the ipywidgets callback each re-implemented tokenisation and
softmax) with one object that both the CLI and the notebook UI call.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import DEFAULT_QUESTION, LABELS
from .data import build_input_text
from .metrics import argmax, softmax
from .modeling import load_trained_model, resolve_device


@dataclass(frozen=True)
class CategoryScore:
    label_id: int
    label: str
    probability: float


@dataclass(frozen=True)
class Prediction:
    title: str
    label_id: int
    category: str
    confidence: float
    scores: tuple[CategoryScore, ...]
    abstained: bool = False
    threshold: float | None = None

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "category": self.category,
            "label_id": self.label_id,
            "confidence": round(self.confidence, 6),
            "abstained": self.abstained,
            "scores": [
                {
                    "label": score.label,
                    "label_id": score.label_id,
                    "probability": round(score.probability, 6),
                }
                for score in self.scores
            ],
        }


class NewsClassifier:
    """Thin, framework-agnostic wrapper around a sequence classifier."""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        labels: Sequence[str] = LABELS,
        max_length: int = 128,
        device: str | None = None,
        question: str = DEFAULT_QUESTION,
    ) -> None:
        if len(labels) == 0:
            raise ValueError("labels must not be empty")
        self.model = model
        self.tokenizer = tokenizer
        self.labels = tuple(labels)
        self.max_length = max_length
        self.device = resolve_device(device)
        self.question = question
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        device: str | None = None,
        max_length: int = 128,
        question: str = DEFAULT_QUESTION,
    ) -> "NewsClassifier":
        model, tokenizer, labels = load_trained_model(model_dir, device=device)
        return cls(
            model,
            tokenizer,
            labels=labels,
            max_length=max_length,
            device=device,
            question=question,
        )

    def predict(
        self,
        title: str,
        *,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> Prediction:
        return self.predict_batch(
            [title], top_k=top_k, threshold=threshold
        )[0]

    def predict_batch(
        self,
        titles: Sequence[str],
        *,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[Prediction]:
        titles = list(titles)
        if not titles:
            return []
        probabilities = self._probabilities(titles)
        return [
            self._build_prediction(title, scores, top_k=top_k, threshold=threshold)
            for title, scores in zip(titles, probabilities)
        ]

    def _probabilities(self, titles: Sequence[str]) -> list[list[float]]:
        import torch

        prompts = [build_input_text(title, self.question) for title in titles]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits.detach().float().cpu().tolist()
        return [softmax(row) for row in logits]

    def _build_prediction(
        self,
        title: str,
        probabilities: Sequence[float],
        *,
        top_k: int | None,
        threshold: float | None,
    ) -> Prediction:
        if len(probabilities) != len(self.labels):
            raise ValueError(
                f"model returned {len(probabilities)} scores but "
                f"{len(self.labels)} labels are configured"
            )
        if threshold is not None and not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")

        best_id = argmax(probabilities)
        confidence = float(probabilities[best_id])
        scored = sorted(
            (
                CategoryScore(
                    label_id=index,
                    label=label,
                    probability=float(probabilities[index]),
                )
                for index, label in enumerate(self.labels)
            ),
            key=lambda score: score.probability,
            reverse=True,
        )
        if top_k is not None:
            if top_k <= 0:
                raise ValueError("top_k must be positive")
            scored = scored[:top_k]
        return Prediction(
            title=title,
            label_id=best_id,
            category=self.labels[best_id],
            confidence=confidence,
            scores=tuple(scored),
            abstained=threshold is not None and confidence < threshold,
            threshold=threshold,
        )


def format_prediction(prediction: Prediction, *, bar_width: int = 20) -> str:
    """Render a prediction the same way the notebook widget did."""

    lines = [
        "=" * 60,
        f'입력 헤드라인 : "{prediction.title}"',
    ]
    if prediction.abstained:
        lines.append(
            f"판단 보류     : 최고 신뢰도 {prediction.confidence:.2%} "
            f"< 임계값 {prediction.threshold:.2%} "
            f"(참고: {prediction.category})"
        )
    else:
        lines.append(
            f"최종 예측     : [{prediction.category}] "
            f"(신뢰도: {prediction.confidence:.2%})"
        )
    lines.extend(
        [
            "=" * 60,
            "카테고리별 확률 점수:",
            "-" * 60,
        ]
    )
    for score in prediction.scores:
        filled = int(round(score.probability * bar_width))
        bar = "■" * filled + "□" * (bar_width - filled)
        lines.append(f"  - {score.label:<8}: {bar} {score.probability:>8.4%}")
    lines.append("-" * 60)
    return "\n".join(lines)
