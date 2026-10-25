from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from news_classifier.config import LABELS
from news_classifier.predict import NewsClassifier, format_prediction


class _Batch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        assert isinstance(texts, list) and texts
        return _Batch(
            input_ids=torch.ones((len(texts), 4), dtype=torch.long),
            attention_mask=torch.ones((len(texts), 4), dtype=torch.long),
        )


class FakeModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("_logits", torch.tensor(logits, dtype=torch.float))

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        batch = input_ids.shape[0]
        return SimpleNamespace(logits=self._logits.expand(batch, -1))


def _classifier(logits):
    return NewsClassifier(FakeModel(logits), FakeTokenizer(), device="cpu")


def test_predict_returns_sorted_scores_and_best_label():
    classifier = _classifier([0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    prediction = classifier.predict("한은, 기준금리 동결")

    assert prediction.category == "경제"
    assert prediction.label_id == 1
    assert prediction.confidence > 0.9
    assert {score.label for score in prediction.scores} == set(LABELS)
    assert prediction.scores[0].label == "경제"
    assert not prediction.abstained


def test_predict_top_k_and_threshold():
    classifier = _classifier([2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    top2 = classifier.predict("t", top_k=2)
    assert len(top2.scores) == 2
    assert top2.category == "정치"

    abstained = classifier.predict("t", threshold=0.99)
    assert abstained.abstained
    assert abstained.category == "정치"


def test_predict_batch_handles_multiple_titles():
    classifier = _classifier([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0])
    predictions = classifier.predict_batch(["a", "b", "c"])
    assert len(predictions) == 3
    assert all(prediction.category == "날씨" for prediction in predictions)


def test_predict_rejects_empty_title_and_bad_options():
    classifier = _classifier([0.0] * len(LABELS))
    with pytest.raises(ValueError):
        classifier.predict("   ")
    with pytest.raises(ValueError):
        classifier.predict("t", top_k=0)
    with pytest.raises(ValueError):
        classifier.predict("t", threshold=1.5)
    assert classifier.predict_batch([]) == []


def test_score_count_must_match_labels():
    classifier = _classifier([0.0, 1.0])
    with pytest.raises(ValueError, match="scores"):
        classifier.predict("t")


def test_format_prediction_renders_scores():
    classifier = _classifier([0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rendered = format_prediction(classifier.predict("금리 동결"))
    assert "금리 동결" in rendered
    assert "경제" in rendered
    assert "■" in rendered


def test_format_prediction_shows_abstain_note():
    classifier = _classifier([0.0] * len(LABELS))
    rendered = format_prediction(classifier.predict("t", threshold=0.9))
    assert "판단 보류" in rendered
