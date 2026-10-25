from __future__ import annotations

import pytest

from news_classifier.metrics import (
    accuracy,
    argmax,
    classification_report,
    confusion_matrix,
    format_confusion_matrix,
    hf_metrics,
    macro_f1,
    per_class_scores,
    softmax,
    top_k_accuracy,
)


def test_softmax_is_stable_and_normalised():
    scores = softmax([1000.0, 1000.0, 1000.0])
    assert scores == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert sum(softmax([0.1, 2.0, -3.0])) == pytest.approx(1.0)


def test_softmax_rejects_empty_input():
    with pytest.raises(ValueError):
        softmax([])


def test_argmax_and_accuracy():
    assert argmax([0.1, 0.9, 0.0]) == 1
    assert accuracy([0, 1, 1], [0, 1, 0]) == pytest.approx(2 / 3)
    assert accuracy([], []) == 0.0
    with pytest.raises(ValueError):
        accuracy([0], [0, 1])


def test_macro_f1_matches_hand_computed_value():
    # class 0: p=1.0 r=0.5 f1=2/3, class 1: p=2/3 r=1.0 f1=0.8
    assert macro_f1([0, 0, 1, 1], [0, 1, 1, 1]) == pytest.approx((2 / 3 + 0.8) / 2)


def test_macro_f1_ignores_absent_classes():
    assert macro_f1([0, 0], [0, 0]) == pytest.approx(1.0)


def test_per_class_scores_support():
    scores = {score.label: score for score in per_class_scores([0, 0, 1], [0, 1, 1])}
    assert scores[0].support == 2
    assert scores[1].support == 1
    assert scores[2].support == 0


def test_confusion_matrix_layout():
    matrix = confusion_matrix([0, 0, 1], [0, 1, 1], labels=[0, 1])
    assert matrix == [[1, 1], [0, 1]]


def test_format_confusion_matrix_contains_labels():
    rendered = format_confusion_matrix([[1, 0], [0, 1]], labels=["정치", "경제"])
    assert "정치" in rendered and "경제" in rendered


def test_classification_report_contains_summary():
    report = classification_report([0, 1], [0, 0])
    assert "macro_f1" in report
    assert "accuracy" in report


def test_top_k_accuracy():
    probabilities = [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]]
    assert top_k_accuracy(probabilities, [0, 2], k=1) == pytest.approx(0.5)
    assert top_k_accuracy(probabilities, [0, 2], k=2) == pytest.approx(1.0)


def test_hf_metrics_adapter_accepts_numpy_like_objects():
    class Array:
        def __init__(self, data):
            self.data = data

        def argmax(self, axis=-1):
            assert axis == -1
            return Array([max(range(len(row)), key=row.__getitem__) for row in self.data])

        def tolist(self):
            return self.data

    result = hf_metrics((Array([[0.9, 0.1], [0.2, 0.8]]), Array([0, 0])))
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["macro_f1"] == pytest.approx(2 / 3)
