from __future__ import annotations

import json
from types import SimpleNamespace

import news_classifier.evaluate as evaluate_module
from news_classifier.evaluate import evaluate_model, write_result


class DummyClassifier:
    def __init__(self, predictions):
        self._predictions = predictions

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise AssertionError("patched instance should be used")

    def predict_batch(self, titles):
        return [
            SimpleNamespace(label_id=label, category=category, confidence=0.9)
            for label, category in self._predictions
        ]


def test_evaluate_model_reports_accuracy_and_errors(tmp_path, monkeypatch):
    data_path = tmp_path / "eval.jsonl"
    rows = [
        {"state": "금리 동결", "target": 1},
        {"state": "북한 미사일", "target": 4},
        {"state": "폭염주의보", "target": 6},
    ]
    data_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    dummy = DummyClassifier([(1, "경제"), (0, "정치"), (6, "날씨")])
    monkeypatch.setattr(
        evaluate_module.NewsClassifier,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: dummy),
    )

    result = evaluate_model("unused", data_path=data_path)

    assert result.total == 3
    assert result.accuracy == 2 / 3
    assert len(result.errors) == 1
    assert result.errors[0]["expected"] == "세계"
    assert result.errors[0]["predicted"] == "정치"
    assert "macro_f1" in result.report

    output = write_result(result, tmp_path / "metrics.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["total"] == 3
    assert payload["errors"][0]["expected"] == "세계"
