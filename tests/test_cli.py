from __future__ import annotations

import json

from news_classifier.cli import main


def test_data_stats_json_output(capsys):
    assert main(["data-stats", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["raw"] == 1171
    assert payload["dedup"]["exact_duplicates"] == 94
    assert payload["after_dedup"]["total"] == 1077
    assert payload["split"]["train"] > 0


def test_predict_reports_missing_model(capsys):
    assert main(["predict", "--model-dir", "does-not-exist", "테스트 제목"]) == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err
