from __future__ import annotations

import json
from pathlib import Path

import pytest

from news_classifier.config import LABELS
from news_classifier.data import (
    DataValidationError,
    Example,
    build_input_text,
    class_counts,
    deduplicate,
    describe,
    load_examples,
    normalize_text,
    split_examples,
    write_examples,
)


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  한은,\t기준금리   동결  ") == "한은, 기준금리 동결"


def test_build_input_text_uses_shared_prompt():
    prompt = build_input_text("  속보:   금리 동결 ")
    assert prompt == "뉴스 제목: 속보: 금리 동결\n질문: 이 뉴스의 주요 카테고리는 무엇인가요?"
    with pytest.raises(ValueError):
        build_input_text("   ")


def test_load_examples_accepts_title_and_label_names(tmp_path):
    path = _write_jsonl(
        tmp_path / "data.jsonl",
        [
            {"state": "금리 동결", "target": 1},
            {"title": "북한 미사일", "label": "세계", "question": "분류?"},
        ],
    )
    examples = load_examples(path)
    assert [example.label for example in examples] == [1, 4]
    assert examples[1].question == "분류?"


def test_load_examples_reads_json_array(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(
        json.dumps([{"state": "폭염주의보", "target": 6}], ensure_ascii=False),
        encoding="utf-8",
    )
    assert load_examples(path)[0].label_name == "날씨"


def test_load_examples_reports_invalid_rows(tmp_path):
    path = _write_jsonl(
        tmp_path / "bad.jsonl",
        [{"state": "ok", "target": 0}, {"state": "", "target": 0}],
    )
    with pytest.raises(DataValidationError, match="row 2"):
        load_examples(path)
    assert len(load_examples(path, strict=False)) == 1


def test_load_examples_rejects_out_of_range_label(tmp_path):
    path = _write_jsonl(tmp_path / "bad.jsonl", [{"state": "ok", "target": 99}])
    with pytest.raises(DataValidationError, match="out of range"):
        load_examples(path)


def test_deduplicate_removes_exact_and_conflicting_rows():
    examples = [
        Example("금리 동결", 1),
        Example("금리  동결", 1),
        Example("금리 동결", 0),
        Example("북한 미사일", 4),
    ]
    cleaned, report = deduplicate(examples)
    assert [example.title for example in cleaned] == ["금리 동결", "북한 미사일"]
    assert report.total == 4
    assert report.unique == 2
    assert report.exact_duplicates == 1
    assert report.dropped_conflicts == 1
    assert report.conflicting_titles == ("금리 동결",)


def test_split_is_deterministic_and_stratified():
    examples = [Example(f"{label}-{index}", label) for label in range(len(LABELS)) for index in range(20)]
    first = split_examples(examples, seed=7)
    second = split_examples(list(reversed(examples)), seed=7)

    assert first.to_dict() == second.to_dict()
    assert len(first.train) + len(first.validation) + len(first.test) == len(examples)
    for part in (first.train, first.validation, first.test):
        assert set(class_counts(part)) == set(range(len(LABELS)))
        assert all(count > 0 for count in class_counts(part).values())


def test_split_without_stratification_still_holds_out_data():
    examples = [Example(f"t{index}", index % len(LABELS)) for index in range(70)]
    split = split_examples(examples, stratify=False, seed=1)

    assert split.to_dict() == {"train": 50, "validation": 10, "test": 10}
    assert {example.title for example in split.train}.isdisjoint(
        {example.title for example in split.test}
    )
    assert {example.title for example in split.train}.isdisjoint(
        {example.title for example in split.validation}
    )


def test_split_without_holdout_keeps_everything_in_train():
    examples = [Example(f"t{index}", index % 7) for index in range(14)]
    split = split_examples(examples, val_ratio=0.0, test_ratio=0.0)
    assert len(split.train) == 14
    assert not split.validation
    assert not split.test


def test_describe_reports_imbalance():
    examples = [Example("a", 0), Example("b", 0), Example("c", 1)]
    payload = describe(examples)
    assert payload["total"] == 3
    assert payload["class_counts"]["정치"] == 2
    assert payload["imbalance_ratio"] == 2.0


def test_write_examples_roundtrip(tmp_path):
    path = tmp_path / "out.jsonl"
    write_examples([Example("금리 동결", 1)], path)
    assert load_examples(path) == [Example("금리 동결", 1)]


def test_seed_dataset_is_valid():
    seed_path = Path(__file__).resolve().parents[1] / "data" / "seed_news.jsonl"
    examples = load_examples(seed_path)
    cleaned, report = deduplicate(examples)
    assert len(examples) == 1171
    assert len(cleaned) == 1077
    assert report.exact_duplicates == 94
    assert set(class_counts(cleaned)) == set(range(len(LABELS)))
    assert all(count > 0 for count in class_counts(cleaned).values())
