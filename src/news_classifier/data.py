"""Dataset loading, validation, cleaning and splitting.

The legacy notebook kept the whole dataset inline and fed it straight into
``datasets.Dataset.from_json`` with no validation, no de-duplication and no
hold-out split.  This module turns that into an explicit, testable pipeline and
drops the heavy ``datasets`` dependency in the process.
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import DEFAULT_QUESTION, LABELS, LABEL2ID, PROMPT_TEMPLATE

_WHITESPACE_RE = re.compile(r"\s+")


class DataValidationError(ValueError):
    """Raised when a data file contains unusable rows."""


@dataclass(frozen=True)
class Example:
    """One ``[state, question, target]`` training example."""

    title: str
    label: int
    question: str = DEFAULT_QUESTION

    @property
    def label_name(self) -> str:
        return LABELS[self.label]

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.title, "question": self.question, "target": self.label}


@dataclass(frozen=True)
class DedupReport:
    total: int
    unique: int
    exact_duplicates: int
    conflicting_titles: tuple[str, ...]

    @property
    def dropped_conflicts(self) -> int:
        return len(self.conflicting_titles)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "unique": self.unique,
            "exact_duplicates": self.exact_duplicates,
            "dropped_conflicts": self.dropped_conflicts,
            "conflicting_titles": list(self.conflicting_titles),
        }


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[Example, ...]
    validation: tuple[Example, ...]
    test: tuple[Example, ...]

    def to_dict(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }


def normalize_text(text: str) -> str:
    """NFC-normalise and collapse whitespace so duplicates compare reliably."""

    return _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def build_input_text(title: str, question: str = DEFAULT_QUESTION) -> str:
    """Build the single prompt string the model was trained on.

    The legacy notebook repeated this f-string in three cells (training,
    inference helper, widget).  One function keeps train/serve consistent.
    """

    title = normalize_text(title)
    if not title:
        raise ValueError("title must not be empty")
    return PROMPT_TEMPLATE.format(title=title, question=normalize_text(question))


def _read_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return []

    if text.lstrip().startswith("["):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DataValidationError(f"{path}: invalid JSON array ({exc.msg})") from exc
        if not isinstance(payload, list):
            raise DataValidationError(f"{path}: expected a JSON array of objects")
        return payload

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataValidationError(
                f"{path}:{line_number}: invalid JSON ({exc.msg})"
            ) from exc
        if not isinstance(row, dict):
            raise DataValidationError(
                f"{path}:{line_number}: expected an object, got {type(row).__name__}"
            )
        rows.append(row)
    return rows


def _coerce_label(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("boolean is not a valid label")
    if isinstance(value, int):
        label = value
    elif isinstance(value, str):
        key = value.strip()
        if key.isdigit():
            label = int(key)
        elif key in LABEL2ID:
            label = LABEL2ID[key]
        else:
            raise ValueError(f"unknown label name {value!r}")
    else:
        raise TypeError(f"label must be int or str, got {type(value).__name__}")
    if not 0 <= label < len(LABELS):
        raise ValueError(f"label {label} out of range 0..{len(LABELS) - 1}")
    return label


def load_examples(
    path: str | Path,
    *,
    question: str | None = None,
    strict: bool = True,
    max_errors: int = 10,
) -> list[Example]:
    """Load a JSONL/JSON dataset and validate every row.

    ``question`` overrides the per-row question, which is useful when a data
    file only stores ``state``/``target``.  With ``strict=False`` invalid rows
    are skipped instead of raising.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")

    examples: list[Example] = []
    errors: list[str] = []
    for index, row in enumerate(_read_rows(path), start=1):
        try:
            title = row.get("state", row.get("title"))
            if not isinstance(title, str) or not normalize_text(title):
                raise ValueError("missing or empty 'state'/'title'")
            raw_label = row.get("target", row.get("label"))
            if raw_label is None:
                raise ValueError("missing 'target'/'label'")
            label = _coerce_label(raw_label)
            row_question = question or row.get("question") or DEFAULT_QUESTION
            if not isinstance(row_question, str) or not row_question.strip():
                raise ValueError("'question' must be a non-empty string")
            examples.append(
                Example(
                    title=normalize_text(title),
                    label=label,
                    question=normalize_text(row_question),
                )
            )
        except (TypeError, ValueError) as exc:
            message = f"{path}: row {index}: {exc}"
            if strict:
                errors.append(message)
                if len(errors) >= max_errors:
                    break
            continue

    if errors:
        raise DataValidationError(
            "invalid rows found:\n  " + "\n  ".join(errors)
        )
    if not examples:
        raise DataValidationError(f"{path}: no valid examples found")
    return examples


def deduplicate(examples: Iterable[Example]) -> tuple[list[Example], DedupReport]:
    """Remove exact duplicate headlines and same-title/different-label conflicts.

    Synthetic augmentation in the legacy notebook produced repeated rows and a
    handful of contradictory labels.  Conflicting titles are dropped rather
    than guessed, because either choice teaches the model noise.
    """

    examples = list(examples)
    seen: dict[str, int] = {}
    kept: list[Example] = []
    exact_duplicates = 0
    conflicts: list[str] = []

    for example in examples:
        key = normalize_text(example.title)
        if key not in seen:
            seen[key] = example.label
            kept.append(example)
            continue
        if seen[key] == example.label:
            exact_duplicates += 1
            continue
        # Same headline, different label: unusable supervision.
        if key not in conflicts:
            conflicts.append(key)

    report = DedupReport(
        total=len(examples),
        unique=len(kept),
        exact_duplicates=exact_duplicates,
        conflicting_titles=tuple(conflicts),
    )
    return kept, report


def class_counts(examples: Iterable[Example]) -> dict[int, int]:
    counts = {label_id: 0 for label_id in range(len(LABELS))}
    for example in examples:
        counts[example.label] += 1
    return counts


def imbalance_ratio(examples: Iterable[Example]) -> float:
    counts = [count for count in class_counts(examples).values() if count > 0]
    if not counts:
        return 0.0
    return max(counts) / min(counts)


def split_examples(
    examples: Sequence[Example],
    *,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify: bool = True,
) -> DatasetSplit:
    """Split into train/validation/test, stratified by label by default.

    The legacy notebook trained on 100% of the data and never measured
    generalisation.  Each class keeps at least one example per split when it
    has three or more rows.
    """

    if not 0.0 <= val_ratio < 1.0 or not 0.0 <= test_ratio < 1.0:
        raise ValueError("ratios must be in [0, 1)")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1")

    rng = random.Random(seed)

    if not stratify:
        shuffled = list(examples)
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_test = (
            max(1, round(n_total * test_ratio))
            if test_ratio > 0 and n_total >= 3
            else 0
        )
        n_val = (
            max(1, round(n_total * val_ratio))
            if val_ratio > 0 and n_total >= 3
            else 0
        )
        while n_test + n_val >= n_total:
            if n_test >= n_val and n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1
            else:
                break
        return DatasetSplit(
            train=tuple(shuffled[n_test + n_val :]),
            validation=tuple(shuffled[n_test : n_test + n_val]),
            test=tuple(shuffled[:n_test]),
        )

    train: list[Example] = []
    validation: list[Example] = []
    test: list[Example] = []

    groups: dict[int, list[Example]] = {}
    for example in examples:
        groups.setdefault(example.label, []).append(example)

    for label_id in sorted(groups):
        group = list(groups[label_id])
        rng.shuffle(group)
        n_total = len(group)
        if n_total < 3:
            train.extend(group)
            continue

        n_test = max(1, round(n_total * test_ratio)) if test_ratio > 0 else 0
        n_val = max(1, round(n_total * val_ratio)) if val_ratio > 0 else 0
        # Always keep at least one training example per class.
        while n_test + n_val >= n_total:
            if n_test >= n_val and n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1
            else:
                break

        test.extend(group[:n_test])
        validation.extend(group[n_test : n_test + n_val])
        train.extend(group[n_test + n_val :])

    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    return DatasetSplit(
        train=tuple(train),
        validation=tuple(validation),
        test=tuple(test),
    )


def write_examples(examples: Iterable[Example], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")


def describe(examples: Sequence[Example]) -> dict[str, Any]:
    """Summary used by ``data-stats`` and by the training report."""

    counts = class_counts(examples)
    total = len(examples)
    return {
        "total": total,
        "class_counts": {LABELS[key]: value for key, value in counts.items()},
        "class_share": {
            LABELS[key]: round(value / total, 4) if total else 0.0
            for key, value in counts.items()
        },
        "imbalance_ratio": round(imbalance_ratio(examples), 3),
    }
