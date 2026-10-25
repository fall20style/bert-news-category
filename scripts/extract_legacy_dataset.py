"""Regenerate ``data/seed_news.jsonl`` from the legacy notebook.

The original notebook hard-coded the synthetic dataset inside a single code
cell.  This script executes only the data-definition statements of that cell
(imports and the file-writing block are stripped) and serialises the result to
JSON Lines, so the dataset has a reproducible provenance instead of living in a
572 KB ``.ipynb`` diff.

Usage:
    python scripts/extract_legacy_dataset.py
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = PROJECT_ROOT / "legacy" / "2026-10-24-EP010-news-cagegory-selector.ipynb"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "seed_news.jsonl"
DATASET_CELL_INDEX = 2


def extract_dataset_source(notebook_path: Path, cell_index: int) -> str:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    if cell_index >= len(cells):
        raise SystemExit(f"cell {cell_index} not found in {notebook_path}")
    return "".join(cells[cell_index]["source"])


def build_dataset(source: str) -> list[dict]:
    tree = ast.parse(source)
    # Drop heavy imports (torch/peft/...) and the trailing file-writing block.
    body = [
        node
        for node in tree.body
        if not isinstance(node, (ast.Import, ast.ImportFrom, ast.With))
    ]
    module = ast.Module(body=body, type_ignores=[])
    namespace: dict = {"json": json}
    exec(compile(module, "<legacy-dataset-cell>", "exec"), namespace)
    return namespace["synthetic_dataset"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--cell-index", type=int, default=DATASET_CELL_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = build_dataset(extract_dataset_source(args.notebook, args.cell_index))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} examples to {args.output}")


if __name__ == "__main__":
    main()
