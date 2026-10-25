# data/

## `seed_news.jsonl`

The 1,171 synthetic training examples that were hard-coded inside the legacy
notebook (`legacy/2026-10-24-EP010-news-cagegory-selector.ipynb`, cell 2).

Format — one JSON object per line, matching the notebook's
`[state, question, target]` layout:

```json
{"state": "한은, 기준금리 동결 '만장일치'…시장 예측 적중", "question": "이 뉴스의 주요 카테고리는 무엇인가요?", "target": 1}
```

| field      | meaning                                            |
| ---------- | -------------------------------------------------- |
| `state`    | news headline (the model input)                    |
| `question` | fixed question appended to the headline (prompt)   |
| `target`   | integer label index (see below)                    |

Label map:

| id | label     |
| -- | --------- |
| 0  | 정치      |
| 1  | 경제      |
| 2  | 사회      |
| 3  | 생활/문화 |
| 4  | 세계      |
| 5  | IT/과학   |
| 6  | 날씨      |

Known issues (handled at load time by `news_classifier.data`):

- 94 exact duplicate headlines,
- 3 rows whose `question` text differs by a trailing `?`/`요`,
- class imbalance (260 vs. 100 rows).

Regenerate with:

```bash
python scripts/extract_legacy_dataset.py
```

The file is generated, not hand-edited. Add new data in a separate JSONL and
pass it via `--data`.
