# Korean News Category Classifier

한국어 뉴스 **제목(headline)** 을 입력하면 7개 카테고리(정치·경제·사회·생활/문화·세계·IT/과학·날씨) 중 하나로 분류하고, 카테고리별 확률까지 보여주는 프로젝트입니다.

기존 Colab 노트북(`legacy/2026-10-24-EP010-news-cagegory-selector.ipynb`)의 아이디어(ModernBERT + LoRA, Jev 스타일 `[state, question, target]` 프롬프트)는 유지하되, **데이터·학습·추론·평가·UI를 모듈로 분리**하고 재현 가능한 파이프라인으로 재구성했습니다.

```text
"한은, 기준금리 동결 '만장일치'…시장 예측 적중"
        ->  [경제]  (신뢰도 97.3%)
```

---

## 1. 무엇이 바뀌었나

| 항목 | 기존 노트북 | 현재 코드 |
| --- | --- | --- |
| 코드 구조 | 셀 1개에 데이터 1,171줄 + 학습 + 추론 | `src/news_classifier/` 8개 모듈 |
| 데이터 | 노트북에 하드코딩, 검증 없음 | `data/seed_news.jsonl` + 로드 시 검증·중복 제거 |
| 라벨 정의 | 3곳에 중복, 문서에는 6개만 기재(날씨 누락) | `config.LABELS` 단일 소스 (7개) |
| 평가 | 없음 (전체 데이터로 학습만) | stratified train/val/test + accuracy·macro-F1·confusion matrix |
| 재현성 | seed 없음 | seed 고정, split/설정/메트릭을 모델과 함께 저장 |
| 추론 코드 | 학습/헬퍼/위젯에 3번 중복 | `NewsClassifier` 하나로 통합 |
| 패딩 | `padding="max_length"` (128 고정) | 배치별 dynamic padding |
| LoRA 저장 | 어댑터만 저장 후 메모리에서 merge | merge 후 **단독 실행 가능한 체크포인트**로 저장 |
| ONNX | 이름만 ONNX, 실제 변환 없음 | 미구현임을 명시하고 개선 항목으로 분리 |
| UI | ipywidgets 셀에 의존 | CLI + Python API + (선택) ipywidgets |
| 테스트 | 없음 | 36개 단위/통합 테스트 |

자세한 설계는 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), 개선 항목과 남은 과제는 [docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md)를 참고하세요.

---

## 2. 빠른 시작

### 2.1 설치

Python 3.10+ 기준입니다.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
pip install -e .          # `news-classifier` 명령 사용 (선택)
```

> **GPU가 없어도 됩니다.** CPU에서도 데이터 검증·추론·평가는 동작합니다. 다만 실제 파인튜닝은 GPU(또는 매우 느린 CPU)가 필요합니다.

### 2.2 데이터 확인 (모델 불필요)

```bash
news-classifier data-stats
# 또는 설치 없이:
PYTHONPATH=src python -m news_classifier data-stats
```

출력 예시:

```text
rows              : 1171 -> 1077 (after cleaning)
exact duplicates  : 94
label conflicts   : 0
imbalance ratio   : 2.29
split             : {'train': 755, 'validation': 161, 'test': 161}
```

`train --dry-run`은 모델을 내려받지 않고 검증·분할 결과만 JSON으로 보여줍니다.

### 2.3 학습

```bash
news-classifier train \
  --data data/seed_news.jsonl \
  --output-dir artifacts/news-classifier \
  --epochs 5 --batch-size 8 --lr 3e-4
```

- 기본값: LoRA `r=32`, `alpha=64`, `all-linear`, max length 128, val/test 각 15%.
- CUDA가 있으면 `bf16`(지원 시) 또는 `fp16`을 자동 사용하고, 없으면 CPU fp32로 동작합니다.
- 검증 macro-F1 기준 best checkpoint + early stopping(patience 2)을 사용합니다.
- 학습이 끝나면 LoRA를 merge하여 `artifacts/news-classifier/`에 저장합니다.

### 2.4 추론

CLI:

```bash
news-classifier predict --model-dir artifacts/news-classifier \
  "한은, 기준금리 동결 '만장일치'…시장 예측 적중"
```

대화형:

```bash
news-classifier predict --model-dir artifacts/news-classifier --interactive
```

Python API:

```python
from news_classifier.predict import NewsClassifier, format_prediction

classifier = NewsClassifier.from_pretrained("artifacts/news-classifier")
prediction = classifier.predict(
    "북한, 동해상으로 미사일 발사…한반도 긴장 고조",
    top_k=3,
    threshold=0.6,   # 0.6 미만이면 abstained=True
)
print(prediction.category, prediction.confidence, prediction.abstained)
print(format_prediction(prediction))
```

Jupyter/Colab 위젯(기존 마지막 셀과 동일한 경험):

```python
from news_classifier.predict import NewsClassifier
from news_classifier.ui import launch_ui

launch_ui(NewsClassifier.from_pretrained("artifacts/news-classifier"))
```

### 2.5 평가

```bash
news-classifier evaluate \
  --model-dir artifacts/news-classifier \
  --data data/seed_news.jsonl \
  --output artifacts/news-classifier/eval.json
```

accuracy, macro-F1, per-class precision/recall/F1, confusion matrix, 오분류 예시(최대 20건)를 출력합니다.

---

## 3. 프로젝트 구조

```text
.
├─ data/
│  ├─ seed_news.jsonl          # 노트북에서 추출한 1,171건 (원본)
│  └─ README.md                # 데이터 스키마·알려진 이슈
├─ docs/
│  ├─ ARCHITECTURE.md          # SW 구조, 데이터 흐름, 설계 결정
│  └─ IMPROVEMENTS.md          # 발견한 문제, 적용/남은 개선 아이디어
├─ legacy/
│  └─ 2026-10-24-EP010-news-cagegory-selector.ipynb
├─ scripts/
│  └─ extract_legacy_dataset.py  # 노트북 -> JSONL 재생성
├─ src/news_classifier/
│  ├─ config.py     # 라벨/경로/프롬프트/학습 설정
│  ├─ data.py       # 로드·검증·중복 제거·stratified split
│  ├─ metrics.py    # softmax, accuracy, macro-F1, confusion matrix
│  ├─ modeling.py   # tokenizer, LoRA, merge, 저장/로드
│  ├─ train.py      # 학습 파이프라인, collator, dataset
│  ├─ predict.py    # NewsClassifier 추론 서비스
│  ├─ evaluate.py   # 체크포인트 오프라인 평가
│  ├─ cli.py        # data-stats/train/predict/evaluate/ui
│  └─ ui.py         # (선택) ipywidgets GUI
└─ tests/           # pytest 36건
```

### 기존 노트북 셀 -> 모듈 매핑

| 노트북 셀 | 내용 | 현재 위치 |
| --- | --- | --- |
| cell 1 | 패키지 설치 | `requirements.txt`, `pyproject.toml` |
| cell 2 | 데이터 + JSONL 저장 | `data/seed_news.jsonl`, `data.py` |
| cell 3–7 | 한글 폰트/분포 시각화 | `data-stats` (텍스트), 필요 시 노트북에서 시각화 |
| cell 8 | ModernBERT + LoRA | `modeling.build_model` |
| cell 9 | 토크나이즈/프롬프트 | `data.build_input_text`, `train.tokenize_examples` |
| cell 10 | Trainer 학습 | `train.run_training` |
| cell 11 | "ONNX" 추론 헬퍼 | `predict.NewsClassifier` (ONNX는 미구현, 문서화) |
| cell 12 | 구현 노트 | `docs/ARCHITECTURE.md`, `docs/IMPROVEMENTS.md` |
| cell 13 | ipywidgets GUI | `ui.py`, `cli ui` |

---

## 4. 데이터와 라벨

- 입력: `data/seed_news.jsonl` (JSON Lines, UTF-8)
- 스키마: `{"state": "뉴스 제목", "question": "질문", "target": 1}`
- 라벨: `0 정치 · 1 경제 · 2 사회 · 3 생활/문화 · 4 세계 · 5 IT/과학 · 6 날씨`

로드 시 자동 처리:

- 빈 제목/범위 밖 라벨/잘못된 JSON -> 명확한 행 번호와 함께 오류
- 동일 제목 + 동일 라벨 -> 중복 제거 (94건)
- 동일 제목 + 다른 라벨 -> 충돌로 간주해 제외 (현재 0건)
- 질문 문구가 다른 3건 -> 학습/평가 시 `DEFAULT_QUESTION`으로 통일

자세한 내용은 [data/README.md](data/README.md).

---

## 5. 산출물

`artifacts/news-classifier/`:

| 파일 | 설명 |
| --- | --- |
| `model.safetensors`, `config.json` | merge 완료된 분류 모델 (PEFT 없이 로드 가능) |
| `tokenizer.json`, `tokenizer_config.json`, ... | 토크나이저 |
| `label_map.json` | 라벨 순서 (추론 시 필수) |
| `model_config.json` | base model, max length, LoRA 설정 |
| `training_config.json` | lr, epochs, split ratio, seed |
| `metrics.json` | validation/test 메트릭, 데이터 리포트, confusion matrix |

`label_map.json`을 함께 저장하므로, 라벨 순서가 바뀌어도 추론 결과가 조용히 틀어지지 않습니다.

---

## 6. 테스트

```bash
pytest            # 36 passed
```

포함 범위:

- 데이터 검증/중복 제거/stratified split 결정성
- softmax, accuracy, macro-F1, top-k, confusion matrix
- 추론 서비스(top-k, threshold abstain, batch) — fake model/tokenizer 사용
- collator의 dynamic padding
- 평가 리포트/오분류 예시
- CLI `data-stats`, 잘못된 모델 경로 처리

---

## 7. 이 환경에서의 검증 범위

이 저장소를 정리한 환경은 **GPU가 없고 `peft`/`accelerate`가 설치되어 있지 않습니다.** 따라서:

- ✅ 데이터 파이프라인, 메트릭, 추론 로직, CLI, 테스트: 실행/검증 완료 (CPU)
- ⚠️ 실제 파인튜닝 및 정확도 수치: **미실행**. `train`은 GPU 환경에서 실행해야 합니다.
- ⚠️ ModernBERT 가중치 다운로드/추론: 네트워크·모델 캐시가 있는 환경에서 확인 필요

이 제약 때문에 코드 개선에 집중했고, 학습을 실행하지 않고도 검증 가능한 경계(데이터/메트릭/추론 인터페이스)를 테스트로 고정했습니다.

---

## 8. 다음 단계

우선순위가 높은 항목만 요약하면:

1. **한국어 특화 base model로 교체** — `answerdotai/ModernBERT-base`는 영어 중심입니다. `klue/roberta-base`, `monologg/koelectra-base-v3-discriminator` 등과 비교 실험을 권장합니다.
2. **증강 문장 누출 방지 split** — 현재 split은 행 단위라, 같은 원문에서 변형된 문장이 train/test에 나뉠 수 있습니다. 원문 그룹 단위 split이 필요합니다.
3. **클래스 불균형 처리** — 최다:최소 = 2.29:1. weighted loss 또는 oversampling + macro-F1 기준 튜닝.
4. **신뢰도 보정 및 abstain 임계값** — validation에서 temperature scaling, threshold 선택.
5. **실제 ONNX export + int8 quantization** — CPU 추론 지연 목표가 있다면 필수.

전체 목록과 실험 계획은 [docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md).

---

## 9. 데이터 출처와 라이선스

- 코드: MIT (프로젝트 정책에 맞게 변경하세요).
- `data/seed_news.jsonl`은 기존 노트북에 포함된 **합성 데이터**를 그대로 추출한 것입니다. 실제 뉴스 기사 수집 데이터가 아니며, 배포 전 라이선스·출처를 반드시 확인하세요.
