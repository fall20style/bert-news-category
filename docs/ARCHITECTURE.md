# SW 구조 (Architecture)

## 1. 목표와 제약

**목표**

- 뉴스 제목 하나를 7개 카테고리로 분류하고, 카테고리별 확률을 투명하게 제공한다.
- 노트북 한 셀에 섞여 있던 데이터·학습·추론·UI를 분리해 **테스트 가능한 경계**를 만든다.
- 학습 조건(split, seed, 하이퍼파라미터)과 라벨 맵을 모델과 함께 저장해 **재현 가능**하게 만든다.
- GPU가 없는 환경에서도 데이터 검증·평가·추론 인터페이스를 검증할 수 있게 한다.

**비목표**

- 이번 작업에서 정확도 개선 실험을 수행하지 않는다 (GPU 부재).
- ONNX/양자화 배포 파이프라인은 구현하지 않는다 (문서화된 후속 과제).
- 실 뉴스 수집·라벨링 파이프라인은 범위 밖이다.

---

## 2. 레이어 구조

```text
┌──────────────────────────────────────────────────────────────┐
│  Interfaces                                                  │
│  cli.py (data-stats / train / predict / evaluate / ui)       │
│  ui.py  (ipywidgets, Jupyter 전용)                            │
└───────────────┬──────────────────────────────┬───────────────┘
                │                              │
┌───────────────▼──────────────┐  ┌────────────▼───────────────┐
│  Application                 │  │  Inference service         │
│  train.run_training          │  │  predict.NewsClassifier    │
│  evaluate.evaluate_model     │  │  - prompt build            │
│  - split / tokenize          │  │  - forward / softmax       │
│  - Trainer / early stopping  │  │  - top-k / abstain         │
│  - merge + save bundle       │  │  - formatting              │
└───────────────┬──────────────┘  └────────────┬───────────────┘
                │                              │
┌───────────────▼──────────────┐  ┌────────────▼───────────────┐
│  Domain                      │  │  Model layer               │
│  data.py                     │  │  modeling.py               │
│  - Example / validation      │  │  - tokenizer               │
│  - dedup / split             │  │  - AutoModel + LoRA        │
│  metrics.py                  │  │  - merge / save / load     │
│  - softmax / F1 / matrix     │  │                            │
└───────────────┬──────────────┘  └────────────┬───────────────┘
                │                              │
┌───────────────▼──────────────────────────────▼───────────────┐
│  config.py — 라벨, 경로, 프롬프트, ModelConfig, TrainConfig     │
└──────────────────────────────────────────────────────────────┘
```

의존 방향은 위에서 아래로만 흐릅니다. `config.py`는 어떤 모듈도 import하지 않으며, `data.py`/`metrics.py`는 torch/transformers 없이 동작합니다. 덕분에 모델 없이도 단위 테스트가 가능합니다.

---

## 3. 모듈별 책임

| 모듈 | 책임 | 주요 공개 API | 외부 의존성 |
| --- | --- | --- | --- |
| `config.py` | 라벨 맵, 경로, 프롬프트, 설정 dataclass | `LABELS`, `ModelConfig`, `TrainConfig`, `build_*` 없음 | 표준 라이브러리 |
| `data.py` | 로드·검증·정규화·중복 제거·stratified split | `load_examples`, `deduplicate`, `split_examples`, `build_input_text`, `describe` | 표준 라이브러리 |
| `metrics.py` | softmax, accuracy, macro-F1, top-k, confusion matrix, report | `softmax`, `macro_f1`, `classification_report`, `hf_metrics` | 표준 라이브러리 |
| `modeling.py` | tokenizer/모델 생성, LoRA, merge, 번들 저장/로드 | `build_model`, `merge_adapter`, `save_bundle`, `load_trained_model` | transformers, peft, torch |
| `train.py` | 학습 파이프라인, collator, dataset | `prepare_data`, `tokenize_examples`, `run_training`, `run_data_report` | torch, transformers |
| `predict.py` | 추론 서비스 | `NewsClassifier`, `Prediction`, `format_prediction` | torch, transformers |
| `evaluate.py` | 체크포인트 오프라인 평가 | `evaluate_model`, `write_result` | torch, transformers |
| `cli.py` | argparse 기반 진입점 | `main`, `build_parser` | (지연 import) |
| `ui.py` | ipywidgets GUI | `build_ui`, `launch_ui` | ipywidgets, IPython (선택) |

> `predict.py`는 `torch`를 top-level에서 import합니다. `news_classifier` 패키지 자체는 `config.py`만 import하므로 `data-stats`나 `--help`는 무겁지 않습니다.

---

## 4. 학습 데이터 흐름

```text
data/seed_news.jsonl
        │
        ▼
load_examples(question=DEFAULT_QUESTION)
  - JSONL/JSON 파싱
  - 필수 키/타입/라벨 범위 검증
  - 제목·질문 NFC 정규화 + 공백 축약
        │  1,171행
        ▼
deduplicate()
  - (정규화 제목, 라벨) 중복 제거          -> 94건 제거
  - 같은 제목·다른 라벨은 충돌로 제외       -> 0건
        │  1,077행
        ▼
split_examples(seed=42, val=0.15, test=0.15, stratify=True)
  - 라벨별 셔플 후 라벨별 최소 1건 보장
        │  train 755 / val 161 / test 161
        ▼
tokenize_examples()
  - build_input_text("뉴스 제목: ...\n질문: ...")
  - truncation=True, max_length=128, padding=False
        │
        ▼
PadCollator (배치마다 최장 길이로 패딩)
        │
        ▼
Trainer(eval_strategy=epoch, save_strategy=epoch,
        load_best_model_at_end, metric=macro_f1,
        EarlyStopping(patience=2))
        │
        ▼
merge_adapter() -> save_bundle()
```

### split 설계

- **stratified**: 라벨별로 train/val/test에 최소 1건을 보장합니다(3건 이상일 때).
- **seed 고정**: `random.Random(seed)`만 사용하므로 프로세스/플랫폼 간 재현됩니다.
- **중복 제거 후 split**: 중복이 train/test에 나뉘어 점수가 부풀려지는 것을 막습니다.
- **한계**: 증강으로 생성된 유사 문장은 "다른 행"이므로 그룹 단위 누출이 남아 있습니다. `docs/IMPROVEMENTS.md`의 P0 항목입니다.

---

## 5. 추론 데이터 흐름

```text
title
  │
  ▼
build_input_text(title, question)      # 학습과 동일한 프롬프트
  │
  ▼
tokenizer(padding=True, truncation=True, max_length)
  │
  ▼
model(**inputs) -> logits
  │
  ▼
softmax -> [7] probabilities
  │
  ├─ argmax -> category, confidence
  ├─ sort desc -> CategoryScore[]
  └─ threshold -> abstained 여부
```

`NewsClassifier`는 `model`/`tokenizer`를 주입받을 수 있어(fake model로 테스트 가능), `from_pretrained` 없이도 단위 테스트가 가능합니다.

---

## 6. 핵심 설계 결정

| # | 결정 | 이유 | 트레이드오프 |
| --- | --- | --- | --- |
| 1 | 라벨 맵을 `config.LABELS` 하나로 통일 | 노트북은 3곳에 라벨을 중복 정의했고 문서에는 6개만 있었다 | 라벨 추가 시 코드 수정 필요 (현재 의도) |
| 2 | `Example` dataclass + 명시적 검증 | 잘못된 행이 조용히 학습되는 것을 방지 | 로드 비용 소폭 증가 (1k행에서는 무시 가능) |
| 3 | `datasets` 의존성 제거, `TokenizedDataset` 직접 구현 | 설치 부담 감소, 동작이 단순/투명 | 대규모 데이터에서 메모리 최적화는 직접 해야 함 |
| 4 | 배치별 dynamic padding | `padding="max_length"`는 짧은 제목에서 연산 낭비 | collator 구현 필요 (30줄) |
| 5 | 프롬프트 빌더 단일화 | 학습/추론 프롬프트 불일치가 성능을 조용히 망침 | 질문을 바꾸려면 재학습 필요 |
| 6 | LoRA merge 후 저장 | `save_model`은 어댑터만 저장해 로드 경로가 불명확했음 | 저장 용량 증가 (base 크기) |
| 7 | `label_map.json` 동봉 | 라벨 순서가 바뀌면 추론이 조용히 틀어짐 | 파일 1개 추가 |
| 8 | 메트릭을 직접 구현 | CPU 환경에서 sklearn 없이 평가/테스트 가능 | 고급 메트릭은 직접 추가해야 함 |
| 9 | CLI + 서비스 분리 | 노트북/Colab 없이도 배치 평가·자동화 가능 | UI는 별도 선택 기능 |
| 10 | `abstain` threshold 옵션 | 낮은 신뢰도를 "오답"으로 내보내지 않기 위한 최소 장치 | 임계값은 데이터로 선택해야 함 (미구현) |

---

## 7. 산출물 스키마

`save_bundle()`이 만드는 디렉터리:

```text
artifacts/news-classifier/
├─ config.json              # HF model config (num_labels=7, id2label 포함)
├─ model.safetensors        # merge된 가중치
├─ tokenizer.json
├─ tokenizer_config.json
├─ special_tokens_map.json
├─ label_map.json
├─ model_config.json
├─ training_config.json
└─ metrics.json
```

`metrics.json` 구조:

```json
{
  "model_config": { "model_id": "...", "max_length": 128, "use_lora": true, ... },
  "train_config": { "learning_rate": 0.0003, "seed": 42, "val_ratio": 0.15, ... },
  "labels": ["정치", "경제", "사회", "생활/문화", "세계", "IT/과학", "날씨"],
  "metrics": {
    "validation": { "eval_loss": 0.42, "eval_accuracy": 0.91, "eval_macro_f1": 0.89, ... },
    "test": {
      "accuracy": 0.90,
      "macro_f1": 0.88,
      "report": "...",
      "confusion_matrix": "...",
      "support": 161
    }
  },
  "data": {
    "raw": 1171,
    "dedup": { "exact_duplicates": 94, "dropped_conflicts": 0, ... },
    "split": { "train": 755, "validation": 161, "test": 161 }
  }
}
```

---

## 8. 오류 처리와 경계 조건

| 상황 | 동작 |
| --- | --- |
| 데이터 파일 없음 | `FileNotFoundError` |
| JSON 파싱 실패 | `DataValidationError` + 파일/행 번호 |
| 라벨 범위 밖 | `DataValidationError` (strict) / skip (strict=False) |
| 같은 제목·다른 라벨 | 충돌 목록에 기록하고 제외 |
| 학습 split이 빔 | `RuntimeError` |
| val/test가 모두 빔 | `RuntimeError` (eval 없이 best checkpoint를 고를 수 없음) |
| LoRA 요청 + peft 미설치 | 설치 안내와 함께 `ImportError` |
| CUDA 요청 + GPU 없음 | `RuntimeError` |
| 모델 디렉터리 없음 | CLI에서 `error: ...` 출력 후 exit code 1 |
| 라벨 수와 logits 수 불일치 | `ValueError` (조용한 오분류 방지) |
| 빈 제목 | `ValueError` |
| ipywidgets/IPython 없음 | 설치/대안 안내와 함께 `ImportError` |

---

## 9. 확장 포인트

- **새 라벨**: `config.LABELS` 수정 -> 데이터 재라벨링 -> 재학습. `num_labels`는 자동 반영됩니다.
- **새 base model**: `--model-id` 또는 `ModelConfig(model_id=...)`. LoRA target은 `all-linear`라 아키텍처 이름에 덜 민감합니다.
- **Full fine-tuning**: `--no-lora`.
- **다른 데이터 형식**: `data.load_examples`가 `state`/`title`, `target`/`label`을 모두 허용합니다.
- **새 메트릭**: `metrics.py`에 함수 추가 후 `hf_metrics`에 연결.
- **배포**: `NewsClassifier`가 유일한 추론 경계이므로 FastAPI/gRPC 래퍼를 그 위에 얹으면 됩니다.
- **실험 관리**: `TrainConfig`를 YAML/JSON으로 직렬화하는 계층을 추가하기 쉽습니다 (`to_dict` 제공).

---

## 10. 런타임 의존성

| 구분 | 패키지 | 비고 |
| --- | --- | --- |
| 필수 | `torch`, `transformers`, `accelerate`, `peft`, `numpy` | 학습/추론 |
| 선택 | `ipywidgets`, `ipython` | 노트북 GUI |
| 선택 | `optimum[onnxruntime]` | 향후 ONNX export |
| 개발 | `pytest` | 36 tests |

CPU만 있는 환경에서도 `torch` CPU wheel로 추론·평가가 가능합니다. 학습은 `accelerate`가 필수입니다(transformers 5.x 기준).
