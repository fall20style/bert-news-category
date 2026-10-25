# 개선 기록과 남은 과제

이 문서는 (1) 기존 노트북에서 발견한 문제, (2) 이번 작업에서 실제로 반영한 개선, (3) 아직 반영하지 않은 아이디어를 우선순위와 함께 정리합니다.

> **검증 범위**: GPU가 없는 환경이라 파인튜닝과 정확도 측정은 수행하지 않았습니다. 데이터 파이프라인·메트릭·추론 인터페이스·CLI는 CPU에서 테스트(36건)로 검증했습니다.

---

## 1. 발견한 문제

### 1.1 구조 / 유지보수

| ID | 위치 | 문제 | 영향 |
| --- | --- | --- | --- |
| S1 | cell 2 | 합성 데이터 1,171건이 노트북 셀에 하드코딩 | diff 리뷰 불가, 재사용 불가, 노트북 572KB |
| S2 | cell 2, 7, 8 | 라벨 리스트가 3곳에 중복 정의 | 한 곳만 수정하면 조용한 오분류 |
| S3 | cell 12 | 구현 노트에 라벨 6개만 기재 (`날씨` 누락) | 문서와 실제 모델 불일치 |
| S4 | cell 2, 9, 11, 13 | 프롬프트 f-string이 3번 중복 | train/serve 프롬프트 불일치 위험 |
| S5 | cell 11, 13 | 토크나이즈+softmax 추론 코드가 2번 중복 | 수정 누락, 테스트 불가 |
| S6 | cell 11 | 함수명 `classify_news_onnx`이지만 ONNX 변환 없음 | 오해, 배포 성능 기대 불일치 |
| S7 | cell 10, 11 | 전역 변수(`model`, `merged_model`, `tokenizer`)에 의존, `NameError`로 복구 | 셀 실행 순서에 취약 |
| S8 | cell 13 | GUI가 ipywidgets/Colab에 강결합 | CLI·서버·배치에서 재사용 불가 |
| S9 | 전체 | 테스트/CI/린트 없음 | 회귀 감지 불가 |

### 1.2 데이터 품질

| ID | 문제 | 측정값 | 영향 |
| --- | --- | --- | --- |
| D1 | 완전 중복 행 | 1,171 → 1,077 (94건, 8.0%) | train/test 누출 시 지표 부풀림 |
| D2 | 질문 문구 불일치 | 3건 (`...무엇인가?` vs `...무엇인가요?`) | 프롬프트 분포 미세 불일치 |
| D3 | 클래스 불균형 | 229 : 100 = **2.29 : 1** (정제 후) | 소수 클래스 recall 저하 |
| D4 | 증강 문장의 near-duplicate | `extended_*`가 원문 + 치환 + 접두어 변형을 생성 | 행 단위 split 시 train/test 누출 |
| D5 | 라벨 경계 모호 | 예: `외교부, 한미일 안보 협력`(정치=0), `이스라엘-하마스 휴전`(세계=4), `반도체株`(경제=1) vs `HBM4 양산`(IT=5) | 사람도 헷갈리는 라벨, 성능 상한 제한 |
| D6 | 합성 데이터 편향 | 특정 사건/표현(총선, 금리, 전세사기) 반복 | 실제 뉴스 분포와 격차 |
| D7 | 데이터 출처/라이선스 불명 | LLM 합성 추정 | 배포 전 확인 필요 |

### 1.3 학습 / 평가

| ID | 문제 | 영향 |
| --- | --- | --- |
| T1 | train/validation/test split 없음 | 일반화 성능 측정 불가 |
| T2 | seed 고정 없음 | 재현 불가 |
| T3 | `padding="max_length"` (128 고정) | 짧은 제목에서 연산/메모리 낭비 |
| T4 | early stopping / best checkpoint 없음 | 과적합 구간을 그대로 사용 |
| T5 | `eval_strategy="no"` | 학습 중 품질 확인 불가 |
| T6 | `save_model`이 LoRA 어댑터만 저장 | 재로드 경로 불명확, 배포 시 PEFT 필요 |
| T7 | 라벨 맵을 저장하지 않음 | 라벨 순서 변경 시 조용한 오분류 |
| T8 | `macro_f1`/클래스별 지표 없음 | 불균형 데이터에서 accuracy 착시 |
| T9 | 클래스 가중치/샘플링 없음 | 소수 클래스 소외 |
| T10 | base model이 영어 중심 ModernBERT | 한국어 형태소/조사 처리에 불리 |
| T11 | 신뢰도 보정/abstain 없음 | 낮은 확신의 예측도 단정적으로 출력 |
| T12 | `transformers` 5.x에서 `warmup_ratio` 제거 | 버전 업 시 학습 인자 오류 |

### 1.4 운영 / 배포

| ID | 문제 | 영향 |
| --- | --- | --- |
| O1 | ONNX/양자화 미구현 | CPU 지연 목표 달성 불가 |
| O2 | 배치 추론 API 없음 | 대량 평가/서비스 비효율 |
| O3 | 모델/설정 버전 관리 없음 | 실험 재현·롤백 어려움 |
| O4 | 로깅/에러 메시지 빈약 | 운영 디버깅 어려움 |
| O5 | 의존성 버전 범위 없음 | 환경마다 다른 동작 |

---

## 2. 이번 작업에서 반영한 개선

| ID | 개선 | 구현 위치 | 검증 |
| --- | --- | --- | --- |
| S1 | 데이터를 JSONL로 분리 + 재생성 스크립트 | `data/seed_news.jsonl`, `scripts/extract_legacy_dataset.py` | `pytest tests/test_data.py` |
| S2/S3 | 라벨 단일 소스(7개) + 모델/UI/CLI가 공유 | `config.LABELS` | `test_seed_dataset_is_valid` |
| S4 | 프롬프트 빌더 단일화 + 질문 통일 | `data.build_input_text`, `prepare_data` | `test_build_input_text_uses_shared_prompt` |
| S5 | 추론 서비스 하나로 통합 | `predict.NewsClassifier` | `tests/test_predict.py` (7건) |
| S6 | "ONNX" 오명 제거, 실제 동작과 이름 일치 | `predict.py` | 문서화 |
| S7 | 전역 상태 제거, 명시적 객체/함수 인자 | 전체 모듈 | CLI/테스트 |
| S8 | CLI + 선택적 UI 분리 | `cli.py`, `ui.py` | `tests/test_cli.py` |
| S9 | pytest 36건, `pyproject.toml` 설정 | `tests/` | `pytest` |
| D1 | 로드 시 중복 제거 + 리포트 | `data.deduplicate` | 94건 제거 확인 |
| D2 | 학습/평가 시 canonical question 강제 | `train.prepare_data`, `evaluate.evaluate_model` | 데이터 통계 |
| D3 | 불균형 비율 리포트 | `data.describe`, `data-stats` | 2.29 출력 |
| T1 | stratified train/val/test split | `data.split_examples` | `test_split_is_deterministic_and_stratified` |
| T2 | seed 고정 (`random`, `torch`, `TrainingArguments`) | `train.set_seed` | split 결정성 테스트 |
| T3 | dynamic padding collator | `train.PadCollator` | `test_pad_collator_*` |
| T4/T5 | epoch 평가 + macro-F1 기준 best + early stopping | `train.run_training` | 코드 리뷰 |
| T6 | LoRA merge 후 standalone 저장 | `modeling.merge_adapter`, `save_bundle` | 코드 리뷰 |
| T7 | `label_map.json` 동봉 | `modeling.save_bundle` | `read_labels` |
| T8 | accuracy/macro-F1/per-class/confusion matrix | `metrics.py` | `tests/test_metrics.py` (11건) |
| T9 | (부분) macro-F1 기준 선택 | `TrainConfig.metric_for_best_model` | — |
| T11 | `threshold` 기반 abstain | `predict.NewsClassifier` | `test_predict_top_k_and_threshold` |
| T12 | `warmup_ratio` -> `warmup_steps` 계산 | `train.run_training` | 코드 리뷰 |
| O2 | 배치 추론 | `predict_batch`, `evaluate_model` | `test_predict_batch_*` |
| O4 | 행 번호 포함 검증 에러, CLI 에러 코드 | `data.load_examples`, `cli.main` | `test_load_examples_*` |
| O5 | 의존성 하한/선택 extras | `requirements.txt`, `pyproject.toml` | — |

---

## 3. 남은 개선 아이디어

우선순위: **P0 = 성능/신뢰성에 직접 영향**, **P1 = 운영/실험 효율**, **P2 = 확장**.

### P0 — 먼저 해야 할 것

#### P0-1. 한국어 특화 base model 비교

- 현재 `answerdotai/ModernBERT-base`는 영어 코퍼스 중심입니다. 한국어 뉴스 제목에는 형태소·조사·줄임말이 많아 vocabulary mismatch가 큽니다.
- 후보:
  - `klue/roberta-base` (한국어 뉴스/문장 유사도에서 검증된 baseline)
  - `monologg/koelectra-base-v3-discriminator`
  - `snunlp/KR-ELECTRA-discriminator`
  - `google/multilingual-e5-base` (다국어, 문장 임베딩)
- 방법: `--model-id`만 바꿔 동일 split/seed로 비교. `all-linear` LoRA라 아키텍처 변경이 쉽습니다.
- 기대: 동일 데이터에서 macro-F1 +3~8%p (가설, 실측 필요).
- 검증: `evaluate`로 test macro-F1, 클래스별 F1, CPU 지연 비교.

#### P0-2. 증강 문장 그룹 split (leakage 제거)

- 현재 split은 행 단위입니다. `extended_*` 증강은 원문을 거의 그대로 복제하므로, 원문과 변형이 train/test에 나뉠 수 있습니다.
- 방법:
  1. 각 행에 `source_id`(원문 해시)를 부여.
  2. `GroupShuffleSplit`처럼 `source_id` 단위로 split.
  3. 또는 near-duplicate(예: char n-gram Jaccard ≥ 0.8) 클러스터를 만들어 클러스터 단위 split.
- 기대: 현재 test 점수가 하락하고, 그 하락폭이 **실제 일반화 갭**입니다. 이후 개선의 기준선이 됩니다.
- 구현 위치: `data.split_examples`에 `group_key` 콜러블 추가.

#### P0-3. 라벨 가이드라인과 재라벨링

- D5의 경계 모호성은 모델로 해결할 수 없습니다. 특히:
  - 정치 vs 세계: 외교·안보 이슈
  - 경제 vs IT: 반도체/플랫폼
  - 사회 vs 생활/문화: 사건·트렌드
- 방법: 라벨 정의서(포함/제외 예시 각 5개), 2인 이상 교차 라벨링, Cohen's kappa 측정.
- 기대: 상한 성능 자체가 올라가고, 오류 분석이 가능해집니다.

#### P0-4. 클래스 불균형 처리

- 현재 2.29:1. `정치/사회`가 다수, `IT/날씨`가 소수.
- 방법(비교 실험):
  1. `CrossEntropyLoss(weight=...)` — `Trainer`에 custom loss 주입.
  2. WeightedRandomSampler.
  3. Focal loss (γ=1~2).
- 평가: macro-F1과 소수 클래스 recall 중심. accuracy는 보조 지표로만.
- 주의: test split은 원본 분포를 유지해야 합니다(리샘플링 금지).

#### P0-5. 신뢰도 보정 + abstain 임계값 선택

- 현재 threshold는 사용자가 임의로 지정합니다.
- 방법: validation에서 temperature scaling 적합 -> ECE 측정 -> 목표 coverage(예: 90%)에 맞는 threshold 선택.
- 기대: "모르면 보류"가 가능해져 운영 리스크 감소.
- 구현: `predict`에 temperature 파라미터, `evaluate`에 ECE/coverage 리포트 추가.

### P1 — 실험/운영 효율

| ID | 아이디어 | 방법 | 기대 효과 |
| --- | --- | --- | --- |
| P1-1 | Hard-negative mining | confusion matrix 상위 쌍(정치↔세계, 경제↔IT)에서 오분류 제목을 수집해 재학습 | 경계 케이스 F1 개선 |
| P1-2 | 실제 ONNX export + int8 | `optimum.onnxruntime.ORTModelForSequenceClassification.from_pretrained(export=True)` -> dynamic quantization | CPU p95 지연 2~4x 개선 (측정 필요) |
| P1-3 | 실험 추적 | `TrainConfig`를 YAML로 저장, MLflow/W&B 연동 | 실험 재현/비교 |
| P1-4 | CI | GitHub Actions: `pytest` + `ruff` + `mypy` (모델 없이) | 회귀 방지 |
| P1-5 | 데이터 버전 관리 | DVC 또는 JSONL 해시를 `metrics.json`에 기록 | 데이터-모델 연결 |
| P1-6 | 교차검증 | 5-fold stratified CV (seed 고정) | 분산 있는 성능 추정 |
| P1-7 | 길이/지연 실험 | `max_length` 64 vs 128, batch size 1/8/32 | CPU 서비스 지연 최적화 |
| P1-8 | LoRA 용량 실험 | r=8/16/32, alpha 비율, dropout | 과적합 vs 표현력 |
| P1-9 | 오류 분석 자동화 | `evaluate`의 errors를 카테고리별로 집계, HTML 리포트 | 개선 루프 단축 |
| P1-10 | 입력 정규화 | HTML 태그/제로폭 문자/이모지 제거, 200자 제한 | 실입력 견고성 |

### P2 — 확장

- **멀티라벨/계층 라벨**: "정치+세계", "경제+IT" 동시 분류. `problem_type="multi_label_classification"`.
- **본문 활용**: 제목만으로 부족한 경우 lead paragraph 1~2문장 추가, `max_length=256`.
- **LLM teacher distillation**: 대형 모델의 확률 분포를 soft label로 증류 (노트북의 원래 아이디어를 제대로 구현).
- **능동 학습**: 낮은 confidence + 높은 entropy 샘플을 우선 라벨링.
- **도메인 적응**: 실제 뉴스 수집 후 continued pretraining (MLM) -> 분류 fine-tuning.
- **서빙**: FastAPI + batching + health/metrics endpoint, drift 모니터링.
- **캐시**: 동일 제목 빈도가 높은 서비스라면 LRU 캐시로 지연/비용 절감.

---

## 4. 실험 계획 (제안)

모든 실험은 동일 split/seed를 사용하고, test는 마지막에 한 번만 사용합니다.

| 실험 | 변경 | 주 지표 | 보조 지표 |
| --- | --- | --- | --- |
| E0 | 현재 코드 baseline (ModernBERT-base) | macro-F1 | accuracy, per-class F1, p95 latency |
| E1 | `klue/roberta-base` | macro-F1 | 위와 동일 |
| E2 | `monologg/koelectra-base-v3` | macro-F1 | 위와 동일 |
| E3 | E1 + class weight | macro-F1, 소수 클래스 recall | accuracy |
| E4 | E3 + group split | macro-F1 (낮아질 것) | leakage 갭 |
| E5 | E4 + hard negatives | macro-F1 | 상위 confusion 쌍 F1 |
| E6 | E5 + temperature scaling | ECE, coverage@90% | macro-F1 |
| E7 | E5 + ONNX int8 | p50/p95 latency | macro-F1 하락폭 |

**리포트 형식**: `metrics.json`에 `experiment_id`, `git_commit`, `data_hash`, `model_id`를 추가하면 비교가 쉬워집니다.

---

## 5. 리스크와 한계

1. **측정되지 않은 성능**: 이번 작업은 구조 개선이며, 실제 정확도 수치는 없습니다. README/문서의 어떤 숫자도 성능 주장으로 해석하면 안 됩니다.
2. **합성 데이터 편향**: 실제 뉴스 분포와 다를 수 있습니다. 배포 전 실데이터 평가 필수.
3. **라벨 경계**: 7개 카테고리는 상호 배타적이지 않습니다. 멀티라벨 요구가 있다면 라벨 체계부터 재설계해야 합니다.
4. **저작권/라이선스**: 데이터 출처가 불명확합니다. 외부 배포 전 확인 필요.
5. **Transformers 5.x 변화**: `warmup_ratio` 제거처럼 API가 바뀝니다. 버전 범위를 고정하거나 CI에서 최신/최소 버전을 모두 테스트하는 것을 권장합니다.
6. **CPU 추론 속도**: 문서의 "10~30ms" 목표는 검증되지 않았습니다. ONNX/양자화 없이는 달성 어려울 수 있습니다.

---

## 6. 작업 로그

| 커밋 | 내용 |
| --- | --- |
| `chore:` | 레거시 노트북 보관, 시드 데이터셋 추출, 프로젝트 메타데이터 |
| `refactor:` | `news_classifier` 패키지 구성 (config/data/metrics/modeling/train/predict/evaluate/cli/ui) |
| `test:` | 데이터/메트릭/추론/학습 유틸/평가/CLI 테스트 36건 |
| `docs:` | README, SW 구조, 개선 아이디어 문서화 |

> 커밋 해시는 `git log --oneline`으로 확인하세요.
