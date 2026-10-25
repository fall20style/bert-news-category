"""Central configuration: paths, label map, prompt template, run settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_DATA_PATH = DATA_DIR / "seed_news.jsonl"
DEFAULT_OUTPUT_DIR = ARTIFACTS_DIR / "news-classifier"

# The legacy notebook documented only six labels and forgot index 6 in its
# implementation notes; the dataset itself uses all seven. Keep a single source
# of truth so the model head, the prompt and the UI cannot drift apart.
LABELS: tuple[str, ...] = (
    "정치",
    "경제",
    "사회",
    "생활/문화",
    "세계",
    "IT/과학",
    "날씨",
)
LABEL2ID: dict[str, int] = {label: index for index, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = dict(enumerate(LABELS))

DEFAULT_QUESTION = "이 뉴스의 주요 카테고리는 무엇인가요?"
PROMPT_TEMPLATE = "뉴스 제목: {title}\n질문: {question}"

# Base model used by the legacy notebook. ModernBERT is English-centric; see
# docs/IMPROVEMENTS.md for Korean-specific alternatives and how to switch.
DEFAULT_MODEL_ID = "answerdotai/ModernBERT-base"


@dataclass(frozen=True)
class ModelConfig:
    """Architecture / tokenization settings."""

    model_id: str = DEFAULT_MODEL_ID
    max_length: int = 128
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_target_modules: str | list[str] = "all-linear"
    num_labels: int = len(LABELS)

    def __post_init__(self) -> None:
        if self.num_labels != len(LABELS):
            raise ValueError(
                f"num_labels({self.num_labels}) must match the label map "
                f"({len(LABELS)} categories)"
            )
        if self.max_length < 8:
            raise ValueError("max_length must be >= 8")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    """Training / evaluation settings.

    Defaults are close to the legacy notebook (LoRA r=32, lr=3e-4, 5 epochs)
    but add the pieces that were missing: a stratified validation/test split,
    early stopping on macro-F1 and a fixed seed.
    """

    output_dir: Path = DEFAULT_OUTPUT_DIR
    learning_rate: float = 3e-4
    num_epochs: float = 5.0
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    logging_steps: int = 20
    seed: int = 42
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    early_stopping_patience: int = 2
    metric_for_best_model: str = "macro_f1"

    def __post_init__(self) -> None:
        if not 0.0 <= self.val_ratio < 1.0:
            raise ValueError("val_ratio must be in [0, 1)")
        if not 0.0 <= self.test_ratio < 1.0:
            raise ValueError("test_ratio must be in [0, 1)")
        if self.val_ratio + self.test_ratio >= 1.0:
            raise ValueError("val_ratio + test_ratio must be < 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload


def label_metadata() -> dict[str, Any]:
    """Serialisable label map written next to every trained model."""

    return {
        "labels": list(LABELS),
        "label2id": dict(LABEL2ID),
        "id2label": {str(key): value for key, value in ID2LABEL.items()},
    }
