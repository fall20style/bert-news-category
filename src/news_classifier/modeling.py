"""Model / tokenizer construction, LoRA wrapping and artifact persistence."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .config import ID2LABEL, LABELS, LABEL2ID, ModelConfig, TrainConfig, label_metadata

logger = logging.getLogger(__name__)

LABEL_MAP_FILENAME = "label_map.json"
MODEL_CONFIG_FILENAME = "model_config.json"
TRAIN_CONFIG_FILENAME = "training_config.json"
METRICS_FILENAME = "metrics.json"


def resolve_device(prefer: str | None = None) -> str:
    """Return ``cuda`` / ``mps`` / ``cpu``, honouring an explicit request."""

    import torch

    if prefer and prefer != "auto":
        if prefer == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if prefer == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS was requested but is not available")
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        fallback = (
            tokenizer.eos_token_id
            if tokenizer.eos_token_id is not None
            else tokenizer.unk_token_id
        )
        if fallback is None:
            raise RuntimeError(
                f"tokenizer for {model_id!r} has no pad/eos/unk token; "
                "set one explicitly before training"
            )
        tokenizer.pad_token_id = fallback
    return tokenizer


def build_model(
    config: ModelConfig,
    *,
    tokenizer=None,
):
    """Build the classification head and optionally wrap it with LoRA.

    ``target_modules="all-linear"`` is kept from the legacy notebook because it
    avoids hard-coding ModernBERT's internal layer names (``Wqkv``/``Wo``/...),
    which differ across versions.  A warning is emitted if the classification
    head ends up frozen.
    """

    from transformers import AutoModelForSequenceClassification

    tokenizer = tokenizer or load_tokenizer(config.model_id)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.model_id,
        num_labels=config.num_labels,
        id2label=dict(ID2LABEL),
        label2id=dict(LABEL2ID),
        problem_type="single_label_classification",
    )
    if tokenizer.pad_token_id is not None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    model = base_model
    if config.use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "LoRA training requires `peft`. Install it with "
                "`pip install peft` or pass use_lora=False for full fine-tuning."
            ) from exc

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
        )
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()

    head_is_trainable = any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "classifier" in name or "score" in name
    )
    if not head_is_trainable:
        logger.warning(
            "The classification head is frozen; LoRA target modules may not "
            "include it. Check lora_target_modules."
        )
    return model, tokenizer


def merge_adapter(model):
    """Merge LoRA weights so the saved checkpoint loads without PEFT."""

    merged = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model
    merged.eval()
    return merged


def save_bundle(
    model,
    tokenizer,
    output_dir: str | Path,
    *,
    model_config: ModelConfig,
    train_config: TrainConfig,
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save a self-contained model directory plus training metadata."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    _write_json(output_dir / LABEL_MAP_FILENAME, label_metadata())
    _write_json(output_dir / MODEL_CONFIG_FILENAME, model_config.to_dict())
    _write_json(output_dir / TRAIN_CONFIG_FILENAME, train_config.to_dict())

    payload: dict[str, Any] = {
        "model_config": model_config.to_dict(),
        "train_config": train_config.to_dict(),
        "labels": list(LABELS),
    }
    if metrics:
        payload["metrics"] = metrics
    if extra:
        payload.update(extra)
    _write_json(output_dir / METRICS_FILENAME, payload)
    return output_dir


def load_trained_model(
    model_dir: str | Path,
    *,
    device: str | None = None,
):
    """Load a saved (already merged) checkpoint for inference."""

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"model directory not found: {model_dir}")

    labels = read_labels(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    target_device = resolve_device(device)
    model.to(target_device)
    model.eval()
    return model, tokenizer, labels


def read_labels(model_dir: str | Path) -> tuple[str, ...]:
    path = Path(model_dir) / LABEL_MAP_FILENAME
    if not path.exists():
        logger.warning("%s not found; falling back to the default label map", path)
        return LABELS
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = tuple(payload.get("labels", LABELS))
    if len(labels) != len(LABELS):
        raise ValueError(
            f"label map in {path} has {len(labels)} entries, expected {len(LABELS)}"
        )
    return labels


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
