"""Command line interface.

Examples:
    news-classifier data-stats
    news-classifier train --dry-run
    news-classifier train --epochs 3 --output-dir artifacts/run-01
    news-classifier predict --model-dir artifacts/run-01 "한은, 기준금리 동결"
    news-classifier evaluate --model-dir artifacts/run-01 --output artifacts/run-01/eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR, LABELS, ModelConfig, TrainConfig


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", type=Path, required=True, help="saved checkpoint directory")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--max-length", type=int, default=128)


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="JSONL/JSON dataset")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="news-classifier",
        description="한국어 뉴스 제목 카테고리 분류기 (ModernBERT + LoRA)",
    )
    parser.add_argument("--verbose", action="store_true", help="show debug logs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- data-stats ---------------------------------------------------------
    stats = subparsers.add_parser(
        "data-stats", help="validate, de-duplicate and split the dataset"
    )
    _add_data_args(stats)
    stats.add_argument("--json", action="store_true", help="print JSON only")

    # --- train --------------------------------------------------------------
    train = subparsers.add_parser("train", help="fine-tune the classifier")
    _add_data_args(train)
    train.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    train.add_argument("--model-id", default=ModelConfig().model_id)
    train.add_argument("--epochs", type=float, default=TrainConfig().num_epochs)
    train.add_argument("--batch-size", type=int, default=TrainConfig().per_device_train_batch_size)
    train.add_argument("--eval-batch-size", type=int, default=TrainConfig().per_device_eval_batch_size)
    train.add_argument("--lr", type=float, default=TrainConfig().learning_rate)
    train.add_argument("--weight-decay", type=float, default=TrainConfig().weight_decay)
    train.add_argument("--warmup-ratio", type=float, default=TrainConfig().warmup_ratio)
    train.add_argument("--val-ratio", type=float, default=TrainConfig().val_ratio)
    train.add_argument("--test-ratio", type=float, default=TrainConfig().test_ratio)
    train.add_argument("--seed", type=int, default=TrainConfig().seed)
    train.add_argument("--max-length", type=int, default=ModelConfig().max_length)
    train.add_argument("--no-lora", action="store_true", help="full fine-tuning instead of LoRA")
    train.add_argument("--lora-r", type=int, default=ModelConfig().lora_r)
    train.add_argument("--lora-alpha", type=int, default=ModelConfig().lora_alpha)
    train.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="only validate/split the data; no model download or training",
    )

    # --- predict ------------------------------------------------------------
    predict = subparsers.add_parser("predict", help="classify one or more headlines")
    _add_model_args(predict)
    predict.add_argument("titles", nargs="*", help="headline(s) to classify")
    predict.add_argument("--top-k", type=int, default=None, help="show only the top-k categories")
    predict.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="abstain when the top probability is below this value (0-1)",
    )
    predict.add_argument("--interactive", action="store_true", help="read headlines from stdin")
    predict.add_argument("--json", action="store_true", help="machine-readable output")

    # --- evaluate -----------------------------------------------------------
    evaluate = subparsers.add_parser("evaluate", help="score a checkpoint on labelled data")
    _add_model_args(evaluate)
    _add_data_args(evaluate)
    evaluate.add_argument("--output", type=Path, default=None, help="write metrics JSON here")
    evaluate.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="do not de-duplicate the evaluation data",
    )

    # --- ui -----------------------------------------------------------------
    ui = subparsers.add_parser("ui", help="open the ipywidgets GUI (Jupyter only)")
    _add_model_args(ui)
    ui.add_argument(
        "--default-text",
        default='"얼굴 점·걸음걸이 똑같나?"…김정은 \'사망설 무색\'하게 보란 듯 등장한 모습 보니',
    )

    return parser


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _cmd_data_stats(args: argparse.Namespace) -> int:
    from .data import deduplicate, describe, load_examples, split_examples

    examples = load_examples(args.data)
    cleaned, report = deduplicate(examples)
    split = split_examples(cleaned)
    payload = {
        "source": str(args.data),
        "raw": len(examples),
        "dedup": report.to_dict(),
        "after_dedup": describe(cleaned),
        "split": split.to_dict(),
        "labels": list(LABELS),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"source            : {args.data}")
    print(f"rows              : {len(examples)} -> {len(cleaned)} (after cleaning)")
    print(f"exact duplicates  : {report.exact_duplicates}")
    print(f"label conflicts   : {report.dropped_conflicts}")
    print(f"imbalance ratio   : {payload['after_dedup']['imbalance_ratio']}")
    print(f"split             : {payload['split']}")
    print("class distribution:")
    for label, count in payload["after_dedup"]["class_counts"].items():
        share = payload["after_dedup"]["class_share"][label]
        print(f"  - {label:<8} {count:>4}  ({share:.1%})")
    if report.conflicting_titles:
        print("conflicting titles (dropped):")
        for title in report.conflicting_titles[:5]:
            print(f"  - {title}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    model_config = ModelConfig(
        model_id=args.model_id,
        max_length=args.max_length,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    train_config = TrainConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    if args.dry_run:
        from .train import run_data_report

        run_data_report(args.data)
        return 0

    from .train import run_training

    run_training(
        data_path=args.data,
        model_config=model_config,
        train_config=train_config,
        device=args.device,
    )
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    from .predict import NewsClassifier, format_prediction

    classifier = NewsClassifier.from_pretrained(
        args.model_dir, device=args.device, max_length=args.max_length
    )

    def render(title: str) -> None:
        prediction = classifier.predict(
            title, top_k=args.top_k, threshold=args.threshold
        )
        if args.json:
            print(json.dumps(prediction.to_dict(), ensure_ascii=False))
        else:
            print(format_prediction(prediction))

    titles = list(args.titles)
    if args.interactive or not titles:
        print("뉴스 제목을 입력하세요. 빈 줄 또는 Ctrl+C로 종료합니다.")
        try:
            while True:
                try:
                    line = input("제목> ").strip()
                except EOFError:
                    break
                if not line:
                    break
                render(line)
        except KeyboardInterrupt:
            print()
        return 0

    for title in titles:
        render(title)
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import evaluate_model, write_result

    result = evaluate_model(
        args.model_dir,
        data_path=args.data,
        device=args.device,
        max_length=args.max_length,
        drop_duplicates=not args.keep_duplicates,
    )
    print(f"examples : {result.total}")
    print(f"accuracy : {result.accuracy:.4f}")
    print(f"macro-F1 : {result.macro_f1:.4f}")
    print(result.report)
    print(result.confusion)
    if result.errors:
        print("\nmisclassified examples (up to 20):")
        for error in result.errors:
            print(
                f"  - expected={error['expected']:<8} predicted={error['predicted']:<8} "
                f"({error['confidence']:.2%}) {error['title']}"
            )
    if args.output:
        path = write_result(result, args.output)
        print(f"\nwrote {path}")
    return 0


def _cmd_ui(args: argparse.Namespace) -> int:
    from .predict import NewsClassifier
    from .ui import launch_ui

    classifier = NewsClassifier.from_pretrained(
        args.model_dir, device=args.device, max_length=args.max_length
    )
    launch_ui(classifier, default_text=args.default_text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    handlers = {
        "data-stats": _cmd_data_stats,
        "train": _cmd_train,
        "predict": _cmd_predict,
        "evaluate": _cmd_evaluate,
        "ui": _cmd_ui,
    }
    try:
        return handlers[args.command](args)
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
