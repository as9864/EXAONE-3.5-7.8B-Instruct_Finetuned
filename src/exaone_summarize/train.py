"""EXAONE-3.5-7.8B-Instruct 문서 요약 LoRA / QLoRA 학습 진입점.

사용 예:
    python -m exaone_summarize.train -c configs/qlora_7.8b.yaml
    python -m exaone_summarize.train -c configs/qlora_7.8b.yaml --set train.num_train_epochs=1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
import transformers
from transformers import Trainer, TrainingArguments, set_seed

from .config import (
    Config,
    add_config_args,
    config_from_args,
    resolve_best_model_setting,
    resolve_eval_strategy,
    to_dict,
    validate,
)
from .data import DataCollatorForCausalSummarization, build_dataset
from .modeling import load_for_training

logger = logging.getLogger(__name__)

# src/exaone_summarize/train.py 맨 위쪽 import 들 바로 아래에 추가

import inspect
import sys
import transformers

print("=" * 60)
print(f"[CHECK 1] Executing script path: {__file__}")
print(f"[CHECK 2] Python executable: {sys.executable}")
print(f"[CHECK 3] Transformers version: {transformers.__version__}")
print(f"[CHECK 4] Transformers location: {transformers.__file__}")
print("=" * 60)


def setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    transformers.utils.logging.set_verbosity_info()


def build_training_args(cfg: Config) -> TrainingArguments:
    tc = cfg.train
    eval_strategy = resolve_eval_strategy(cfg)
    load_best = resolve_best_model_setting(tc, eval_strategy)

    if tc.bf16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise ValueError(
            "이 GPU는 bf16을 지원하지 않습니다. train.bf16=false, train.fp16=true 로 바꾸세요."
        )

    # 1. 전달하고자 하는 인자 전체 모음
    raw_args = {
        "output_dir": tc.output_dir,
        "seed": tc.seed,
        "num_train_epochs": tc.num_train_epochs,
        "max_steps": tc.max_steps,
        "per_device_train_batch_size": tc.per_device_train_batch_size,
        "per_device_eval_batch_size": tc.per_device_eval_batch_size,
        "gradient_accumulation_steps": tc.gradient_accumulation_steps,
        "learning_rate": tc.learning_rate,
        "weight_decay": tc.weight_decay,
        "warmup_ratio": tc.warmup_ratio,
        "lr_scheduler_type": tc.lr_scheduler_type,
        "max_grad_norm": tc.max_grad_norm,
        "optim": tc.optim,
        "bf16": tc.bf16,
        "fp16": tc.fp16,
        "logging_steps": tc.logging_steps,
        "eval_strategy": eval_strategy,
        "evaluation_strategy": eval_strategy,
        "eval_steps": tc.eval_steps if eval_strategy == "steps" else None,
        "save_strategy": tc.save_strategy,
        "save_steps": tc.save_steps,
        "save_total_limit": tc.save_total_limit,
        "load_best_model_at_end": load_best,
        "metric_for_best_model": tc.metric_for_best_model,
        "greater_is_better": tc.greater_is_better,
        "report_to": tc.report_to or "none",
        "remove_unused_columns": False,
        # 16GB에서 7.8B를 돌리려면 필수. 끄면 활성값만 ~16GB라 sysmem으로 스필된다.
        "gradient_checkpointing": cfg.model.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "dataloader_pin_memory": torch.cuda.is_available(),
        "dataloader_num_workers": 0,
        # 길이 편차가 커서(100~3000토큰) batch>1이면 패딩 낭비가 크다.
        "group_by_length": tc.group_by_length,
        "length_column_name": "length",
        "label_names": ["labels"],
    }

    # 2. 현재 설치된 TrainingArguments가 실제로 받는 인자만 필터링 (warmup_ratio 자동 제거됨)
    valid_keys = inspect.signature(TrainingArguments.__init__).parameters.keys()
    filtered_args = {k: v for k, v in raw_args.items() if k in valid_keys}

    return TrainingArguments(**filtered_args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXAONE-3.5 요약 LoRA 학습")
    add_config_args(parser)
    args = parser.parse_args(argv)

    setup_logging()
    cfg = validate(config_from_args(args))
    set_seed(cfg.train.seed)

    if not torch.cuda.is_available():
        logger.warning("CUDA를 찾지 못했습니다. 7.8B 모델의 CPU 학습은 현실적으로 불가능합니다.")

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as fp:
        json.dump(to_dict(cfg), fp, ensure_ascii=False, indent=2)

    logger.info("모델 로딩: %s (4bit=%s)", cfg.model.model_name_or_path, cfg.model.load_in_4bit)
    model, tokenizer = load_for_training(cfg)

    train_dataset = build_dataset(
        cfg.data.train_file, tokenizer, cfg.data, cfg.data.max_train_samples, desc="train"
    )
    eval_dataset = None
    if cfg.data.eval_file:
        eval_dataset = build_dataset(
            cfg.data.eval_file, tokenizer, cfg.data, cfg.data.max_eval_samples, desc="eval"
        )

    trainer = Trainer(
        model=model,
        args=build_training_args(cfg),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForCausalSummarization(tokenizer),
    )

    resume = cfg.train.resume_from_checkpoint
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)

    if eval_dataset is not None:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    trainer.save_state()
    logger.info("LoRA 어댑터 저장 완료: %s", adapter_dir)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
