"""요약 데이터 로딩 / 토크나이즈 / 콜레이트.

핵심은 **completion-only masking** 이다. 프롬프트(system + user + assistant 헤더)
토큰의 label을 -100으로 채워서, 요약문 구간에만 loss가 걸리도록 한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from .config import DataConfig
from .jsonl import read_jsonl, write_jsonl
from .prompt import build_messages, truncate_document

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

__all__ = [
    "IGNORE_INDEX",
    "DataCollatorForCausalSummarization",
    "SummarizationEncoder",
    "build_dataset",
    "read_jsonl",
    "write_jsonl",
]


def _template_overhead(tokenizer: PreTrainedTokenizerBase, cfg: DataConfig) -> int:
    """문서 본문을 뺀 chat template 자체가 차지하는 토큰 수."""
    rendered = tokenizer.apply_chat_template(
        build_messages("", cfg.system_prompt, cfg.user_template),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    return len(rendered)


class SummarizationEncoder:
    """(document, summary) -> input_ids / labels / attention_mask"""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, cfg: DataConfig) -> None:
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.overhead = _template_overhead(tokenizer, cfg)
        self.document_budget = cfg.max_seq_len - cfg.max_target_tokens - self.overhead
        if self.document_budget <= 0:
            raise ValueError(
                f"max_seq_len({cfg.max_seq_len})이 너무 작습니다. "
                f"template overhead={self.overhead}, max_target_tokens={cfg.max_target_tokens}"
            )
        logger.info(
            "토큰 예산: template=%d, document<=%d, target<=%d, total<=%d",
            self.overhead,
            self.document_budget,
            cfg.max_target_tokens,
            cfg.max_seq_len,
        )

    def __call__(self, example: dict[str, Any]) -> dict[str, list[int]]:
        document = (example[self.cfg.document_key] or "").strip()
        summary = (example[self.cfg.summary_key] or "").strip()

        document = truncate_document(self.tokenizer, document, self.document_budget)

        # //원본 변경 20260815
        # prompt_ids: list[int] = self.tokenizer.apply_chat_template(
        #     build_messages(document, self.cfg.system_prompt, self.cfg.user_template),
        #     tokenize=True,
        #     add_generation_prompt=True,
        #     return_dict=False,
        # )

        # target_ids: list[int] = self.tokenizer(summary, add_special_tokens=False)["input_ids"]
        # target_ids = target_ids[: self.cfg.max_target_tokens - 1]
        # target_ids = target_ids + [self.tokenizer.eos_token_id]


        # apply_chat_template 결과가 BatchEncoding 혹은 다른 형태로 반환될 수 있으므로 예외 방지 처리를 추가합니다.
        prompt_output = self.tokenizer.apply_chat_template(
            build_messages(document, self.cfg.system_prompt, self.cfg.user_template),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        
        # 만약 반환값이 객체 형태라면 .input_ids를 추출하고, 아니면 그대로 list로 변환
        if hasattr(prompt_output, "input_ids"):
            prompt_ids: list[int] = prompt_output["input_ids"]
        else:
            prompt_ids = list(prompt_output)

        target_ids: list[int] = self.tokenizer(summary, add_special_tokens=False)["input_ids"]
        target_ids = target_ids[: self.cfg.max_target_tokens - 1]
        target_ids = target_ids + [self.tokenizer.eos_token_id]

        input_ids = prompt_ids + target_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + list(target_ids)

        # 예산 계산이 어긋나는 예외적 경우에만 동작하는 안전망
        if len(input_ids) > self.cfg.max_seq_len:
            input_ids = input_ids[: self.cfg.max_seq_len]
            labels = labels[: self.cfg.max_seq_len]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
            # group_by_length가 쓰는 컬럼. 없으면 Trainer가 매 에폭 전체를 훑는다.
            "length": len(input_ids),
        }


def build_dataset(
    path: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    cfg: DataConfig,
    max_samples: int | None = None,
    *,
    desc: str = "dataset",
) -> Dataset:
    rows = read_jsonl(path)

    missing = [
        i
        for i, row in enumerate(rows)
        if cfg.document_key not in row or cfg.summary_key not in row
    ]
    if missing:
        raise ValueError(
            f"{path}: '{cfg.document_key}' / '{cfg.summary_key}' 키가 없는 레코드가 "
            f"{len(missing)}건 있습니다 (예: index {missing[:5]})."
        )

    rows = [
        row
        for row in rows
        if (row[cfg.document_key] or "").strip() and (row[cfg.summary_key] or "").strip()
    ]
    if max_samples is not None:
        rows = rows[:max_samples]

    dataset = Dataset.from_list(rows)
    encoder = SummarizationEncoder(tokenizer, cfg)
    dataset = dataset.map(
        encoder,
        remove_columns=dataset.column_names,
        num_proc=cfg.num_proc if cfg.num_proc > 1 else None,
        desc=f"tokenizing {desc}",
    )

    lengths = [len(x) for x in dataset["input_ids"]]
    logger.info(
        "%s: %d건 / 토큰 길이 min=%d mean=%.0f max=%d",
        desc,
        len(dataset),
        min(lengths),
        sum(lengths) / len(lengths),
        max(lengths),
    )
    return dataset


@dataclass
class DataCollatorForCausalSummarization:
    """가변 길이 배치를 오른쪽 패딩하고 labels는 -100으로 채운다."""

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of > 1:
            remainder = max_len % self.pad_to_multiple_of
            if remainder:
                max_len += self.pad_to_multiple_of - remainder

        pad_id = self.tokenizer.pad_token_id
        input_ids, labels, attention_mask = [], [], []
        for f in features:
            n_pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * n_pad)
            labels.append(f["labels"] + [IGNORE_INDEX] * n_pad)
            attention_mask.append(f["attention_mask"] + [0] * n_pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
