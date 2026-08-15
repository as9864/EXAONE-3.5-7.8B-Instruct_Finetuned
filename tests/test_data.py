"""데이터 인코딩 / 마스킹 / 콜레이트 검증."""

from __future__ import annotations

import pytest

from exaone_summarize.config import DataConfig
from exaone_summarize.data import (
    IGNORE_INDEX,
    DataCollatorForCausalSummarization,
    SummarizationEncoder,
    build_dataset,
)
from exaone_summarize.prompt import truncate_document


def make_cfg(**kwargs) -> DataConfig:
    defaults = {"max_seq_len": 512, "max_target_tokens": 128}
    return DataConfig(**{**defaults, **kwargs})


def test_prompt_is_masked_and_target_is_not(tokenizer):
    encoder = SummarizationEncoder(tokenizer, make_cfg())
    out = encoder({"document": "본문 내용입니다." * 5, "summary": "요약문."})

    assert len(out["input_ids"]) == len(out["labels"]) == len(out["attention_mask"])
    assert set(out["attention_mask"]) == {1}

    n_masked = sum(1 for label in out["labels"] if label == IGNORE_INDEX)
    assert n_masked > 0, "프롬프트 구간이 마스킹되지 않았다"

    # 마스킹은 앞쪽에 연속으로만 존재해야 한다 (중간에 다시 나오면 안 됨).
    supervised = out["labels"][n_masked:]
    assert IGNORE_INDEX not in supervised

    # 지도 구간은 input_ids와 정확히 일치하고 eos로 끝난다.
    assert supervised == out["input_ids"][n_masked:]
    assert supervised[-1] == tokenizer.eos_token_id
    assert tokenizer.decode(supervised, skip_special_tokens=True) == "요약문."


def test_long_document_is_truncated_within_budget(tokenizer):
    cfg = make_cfg(max_seq_len=256, max_target_tokens=64)
    encoder = SummarizationEncoder(tokenizer, cfg)
    out = encoder({"document": "가" * 100_000, "summary": "짧은 요약"})

    assert len(out["input_ids"]) <= cfg.max_seq_len
    assert encoder.document_budget > 0
    # 예산이 다 소진될 만큼 긴 문서이므로 프롬프트가 예산 상한에 붙어 있어야 한다.
    n_masked = sum(1 for label in out["labels"] if label == IGNORE_INDEX)
    assert n_masked == encoder.overhead + encoder.document_budget


def test_long_summary_is_truncated_but_keeps_eos(tokenizer):
    cfg = make_cfg(max_seq_len=1024, max_target_tokens=32)
    encoder = SummarizationEncoder(tokenizer, cfg)
    out = encoder({"document": "문서" * 50, "summary": "요" * 500})

    n_masked = sum(1 for label in out["labels"] if label == IGNORE_INDEX)
    target = out["labels"][n_masked:]
    assert len(target) == cfg.max_target_tokens
    assert target[-1] == tokenizer.eos_token_id


def test_max_seq_len_too_small_raises(tokenizer):
    with pytest.raises(ValueError, match="max_seq_len"):
        SummarizationEncoder(tokenizer, make_cfg(max_seq_len=16, max_target_tokens=128))


def test_truncate_document_is_noop_when_short(tokenizer):
    text = "짧은 문서"
    assert truncate_document(tokenizer, text, 1000) == text


def test_collator_pads_right_with_ignore_index(tokenizer):
    collator = DataCollatorForCausalSummarization(tokenizer, pad_to_multiple_of=8)
    features = [
        {"input_ids": [1, 2, 3], "labels": [IGNORE_INDEX, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [4, 5], "labels": [IGNORE_INDEX, 5], "attention_mask": [1, 1]},
    ]
    batch = collator(features)

    assert batch["input_ids"].shape == (2, 8)  # max_len 3 -> 8의 배수로 패딩
    assert batch["labels"].shape == batch["attention_mask"].shape == (2, 8)

    # 짧은 샘플의 패딩 구간
    assert batch["input_ids"][1, 2:].tolist() == [tokenizer.pad_token_id] * 6
    assert batch["labels"][1, 2:].tolist() == [IGNORE_INDEX] * 6
    assert batch["attention_mask"][1].tolist() == [1, 1, 0, 0, 0, 0, 0, 0]


def test_build_dataset_from_sample_files(tokenizer, sample_dir):
    cfg = make_cfg(max_seq_len=1024, max_target_tokens=256)
    dataset = build_dataset(sample_dir / "train.jsonl", tokenizer, cfg, desc="test")

    assert len(dataset) == 8
    assert set(dataset.column_names) == {"input_ids", "labels", "attention_mask"}
    for row in dataset:
        assert len(row["input_ids"]) == len(row["labels"])
        assert len(row["input_ids"]) <= cfg.max_seq_len
        assert any(label != IGNORE_INDEX for label in row["labels"])


def test_build_dataset_missing_key_raises(tokenizer, sample_dir, tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"text": "본문만 있음"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="키가 없는 레코드"):
        build_dataset(bad, tokenizer, make_cfg(), desc="bad")


def test_build_dataset_respects_max_samples(tokenizer, sample_dir):
    dataset = build_dataset(
        sample_dir / "train.jsonl", tokenizer, make_cfg(max_seq_len=1024), max_samples=3
    )
    assert len(dataset) == 3
