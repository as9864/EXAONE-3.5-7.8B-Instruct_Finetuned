"""Summarizer 고수준 API 검증 (모델 가중치 없이)."""

from __future__ import annotations

import pytest

from exaone_summarize.api import (
    DEFAULT_ADAPTER_PATH,
    DEFAULT_CONFIG_PATH,
    MIN_DOCUMENT_BUDGET,
    Summarizer,
    merge_generation,
    resolve_repo_path,
)
from exaone_summarize.config import Config, GenerationConfig

# ------------------------------------------------------------------ 생성 옵션 병합


def test_merge_generation_keeps_base_when_none():
    base = GenerationConfig(max_new_tokens=256, do_sample=False)
    merged = merge_generation(base, max_new_tokens=None, temperature=None)
    assert merged == base


def test_merge_generation_applies_overrides():
    base = GenerationConfig(max_new_tokens=512, do_sample=False)
    merged = merge_generation(base, max_new_tokens=128, do_sample=True, temperature=0.3)
    assert merged.max_new_tokens == 128
    assert merged.do_sample is True
    assert merged.temperature == pytest.approx(0.3)
    # 지정하지 않은 항목은 그대로 남는다.
    assert merged.top_p == base.top_p


def test_merge_generation_rejects_unknown_option():
    with pytest.raises(ValueError, match="알 수 없는 생성 옵션"):
        merge_generation(GenerationConfig(), num_beams=4)


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_new_tokens": 0},
        {"temperature": 0},
        {"temperature": 99},
        {"top_p": 0},
        {"top_p": 1.5},
        {"repetition_penalty": 0},
    ],
)
def test_merge_generation_rejects_out_of_range(overrides):
    with pytest.raises(ValueError):
        merge_generation(GenerationConfig(), **overrides)


# ------------------------------------------------------------------ 경로 해석


def test_resolve_repo_path_joins_relative_to_root(tmp_path):
    assert resolve_repo_path("configs/x.yaml", tmp_path) == (tmp_path / "configs/x.yaml").resolve()


def test_resolve_repo_path_keeps_absolute(tmp_path):
    absolute = (tmp_path / "a.yaml").resolve()
    assert resolve_repo_path(absolute, tmp_path / "other") == absolute


def test_default_paths_exist_in_repo(project_root):
    """기본 경로가 저장소 구조와 어긋나면 다른 프로젝트에서 load()가 깨진다."""
    assert (project_root / DEFAULT_CONFIG_PATH).is_file()
    assert (project_root / DEFAULT_ADAPTER_PATH).is_dir()


# ------------------------------------------------------------------ 로딩 사전 검증


def test_load_reports_missing_config_before_loading_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="설정 파일을 찾을 수 없습니다"):
        Summarizer.load(config="configs/nope.yaml", repo_root=tmp_path)


def test_load_reports_missing_adapter_before_loading_model(tmp_path, project_root):
    """어댑터 경로 오류는 몇 분짜리 모델 로딩 *전에* 잡혀야 한다."""
    with pytest.raises(FileNotFoundError, match="LoRA 어댑터가 없습니다"):
        Summarizer.load(
            config=project_root / DEFAULT_CONFIG_PATH,
            adapter=tmp_path / "missing-adapter",
        )


# ------------------------------------------------------------------ 요약 흐름


class RecordingSummarizer(Summarizer):
    """generate 없이 호출 기록만 남기는 Summarizer."""

    def __init__(self, cfg, tokenizer):
        super().__init__(cfg, model=None, tokenizer=tokenizer)
        self.calls: list[list[str]] = []


@pytest.fixture
def summarizer(monkeypatch, tokenizer):
    cfg = Config()
    cfg.data.max_seq_len = 1536
    cfg.generation.max_new_tokens = 512
    instance = RecordingSummarizer(cfg, tokenizer)

    def fake_summarize_batch(model, tok, config, documents):
        instance.calls.append(list(documents))
        return [f"요약<{doc[:4]}>" for doc in documents]

    monkeypatch.setattr("exaone_summarize.infer.summarize_batch", fake_summarize_batch)
    return instance


def test_summarize_single_document(summarizer):
    assert summarizer.summarize("한국은행이 기준금리를 인하했다") == "요약<한국은행>"


def test_summarize_many_preserves_order(summarizer):
    documents = [f"문서{i}입니다" for i in range(5)]
    summaries = summarizer.summarize_many(documents, batch_size=2)
    assert summaries == [f"요약<문서{i}입>" for i in range(5)]
    # batch_size=2 -> 2+2+1
    assert [len(call) for call in summarizer.calls] == [2, 2, 1]


def test_summarize_detailed_flags_truncation(summarizer):
    """본문이 예산(max_seq_len - max_new_tokens)을 넘으면 잘렸다고 알려야 한다."""
    budget = summarizer.document_budget()
    assert budget == 1536 - 512

    short = summarizer.summarize_detailed(["가" * 100])[0]
    assert short.input_tokens == 100
    assert short.truncated is False

    long = summarizer.summarize_detailed(["나" * (budget + 50)])[0]
    assert long.input_tokens == budget + 50
    assert long.truncated is True
    assert long.document_budget == budget


def test_empty_list_returns_empty(summarizer):
    assert summarizer.summarize_detailed([]) == []
    assert summarizer.calls == []


def test_string_instead_of_list_is_rejected(summarizer):
    with pytest.raises(TypeError, match="문자열 리스트"):
        summarizer.summarize_many("문서 하나")


@pytest.mark.parametrize("documents", [["   "], ["정상", ""], ["정상", "\n\t"]])
def test_blank_documents_are_rejected(summarizer, documents):
    with pytest.raises(ValueError, match="비어 있습니다"):
        summarizer.summarize_many(documents)


def test_non_string_document_is_rejected(summarizer):
    with pytest.raises(TypeError, match=r"documents\[1\]"):
        summarizer.summarize_many(["정상", 123])


def test_documents_are_stripped_before_generation(summarizer):
    summarizer.summarize("  앞뒤 공백  ")
    assert summarizer.calls == [["앞뒤 공백"]]


def test_invalid_batch_size_is_rejected(summarizer):
    with pytest.raises(ValueError, match="batch_size"):
        summarizer.summarize_many(["문서"], batch_size=0)


def test_max_new_tokens_cannot_starve_the_document_budget(summarizer):
    """max_new_tokens를 max_seq_len 가까이 올리면 본문이 남지 않는다."""
    with pytest.raises(ValueError, match="본문 예산"):
        summarizer.summarize("문서", max_new_tokens=1536 - MIN_DOCUMENT_BUDGET + 1)


def test_per_request_override_does_not_mutate_shared_config(summarizer):
    summarizer.summarize("문서", max_new_tokens=64)
    assert summarizer.cfg.generation.max_new_tokens == 512
