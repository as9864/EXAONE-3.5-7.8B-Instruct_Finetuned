"""ROUGE 계산 및 JSONL 유틸 검증."""

from __future__ import annotations

import pytest

from exaone_summarize.evaluate import CharTokenizer, WordTokenizer, compute_rouge
from exaone_summarize.jsonl import read_jsonl, write_jsonl


def test_word_tokenizer_splits_korean_and_latin():
    assert WordTokenizer().tokenize("한국은행이 GDP 3.2% 발표") == [
        "한국은행이",
        "gdp",
        "3",
        "2",
        "발표",
    ]


def test_char_tokenizer_drops_whitespace():
    assert CharTokenizer().tokenize("금리 인하") == ["금", "리", "인", "하"]


def test_identical_text_scores_100():
    text = "한국은행이 기준금리를 인하했다."
    metrics = compute_rouge([text], [text], "char")
    assert metrics["rouge1"] == pytest.approx(100.0)
    assert metrics["rouge2"] == pytest.approx(100.0)
    assert metrics["rougeL"] == pytest.approx(100.0)
    assert metrics["n_samples"] == 1


def test_disjoint_text_scores_zero():
    metrics = compute_rouge(["가나다라"], ["ABCD"], "char")
    assert metrics["rouge1"] == pytest.approx(0.0)


def test_partial_overlap_between_bounds():
    metrics = compute_rouge(
        ["한국은행이 기준금리를 동결했다"],
        ["한국은행이 기준금리를 인하했다"],
        "char",
    )
    assert 0 < metrics["rouge1"] < 100
    assert metrics["rougeL"] <= metrics["rouge1"]


def test_length_stats_are_reported():
    metrics = compute_rouge(["짧다"], ["조금 더 긴 참조 요약"], "word")
    assert metrics["pred_len_mean"] == 2
    assert metrics["ref_len_mean"] == len("조금 더 긴 참조 요약")


def test_jsonl_roundtrip_preserves_korean(tmp_path):
    rows = [{"document": "본문 가나다", "summary": "요약 라마바"}]
    path = tmp_path / "out.jsonl"
    write_jsonl(path, rows)

    assert "가나다" in path.read_text(encoding="utf-8")  # ensure_ascii=False 확인
    assert read_jsonl(path) == rows


def test_read_missing_file_raises_with_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_data.py"):
        read_jsonl(tmp_path / "nope.jsonl")


def test_sample_data_is_wellformed(sample_dir):
    for name, expected in (("train.jsonl", 8), ("validation.jsonl", 3)):
        rows = read_jsonl(sample_dir / name)
        assert len(rows) == expected
        for row in rows:
            assert set(row) == {"document", "summary"}
            assert len(row["summary"]) < len(row["document"])
            assert len(row["document"]) >= 100  # prepare_data 기본 필터 통과
