"""AI Hub 변환 / 데이터셋 병합 스크립트 검증."""

from __future__ import annotations

import io
import json
import sys
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import merge_datasets  # noqa: E402
import prepare_aihub  # noqa: E402


def make_doc(doc_id: str, n_sentences: int = 6, abstractive: str | None = None) -> dict:
    sentences = [
        {"index": i, "sentence": f"{doc_id}번 문서의 {i}번째 문장입니다." * 3, "highlight_indices": ""}
        for i in range(n_sentences)
    ]
    return {
        "id": doc_id,
        "category": "정치",
        "title": f"{doc_id} 제목",
        "text": [[s] for s in sentences],
        "extractive": [0, 2],
        "abstractive": [
            abstractive if abstractive is not None else f"{doc_id}번 문서의 요약문입니다."
        ],
    }


def dump(documents: list[dict]) -> str:
    return json.dumps({"name": "테스트", "documents": documents}, ensure_ascii=False, indent="\t")


def default_args(**kwargs) -> Namespace:
    base = {
        "summary_type": "abstractive",
        "include_title": False,
        "min_doc_chars": 100,
        "min_summary_chars": 10,
        "max_doc_chars": None,
        "progress": 0,
    }
    return Namespace(**{**base, **kwargs})


# ---------------------------------------------------------------- 스트리밍 파서


def test_iter_documents_streams_every_record():
    payload = dump([make_doc(str(i)) for i in range(25)])
    docs = list(prepare_aihub.iter_documents(io.StringIO(payload)))
    assert [d["id"] for d in docs] == [str(i) for i in range(25)]


def test_iter_documents_survives_tiny_chunks(monkeypatch):
    """청크 경계가 객체 중간을 잘라도 복구해야 한다."""
    monkeypatch.setattr(prepare_aihub, "_CHUNK", 7)
    payload = dump([make_doc(str(i)) for i in range(5)])
    docs = list(prepare_aihub.iter_documents(io.StringIO(payload)))
    assert len(docs) == 5
    assert docs[-1]["abstractive"] == ["4번 문서의 요약문입니다."]


def test_iter_documents_rejects_unknown_shape():
    with pytest.raises(ValueError):
        list(prepare_aihub.iter_documents(io.StringIO('{"rows": []}')))


# ---------------------------------------------------------------- 레코드 변환


def test_to_record_flattens_text_and_takes_abstractive():
    record = prepare_aihub.to_record(make_doc("7"), "news", "abstractive", include_title=False)
    assert record["summary"] == "7번 문서의 요약문입니다."
    assert record["source"] == "aihub_news"
    assert record["id"] == "7"
    assert record["category"] == "정치"
    assert "\n" not in record["document"]
    assert record["document"].startswith("7번 문서의 0번째 문장입니다.")


def test_to_record_extractive_joins_selected_indices():
    record = prepare_aihub.to_record(make_doc("7"), "news", "extractive", include_title=False)
    assert record["summary"].startswith("7번 문서의 0번째 문장입니다.")
    assert "2번째 문장" in record["summary"]
    assert "1번째 문장" not in record["summary"]


def test_to_record_include_title_prepends_title():
    record = prepare_aihub.to_record(make_doc("7"), "news", "abstractive", include_title=True)
    assert record["document"].startswith("7 제목\n")


def test_to_record_returns_none_without_summary():
    doc = make_doc("7")
    doc["abstractive"] = []
    assert prepare_aihub.to_record(doc, "news", "abstractive", include_title=False) is None


def test_convert_file_filters_and_dedups(tmp_path: Path):
    documents = [
        make_doc("1"),
        make_doc("1"),  # 본문이 같은 중복
        make_doc("2", n_sentences=1),  # min_doc_chars 미달
        make_doc("3", abstractive="짧음"),  # min_summary_chars 미달
        make_doc("4"),
    ]
    documents[2]["text"] = [[{"index": 0, "sentence": "짧은 본문.", "highlight_indices": ""}]]
    path = tmp_path / "신문기사_train_original.json"
    path.write_text(dump(documents), encoding="utf-8")

    rows = prepare_aihub.convert_file(path, "news", default_args())
    assert [r["id"] for r in rows] == ["1", "4"]


def test_convert_file_reads_zip(tmp_path: Path):
    path = tmp_path / "법률_valid_original.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("valid_original.json", dump([make_doc("9")]))

    rows = prepare_aihub.convert_file(path, "law", default_args())
    assert len(rows) == 1
    assert rows[0]["source"] == "aihub_law"


# ---------------------------------------------------------------- 파일 탐색


def test_discover_maps_domain_and_split(tmp_path: Path):
    (tmp_path / "Training").mkdir()
    (tmp_path / "Validation").mkdir()
    for name in ("법률_train_original.zip", "사설_train_original.zip", "신문기사_train_original.zip"):
        (tmp_path / "Training" / name).write_bytes(b"")
    (tmp_path / "Validation" / "신문기사_valid_original.zip").write_bytes(b"")
    (tmp_path / "Training" / "README.txt").write_text("무시", encoding="utf-8")

    found = {(p.name, domain, split) for p, domain, split in prepare_aihub.discover(tmp_path)}
    assert found == {
        ("법률_train_original.zip", "law", "train"),
        ("사설_train_original.zip", "editorial", "train"),
        ("신문기사_train_original.zip", "news", "train"),
        ("신문기사_valid_original.zip", "news", "valid"),
    }


# ---------------------------------------------------------------- 병합


def test_parse_spec_handles_limit_and_windows_path():
    assert merge_datasets.parse_spec("data/a.jsonl") == (Path("data/a.jsonl"), None)
    assert merge_datasets.parse_spec("data/a.jsonl:500") == (Path("data/a.jsonl"), 500)
    path, limit = merge_datasets.parse_spec(r"C:\data\a.jsonl")
    assert (path, limit) == (Path(r"C:\data\a.jsonl"), None)


def write_rows(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return path


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_merge_dedups_and_excludes(tmp_path: Path):
    a = write_rows(
        tmp_path / "a.jsonl",
        [
            {"document": "본문 A" * 20, "summary": "요약 A", "source": "x"},
            {"document": "본문 B" * 20, "summary": "요약 B", "source": "x"},
        ],
    )
    b = write_rows(
        tmp_path / "b.jsonl",
        [
            {"document": "본문 A" * 20, "summary": "다른 요약", "source": "y"},  # 중복
            {"document": "본문 C" * 20, "summary": "요약 C", "source": "y"},
            {"document": "본문 D" * 20, "summary": "", "source": "y"},  # 빈 요약
        ],
    )
    holdout = write_rows(
        tmp_path / "val.jsonl", [{"document": "본문 B" * 20, "summary": "요약 B"}]
    )
    out = tmp_path / "merged.jsonl"

    merge_datasets.main(
        [
            "--input", str(a),
            "--input", str(b),
            "--exclude", str(holdout),
            "--output", str(out),
        ]
    )

    rows = read_rows(out)
    summaries = {r["summary"] for r in rows}
    assert summaries == {"요약 A", "요약 C"}


LONG_ARTICLE = (
    "수협은행은 최고 연 3.1% 금리 혜택을 받을 수 있는 플러스알파예금과 최고 연 3.2% "
    "금리의 플러스알파적금 상품을 신규 출시한다고 4일 밝혔다. 개인고객 누구나 가입할 수 "
    "있는 플러스알파예금은 1인당 최대 5억원 한도로 가입할 수 있으며 만기는 12개월이다. "
    "적금은 매월 100만원까지 납입할 수 있다. 관계자는 금리 상승기에 맞춘 상품이라고 설명했다."
)
REPUBLISHED_ARTICLE = "서울 뉴시스 이주혜 기자 사진 제공 " + LONG_ARTICLE


def test_merge_drops_republished_article(tmp_path: Path):
    """문자열은 다르지만 사실상 같은 기사는 한 건만 남아야 한다."""
    a = write_rows(tmp_path / "a.jsonl", [{"document": LONG_ARTICLE, "summary": "예적금 출시"}])
    b = write_rows(
        tmp_path / "b.jsonl", [{"document": REPUBLISHED_ARTICLE, "summary": "예적금 출시 재게재"}]
    )
    out = tmp_path / "merged.jsonl"

    merge_datasets.main(["--input", str(a), "--input", str(b), "--output", str(out)])
    assert len(read_rows(out)) == 1

    # 임계값 0이면 완전 일치만 보므로 두 건 다 남는다.
    merge_datasets.main(
        [
            "--input", str(a),
            "--input", str(b),
            "--output", str(out),
            "--near-dup-threshold", "0",
        ]
    )
    assert len(read_rows(out)) == 2


def test_merge_exclude_catches_republished_article(tmp_path: Path):
    """평가 세트에 있는 기사의 재게재본은 학습 세트에서 빠져야 한다."""
    train = write_rows(
        tmp_path / "train.jsonl",
        [
            {"document": REPUBLISHED_ARTICLE, "summary": "재게재본"},
            {"document": "완전히 다른 주제의 긴 문서입니다. " * 12, "summary": "다른 문서"},
        ],
    )
    holdout = write_rows(tmp_path / "test.jsonl", [{"document": LONG_ARTICLE, "summary": "원본"}])
    out = tmp_path / "merged.jsonl"

    merge_datasets.main(
        ["--input", str(train), "--exclude", str(holdout), "--output", str(out)]
    )

    rows = read_rows(out)
    assert [r["summary"] for r in rows] == ["다른 문서"]


def test_merge_limit_and_backup(tmp_path: Path):
    src = write_rows(
        tmp_path / "a.jsonl",
        [{"document": f"본문 {i}" * 20, "summary": f"요약 {i}"} for i in range(10)],
    )
    out = write_rows(tmp_path / "merged.jsonl", [{"document": "기존" * 60, "summary": "기존 요약"}])

    merge_datasets.main(["--input", f"{src}:4", "--output", str(out)])

    assert len(read_rows(out)) == 4
    backup = out.with_suffix(out.suffix + ".bak")
    assert read_rows(backup)[0]["summary"] == "기존 요약"
