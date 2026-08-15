"""요약 학습용 JSONL 데이터셋 준비.

출력 형식은 한 줄에 하나의 JSON 객체:
    {"document": "...본문...", "summary": "...정답 요약..."}

사용 예:
    # 1) 번들 샘플을 data/processed 로 복사 (오프라인 스모크 테스트)
    python scripts/prepare_data.py --from-sample

    # 2) HuggingFace 한국어 뉴스 요약 데이터셋 (기본값)
    python scripts/prepare_data.py --hf-dataset daekeun-ml/naver-news-summarization-ko

    # 3) 내 로컬 파일 (jsonl / csv)
    python scripts/prepare_data.py --input-file mydata.jsonl \
        --document-column body --summary-column abstract
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exaone_summarize.jsonl import read_jsonl, write_jsonl  # noqa: E402

OUT_DIR = ROOT / "data" / "processed"
SAMPLE_DIR = ROOT / "data" / "sample"


def _normalize(
    rows: list[dict],
    document_column: str,
    summary_column: str,
    min_doc_chars: int,
    min_summary_chars: int,
    max_doc_chars: int | None,
) -> list[dict]:
    out, skipped = [], 0
    for row in rows:
        doc = str(row.get(document_column) or "").strip()
        summary = str(row.get(summary_column) or "").strip()
        if len(doc) < min_doc_chars or len(summary) < min_summary_chars:
            skipped += 1
            continue
        if len(summary) >= len(doc):
            # 요약이 본문보다 길면 라벨 오류일 가능성이 높다.
            skipped += 1
            continue
        if max_doc_chars:
            doc = doc[:max_doc_chars]
        out.append({"document": doc, "summary": summary})
    if skipped:
        print(f"  필터링으로 제외된 레코드: {skipped}건")
    return out


def _load_local(path: Path, document_column: str, summary_column: str) -> list[dict]:
    if path.suffix.lower() in {".jsonl", ".json"}:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else data.get("data", [])
        return read_jsonl(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as fp:
            return list(csv.DictReader(fp))
    raise SystemExit(f"지원하지 않는 확장자: {path.suffix} (.jsonl / .json / .csv)")


def _load_hf(name: str, config: str | None, document_column: str, summary_column: str):
    from datasets import load_dataset

    print(f"HuggingFace 데이터셋 로딩: {name}" + (f" ({config})" if config else ""))
    dsd = load_dataset(name, config) if config else load_dataset(name)
    print(f"  스플릿: { {k: len(v) for k, v in dsd.items()} }")

    for split in dsd:
        cols = dsd[split].column_names
        for col in (document_column, summary_column):
            if col not in cols:
                raise SystemExit(
                    f"'{split}' 스플릿에 '{col}' 컬럼이 없습니다. 존재하는 컬럼: {cols}\n"
                    "--document-column / --summary-column 로 지정하세요."
                )
        break
    return dsd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="요약 데이터셋 준비")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-sample", action="store_true", help="번들 샘플 사용")
    source.add_argument("--hf-dataset", default=None, help="HuggingFace 데이터셋 이름")
    source.add_argument("--input-file", default=None, help="로컬 jsonl/json/csv")

    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--document-column", default="document")
    parser.add_argument("--summary-column", default="summary")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--max-train", type=int, default=None, help="학습 데이터 최대 건수")
    parser.add_argument("--max-eval", type=int, default=1000)
    parser.add_argument("--val-ratio", type=float, default=0.05, help="스플릿이 없을 때 검증 비율")
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--min-doc-chars", type=int, default=100)
    parser.add_argument("--min-summary-chars", type=int, default=10)
    parser.add_argument("--max-doc-chars", type=int, default=None, help="본문 문자 상한(선택)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    rng = random.Random(args.seed)
    splits: dict[str, list[dict]] = {}

    if args.from_sample:
        for name in ("train", "validation"):
            src = SAMPLE_DIR / f"{name}.jsonl"
            splits[name] = read_jsonl(src)
        print(f"번들 샘플 사용: { {k: len(v) for k, v in splits.items()} }")

    elif args.hf_dataset:
        dsd = _load_hf(args.hf_dataset, args.hf_config, args.document_column, args.summary_column)
        alias = {"train": "train", "validation": "validation", "valid": "validation",
                 "dev": "validation", "test": "test"}
        for split_name, ds in dsd.items():
            target = alias.get(split_name, split_name)
            limit = args.max_train if target == "train" else args.max_eval
            if limit is not None and len(ds) > limit:
                ds = ds.shuffle(seed=args.seed).select(range(limit))
            splits.setdefault(target, []).extend(ds.to_list())

    else:
        rows = _load_local(Path(args.input_file), args.document_column, args.summary_column)
        print(f"로컬 파일 {len(rows)}건 로딩")
        rng.shuffle(rows)
        n = len(rows)
        n_val = max(1, int(n * args.val_ratio))
        n_test = max(1, int(n * args.test_ratio)) if args.test_ratio > 0 else 0
        splits["validation"] = rows[:n_val]
        if n_test:
            splits["test"] = rows[n_val : n_val + n_test]
        train_rows = rows[n_val + n_test :]
        if args.max_train:
            train_rows = train_rows[: args.max_train]
        splits["train"] = train_rows

    if "train" not in splits or not splits["train"]:
        raise SystemExit("train 스플릿을 만들 수 없습니다.")

    print("\n정규화 및 저장:")
    summary_report = {}
    for name, rows in splits.items():
        cleaned = _normalize(
            rows,
            args.document_column,
            args.summary_column,
            args.min_doc_chars,
            args.min_summary_chars,
            args.max_doc_chars,
        )
        if not cleaned:
            print(f"  {name}: 남은 레코드 0건 -> 건너뜀")
            continue
        path = out_dir / f"{name}.jsonl"
        write_jsonl(path, cleaned)
        doc_avg = sum(len(r["document"]) for r in cleaned) / len(cleaned)
        sum_avg = sum(len(r["summary"]) for r in cleaned) / len(cleaned)
        summary_report[name] = len(cleaned)
        print(f"  {name}: {len(cleaned)}건 -> {path}  (본문 평균 {doc_avg:.0f}자 / 요약 {sum_avg:.0f}자)")

    if "validation" not in summary_report:
        print("\n주의: validation.jsonl이 없습니다. 학습 설정에서 data.eval_file=null 로 두세요.")
    print("\n완료.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
