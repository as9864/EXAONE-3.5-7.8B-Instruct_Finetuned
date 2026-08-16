"""여러 JSONL 데이터셋을 하나로 합친다(중복 제거 + 샘플링 + 셔플).

AI Hub 문서요약 데이터(scripts/prepare_aihub.py 출력)와 기존 HF 뉴스 요약
데이터를 섞어 하나의 학습 세트를 만들 때 쓴다.

입력 지정은 `경로` 또는 `경로:최대건수` 형태다. 최대건수를 주면 해당 파일을
먼저 셔플한 뒤 앞에서 N건만 취한다(앞부분만 잘라 편향되는 것을 피한다).

**근사 중복까지 제거한다.** 한국어 뉴스는 통신사 기사 재배포와 재게재가 많아
문자열 일치만으로는 걸러지지 않는다. 기본값(`--near-dup-threshold 0.5`)으로
어절 5-gram 유사도 0.5 이상이면 같은 문서로 보고 버린다. 이 판정은
`--exclude` 대조에도 그대로 적용되므로, 평가 세트와 사실상 같은 기사가
학습 세트에 남지 않는다.

사용 예:
    # 학습 세트: 기존 뉴스 전체 + AI Hub 도메인별 2만건씩
    python scripts/merge_datasets.py --output data/processed/train.jsonl \
        --input data/processed/naver_news/train.jsonl \
        --input data/processed/aihub/news_train.jsonl:20000 \
        --input data/processed/aihub/editorial_train.jsonl:20000 \
        --input data/processed/aihub/law_train.jsonl:20000 \
        --exclude data/processed/validation.jsonl \
        --exclude data/processed/test.jsonl

    # 검증 세트: 각 출처에서 조금씩
    python scripts/merge_datasets.py --output data/processed/validation.jsonl \
        --input data/processed/naver_news/validation.jsonl:300 \
        --input data/processed/aihub/news_valid.jsonl:300 \
        --input data/processed/aihub/editorial_valid.jsonl:200 \
        --input data/processed/aihub/law_valid.jsonl:200
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exaone_summarize.dedup import ShingleIndex, exact_key  # noqa: E402
from exaone_summarize.jsonl import read_jsonl, write_jsonl  # noqa: E402


def parse_spec(spec: str) -> tuple[Path, int | None]:
    """'path' 또는 'path:1000' 을 (경로, 상한)으로 분해한다.

    Windows 드라이브 문자(C:\\...)를 상한으로 오인하지 않도록 뒤쪽 조각이
    숫자일 때만 상한으로 본다.
    """
    head, sep, tail = spec.rpartition(":")
    if sep and tail.isdigit():
        return Path(head), int(tail)
    return Path(spec), None


def load_excluded(
    paths: list[Path], document_key: str, near_dup: bool
) -> tuple[set[str], ShingleIndex | None]:
    """제외 대상 본문의 완전 일치 키 집합과 (선택) 근사 중복 색인."""
    keys: set[str] = set()
    index = ShingleIndex() if near_dup else None
    for path in paths:
        if not path.exists():
            raise SystemExit(f"--exclude 파일이 없습니다: {path}")
        rows = read_jsonl(path)
        for row in rows:
            document = str(row.get(document_key) or "")
            keys.add(exact_key(document))
            if index is not None:
                index.add(document)
        print(f"  제외 대상 로딩: {path} ({len(rows):,}건)")
    return keys, index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="JSONL 데이터셋 병합")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="PATH[:N]",
        help="합칠 JSONL. ':N'을 붙이면 셔플 후 N건만 사용 (반복 지정 가능)",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="PATH",
        help="여기 들어 있는 본문은 결과에서 뺀다 (train/eval 누수 방지)",
    )
    parser.add_argument("--document-key", default="document")
    parser.add_argument("--summary-key", default="summary")
    parser.add_argument("--no-dedup", action="store_true", help="중복 제거 끄기")
    parser.add_argument(
        "--near-dup-threshold",
        type=float,
        default=0.5,
        help="어절 5-gram 유사도가 이 값 이상이면 중복으로 본다. 0이면 완전 일치만 (기본 0.5)",
    )
    parser.add_argument("--no-shuffle", action="store_true", help="셔플 끄기(입력 순서 유지)")
    parser.add_argument("--no-backup", action="store_true", help="기존 출력 파일 백업 안 함")
    parser.add_argument("--max-total", type=int, default=None, help="최종 건수 상한")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    specs = [parse_spec(s) for s in args.input]
    missing = [p for p, _ in specs if not p.exists()]
    if missing:
        raise SystemExit("입력 파일이 없습니다:\n  " + "\n  ".join(str(p) for p in missing))

    output = Path(args.output)
    near_dup = args.near_dup_threshold > 0 and not args.no_dedup
    excluded_keys, excluded_index = (
        load_excluded([Path(p) for p in args.exclude], args.document_key, near_dup)
        if args.exclude
        else (set(), None)
    )
    if near_dup:
        print(f"근사 중복 판정: 어절 5-gram 유사도 ≥ {args.near_dup_threshold:.2f}")

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    kept_index = ShingleIndex() if near_dup else None

    print("\n병합:")
    for path, limit in specs:
        rows = read_jsonl(path)
        rng = random.Random(args.seed)
        if limit is not None:
            rng.shuffle(rows)
            rows = rows[:limit]

        kept = dropped_empty = dropped_dup = dropped_near = 0
        dropped_excluded = dropped_excluded_near = 0
        for row in rows:
            document = str(row.get(args.document_key) or "").strip()
            summary = str(row.get(args.summary_key) or "").strip()
            if not document or not summary:
                dropped_empty += 1
                continue
            key = exact_key(document)
            if key in excluded_keys:
                dropped_excluded += 1
                continue
            if excluded_index is not None:
                score, _ = excluded_index.best_match(document)
                if score >= args.near_dup_threshold:
                    dropped_excluded_near += 1
                    continue
            if not args.no_dedup:
                if key in seen:
                    dropped_dup += 1
                    continue
                seen.add(key)
                if kept_index is not None:
                    score, _ = kept_index.best_match(document)
                    if score >= args.near_dup_threshold:
                        dropped_near += 1
                        continue
                    kept_index.add(document)
            merged.append(row)
            kept += 1

        detail = []
        if limit is not None:
            detail.append(f"상한 {limit:,}")
        if dropped_empty:
            detail.append(f"빈값 {dropped_empty:,}")
        if dropped_dup:
            detail.append(f"중복 {dropped_dup:,}")
        if dropped_near:
            detail.append(f"근사중복 {dropped_near:,}")
        if dropped_excluded:
            detail.append(f"제외목록 {dropped_excluded:,}")
        if dropped_excluded_near:
            detail.append(f"제외목록-근사 {dropped_excluded_near:,}")
        suffix = f"  ({', '.join(detail)})" if detail else ""
        print(f"  {path}: {kept:,}건 사용{suffix}")

    if not merged:
        raise SystemExit("병합 결과가 0건입니다.")

    if not args.no_shuffle:
        random.Random(args.seed).shuffle(merged)
    if args.max_total and len(merged) > args.max_total:
        print(f"  최종 상한 적용: {len(merged):,} -> {args.max_total:,}")
        merged = merged[: args.max_total]

    if output.exists() and not args.no_backup:
        backup = output.with_suffix(output.suffix + ".bak")
        shutil.copy2(output, backup)
        print(f"\n기존 파일 백업: {backup}")

    write_jsonl(output, merged)

    doc_avg = sum(len(str(r[args.document_key])) for r in merged) / len(merged)
    sum_avg = sum(len(str(r[args.summary_key])) for r in merged) / len(merged)
    print(f"\n{output}: {len(merged):,}건 (본문 평균 {doc_avg:.0f}자 / 요약 {sum_avg:.0f}자)")

    sources = Counter(str(r.get("source", "unknown")) for r in merged)
    print("출처 구성:")
    for name, count in sources.most_common():
        print(f"  {name}: {count:,}건 ({count / len(merged) * 100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
