"""학습 세트와 평가 세트 사이의 데이터 누수를 검사한다.

평가 점수가 의심스러울 만큼 높을 때 가장 먼저 확인할 항목이다. 세 층위로 본다.

1. 완전 일치  : 공백/대소문자 정규화 후 본문이 같음
2. 근사 중복  : 학습 문서 **한 건**과의 shingle overlap이 임계값 이상
                — 재게재·통신사 재배포·앞뒤만 잘린 판본을 잡는다
3. 요약 일치  : 본문은 달라도 정답 요약문이 같음

판정 로직은 `exaone_summarize.dedup`에 있다.

--predictions 를 주면 누수된 샘플과 깨끗한 샘플의 ROUGE를 따로 계산한다.
누수가 점수를 얼마나 밀어 올렸는지가 최종 판단 근거다.

사용 예:
    python scripts/check_leakage.py --train data/processed/train.jsonl \
        --eval data/processed/validation.jsonl --eval data/processed/test.jsonl

    python scripts/check_leakage.py --train data/processed/train.jsonl.bak \
        --eval data/processed/test.jsonl.bak \
        --predictions outputs/exaone-3.5-7.8b-summary-qlora/predictions.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exaone_summarize.dedup import SAMPLE_MOD, ShingleIndex, exact_key, normalize  # noqa: E402
from exaone_summarize.jsonl import read_jsonl  # noqa: E402


def rouge_of(pairs: list[tuple[str, str]], tokenizer: str) -> dict[str, float] | None:
    if not pairs:
        return None
    from exaone_summarize.evaluate import compute_rouge

    return compute_rouge([p for p, _ in pairs], [r for _, r in pairs], tokenizer)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="학습/평가 데이터 누수 검사")
    parser.add_argument("--train", required=True, help="학습에 쓴 JSONL")
    parser.add_argument("--eval", action="append", required=True, help="평가 JSONL (반복 가능)")
    parser.add_argument(
        "--predictions", default=None, help="infer/evaluate가 남긴 predictions.jsonl"
    )
    parser.add_argument("--document-key", default="document")
    parser.add_argument("--summary-key", default="summary")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="근사 중복 판정 임계값 (기본 0.5)"
    )
    parser.add_argument("--rouge-tokenizer", default="char", choices=("word", "char", "morph"))
    parser.add_argument("--show", type=int, default=3, help="누수 사례 출력 개수")
    args = parser.parse_args(argv)

    train_rows = read_jsonl(args.train)
    print(f"학습 세트: {args.train} ({len(train_rows):,}건)")

    train_documents = [str(row.get(args.document_key) or "") for row in train_rows]
    train_exact = {exact_key(d) for d in train_documents}
    train_summaries = {normalize(str(row.get(args.summary_key) or "")) for row in train_rows}
    index = ShingleIndex(train_documents)
    print(f"  shingle 색인 {len(index.postings):,}개 (1/{SAMPLE_MOD} 표본)")

    leaked_docs: dict[str, float] = {}  # 평가 본문 exact_key -> 최대 유사도
    total_flagged = 0

    for eval_path in args.eval:
        rows = read_jsonl(eval_path)
        exact_hits, near_hits, summary_hits = [], [], 0
        histogram: Counter[float] = Counter()
        for row in rows:
            document = str(row.get(args.document_key) or "")
            key = exact_key(document)
            score, match_id = index.best_match(document)
            histogram[min(round(score, 1), 1.0)] += 1
            leaked_docs[key] = max(leaked_docs.get(key, 0.0), score)
            if key in train_exact:
                exact_hits.append((row, 1.0, match_id))
            elif score >= args.threshold:
                near_hits.append((row, score, match_id))
            if normalize(str(row.get(args.summary_key) or "")) in train_summaries:
                summary_hits += 1

        flagged = len(exact_hits) + len(near_hits)
        total_flagged += flagged
        print(f"\n평가 세트: {eval_path} ({len(rows):,}건)")
        print(f"  완전 일치        : {len(exact_hits):,}건")
        print(f"  근사 중복(≥{args.threshold:.2f}) : {len(near_hits):,}건")
        print(f"  요약문 일치      : {summary_hits:,}건")
        print(f"  => 누수 의심 합계 : {flagged:,}건 ({flagged / len(rows) * 100:.2f}%)")
        bins = " ".join(f"{b:.1f}:{histogram[b]}" for b in sorted(histogram))
        print(f"  최대 유사도 분포 : {bins}")

        for row, score, match_id in (exact_hits + near_hits)[: args.show]:
            document = normalize(str(row.get(args.document_key) or ""))
            print(f"    [유사도 {score:.2f}] EVAL : {document[:86]}...")
            if match_id is not None:
                print(f"                  TRAIN: {normalize(train_documents[match_id])[:86]}...")

    if not args.predictions:
        if total_flagged == 0:
            print("\n누수 없음.")
        else:
            print("\n누수를 학습 세트에서 제거하세요:")
            print(
                "  python scripts/merge_datasets.py --output <train> --input ... "
                "--exclude <eval> --near-dup-threshold 0.5"
            )
        return 0

    # ---- 누수 샘플 / 깨끗한 샘플의 점수 비교
    preds = read_jsonl(args.predictions)
    leaked_pairs, clean_pairs = [], []
    for row in preds:
        document = str(row.get(args.document_key) or "")
        pair = (str(row.get("prediction") or ""), str(row.get("reference") or ""))
        score = leaked_docs.get(exact_key(document))
        if score is None:  # 평가 세트에 없던 문서
            score, _ = index.best_match(document)
        (leaked_pairs if score >= args.threshold else clean_pairs).append(pair)

    print(f"\n예측 파일: {args.predictions} ({len(preds):,}건)")
    print(f"  누수 {len(leaked_pairs):,}건 / 깨끗 {len(clean_pairs):,}건")

    rows_out = [
        ("전체", rouge_of(leaked_pairs + clean_pairs, args.rouge_tokenizer)),
        ("누수 샘플", rouge_of(leaked_pairs, args.rouge_tokenizer)),
        ("깨끗한 샘플", rouge_of(clean_pairs, args.rouge_tokenizer)),
    ]
    print(f"\nROUGE ({args.rouge_tokenizer} 분절)")
    print(f"  {'구분':<12} {'n':>5} {'R-1':>7} {'R-2':>7} {'R-L':>7}")
    for name, metrics in rows_out:
        if metrics is None:
            print(f"  {name:<12} {0:>5}       -       -       -")
            continue
        print(
            f"  {name:<12} {metrics['n_samples']:>5}"
            f" {metrics['rouge1']:>7.2f} {metrics['rouge2']:>7.2f} {metrics['rougeL']:>7.2f}"
        )

    both = [m for _, m in rows_out[1:] if m]
    if len(both) == 2:
        print(f"\n  누수 샘플이 R-1 기준 {both[0]['rouge1'] - both[1]['rouge1']:+.2f}점 높습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
