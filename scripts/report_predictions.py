"""predictions.jsonl을 논문식 표로 정리한다.

`evaluate.py`가 내주지 않는 세 가지를 채운다.

1. **출처별 lead-N 베이스라인** — 전체 평균만 보면 도메인별로 서로 반대 방향인
   결과가 상쇄돼 "베이스라인과 차이 없음"으로 보인다.
2. **부트스트랩 신뢰구간** — 평가 표본이 수백 건이면 ±2~3점은 표본 오차다.
   베이스라인 대비 차이(Δ)가 0을 포함하는지 봐야 한다.
3. **추상성·붕괴 지표** — 신규 4-gram 비율(복사가 아닌지), 길이비, 빈 출력,
   문장 미완결, 반복 5-gram.

ROUGE는 `evaluate.py`와 같은 분절기를 쓰므로 값이 일치한다.

사용 예:
    python scripts\\report_predictions.py `
        --predictions outputs\\exaone-3.5-7.8b-summary-qlora\\predictions.jsonl --markdown
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exaone_summarize.evaluate import ROUGE_TYPES, get_tokenizer, lead_sentences  # noqa: E402
from exaone_summarize.jsonl import read_jsonl  # noqa: E402

_WORD_RE = re.compile(r"[가-힣]+|[a-zA-Z]+|[0-9]+")
_SENTENCE_END = tuple(".!?\"')”’")


def per_sample_scores(predictions, references, tokenizer_name):
    """샘플별 ROUGE F1. 부트스트랩을 위해 평균 전 값을 남긴다."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        list(ROUGE_TYPES), use_stemmer=False, tokenizer=get_tokenizer(tokenizer_name)
    )
    rows = []
    for prediction, reference in zip(predictions, references, strict=True):
        scored = scorer.score(reference, prediction)
        rows.append({t: scored[t].fmeasure * 100 for t in ROUGE_TYPES})
    return rows


def mean(values):
    return statistics.mean(values) if values else 0.0


def bootstrap_delta_ci(model_scores, baseline_scores, key, *, rounds, seed, alpha=0.05):
    """Δ(모델 − 베이스라인)의 백분위 부트스트랩 신뢰구간.

    같은 문서에 대한 두 점수를 짝지어 재표집한다(paired bootstrap).
    """
    paired = [m[key] - b[key] for m, b in zip(model_scores, baseline_scores, strict=True)]
    n = len(paired)
    if n < 2:
        return mean(paired), 0.0, 0.0

    rng = random.Random(seed)
    means = []
    for _ in range(rounds):
        means.append(mean([paired[rng.randrange(n)] for _ in range(n)]))
    means.sort()
    low = means[int(alpha / 2 * rounds)]
    high = means[min(int((1 - alpha / 2) * rounds), rounds - 1)]
    return mean(paired), low, high


def novel_ngram_ratio(summary: str, document: str, n: int) -> float:
    """요약의 n-gram 중 원문에 없는 비율 — 높을수록 추상적(복사가 아님)."""
    summary_tokens = _WORD_RE.findall(summary.lower())
    document_tokens = _WORD_RE.findall(document.lower())
    summary_grams = Counter(
        tuple(summary_tokens[i : i + n]) for i in range(len(summary_tokens) - n + 1)
    )
    document_grams = Counter(
        tuple(document_tokens[i : i + n]) for i in range(len(document_tokens) - n + 1)
    )
    total = sum(summary_grams.values())
    if total == 0:
        return 0.0
    return 1 - sum((summary_grams & document_grams).values()) / total


def repeats_ngram(text: str, n: int) -> bool:
    tokens = _WORD_RE.findall(text.lower())
    grams = Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return bool(grams) and max(grams.values()) > 1


def analyse(rows, args):
    predictions = [r["prediction"] for r in rows]
    references = [r["reference"] for r in rows]
    documents = [str(r.get("document") or "") for r in rows]
    baselines = [lead_sentences(d, args.lead) for d in documents]

    model = per_sample_scores(predictions, references, args.tokenizer)
    baseline = per_sample_scores(baselines, references, args.tokenizer)

    stats = {
        "n": len(rows),
        "model": {t: mean([s[t] for s in model]) for t in ROUGE_TYPES},
        "baseline": {t: mean([s[t] for s in baseline]) for t in ROUGE_TYPES},
        "delta": {},
        "pred_len": mean([len(p) for p in predictions]),
        "ref_len": mean([len(r) for r in references]),
        "pred_novelty": mean([novel_ngram_ratio(p, d, args.novelty_n)
                              for p, d in zip(predictions, documents, strict=True)]),
        "ref_novelty": mean([novel_ngram_ratio(r, d, args.novelty_n)
                             for r, d in zip(references, documents, strict=True)]),
        "empty": sum(1 for p in predictions if not p.strip()),
        "unfinished": sum(1 for p in predictions if not p.strip().endswith(_SENTENCE_END)),
        "repeated": sum(1 for p in predictions if repeats_ngram(p, args.repeat_n)),
    }
    for rouge_type in ROUGE_TYPES:
        stats["delta"][rouge_type] = bootstrap_delta_ci(
            model, baseline, rouge_type, rounds=args.bootstrap, seed=args.seed
        )
    return stats


def print_plain(groups, args):
    print(f"\n=== ROUGE ({args.tokenizer} 분절, F1 %) — lead-{args.lead} 베이스라인 대비 ===")
    header = (f"{'group':<18}{'n':>5}{'R-1':>8}{'R-2':>8}{'R-L':>8}"
              f"   {'lead R-1':>9}{'ΔR-1':>8}  95% CI")
    print(header)
    print("-" * len(header))
    for name, s in groups.items():
        low, high = s["delta"]["rouge1"][1], s["delta"]["rouge1"][2]
        print(
            f"{name:<18}{s['n']:>5}"
            f"{s['model']['rouge1']:>8.2f}{s['model']['rouge2']:>8.2f}{s['model']['rougeL']:>8.2f}"
            f"   {s['baseline']['rouge1']:>9.2f}{s['delta']['rouge1'][0]:>+8.2f}"
            f"  [{low:+.2f}, {high:+.2f}]"
        )

    print(f"\n=== 추상성 · 생성 안정성 (신규 {args.novelty_n}-gram 비율) ===")
    header = (f"{'group':<18}{'예측':>8}{'정답':>8}{'길이비':>9}"
              f"{'빈출력':>8}{'미완결':>8}{'반복':>7}")
    print(header)
    print("-" * len(header))
    for name, s in groups.items():
        print(
            f"{name:<18}{100 * s['pred_novelty']:>7.1f}%{100 * s['ref_novelty']:>7.1f}%"
            f"{s['pred_len'] / s['ref_len']:>9.2f}"
            f"{s['empty']:>8}{s['unfinished']:>8}{s['repeated']:>7}"
        )


def print_markdown(groups, args):
    print(f"\n#### ROUGE ({args.tokenizer} 분절, F1 %)\n")
    print("| 데이터 | n | R-1 | R-2 | R-L | lead-3 R-1 | ΔR-1 | 95% CI |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for name, s in groups.items():
        low, high = s["delta"]["rouge1"][1], s["delta"]["rouge1"][2]
        print(
            f"| {name} | {s['n']} | {s['model']['rouge1']:.2f} | {s['model']['rouge2']:.2f} "
            f"| {s['model']['rougeL']:.2f} | {s['baseline']['rouge1']:.2f} "
            f"| {s['delta']['rouge1'][0]:+.2f} | [{low:+.2f}, {high:+.2f}] |"
        )

    print(f"\n#### 추상성 · 생성 안정성 (신규 {args.novelty_n}-gram)\n")
    print("| 데이터 | 예측 | 정답 | 길이비 | 빈 출력 | 미완결 | 반복 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, s in groups.items():
        print(
            f"| {name} | {100 * s['pred_novelty']:.1f}% | {100 * s['ref_novelty']:.1f}% "
            f"| {s['pred_len'] / s['ref_len']:.2f} | {s['empty']} | {s['unfinished']} "
            f"| {s['repeated']} |"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="예측 파일 리포트 (출처별 · 신뢰구간)")
    parser.add_argument("--predictions", required=True, help="infer/evaluate가 만든 JSONL")
    parser.add_argument("--tokenizer", choices=["word", "char", "morph"], default="word")
    parser.add_argument("--lead", type=int, default=3, help="lead-N 베이스라인")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--novelty-n", type=int, default=4)
    parser.add_argument("--repeat-n", type=int, default=5)
    parser.add_argument("--markdown", action="store_true", help="마크다운 표로 출력")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args(argv)

    rows = read_jsonl(args.predictions)
    missing = [i for i, r in enumerate(rows) if not r.get("reference")]
    if missing:
        raise SystemExit(f"reference가 없는 레코드 {len(missing)}건 (예: {missing[:5]})")
    if not any(r.get("document") for r in rows):
        raise SystemExit("document 필드가 없어 lead 베이스라인을 계산할 수 없습니다.")

    groups = {"전체": analyse(rows, args)}
    by_source: dict[str, list] = {}
    for row in rows:
        if row.get("source"):
            by_source.setdefault(row["source"], []).append(row)
    for name in sorted(by_source):
        groups[name] = analyse(by_source[name], args)

    (print_markdown if args.markdown else print_plain)(groups, args)

    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n저장: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
