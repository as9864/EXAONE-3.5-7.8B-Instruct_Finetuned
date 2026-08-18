"""두 예측 파일(A/B)을 같은 문서끼리 짝지어 비교한다.

`report_predictions.py`는 한 실행을 lead-N 베이스라인과 비교한다. 이 스크립트는
**두 실행을 서로** 비교한다 — 주 용도는 파인튜닝 어댑터 vs 베이스 모델 zero-shot.

lead-N은 하한선일 뿐이므로, 어댑터가 실제로 무엇을 더했는지는 같은 프롬프트·같은
생성 설정으로 돌린 베이스 모델과 비교해야만 알 수 있다.

문서 문자열로 짝을 맞추므로 두 파일의 행 순서가 달라도 되고, 한쪽에만 있는 문서는
제외한다(제외 건수를 보고한다).

사용 예:
    python scripts\\compare_runs.py `
        --a outputs\\baseline-zeroshot\\predictions.jsonl --a-label base-zeroshot `
        --b outputs\\exaone-3.5-7.8b-summary-qlora_task2\\predictions.jsonl --b-label qlora `
        --markdown --output-json outputs\\comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # report_predictions 재사용

from exaone_summarize.evaluate import ROUGE_TYPES, lead_sentences  # noqa: E402
from exaone_summarize.jsonl import read_jsonl  # noqa: E402
from report_predictions import (  # noqa: E402
    _SENTENCE_END,
    bootstrap_delta_ci,
    mean,
    novel_ngram_ratio,
    per_sample_scores,
    repeats_ngram,
)


def _key(row: dict) -> str:
    """짝을 맞추는 키 — 본문 공백 정규화."""
    return " ".join(str(row.get("document") or "").split())


def load_paired(path_a: str, path_b: str) -> tuple[list[dict], list[dict], dict]:
    rows_a, rows_b = read_jsonl(path_a), read_jsonl(path_b)
    for name, rows in (("A", rows_a), ("B", rows_b)):
        missing = [i for i, r in enumerate(rows) if not r.get("reference")]
        if missing:
            raise SystemExit(f"{name}: reference 없는 레코드 {len(missing)}건 (예: {missing[:5]})")
        if not any(r.get("document") for r in rows):
            raise SystemExit(f"{name}: document 필드가 없어 짝을 맞출 수 없습니다.")

    index_b: dict[str, dict] = {}
    for row in rows_b:
        index_b.setdefault(_key(row), row)  # 중복 본문은 첫 등장만 사용

    paired_a, paired_b, unmatched = [], [], []
    used: set[str] = set()
    for row in rows_a:
        key = _key(row)
        if key in index_b and key not in used:
            used.add(key)
            paired_a.append(row)
            paired_b.append(index_b[key])
        else:
            unmatched.append(row)

    mismatched_refs = sum(
        1
        for a, b in zip(paired_a, paired_b, strict=True)
        if " ".join(a["reference"].split()) != " ".join(b["reference"].split())
    )
    info = {
        "n_a": len(rows_a),
        "n_b": len(rows_b),
        "n_paired": len(paired_a),
        "n_unmatched_a": len(unmatched),
        "n_mismatched_reference": mismatched_refs,
    }
    if not paired_a:
        raise SystemExit("두 파일에 공통 문서가 없습니다. 같은 평가 세트로 돌린 파일인지 확인하세요.")
    if mismatched_refs:
        raise SystemExit(
            f"같은 문서인데 정답 요약이 다른 레코드 {mismatched_refs}건 - 다른 데이터 버전입니다."
        )
    return paired_a, paired_b, info


def describe(rows: list[dict], scores: list[dict], args) -> dict:
    predictions = [r["prediction"] for r in rows]
    references = [r["reference"] for r in rows]
    documents = [str(r.get("document") or "") for r in rows]
    return {
        **{t: mean([s[t] for s in scores]) for t in ROUGE_TYPES},
        "pred_len": mean([len(p) for p in predictions]),
        "ref_len": mean([len(r) for r in references]),
        "novelty": mean(
            [
                novel_ngram_ratio(p, d, args.novelty_n)
                for p, d in zip(predictions, documents, strict=True)
            ]
        ),
        "empty": sum(1 for p in predictions if not p.strip()),
        "unfinished": sum(1 for p in predictions if not p.strip().endswith(_SENTENCE_END)),
        "repeated": sum(1 for p in predictions if repeats_ngram(p, args.repeat_n)),
    }


def analyse(rows_a: list[dict], rows_b: list[dict], args) -> dict:
    references = [r["reference"] for r in rows_a]
    documents = [str(r.get("document") or "") for r in rows_a]

    scores_a = per_sample_scores([r["prediction"] for r in rows_a], references, args.tokenizer)
    scores_b = per_sample_scores([r["prediction"] for r in rows_b], references, args.tokenizer)
    scores_lead = per_sample_scores(
        [lead_sentences(d, args.lead) for d in documents], references, args.tokenizer
    )

    stats = {
        "n": len(rows_a),
        "a": describe(rows_a, scores_a, args),
        "b": describe(rows_b, scores_b, args),
        "lead": {t: mean([s[t] for s in scores_lead]) for t in ROUGE_TYPES},
        "delta_b_minus_a": {},
        "delta_a_minus_lead": {},
        "delta_b_minus_lead": {},
        "b_wins": sum(1 for x, y in zip(scores_a, scores_b, strict=True)
                      if y["rouge1"] > x["rouge1"]),
    }
    for rouge_type in ROUGE_TYPES:
        stats["delta_b_minus_a"][rouge_type] = bootstrap_delta_ci(
            scores_b, scores_a, rouge_type, rounds=args.bootstrap, seed=args.seed
        )
    for rouge_type in ROUGE_TYPES:
        stats["delta_a_minus_lead"][rouge_type] = bootstrap_delta_ci(
            scores_a, scores_lead, rouge_type, rounds=args.bootstrap, seed=args.seed
        )
        stats["delta_b_minus_lead"][rouge_type] = bootstrap_delta_ci(
            scores_b, scores_lead, rouge_type, rounds=args.bootstrap, seed=args.seed
        )
    return stats


def verdict(delta: tuple[float, float, float]) -> str:
    _, low, high = delta
    if low > 0:
        return "B 유의하게 우세"
    if high < 0:
        return "A 유의하게 우세"
    return "차이 없음(CI가 0 포함)"


def print_plain(groups: dict, args) -> None:
    a, b = args.a_label, args.b_label
    # 콘솔이 cp949일 수 있어 em dash/minus sign 같은 문자는 쓰지 않는다.
    print(f"\n=== ROUGE-1 F1 ({args.tokenizer} 분절, %) : B({b}) - A({a}) ===")
    header = (f"{'group':<18}{'n':>5}{'A':>8}{'B':>8}{'lead':>8}"
              f"{'ΔB-A':>8}  {'95% CI':<18} 판정")
    print(header)
    print("-" * (len(header) + 8))
    for name, s in groups.items():
        d = s["delta_b_minus_a"]["rouge1"]
        print(
            f"{name:<18}{s['n']:>5}{s['a']['rouge1']:>8.2f}{s['b']['rouge1']:>8.2f}"
            f"{s['lead']['rouge1']:>8.2f}{d[0]:>+8.2f}  "
            f"{f'[{d[1]:+.2f}, {d[2]:+.2f}]':<18} {verdict(d)}"
        )

    print(f"\n=== R-2 / R-L 및 lead-{args.lead} 대비 ===")
    header = (f"{'group':<18}{'A R-2':>8}{'B R-2':>8}{'A R-L':>8}{'B R-L':>8}"
              f"{'A-lead':>9}{'B-lead':>9}")
    print(header)
    print("-" * len(header))
    for name, s in groups.items():
        print(
            f"{name:<18}{s['a']['rouge2']:>8.2f}{s['b']['rouge2']:>8.2f}"
            f"{s['a']['rougeL']:>8.2f}{s['b']['rougeL']:>8.2f}"
            f"{s['delta_a_minus_lead']['rouge1'][0]:>+9.2f}"
            f"{s['delta_b_minus_lead']['rouge1'][0]:>+9.2f}"
        )

    print(f"\n=== 길이 · 추상성 · 붕괴 (신규 {args.novelty_n}-gram, 반복 {args.repeat_n}-gram) ===")
    header = (f"{'group':<18}{'A 길이비':>10}{'B 길이비':>10}{'A 신규':>9}{'B 신규':>9}"
              f"{'A 반복':>8}{'B 반복':>8}{'A 미완결':>10}{'B 미완결':>10}")
    print(header)
    print("-" * len(header))
    for name, s in groups.items():
        print(
            f"{name:<18}{s['a']['pred_len'] / s['a']['ref_len']:>10.2f}"
            f"{s['b']['pred_len'] / s['b']['ref_len']:>10.2f}"
            f"{100 * s['a']['novelty']:>8.1f}%{100 * s['b']['novelty']:>8.1f}%"
            f"{s['a']['repeated']:>8}{s['b']['repeated']:>8}"
            f"{s['a']['unfinished']:>10}{s['b']['unfinished']:>10}"
        )


def print_markdown(groups: dict, args) -> None:
    a, b = args.a_label, args.b_label
    print(f"\n#### ROUGE-1 F1 ({args.tokenizer} 분절, %)\n")
    print(f"| 데이터 | n | {a} | {b} | lead-{args.lead} | Δ({b}-{a}) | 95% CI | 판정 |")
    print("|---|---:|---:|---:|---:|---:|---|---|")
    for name, s in groups.items():
        d = s["delta_b_minus_a"]["rouge1"]
        print(
            f"| {name} | {s['n']} | {s['a']['rouge1']:.2f} | {s['b']['rouge1']:.2f} "
            f"| {s['lead']['rouge1']:.2f} | {d[0]:+.2f} | [{d[1]:+.2f}, {d[2]:+.2f}] "
            f"| {verdict(d)} |"
        )

    print("\n#### R-2 / R-L\n")
    print(f"| 데이터 | {a} R-2 | {b} R-2 | {a} R-L | {b} R-L |")
    print("|---|---:|---:|---:|---:|")
    for name, s in groups.items():
        print(
            f"| {name} | {s['a']['rouge2']:.2f} | {s['b']['rouge2']:.2f} "
            f"| {s['a']['rougeL']:.2f} | {s['b']['rougeL']:.2f} |"
        )

    print(f"\n#### 길이 · 추상성 · 붕괴 (신규 {args.novelty_n}-gram)\n")
    print(f"| 데이터 | {a} 길이비 | {b} 길이비 | {a} 신규 | {b} 신규 "
          f"| {a} 반복 | {b} 반복 | {a} 미완결 | {b} 미완결 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, s in groups.items():
        print(
            f"| {name} | {s['a']['pred_len'] / s['a']['ref_len']:.2f} "
            f"| {s['b']['pred_len'] / s['b']['ref_len']:.2f} "
            f"| {100 * s['a']['novelty']:.1f}% | {100 * s['b']['novelty']:.1f}% "
            f"| {s['a']['repeated']} | {s['b']['repeated']} "
            f"| {s['a']['unfinished']} | {s['b']['unfinished']} |"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="두 예측 파일 비교 (paired bootstrap)")
    parser.add_argument("--a", required=True, help="기준 예측 JSONL (예: 베이스 zero-shot)")
    parser.add_argument("--b", required=True, help="비교 대상 예측 JSONL (예: 파인튜닝)")
    parser.add_argument("--a-label", default="A")
    parser.add_argument("--b-label", default="B")
    parser.add_argument("--tokenizer", choices=["word", "char", "morph"], default="word")
    parser.add_argument("--lead", type=int, default=3, help="참고용 lead-N 베이스라인")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--novelty-n", type=int, default=4)
    parser.add_argument("--repeat-n", type=int, default=5)
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--examples", type=int, default=0,
                        help="점수 차가 큰 샘플 N건을 양쪽 요약과 함께 출력")
    args = parser.parse_args(argv)

    rows_a, rows_b, info = load_paired(args.a, args.b)
    print(
        f"짝지은 문서 {info['n_paired']}건 "
        f"(A {info['n_a']}건 / B {info['n_b']}건, 미매칭 A {info['n_unmatched_a']}건)"
    )

    groups = {"전체": analyse(rows_a, rows_b, args)}
    by_source: dict[str, list[int]] = {}
    for i, row in enumerate(rows_a):
        if row.get("source"):
            by_source.setdefault(row["source"], []).append(i)
    for name in sorted(by_source):
        picked = by_source[name]
        groups[name] = analyse([rows_a[i] for i in picked], [rows_b[i] for i in picked], args)

    (print_markdown if args.markdown else print_plain)(groups, args)

    if args.examples:
        references = [r["reference"] for r in rows_a]
        scores_a = per_sample_scores([r["prediction"] for r in rows_a], references, args.tokenizer)
        scores_b = per_sample_scores([r["prediction"] for r in rows_b], references, args.tokenizer)
        diffs = sorted(
            range(len(rows_a)),
            key=lambda i: abs(scores_b[i]["rouge1"] - scores_a[i]["rouge1"]),
            reverse=True,
        )[: args.examples]
        print(f"\n=== 점수 차가 큰 샘플 {len(diffs)}건 ===")
        for i in diffs:
            gap = scores_b[i]["rouge1"] - scores_a[i]["rouge1"]
            print(f"\n--- [{rows_a[i].get('source', '?')}] ΔR-1 {gap:+.1f} ---")
            print(f"정답      : {rows_a[i]['reference'][:300]}")
            print(f"{args.a_label:<10}: {rows_a[i]['prediction'][:300]}")
            print(f"{args.b_label:<10}: {rows_b[i]['prediction'][:300]}")

    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "a": {"path": args.a, "label": args.a_label},
            "b": {"path": args.b, "label": args.b_label},
            "tokenizer": args.tokenizer,
            "pairing": info,
            "groups": groups,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n저장: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
