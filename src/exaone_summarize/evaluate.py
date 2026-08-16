"""요약 결과 평가 (ROUGE-1/2/L).

infer.py가 만든 predictions.jsonl(prediction/reference)을 그대로 먹거나,
--adapter를 주면 추론+평가를 한 번에 수행한다.

한국어 ROUGE 주의: 기본 rouge-score 토크나이저는 영문 기준이라 한국어에서
점수가 왜곡된다. --tokenizer 로 분절 방식을 고르세요.
  word  : 정규식 어절/숫자/영문 단위 (기본, 의존성 없음)
  char  : 음절 단위 — 조사 변화에 관대하지만 **점수가 크게 부풀려진다**
  morph : konlpy 형태소 (konlpy + JDK 설치 필요, 가장 정확)

ROUGE 절대값은 그 자체로는 의미가 없다. 분절 방식과 데이터셋 성질에 따라
10점 이상 움직인다. 그래서 두 가지를 함께 낸다.

* **lead-N 베이스라인** — 본문 앞 N문장을 그대로 복사한 "요약". 학습된 모델이
  이 값을 크게 넘지 못하면 요약을 배운 게 아니라 복사를 배운 것이다.
  (실측: naver 뉴스 데이터는 정답 요약의 85%가 본문 복붙이라 lead-3만으로
  char ROUGE-1 63점이 나온다. 같은 세트에서 학습 모델은 67.7점이었다.)
* **출처별 분해** — 데이터에 `source` 필드가 있으면 출처별로 따로 낸다.
  섞인 데이터셋의 단일 평균은 도메인별 실력 차이를 가린다.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
from pathlib import Path

from .config import add_config_args, config_from_args
from .jsonl import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)

ROUGE_TYPES = ("rouge1", "rouge2", "rougeL")
_WORD_RE = re.compile(r"[가-힣]+|[a-zA-Z]+|[0-9]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def lead_sentences(document: str, n: int) -> str:
    """본문 앞 n문장 — 요약 베이스라인."""
    return " ".join(_SENT_SPLIT_RE.split(document.strip())[:n]).strip()


class WordTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return _WORD_RE.findall(text.lower())


class CharTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return [ch for ch in text.lower() if not ch.isspace()]


class MorphTokenizer:
    def __init__(self) -> None:
        try:
            from konlpy.tag import Okt
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "morph 토크나이저는 konlpy가 필요합니다: pip install konlpy (JDK 설치 필요)"
            ) from exc
        self._okt = Okt()

    def tokenize(self, text: str) -> list[str]:
        return self._okt.morphs(text)


def get_tokenizer(name: str):
    return {"word": WordTokenizer, "char": CharTokenizer, "morph": MorphTokenizer}[name]()


def compute_rouge(
    predictions: list[str],
    references: list[str],
    tokenizer_name: str = "word",
) -> dict[str, float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        list(ROUGE_TYPES), use_stemmer=False, tokenizer=get_tokenizer(tokenizer_name)
    )

    per_type: dict[str, list[float]] = {t: [] for t in ROUGE_TYPES}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for rouge_type in ROUGE_TYPES:
            per_type[rouge_type].append(scores[rouge_type].fmeasure)

    metrics = {t: statistics.mean(v) * 100 for t, v in per_type.items()}
    metrics["pred_len_mean"] = statistics.mean(len(p) for p in predictions)
    metrics["ref_len_mean"] = statistics.mean(len(r) for r in references)
    metrics["n_samples"] = len(predictions)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="요약 결과 ROUGE 평가")
    add_config_args(parser)
    parser.add_argument(
        "--predictions",
        default=None,
        help="prediction/reference가 담긴 JSONL (infer.py 출력)",
    )
    parser.add_argument("--input-jsonl", default=None, help="평가용 원본 JSONL (추론까지 수행)")
    parser.add_argument("--adapter", default=None, help="--input-jsonl과 함께 쓰는 LoRA 경로")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tokenizer", choices=["word", "char", "morph"], default="word")
    parser.add_argument(
        "--lead-baseline",
        type=int,
        default=3,
        help="본문 앞 N문장 복사 베이스라인을 함께 계산 (0이면 끔)",
    )
    parser.add_argument("--output-json", default=None, help="메트릭 저장 경로")
    parser.add_argument(
        "--save-predictions", default=None, help="추론 수행 시 예측 JSONL 저장 경로"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if bool(args.predictions) == bool(args.input_jsonl):
        parser.error("--predictions 또는 --input-jsonl 중 하나만 지정하세요.")

    if args.predictions:
        rows = read_jsonl(args.predictions)
        if args.limit:
            rows = rows[: args.limit]
        missing = [i for i, r in enumerate(rows) if not r.get("reference")]
        if missing:
            raise SystemExit(
                f"reference가 비어 있는 레코드 {len(missing)}건 (예: {missing[:5]}). "
                "평가에는 정답 요약이 필요합니다."
            )
        preds = [r["prediction"] for r in rows]
        refs = [r["reference"] for r in rows]
        docs = [str(r.get("document") or "") for r in rows]
        sources = [str(r.get("source") or "") for r in rows]
    else:
        # 지연 임포트: 순수 평가만 할 때 torch 로딩을 피한다.
        from .infer import summarize_batch
        from .modeling import load_for_inference

        cfg = config_from_args(args)
        rows = read_jsonl(args.input_jsonl)
        if args.limit:
            rows = rows[: args.limit]

        model, tokenizer = load_for_inference(cfg, args.adapter)
        doc_key, sum_key = cfg.data.document_key, cfg.data.summary_key

        preds, refs, docs, sources, records = [], [], [], [], []
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start : start + args.batch_size]
            outs = summarize_batch(model, tokenizer, cfg, [r[doc_key] for r in chunk])
            for row, pred in zip(chunk, outs):
                preds.append(pred)
                refs.append(row[sum_key])
                docs.append(str(row[doc_key]))
                sources.append(str(row.get("source") or ""))
                record = {doc_key: row[doc_key], "reference": row[sum_key], "prediction": pred}
                if row.get("source"):
                    # 출처별 분해를 나중에 다시 할 수 있도록 같이 남긴다.
                    record["source"] = row["source"]
                records.append(record)
            logger.info("추론 %d/%d", min(start + args.batch_size, len(rows)), len(rows))

        if args.save_predictions:
            write_jsonl(args.save_predictions, records)

    metrics = compute_rouge(preds, refs, args.tokenizer)
    metrics["rouge_tokenizer"] = args.tokenizer

    print("\n=== ROUGE (F1, %) ===")
    for key in ROUGE_TYPES:
        print(f"{key:>8}: {metrics[key]:6.2f}")
    print(f"\nsamples={metrics['n_samples']}  "
          f"pred_len={metrics['pred_len_mean']:.0f}자  ref_len={metrics['ref_len_mean']:.0f}자")

    if args.lead_baseline > 0 and any(docs):
        baseline = compute_rouge(
            [lead_sentences(d, args.lead_baseline) for d in docs], refs, args.tokenizer
        )
        metrics["lead_baseline"] = {f"lead{args.lead_baseline}": baseline}
        print(f"\n--- lead-{args.lead_baseline} 베이스라인 (본문 앞 문장 복사) ---")
        for key in ROUGE_TYPES:
            print(f"{key:>8}: {baseline[key]:6.2f}   (모델 대비 {metrics[key] - baseline[key]:+.2f})")
        if metrics["rouge1"] - baseline["rouge1"] < 3:
            print("  주의: 베이스라인과 차이가 거의 없습니다. 모델이 복사만 하고 있을 수 있습니다.")

    if any(sources):
        by_source: dict[str, dict[str, float]] = {}
        for name in sorted(set(sources)):
            picked = [i for i, s in enumerate(sources) if s == name]
            by_source[name or "unknown"] = compute_rouge(
                [preds[i] for i in picked], [refs[i] for i in picked], args.tokenizer
            )
        metrics["by_source"] = by_source
        print("\n--- 출처별 ---")
        print(f"  {'source':<18} {'n':>5} {'R-1':>7} {'R-2':>7} {'R-L':>7}")
        for name, scores in by_source.items():
            print(
                f"  {name:<18} {scores['n_samples']:>5}"
                f" {scores['rouge1']:>7.2f} {scores['rouge2']:>7.2f} {scores['rougeL']:>7.2f}"
            )

    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("메트릭 저장: %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
