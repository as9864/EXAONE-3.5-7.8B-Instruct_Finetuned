"""학습한 LoRA 어댑터로 문서 요약 추론.

사용 예:
    # 단일 텍스트
    python -m exaone_summarize.infer -c configs/qlora_7.8b.yaml \
        --adapter outputs/exaone-3.5-7.8b-summary-qlora/adapter \
        --text "요약할 문서 본문..."

    # 파일 하나
    python -m exaone_summarize.infer --adapter ... --input-file article.txt

    # JSONL 배치 -> 예측 JSONL
    python -m exaone_summarize.infer --adapter ... \
        --input-jsonl data/processed/test.jsonl \
        --output-jsonl outputs/preds.jsonl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .config import Config, add_config_args, config_from_args
from .jsonl import read_jsonl, write_jsonl
from .modeling import load_for_inference
from .prompt import build_messages, truncate_document

logger = logging.getLogger(__name__)


def _generation_kwargs(cfg: Config, tokenizer: PreTrainedTokenizerBase) -> dict:
    gc = cfg.generation
    kwargs = {
        "max_new_tokens": gc.max_new_tokens,
        "do_sample": gc.do_sample,
        "repetition_penalty": gc.repetition_penalty,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if gc.do_sample:
        kwargs["temperature"] = gc.temperature
        kwargs["top_p"] = gc.top_p
    return kwargs


@torch.inference_mode()
def summarize_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    cfg: Config,
    documents: list[str],
) -> list[str]:
    """문서 리스트를 한 번에 요약한다 (left padding 전제)."""
    document_budget = cfg.data.max_seq_len - cfg.generation.max_new_tokens
    prompts = [
        tokenizer.apply_chat_template(
            build_messages(
                truncate_document(tokenizer, doc.strip(), max(document_budget, 128)),
                cfg.data.system_prompt,
                cfg.data.user_template,
            ),
            tokenize=False,
            add_generation_prompt=True,
        )
        for doc in documents
    ]

    batch = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    batch = {k: v.to(model.device) for k, v in batch.items()}

    outputs = model.generate(**batch, **_generation_kwargs(cfg, tokenizer))
    # 프롬프트 구간을 잘라내고 생성분만 디코드한다.
    generated = outputs[:, batch["input_ids"].shape[1] :]
    return [tokenizer.decode(seq, skip_special_tokens=True).strip() for seq in generated]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXAONE-3.5 요약 추론")
    add_config_args(parser)
    parser.add_argument("--adapter", default=None, help="LoRA 어댑터 경로 (없으면 베이스 모델)")
    parser.add_argument("--text", default=None, help="요약할 문서 문자열")
    parser.add_argument("--input-file", default=None, help="요약할 문서 텍스트 파일")
    parser.add_argument("--input-jsonl", default=None, help="배치 입력 JSONL")
    parser.add_argument("--output-jsonl", default=None, help="예측 결과 JSONL 경로")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="입력 JSONL 앞 N건만 처리")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = config_from_args(args)

    sources = [bool(args.text), bool(args.input_file), bool(args.input_jsonl)]
    if sum(sources) != 1:
        parser.error("--text / --input-file / --input-jsonl 중 정확히 하나를 지정하세요.")

    model, tokenizer = load_for_inference(cfg, args.adapter)

    if args.text or args.input_file:
        document = args.text or Path(args.input_file).read_text(encoding="utf-8")
        summary = summarize_batch(model, tokenizer, cfg, [document])[0]
        print("\n=== 요약 ===")
        print(summary)
        return 0

    rows = read_jsonl(args.input_jsonl)
    if args.limit:
        rows = rows[: args.limit]

    doc_key, sum_key = cfg.data.document_key, cfg.data.summary_key
    predictions: list[dict] = []
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start : start + args.batch_size]
        summaries = summarize_batch(model, tokenizer, cfg, [r[doc_key] for r in chunk])
        for row, pred in zip(chunk, summaries):
            predictions.append(
                {doc_key: row[doc_key], "reference": row.get(sum_key), "prediction": pred}
            )
        done = min(start + args.batch_size, len(rows))
        print(f"\r{done}/{len(rows)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    out_path = args.output_jsonl or "outputs/predictions.jsonl"
    write_jsonl(out_path, predictions)
    logger.info("예측 %d건 저장: %s", len(predictions), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
