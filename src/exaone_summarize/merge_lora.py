"""LoRA 어댑터를 베이스 모델에 병합해 단일 모델로 저장한다.

vLLM / TGI 등으로 서빙할 때 유용하다.

중요: 병합은 4bit 양자화 상태에서 하면 품질이 크게 손상된다. 반드시
베이스 모델을 bf16(비양자화)으로 올려서 병합해야 하므로 CPU RAM 또는
VRAM이 ~16GB+ 필요하다. 기본값은 CPU에서 병합한다.

사용 예:
    python -m exaone_summarize.merge_lora \
        --adapter outputs/exaone-3.5-7.8b-summary-qlora/adapter \
        --output merged/exaone-3.5-7.8b-summary
"""

from __future__ import annotations

import argparse
import logging

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LoRA 어댑터 병합")
    parser.add_argument("--base", default="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--max-shard-size", default="4GB")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    logger.info("베이스 모델을 %s / %s 로 로딩 중 (양자화 없음)", args.device, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map={"": args.device},
        low_cpu_mem_usage=True,
    )

    logger.info("어댑터 적용: %s", args.adapter)
    model = PeftModel.from_pretrained(model, args.adapter, torch_dtype=dtype)
    model = model.merge_and_unload()
    model.config.use_cache = True

    logger.info("병합 모델 저장: %s", args.output)
    model.save_pretrained(args.output, max_shard_size=args.max_shard_size, safe_serialization=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tokenizer.save_pretrained(args.output)

    logger.info("완료.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
