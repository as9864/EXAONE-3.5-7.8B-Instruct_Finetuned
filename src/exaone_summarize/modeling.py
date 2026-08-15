"""EXAONE-3.5 모델/토크나이저 로딩과 LoRA 부착."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .config import Config, ModelConfig

logger = logging.getLogger(__name__)

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "auto": "auto",
}

# LoRA를 붙이면 안 되는 이름들 (출력 헤드, 임베딩)
_LORA_EXCLUDE = {"lm_head", "wte", "embed_tokens", "embed_out"}


def resolve_dtype(name: str):
    if name not in _DTYPES:
        raise ValueError(f"지원하지 않는 dtype: {name} (가능: {sorted(_DTYPES)})")
    return _DTYPES[name]


def load_tokenizer(cfg: ModelConfig) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
    )
    # EXAONE-3.5는 pad 토큰이 정의되어 있지만, 없을 경우를 대비한다.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # causal LM 학습은 right padding, 생성은 left padding이 맞다.
    tokenizer.padding_side = "right"
    return tokenizer


def _quantization_config(cfg: ModelConfig) -> BitsAndBytesConfig | None:
    if not cfg.load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=resolve_dtype(cfg.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )


def load_base_model(cfg: ModelConfig, *, for_training: bool = True) -> PreTrainedModel:
    quant_config = _quantization_config(cfg)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
        quantization_config=quant_config,
        torch_dtype=resolve_dtype(cfg.torch_dtype),
        attn_implementation=cfg.attn_implementation,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
    )
    model.config.use_cache = not for_training
    return model


def find_linear_module_names(model: nn.Module) -> list[str]:
    """LoRA를 붙일 Linear 계열 모듈의 leaf 이름을 수집한다.

    EXAONE-3.5는 어텐션이 q_proj/k_proj/v_proj/out_proj, MLP가 c_fc_0/c_fc_1/c_proj
    라서 llama 계열 기본 target_modules가 하나도 매칭되지 않는다. 모델 구조를
    직접 스캔하는 편이 버전 변화에도 안전하다.
    """
    try:
        import bitsandbytes as bnb

        linear_types: tuple[type, ...] = (nn.Linear, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
    except ImportError:  # bitsandbytes 미설치 (bf16 LoRA 경로)
        linear_types = (nn.Linear,)

    names: set[str] = set()
    for full_name, module in model.named_modules():
        if not isinstance(module, linear_types):
            continue
        leaf = full_name.rsplit(".", 1)[-1]
        if leaf in _LORA_EXCLUDE:
            continue
        names.add(leaf)
    return sorted(names)


def build_peft_model(cfg: Config, model: PreTrainedModel) -> PeftModel:
    ckpt_kwargs = {"use_reentrant": False}
    if cfg.model.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg.model.gradient_checkpointing,
            gradient_checkpointing_kwargs=ckpt_kwargs,
        )
    elif cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=ckpt_kwargs)

    # 베이스가 동결된 PEFT + grad checkpointing 조합에서는 입력 임베딩이 grad를
    # 요구하지 않으면 체크포인트 구간으로 gradient가 흐르지 않는다.
    if cfg.model.gradient_checkpointing:
        model.enable_input_require_grads()

    target_modules = cfg.lora.target_modules
    if not target_modules:
        target_modules = find_linear_module_names(model)
        logger.info("target_modules 자동 감지: %s", target_modules)
        if not target_modules:
            raise RuntimeError("LoRA를 붙일 Linear 모듈을 찾지 못했습니다.")

    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        task_type="CAUSAL_LM",
        target_modules=list(target_modules),
        modules_to_save=cfg.lora.modules_to_save or None,
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def load_for_training(cfg: Config) -> tuple[PeftModel, PreTrainedTokenizerBase]:
    model = load_base_model(cfg.model, for_training=True)
    tokenizer = load_tokenizer(cfg.model)

    # sdpa가 실제로 적용됐는지 확인한다. eager로 폴백되면 3072 토큰에서
    # 어텐션 행렬만 레이어당 수백 MB라 속도/메모리가 모두 무너진다.
    logger.info(
        "attn_implementation=%s, gradient_checkpointing=%s",
        getattr(model.config, "_attn_implementation", "unknown"),
        cfg.model.gradient_checkpointing,
    )
    return build_peft_model(cfg, model), tokenizer


def load_for_inference(
    cfg: Config,
    adapter_path: str | Path | None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """추론용 로딩. adapter_path가 있으면 LoRA를 얹고, 없으면 베이스 모델만 쓴다."""
    tokenizer_source = str(adapter_path) if adapter_path and _has_tokenizer(adapter_path) else None
    if tokenizer_source:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source, trust_remote_code=cfg.model.trust_remote_code
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = load_tokenizer(cfg.model)
    tokenizer.padding_side = "left"  # 생성은 left padding

    model = load_base_model(cfg.model, for_training=False)
    if adapter_path:
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    model.eval()
    return model, tokenizer


def _has_tokenizer(path: str | Path) -> bool:
    path = Path(path)
    return any((path / name).exists() for name in ("tokenizer.json", "tokenizer_config.json"))
