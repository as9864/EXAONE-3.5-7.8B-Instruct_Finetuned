"""YAML + CLI 오버라이드를 지원하는 설정 로더."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import logging
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

from .prompt import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_TEMPLATE

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_name_or_path: str = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    trust_remote_code: bool = True
    attn_implementation: str = "sdpa"  # sdpa | eager | flash_attention_2
    torch_dtype: str = "bfloat16"

    # 양자화 (16GB VRAM에서 7.8B를 학습하려면 4bit가 사실상 필수)
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True

    gradient_checkpointing: bool = True


@dataclass
class LoraConfigSpec:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    # None(= null)이면 nn.Linear 이름을 스캔해 자동으로 채운다.
    # EXAONE-3.5는 q_proj/k_proj/v_proj/out_proj + c_fc_0/c_fc_1/c_proj 라서
    # llama 계열 기본값(o_proj, gate_proj...)이 매칭되지 않는다.
    target_modules: list[str] | None = None
    modules_to_save: list[str] | None = None


@dataclass
class DataConfig:
    train_file: str = "data/processed/train.jsonl"
    eval_file: str | None = "data/processed/validation.jsonl"
    document_key: str = "document"
    summary_key: str = "summary"

    max_seq_len: int = 3072
    max_target_tokens: int = 512

    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_template: str = DEFAULT_USER_TEMPLATE

    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    num_proc: int = 1  # Windows에서는 1 권장


@dataclass
class TrainConfig:
    output_dir: str = "outputs/exaone-3.5-7.8b-summary-qlora"
    seed: int = 42

    num_train_epochs: float = 3.0
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16

    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    optim: str = "paged_adamw_8bit"

    bf16: bool = True
    fp16: bool = False

    # 길이가 비슷한 샘플을 같은 배치로 모아 패딩 낭비를 줄인다.
    # per_device_train_batch_size > 1 일 때만 의미가 있다.
    group_by_length: bool = True

    logging_steps: int = 10
    eval_strategy: str = "steps"  # no | steps | epoch
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    report_to: list[str] = field(default_factory=list)  # ["tensorboard"] 등
    resume_from_checkpoint: str | None = None


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.05


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfigSpec = field(default_factory=LoraConfigSpec)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


_SECTIONS = {
    "model": ModelConfig,
    "lora": LoraConfigSpec,
    "data": DataConfig,
    "train": TrainConfig,
    "generation": GenerationConfig,
}


def _build_section(cls: type, raw: dict[str, Any] | None) -> Any:
    raw = raw or {}
    valid = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"{cls.__name__}에 없는 키: {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | Path | None) -> Config:
    """YAML 파일에서 Config를 만든다. path가 None이면 기본값을 쓴다."""
    if path is None:
        return Config()

    path = Path(path)
    with path.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp) or {}

    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"설정 파일에 알 수 없는 섹션이 있습니다: {sorted(unknown)}")

    return Config(**{name: _build_section(cls, raw.get(name)) for name, cls in _SECTIONS.items()})


@functools.lru_cache(maxsize=None)
def _field_types(cls: type) -> dict[str, Any]:
    """dataclass 필드의 선언 타입을 해석해 캐시한다.

    현재 값의 타입으로 추론하면 기본값이 None인 필드(max_train_samples,
    target_modules 등)에서 문자열이 그대로 새어 들어간다. 선언 타입을 봐야 한다.
    """
    return get_type_hints(cls)


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """`X | None` -> (X, True), 그 외 -> (annotation, False)"""
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
        return annotation, True
    return annotation, False


def _coerce(annotation: Any, value: str, *, where: str) -> Any:
    """CLI 문자열을 필드의 선언 타입에 맞춰 변환한다."""
    target, optional = _unwrap_optional(annotation)

    if value.lower() in {"none", "null", ""} and get_origin(target) is not list:
        if optional:
            return None
        raise ValueError(f"{where}: None을 허용하지 않는 필드입니다 (타입 {annotation}).")

    if target is bool:
        lowered = value.lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(f"{where}: bool로 해석할 수 없는 값 '{value}'")

    if target is int:
        try:
            return int(value)
        except ValueError:
            # 1e3 같은 지수 표기도 받아준다.
            as_float = float(value)
            if not as_float.is_integer():
                raise ValueError(f"{where}: 정수 필드에 '{value}'는 쓸 수 없습니다.") from None
            return int(as_float)

    if target is float:
        return float(value)

    if get_origin(target) is list:
        return [v.strip() for v in value.split(",") if v.strip()]

    return value


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    """`--set train.learning_rate=2e-4` 형태의 오버라이드를 적용한다."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"오버라이드 형식이 잘못됨 (section.key=value): {item}")
        dotted, value = item.split("=", 1)
        if "." not in dotted:
            raise ValueError(f"오버라이드는 section.key 형태여야 합니다: {item}")
        section, key = dotted.split(".", 1)
        if section not in _SECTIONS:
            raise ValueError(f"알 수 없는 섹션: {section} (가능: {sorted(_SECTIONS)})")

        cls = _SECTIONS[section]
        types_map = _field_types(cls)
        if key not in types_map:
            raise ValueError(f"{section}에 없는 키: {key} (가능: {sorted(types_map)})")

        coerced = _coerce(types_map[key], value, where=f"--set {dotted}")
        setattr(getattr(cfg, section), key, coerced)
    return cfg


def resolve_eval_strategy(cfg: Config) -> str:
    """eval_file이 없으면 평가 전략을 강제로 끈다."""
    return cfg.train.eval_strategy if cfg.data.eval_file else "no"


def resolve_best_model_setting(tc: TrainConfig, eval_strategy: str) -> bool:
    """load_best_model_at_end를 실제로 켤 수 있는지 판정한다.

    Trainer는 (1) eval이 없거나 (2) save_strategy='no'이거나 (3) steps 전략에서
    save_steps가 eval_steps의 배수가 아니면 예외를 던진다. 몇 시간 학습한 뒤가
    아니라 시작 시점에 걸러낸다.
    """
    if not tc.load_best_model_at_end:
        return False

    if eval_strategy == "no":
        logger.warning("평가가 비활성화되어 load_best_model_at_end를 끕니다.")
        return False

    if tc.save_strategy == "no":
        logger.warning("save_strategy='no'이므로 load_best_model_at_end를 끕니다.")
        return False

    if eval_strategy == "steps":
        if tc.save_strategy != "steps":
            raise ValueError(
                "load_best_model_at_end=true 이면 eval_strategy와 save_strategy가 같아야 합니다 "
                f"(현재 eval={eval_strategy}, save={tc.save_strategy})."
            )
        if tc.eval_steps <= 0 or tc.save_steps % tc.eval_steps != 0:
            raise ValueError(
                f"save_steps({tc.save_steps})는 eval_steps({tc.eval_steps})의 배수여야 합니다. "
                "그렇지 않으면 Trainer가 best checkpoint를 찾지 못합니다."
            )
    elif eval_strategy == "epoch" and tc.save_strategy != "epoch":
        raise ValueError(
            "load_best_model_at_end=true 이면 eval_strategy='epoch'일 때 "
            f"save_strategy도 'epoch'여야 합니다 (현재 {tc.save_strategy})."
        )
    return True


def validate(cfg: Config) -> Config:
    """학습 시작 전 설정 정합성을 검사한다."""
    if cfg.train.bf16 and cfg.train.fp16:
        raise ValueError("train.bf16과 train.fp16을 동시에 켤 수 없습니다.")
    if cfg.data.max_seq_len <= cfg.data.max_target_tokens:
        raise ValueError(
            f"data.max_seq_len({cfg.data.max_seq_len})은 "
            f"data.max_target_tokens({cfg.data.max_target_tokens})보다 커야 합니다."
        )
    if cfg.lora.r <= 0:
        raise ValueError("lora.r은 1 이상이어야 합니다.")
    if not cfg.model.load_in_4bit and "paged" in cfg.train.optim:
        logger.warning(
            "비양자화 학습에 paged optimizer(%s)를 쓰고 있습니다. "
            "adamw_torch가 더 적합합니다.",
            cfg.train.optim,
        )
    resolve_best_model_setting(cfg.train, resolve_eval_strategy(cfg))
    return cfg


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", "-c", default=None, help="YAML 설정 파일 경로")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="설정 값 오버라이드 (여러 번 사용 가능)",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    return apply_overrides(load_config(args.config), args.overrides)


def to_dict(cfg: Config) -> dict[str, Any]:
    return dataclasses.asdict(cfg)
