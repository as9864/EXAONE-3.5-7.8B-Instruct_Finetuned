"""설정 로딩 / 오버라이드 검증."""

from __future__ import annotations

import pytest
import yaml

from exaone_summarize.config import (
    Config,
    apply_overrides,
    load_config,
    resolve_best_model_setting,
    resolve_eval_strategy,
    validate,
)


def test_default_config_targets_qlora():
    cfg = Config()
    assert cfg.model.load_in_4bit is True
    assert cfg.lora.target_modules is None  # 자동 감지
    assert cfg.train.bf16 is True


@pytest.mark.parametrize(
    "config_name",
    ["qlora_7.8b.yaml", "lora_bf16_7.8b.yaml", "smoke.yaml"],
)
def test_shipped_configs_load(project_root, config_name):
    cfg = load_config(project_root / "configs" / config_name)
    assert cfg.model.model_name_or_path.startswith("LGAI-EXAONE/EXAONE-3.5")
    assert cfg.data.max_seq_len > cfg.data.max_target_tokens


def test_bf16_config_does_not_use_paged_optimizer(project_root):
    """비양자화 경로에서 paged optimizer를 쓰면 bitsandbytes에 불필요하게 묶인다."""
    cfg = load_config(project_root / "configs" / "lora_bf16_7.8b.yaml")
    assert cfg.model.load_in_4bit is False
    assert "paged" not in cfg.train.optim


def test_overrides_coerce_types():
    cfg = Config()
    apply_overrides(
        cfg,
        [
            "train.learning_rate=2e-4",
            "train.num_train_epochs=1",
            "model.load_in_4bit=false",
            "data.max_train_samples=500",
            "lora.target_modules=q_proj,k_proj,v_proj",
            "data.eval_file=none",
        ],
    )
    assert cfg.train.learning_rate == pytest.approx(2e-4)
    assert cfg.train.num_train_epochs == 1.0
    assert cfg.model.load_in_4bit is False
    assert cfg.data.max_train_samples == 500
    assert cfg.lora.target_modules == ["q_proj", "k_proj", "v_proj"]
    assert cfg.data.eval_file is None


def test_override_on_none_default_field_keeps_declared_type():
    """기본값이 None인 필드도 선언 타입으로 변환되어야 한다.

    문자열이 새어 들어가면 rows[:'500'] 같은 곳에서 런타임에 터진다.
    """
    cfg = Config()
    apply_overrides(cfg, ["data.max_train_samples=500", "lora.target_modules=q_proj"])
    assert isinstance(cfg.data.max_train_samples, int)
    assert cfg.lora.target_modules == ["q_proj"]


def test_int_field_accepts_exponent_notation():
    cfg = Config()
    apply_overrides(cfg, ["train.max_steps=1e3"])
    assert cfg.train.max_steps == 1000


def test_empty_list_override():
    cfg = Config()
    apply_overrides(cfg, ["train.report_to="])
    assert cfg.train.report_to == []


@pytest.mark.parametrize(
    "override",
    [
        "train.learning_rate",  # '=' 없음
        "nosuch.key=1",  # 없는 섹션
        "train.nosuch=1",  # 없는 키
        "learning_rate=1",  # 섹션 누락
        "model.load_in_4bit=maybe",  # bool 파싱 불가
        "train.max_steps=abc",  # int 파싱 불가
        "train.max_steps=1.5",  # 정수 아님
        "train.output_dir=none",  # Optional이 아닌 필드에 None
    ],
)
def test_bad_overrides_raise(override):
    with pytest.raises(ValueError):
        apply_overrides(Config(), [override])


def test_unknown_yaml_key_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"train": {"nonexistent_option": 1}}), encoding="utf-8")
    with pytest.raises(ValueError, match="없는 키"):
        load_config(path)


def test_unknown_yaml_section_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"trainer": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="알 수 없는 섹션"):
        load_config(path)


# --- 학습 시작 전 정합성 검사 -------------------------------------------------


def test_eval_strategy_disabled_without_eval_file():
    cfg = Config()
    assert resolve_eval_strategy(cfg) == "steps"
    cfg.data.eval_file = None
    assert resolve_eval_strategy(cfg) == "no"


def test_best_model_disabled_when_no_eval_or_no_save():
    cfg = Config()
    assert resolve_best_model_setting(cfg.train, "no") is False

    cfg.train.save_strategy = "no"
    assert resolve_best_model_setting(cfg.train, "steps") is False


def test_best_model_enabled_when_steps_align():
    cfg = Config()
    cfg.train.eval_steps = 50
    cfg.train.save_steps = 100  # 배수 -> OK
    assert resolve_best_model_setting(cfg.train, "steps") is True


def test_misaligned_save_steps_raises_early():
    """학습 몇 시간 뒤가 아니라 시작 직후에 잡혀야 한다."""
    cfg = Config()
    cfg.train.eval_steps = 30
    cfg.train.save_steps = 100  # 100 % 30 != 0
    with pytest.raises(ValueError, match="배수여야"):
        resolve_best_model_setting(cfg.train, "steps")


def test_mismatched_strategies_raise():
    cfg = Config()
    cfg.train.save_strategy = "epoch"
    with pytest.raises(ValueError, match="같아야"):
        resolve_best_model_setting(cfg.train, "steps")


def test_validate_accepts_shipped_configs(project_root):
    for name in ("qlora_7.8b.yaml", "lora_bf16_7.8b.yaml", "smoke.yaml"):
        validate(load_config(project_root / "configs" / name))


def test_validate_rejects_bf16_and_fp16():
    cfg = Config()
    cfg.train.fp16 = True  # bf16은 기본 True
    with pytest.raises(ValueError, match="동시에"):
        validate(cfg)


def test_validate_rejects_target_longer_than_seq_len():
    cfg = Config()
    cfg.data.max_target_tokens = cfg.data.max_seq_len
    with pytest.raises(ValueError, match="max_seq_len"):
        validate(cfg)


def test_validate_rejects_nonpositive_lora_rank():
    cfg = Config()
    cfg.lora.r = 0
    with pytest.raises(ValueError, match="lora.r"):
        validate(cfg)
