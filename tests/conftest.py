"""테스트 공용 픽스처.

실제 EXAONE 모델을 내려받지 않고도 마스킹/토큰 예산 로직을 검증하기 위해
문자 단위 스텁 토크나이저를 쓴다. EXAONE의 chat template 골격만 흉내낸다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EOS_ID = 1_000_001
PAD_ID = 1_000_002
_SPECIAL = {EOS_ID, PAD_ID}


class StubTokenizer:
    """문자 하나 = 토큰 하나인 최소 토크나이저."""

    def __init__(self) -> None:
        self.eos_token_id = EOS_ID
        self.pad_token_id = PAD_ID
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.padding_side = "right"

    def __call__(self, text, add_special_tokens: bool = True, **kwargs):
        if isinstance(text, str):
            return {"input_ids": [ord(c) for c in text]}
        return {"input_ids": [[ord(c) for c in t] for t in text]}

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return "".join(chr(i) for i in ids if not (skip_special_tokens and i in _SPECIAL))

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **kwargs,
    ):
        parts = [f"[|{m['role']}|]{m['content']}[|endofturn|]" for m in messages]
        if add_generation_prompt:
            parts.append("[|assistant|]")
        text = "\n".join(parts)
        return [ord(c) for c in text] if tokenize else text


@pytest.fixture
def tokenizer() -> StubTokenizer:
    return StubTokenizer()


@pytest.fixture
def sample_dir() -> Path:
    return ROOT / "data" / "sample"


@pytest.fixture
def project_root() -> Path:
    return ROOT
