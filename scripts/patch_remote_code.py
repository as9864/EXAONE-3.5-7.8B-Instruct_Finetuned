"""EXAONE-3.5 remote code(modeling_exaone.py)의 attention mask 생성부를 고친다.

## 왜 필요한가

`trust_remote_code`로 받는 `modeling_exaone.py`는 transformers 5.15에서
`create_causal_mask()`를 **옛 시그니처**로 호출한다(`input_embeds=`, `cache_position=`).
5.15의 인자명은 `inputs_embeds`이고 `cache_position`은 아예 없어서 TypeError가 난다.

이걸 try/except로 감싸 `causal_mask = None`으로 흘려보내면 **조용히 더 나쁜 일**이
벌어진다. SDPA는 mask가 None이면 `is_causal=True`로 순수 causal 어텐션을 하므로
**패딩 토큰을 실제 토큰처럼 attend한다.**

- 학습(right padding)은 pad가 뒤에 있어 각 실토큰의 앞쪽이 전부 실토큰이라 거의 무해하다.
- **생성(left padding)은 치명적이다.** 배치 안에서 짧은 문서일수록 앞에 pad가 많이
  붙고, 그 pad를 문맥으로 읽어 요약이 반복 루프·환각으로 붕괴한다. 배치 내 최장
  문서(=pad 0개)만 정상이라 원인 파악이 어렵다. 실측: 베이스 모델 200건 batch=4
  평가에서 113건에 5-gram 반복이 나오고 ROUGE-1이 8.83까지 떨어졌다.

## 하는 일

`causal_mask = ...` 대입부를 5.15 시그니처에 맞춘 호출로 바꾼다.

    causal_mask = create_causal_mask(
        config=self.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

대상은 transformers가 실제로 import하는 쪽, 즉
`~/.cache/huggingface/modules/transformers_modules/**/modeling_exaone.py`다.
(hub 스냅샷은 원본 그대로 두므로 `--restore`로 언제든 되돌릴 수 있다.)

## 사용법

    python scripts\\patch_remote_code.py --check     # 상태만 확인 (종료코드 1 = 패치 필요)
    python scripts\\patch_remote_code.py             # 패치 (원본은 .orig로 백업)
    python scripts\\patch_remote_code.py --restore   # 백업으로 되돌리기

모델 캐시를 지우고 다시 받으면 원래 코드로 돌아오므로 **그때는 다시 실행해야 한다.**
"""

from __future__ import annotations

import argparse
import ast
import shutil
import sys
from pathlib import Path

MODULES_GLOB = "transformers_modules/**/modeling_exaone.py"

# transformers 5.15의 create_causal_mask 시그니처:
#   (config, inputs_embeds, attention_mask, past_key_values, position_ids=None, ...)
FIXED_CALL = """        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
"""

# 원본(업로드된 remote code) — 5.15에 없는 인자명(`input_embeds`, `cache_position`)을 써서 TypeError.
UPSTREAM_CALL = """        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
"""

# 위 TypeError를 try/except로 덮어 마스크를 통째로 버리던 손수정 버전.
BROKEN_CALL = """        # [수정 후 코드]
        try:
            # 최신 transformers 호환용
            from transformers.masking_utils import _prepare_4d_causal_attention_mask
            causal_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length=past_key_values.get_usable_length(seq_length) if past_key_values is not None else 0
            )
        except Exception:
            # 혹시 실패할 경우 None으로 넘기면 내부 sdpa / flash_attention2가 알아서 처리
            causal_mask = None
"""

_IMPORT_LINE = "from transformers.masking_utils import create_causal_mask"


def modules_root() -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    # modules는 hub 캐시와 형제 디렉터리다: ~/.cache/huggingface/{hub,modules}
    return Path(HF_HUB_CACHE).parent / "modules"


def find_targets(root: Path) -> list[Path]:
    return sorted(p for p in root.glob(MODULES_GLOB) if "EXAONE" in str(p).upper())


def read_source(path: Path) -> tuple[str, str]:
    """파일을 LF로 정규화해 읽고, 원래 줄바꿈 방식을 함께 돌려준다.

    Windows에서 그냥 read_text/write_text를 쓰면 줄바꿈이 CRLF로 바뀌어 파일 전체가
    변경된 것처럼 보인다(내용 비교·diff가 무의미해진다).
    """
    raw = path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in raw else "\n"
    return raw.replace("\r\n", "\n"), newline


def write_source(path: Path, text: str, newline: str) -> None:
    path.write_bytes(text.replace("\n", newline).encode("utf-8"))


def classify(text: str) -> str:
    """파일 상태를 판정한다: fixed / broken / upstream / unknown."""
    if FIXED_CALL in text:
        return "fixed"
    if BROKEN_CALL in text:
        return "broken"
    if UPSTREAM_CALL in text:
        return "upstream"  # 원본 = 5.15에서 TypeError
    return "unknown"


def patch_text(text: str) -> str:
    """causal_mask 대입 구간을 올바른 호출로 교체한다.

    구간을 **문자열 그대로** 찾아 바꾼다. 인덱스 탐색으로 범위를 잡으면 다른 위치의
    비슷한 코드에 걸려 파일을 크게 훼손할 수 있다(실제로 그랬다).
    """
    target = next((block for block in (BROKEN_CALL, UPSTREAM_CALL) if block in text), None)
    if target is None:
        raise SystemExit(
            "알려진 causal_mask 블록을 찾지 못했습니다. remote code가 업데이트된 것 같으니 "
            "직접 확인하세요(create_causal_mask 호출부)."
        )
    if text.count(target) != 1:
        raise SystemExit(f"교체 대상이 {text.count(target)}곳입니다. 1곳이어야 합니다.")

    new_text = text.replace(target, FIXED_CALL)
    if _IMPORT_LINE not in new_text:
        raise SystemExit(f"import가 없습니다: {_IMPORT_LINE}")
    ast.parse(new_text)  # 문법 검증 — 깨진 파일을 남기지 않는다.

    # 교체 후에도 줄 수가 크게 달라지면 의도치 않은 삭제가 있었다는 뜻이다.
    delta = len(text.splitlines()) - len(new_text.splitlines())
    if abs(delta) > 12:
        raise SystemExit(f"줄 수가 {delta}줄 변했습니다. 교체 범위가 잘못됐을 수 있습니다.")
    return new_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXAONE remote code attention mask 패치")
    parser.add_argument("--check", action="store_true", help="상태만 확인 (수정하지 않음)")
    parser.add_argument("--restore", action="store_true", help=".orig 백업으로 되돌리기")
    args = parser.parse_args(argv)

    root = modules_root()
    targets = find_targets(root)
    if not targets:
        print(f"modeling_exaone.py를 찾지 못했습니다: {root / MODULES_GLOB}")
        print("모델을 한 번 로딩하면(예: python -m exaone_summarize.infer ...) 캐시에 생깁니다.")
        return 1

    need_patch = 0
    for path in targets:
        text, newline = read_source(path)
        state = classify(text)
        backup = path.with_suffix(".py.orig")

        if args.restore:
            if backup.exists():
                shutil.copy(backup, path)
                print(f"되돌림: {path}  <- {backup.name}")
            else:
                print(f"백업 없음, 건너뜀: {path}")
            continue

        print(f"{state:>8} | {path}")
        if state == "fixed":
            continue
        need_patch += 1
        if args.check:
            continue

        if not backup.exists():
            shutil.copy(path, backup)
        write_source(path, patch_text(text), newline)
        # __pycache__가 남아 있으면 옛 바이트코드를 쓸 수 있다.
        for cached in (path.parent / "__pycache__").glob("modeling_exaone*.pyc"):
            cached.unlink()
        print(f"   -> 패치 완료 (백업: {backup.name})")

    if args.check:
        if need_patch:
            print(f"\n패치가 필요한 파일 {need_patch}개. `python scripts\\patch_remote_code.py` 실행하세요.")
            return 1
        print("\n모두 정상입니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
