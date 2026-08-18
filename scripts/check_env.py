"""학습 전 환경 점검. 설치가 꼬였을 때 원인을 빨리 찾기 위한 스크립트.

    python scripts/check_env.py
"""

from __future__ import annotations

import importlib
import platform
import sys

CHECKS = [
    ("torch", None),
    ("transformers", "5.15"),
    ("peft", "0.14"),
    ("accelerate", "1.3"),
    ("datasets", "3.2"),
    ("bitsandbytes", "0.46"),
    ("rouge_score", None),
    ("yaml", None),
]

# requirements.txt에서 '==' 로 정확히 고정한 패키지. EXAONE-3.5는
# trust_remote_code 기반 커스텀 모델링 코드를 쓰기 때문에, transformers가
# 이 버전보다 낮아도 높아도(원격 코드가 최신 API 가정과 어긋나 조용히 깨짐)
# 문제가 된다. 최소 버전 체크만으로는 이런 드리프트를 못 잡으므로 정확히 비교한다.
#
# 5.15.x = QLoRA 학습 1에폭 + 추론 + HTTP 서버까지 완주한 버전.
EXACT_PINS = {"transformers": "5.15"}

OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"


def _cuda_version_tuple(version: str | None) -> tuple[int, ...]:
    """'12.10' > '12.8' 을 올바르게 비교하기 위해 튜플로 변환한다.

    문자열 비교로는 '12.10' < '12.8' 이 되어 오탐이 난다.
    버전을 알 수 없으면 경고를 띄우지 않도록 매우 큰 값을 돌려준다.
    """
    if not version:
        return (99, 99)
    try:
        return tuple(int(part) for part in version.split(".")[:2])
    except ValueError:
        return (99, 99)


def check_remote_code() -> int:
    """캐시된 EXAONE remote code의 attention mask 생성부 상태를 본다.

    원본 코드는 transformers 5.15에 없는 인자명으로 `create_causal_mask()`를 부른다.
    이걸 방치하거나 try/except로 덮으면 **패딩 마스크가 버려져** 배치 생성(left padding)이
    조용히 망가진다. 상세와 수정은 `scripts/patch_remote_code.py`.
    """
    try:
        from patch_remote_code import classify, find_targets, modules_root, read_source
    except ImportError as exc:  # pragma: no cover
        print(f"{WARN} patch_remote_code 임포트 실패: {exc}")
        return 1

    targets = find_targets(modules_root())
    if not targets:
        print(f"{OK} EXAONE remote code 캐시 없음 (첫 로딩 시 내려받습니다)")
        return 0

    problems = 0
    for path in targets:
        state = classify(read_source(path)[0])
        if state == "fixed":
            print(f"{OK} EXAONE remote code   attention mask 패치 적용됨")
        else:
            problems += 1
            reason = {
                "upstream": "5.15에서 create_causal_mask 호출이 TypeError로 실패합니다",
                "broken": "마스크를 버리고 있습니다(causal_mask=None) — 배치 생성이 망가집니다",
            }.get(state, "상태를 알 수 없습니다")
            print(f"{WARN} EXAONE remote code   {state}: {reason}")
            print("       -> python scripts\\patch_remote_code.py")
    return problems


def main() -> int:
    problems = 0
    print(f"Python  : {sys.version.split()[0]}  ({platform.system()} {platform.release()})")
    print("-" * 62)

    for name, min_version in CHECKS:
        try:
            module = importlib.import_module(name)
        except ImportError as exc:
            print(f"{FAIL} {name:<14} 임포트 실패: {exc}")
            problems += 1
            continue
        version = getattr(module, "__version__", "?")
        status = OK
        note = ""
        if min_version and version != "?":
            got = tuple(int(p) for p in version.split(".")[:2] if p.isdigit())
            want = tuple(int(p) for p in min_version.split(".")[:2])
            if got and got < want:
                status = WARN
        pin = EXACT_PINS.get(name)
        if pin and version != "?":
            got = tuple(int(p) for p in version.split(".")[:2] if p.isdigit())
            want = tuple(int(p) for p in pin.split(".")[:2])
            if got and got != want:
                status = WARN
                problems += 1
                direction = "낮음" if got < want else "높음"
                note = (
                    f"  <- requirements.txt는 {name}=={pin}.x 로 정확히 고정. "
                    f"설치된 버전이 그보다 {direction} (trust_remote_code 커스텀 모델링과 "
                    "어긋날 수 있음). `pip install -r requirements.txt` 재실행 권장"
                )
        print(f"{status} {name:<14} {version}{note}")

    print("-" * 62)
    try:
        import torch

        print(f"CUDA available : {torch.cuda.is_available()}")
        print(f"torch CUDA ver : {torch.version.cuda}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                cc = f"sm_{props.major}{props.minor}"
                total = props.total_memory / 1024**3
                print(f"  GPU {i}: {props.name}  {total:.1f}GiB  {cc}")
                if props.major >= 12 and _cuda_version_tuple(torch.version.cuda) < (12, 8):
                    print(
                        f"  {WARN} Blackwell({cc})에는 CUDA 12.8+ 빌드가 필요합니다. "
                        "pip install torch --index-url https://download.pytorch.org/whl/cu128"
                    )
                    problems += 1
                if total < 15:
                    print(
                        f"  {WARN} VRAM {total:.1f}GiB — 7.8B QLoRA에는 16GB 이상을 권장합니다. "
                        "data.max_seq_len을 줄이세요."
                    )
            print(f"bf16 지원      : {torch.cuda.is_bf16_supported()}")
        else:
            print(f"{FAIL} CUDA를 찾지 못했습니다. 7.8B 학습은 GPU가 필수입니다.")
            problems += 1
    except ImportError:
        problems += 1

    # bitsandbytes는 임포트만 되고 실제 4bit 커널이 안 도는 경우가 흔하다.
    try:
        import bitsandbytes  # noqa: F401
        import torch

        if torch.cuda.is_available():
            from bitsandbytes.nn import Linear4bit

            layer = Linear4bit(64, 64, compute_dtype=torch.bfloat16).cuda()
            out = layer(torch.randn(2, 64, device="cuda", dtype=torch.bfloat16))
            print(f"{OK} bitsandbytes 4bit 커널 동작 확인 (out {tuple(out.shape)})")
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} bitsandbytes 4bit 커널 실행 실패: {type(exc).__name__}: {exc}")
        print("       -> QLoRA를 못 씁니다. bitsandbytes 업그레이드 또는 bf16 LoRA 설정을 쓰세요.")
        problems += 1

    problems += check_remote_code()

    print("-" * 62)
    print("문제 없음." if problems == 0 else f"확인이 필요한 항목 {problems}건.")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
