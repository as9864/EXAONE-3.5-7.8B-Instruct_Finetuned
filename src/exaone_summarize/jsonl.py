"""JSONL 입출력 유틸.

data.py와 분리해 둔 이유: 데이터 전처리(scripts/prepare_data.py)만 돌릴 때
torch / datasets 임포트 비용을 물지 않도록 하기 위함이다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"데이터 파일이 없습니다: {path}\n"
            "scripts/prepare_data.py 를 먼저 실행해 data/processed/*.jsonl 을 만드세요."
        )
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for lineno, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} JSON 파싱 실패: {exc}") from exc
    if not rows:
        raise ValueError(f"{path} 에 유효한 레코드가 없습니다.")
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
