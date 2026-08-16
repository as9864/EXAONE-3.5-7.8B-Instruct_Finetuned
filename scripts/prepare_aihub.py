"""AI Hub '문서요약 텍스트' 데이터를 학습용 JSONL로 변환한다.

입력은 AI Hub에서 받은 zip을 그대로 둔 디렉터리다(압축 해제 불필요).

    data/AIHUB_DocSummaryData/
      Training/   법률_train_original.zip  사설_train_original.zip  신문기사_train_original.zip
      Validation/ 법률_valid_original.zip  사설_valid_original.zip  신문기사_valid_original.zip

각 zip 안에는 JSON 하나가 들어 있고 구조는 다음과 같다.

    {"name": ..., "documents": [
        {"id": ..., "category": ..., "title": ...,
         "text": [[{"index": 0, "sentence": "...", "highlight_indices": "..."}, ...], ...],
         "extractive": [0, 4, 6],
         "abstractive": ["사람이 쓴 생성 요약문"]}, ...]}

신문기사 train_original.json은 1.1GB라 통째로 파싱하면 메모리를 크게 먹는다.
그래서 `documents` 배열을 객체 단위로 스트리밍 파싱한다.

출력은 prepare_data.py와 같은 스키마 + 출처 메타데이터:

    {"document": "...본문...", "summary": "...요약...",
     "source": "aihub_news", "category": "정치", "id": "340626877"}

Validation zip은 test 스플릿이 따로 없으므로 `--test-ratio`(기본 0.5)로 쪼개
validation / test를 서로 겹치지 않게 만든다.

사용 예:
    python scripts/prepare_aihub.py
    python scripts/prepare_aihub.py --summary-type extractive --include-title
    python scripts/prepare_aihub.py --max-train-per-domain 20000
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exaone_summarize.dedup import exact_key  # noqa: E402
from exaone_summarize.jsonl import write_jsonl  # noqa: E402

IN_DIR = ROOT / "data" / "AIHUB_DocSummaryData"
OUT_DIR = ROOT / "data" / "processed" / "aihub"

# 파일명에 박혀 있는 한글 도메인 -> 출력 파일에 쓸 슬러그
DOMAINS = {
    "신문기사": "news",
    "사설": "editorial",
    "법률": "law",
}

_CHUNK = 1 << 20  # 스트리밍 파서가 한 번에 읽는 문자 수


# ---------------------------------------------------------------- 스트리밍 파서


def iter_documents(fp: TextIO) -> Iterator[dict[str, Any]]:
    """`{"documents": [...]}` 의 배열 원소를 하나씩 흘려보낸다.

    json.load()는 1.1GB 파일에서 수 GB의 파이썬 객체를 한꺼번에 만든다.
    여기서는 raw_decode로 객체 하나씩 떼어내고 소비한 앞부분을 버려서
    상수 메모리로 훑는다.
    """
    decoder = json.JSONDecoder()
    buf = ""
    eof = False

    def fill() -> bool:
        """버퍼를 늘린다. 더 읽을 게 없으면 False."""
        nonlocal buf, eof
        if eof:
            return False
        chunk = fp.read(_CHUNK)
        if not chunk:
            eof = True
            return False
        buf += chunk
        return True

    # "documents": [ 위치까지 전진
    key = '"documents"'
    while key not in buf:
        if not fill():
            raise ValueError("'documents' 키를 찾지 못했습니다 (예상과 다른 포맷)")
    pos = buf.index(key) + len(key)
    while True:
        bracket = buf.find("[", pos)
        if bracket != -1:
            pos = bracket + 1
            break
        if not fill():
            raise ValueError("'documents' 배열의 '['를 찾지 못했습니다")

    while True:
        # 원소 사이의 공백 / 쉼표 건너뛰기
        while True:
            while pos < len(buf) and buf[pos] in " \t\r\n,":
                pos += 1
            if pos < len(buf):
                break
            if not fill():
                return  # 배열이 닫히지 않았지만 더 읽을 게 없다
        if buf[pos] == "]":
            return

        while True:
            try:
                obj, end = decoder.raw_decode(buf, pos)
                break
            except ValueError:
                # 객체가 아직 덜 읽혔을 수 있다. 더 읽어도 안 되면 진짜 파싱 오류.
                if not fill():
                    raise
        yield obj
        pos = end

        # 소비한 앞부분을 잘라 버퍼가 무한정 커지지 않게 한다.
        if pos > _CHUNK:
            buf = buf[pos:]
            pos = 0


def open_json_stream(path: Path):
    """zip 또는 raw .json 을 텍스트 스트림으로 연다."""
    if path.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(path)
        names = [n for n in zf.namelist() if n.lower().endswith(".json")]
        if not names:
            zf.close()
            raise SystemExit(f"{path.name}: zip 안에 .json이 없습니다")
        raw = zf.open(names[0])
        stream = io.TextIOWrapper(raw, encoding="utf-8")

        def close() -> None:
            stream.close()
            zf.close()

        return stream, close

    stream = path.open("r", encoding="utf-8")
    return stream, stream.close


# ---------------------------------------------------------------- 레코드 변환


def _sentences(doc: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for paragraph in doc.get("text") or []:
        # 방어적으로: 문단이 dict 하나로 오는 변종도 있다.
        items = paragraph if isinstance(paragraph, list) else [paragraph]
        for item in items:
            if isinstance(item, dict) and (item.get("sentence") or "").strip():
                out.append(item)
    return out


def to_record(
    doc: dict[str, Any],
    domain: str,
    summary_type: str,
    include_title: bool,
) -> dict[str, Any] | None:
    sentences = _sentences(doc)
    if not sentences:
        return None

    body = " ".join(s["sentence"].strip() for s in sentences)
    title = str(doc.get("title") or "").strip()
    document = f"{title}\n{body}" if include_title and title else body

    if summary_type == "extractive":
        by_index = {s.get("index"): s["sentence"].strip() for s in sentences}
        picked = [by_index.get(i) for i in (doc.get("extractive") or [])]
        summary = " ".join(p for p in picked if p)
    else:
        abstractive = doc.get("abstractive") or []
        if isinstance(abstractive, str):
            abstractive = [abstractive]
        summary = " ".join(str(a).strip() for a in abstractive if str(a).strip())

    if not summary:
        return None

    record = {
        "document": document,
        "summary": summary,
        "source": f"aihub_{domain}",
    }
    if doc.get("category"):
        record["category"] = str(doc["category"])
    if doc.get("id") is not None:
        record["id"] = str(doc["id"])
    return record


def _doc_key(text: str) -> str:
    """도메인 파일 내부의 완전 중복 제거용 키.

    근사 중복(재게재·재배포)은 여기서 손대지 않는다. 병합 단계에서
    scripts/merge_datasets.py가 shingle 유사도로 처리한다.
    """
    return exact_key(text)


def _rel(path: Path) -> str:
    """출력용 짧은 경로. 프로젝트 밖이면 그대로 보여 준다."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def convert_file(
    path: Path,
    domain: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    stream, close = open_json_stream(path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    stats = {"total": 0, "no_summary": 0, "too_short": 0, "summary_too_long": 0, "dup": 0}
    try:
        for doc in iter_documents(stream):
            stats["total"] += 1
            record = to_record(doc, domain, args.summary_type, args.include_title)
            if record is None:
                stats["no_summary"] += 1
                continue
            document, summary = record["document"], record["summary"]
            if len(document) < args.min_doc_chars or len(summary) < args.min_summary_chars:
                stats["too_short"] += 1
                continue
            if len(summary) >= len(document):
                # 요약이 본문보다 길면 라벨 오류일 가능성이 높다.
                stats["summary_too_long"] += 1
                continue
            key = _doc_key(document)
            if key in seen:
                stats["dup"] += 1
                continue
            seen.add(key)
            if args.max_doc_chars:
                record["document"] = document[: args.max_doc_chars]
            rows.append(record)
            if args.progress and stats["total"] % args.progress == 0:
                print(f"    ... {stats['total']:,}건 읽음 (유효 {len(rows):,})", flush=True)
    finally:
        close()

    dropped = stats["total"] - len(rows)
    print(
        f"    원본 {stats['total']:,}건 -> 유효 {len(rows):,}건"
        f" (제외 {dropped:,}: 요약없음 {stats['no_summary']:,} /"
        f" 너무짧음 {stats['too_short']:,} /"
        f" 요약>본문 {stats['summary_too_long']:,} /"
        f" 중복 {stats['dup']:,})"
    )
    return rows


# ---------------------------------------------------------------- 파일 탐색


def discover(input_dir: Path) -> list[tuple[Path, str, str]]:
    """(경로, 도메인 슬러그, 스플릿) 목록. 스플릿은 train / valid."""
    found: list[tuple[Path, str, str]] = []
    candidates = sorted(
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".zip", ".json"}
    )
    for path in candidates:
        name = path.name
        domain = next((slug for ko, slug in DOMAINS.items() if ko in name), None)
        if domain is None:
            domain = next(
                (slug for ko, slug in DOMAINS.items() if ko in str(path.parent)), None
            )
        if domain is None:
            print(f"  건너뜀(도메인 판별 불가): {path.relative_to(input_dir)}")
            continue

        lowered = f"{path.parent.name}/{name}".lower()
        if "valid" in lowered or "dev" in lowered:
            split = "valid"
        elif "train" in lowered:
            split = "train"
        else:
            print(f"  건너뜀(스플릿 판별 불가): {path.relative_to(input_dir)}")
            continue
        found.append((path, domain, split))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI Hub 문서요약 텍스트 -> JSONL")
    parser.add_argument("--input-dir", default=str(IN_DIR))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument(
        "--summary-type",
        choices=("abstractive", "extractive"),
        default="abstractive",
        help="abstractive=사람이 쓴 생성 요약(기본), extractive=핵심문장 3개 이어붙이기",
    )
    parser.add_argument("--include-title", action="store_true", help="본문 앞에 제목 추가")
    parser.add_argument("--max-train-per-domain", type=int, default=None)
    parser.add_argument("--max-eval-per-domain", type=int, default=None)
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.5,
        help="Validation 원본에서 test로 떼어낼 비율 (0이면 test 생성 안 함)",
    )
    parser.add_argument("--min-doc-chars", type=int, default=100)
    parser.add_argument("--min-summary-chars", type=int, default=10)
    parser.add_argument("--max-doc-chars", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--progress", type=int, default=50000, help="N건마다 진행 출력 (0=끄기)"
    )
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"입력 디렉터리가 없습니다: {input_dir}")
    out_dir = Path(args.output_dir)

    targets = discover(input_dir)
    if not targets:
        raise SystemExit(f"{input_dir} 에서 변환할 zip/json을 찾지 못했습니다.")

    print(f"입력: {input_dir}")
    for path, domain, split in targets:
        print(f"  {path.relative_to(input_dir)} -> {domain}/{split}")

    written: dict[str, int] = {}
    for path, domain, split in targets:
        print(f"\n[{domain}/{split}] {path.name}")
        rows = convert_file(path, domain, args)
        if not rows:
            print("    유효 레코드 0건 -> 건너뜀")
            continue

        rng = random.Random(args.seed)
        rng.shuffle(rows)

        if split == "train":
            if args.max_train_per_domain:
                rows = rows[: args.max_train_per_domain]
            groups = {"train": rows}
        else:
            n_test = int(len(rows) * args.test_ratio) if args.test_ratio > 0 else 0
            groups = {"valid": rows[n_test:], "test": rows[:n_test]}
            if args.max_eval_per_domain:
                groups = {k: v[: args.max_eval_per_domain] for k, v in groups.items()}

        for group, group_rows in groups.items():
            if not group_rows:
                continue
            out_path = out_dir / f"{domain}_{group}.jsonl"
            write_jsonl(out_path, group_rows)
            doc_avg = sum(len(r["document"]) for r in group_rows) / len(group_rows)
            sum_avg = sum(len(r["summary"]) for r in group_rows) / len(group_rows)
            written[_rel(out_path)] = len(group_rows)
            print(
                f"    {_rel(out_path)}: {len(group_rows):,}건"
                f" (본문 평균 {doc_avg:.0f}자 / 요약 {sum_avg:.0f}자)"
            )

    print("\n생성된 파일:")
    for name, count in sorted(written.items()):
        print(f"  {name}: {count:,}건")
    print(
        "\n다음 단계: scripts/merge_datasets.py 로 기존 데이터와 합치세요.\n"
        "  python scripts/merge_datasets.py --output data/processed/train.jsonl \\\n"
        "      --input data/processed/naver_news/train.jsonl \\\n"
        "      --input data/processed/aihub/news_train.jsonl:20000 \\\n"
        "      --input data/processed/aihub/editorial_train.jsonl:20000 \\\n"
        "      --input data/processed/aihub/law_train.jsonl:20000"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
