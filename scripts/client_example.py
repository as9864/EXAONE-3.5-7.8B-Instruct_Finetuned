"""`exaone_summarize.serve` 서버를 호출하는 클라이언트 예제.

**이 파일은 다른 프로젝트로 복사해서 쓰도록 만들었습니다.** 표준 라이브러리만
쓰므로 requests도, torch도, transformers도 필요 없습니다.

    # 이 저장소 쪽에서 서버를 먼저 띄운다
    python -m exaone_summarize.serve --port 8000

    # 다른 프로젝트에서
    python client_example.py --text "요약할 본문..."
    python client_example.py --file article.txt
    python client_example.py --jsonl docs.jsonl --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


class SummarizeError(RuntimeError):
    """서버가 4xx/5xx로 응답했을 때."""


class SummarizeClient:
    """요약 서버 클라이언트.

    timeout은 넉넉해야 합니다. 7.8B 모델이 512토큰을 생성하는 데 수십 초가 걸리고,
    서버는 요청을 직렬로 처리하므로 앞선 요청을 기다릴 수도 있습니다.
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise SummarizeError(f"HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise SummarizeError(
                f"서버에 접속할 수 없습니다({self.base_url}). "
                f"`python -m exaone_summarize.serve`가 떠 있는지 확인하세요: {exc.reason}"
            ) from exc

    def health(self) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise SummarizeError(f"서버 응답 없음: {exc.reason}") from exc

    def summarize(self, document: str, **options: Any) -> str:
        """문서 한 건 -> 요약문."""
        return self.summarize_detailed(document, **options)["summary"]

    def summarize_detailed(self, document: str, **options: Any) -> dict[str, Any]:
        """요약 + 입력 토큰 수 + 절단 여부(truncated)까지 함께."""
        payload = {"document": document, **{k: v for k, v in options.items() if v is not None}}
        return self._post("/summarize", payload)

    def summarize_many(
        self, documents: list[str], *, batch_size: int = 1, **options: Any
    ) -> list[str]:
        """문서 여러 건 -> 요약문 리스트 (입력 순서 유지).

        서버는 한 요청에 32건까지 받으므로 그보다 길면 나눠 보냅니다.
        """
        summaries: list[str] = []
        chunk_limit = 32
        for start in range(0, len(documents), chunk_limit):
            payload = {
                "documents": documents[start : start + chunk_limit],
                "batch_size": batch_size,
                **{k: v for k, v in options.items() if v is not None},
            }
            response = self._post("/summarize/batch", payload)
            summaries.extend(item["summary"] for item in response["results"])
        return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="요약 서버 호출 예제")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--text", help="요약할 본문 문자열")
    parser.add_argument("--file", help="요약할 텍스트 파일")
    parser.add_argument("--jsonl", help="document 필드가 있는 JSONL (배치)")
    parser.add_argument("--document-key", default="document")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--health", action="store_true", help="서버 상태만 확인")
    args = parser.parse_args(argv)

    client = SummarizeClient(args.base_url)

    try:
        if args.health:
            print(json.dumps(client.health(), ensure_ascii=False, indent=2))
            return 0

        given = [bool(args.text), bool(args.file), bool(args.jsonl)]
        if sum(given) != 1:
            parser.error("--text / --file / --jsonl 중 정확히 하나를 지정하세요.")

        if args.jsonl:
            documents = [
                json.loads(line)[args.document_key]
                for line in Path(args.jsonl).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            summaries = client.summarize_many(
                documents, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens
            )
            for index, summary in enumerate(summaries):
                print(f"[{index}] {summary}")
            return 0

        document = args.text or Path(args.file).read_text(encoding="utf-8")
        result = client.summarize_detailed(document, max_new_tokens=args.max_new_tokens)
        print(result["summary"])
        if result["truncated"]:
            print(
                f"\n(경고: 본문 {result['input_tokens']}토큰이 예산 "
                f"{result['document_budget']}토큰으로 잘렸습니다. 뒷부분은 반영되지 않았습니다.)",
                file=sys.stderr,
            )
        return 0
    except SummarizeError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
