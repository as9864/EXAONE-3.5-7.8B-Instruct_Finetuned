"""요약 모델을 상주시키는 로컬 HTTP 서버.

다른 프로젝트가 이 저장소의 무거운 의존성(torch/transformers/peft/bitsandbytes)을
공유하지 않고도 요약을 쓸 수 있게 한다. 모델은 서버 프로세스에 **한 번만** 올라가고,
클라이언트는 JSON만 주고받는다.

    pip install fastapi "uvicorn[standard]"
    python -m exaone_summarize.serve --port 8000

    GET  /health              모델·어댑터·본문 예산 확인
    POST /summarize           {"document": "..."} -> {"summary": "..."}
    POST /summarize/batch     {"documents": [...]} -> {"results": [...]}

GPU가 하나이므로 생성은 `Summarizer` 내부 락으로 직렬화된다. 동시에 들어온 요청은
큐에 쌓여 순서대로 처리되며, 엔드포인트를 동기 함수로 둬서 FastAPI 워커 스레드에서
실행되게 했다(이벤트 루프가 생성 중에 멈추지 않는다).
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, ConfigDict, Field, field_validator
except ImportError as exc:  # pragma: no cover - 선택 의존성
    raise ImportError(
        'HTTP 서버에는 fastapi가 필요합니다: pip install fastapi "uvicorn[standard]"'
    ) from exc

from .api import DEFAULT_ADAPTER_PATH, DEFAULT_CONFIG_PATH, Summarizer
from .config import add_config_args

logger = logging.getLogger(__name__)

# 한 요청에 담을 수 있는 문서 수. 무제한이면 요청 하나가 서버를 몇 시간 점유한다.
MAX_BATCH_DOCUMENTS = 32
MAX_DOCUMENT_CHARS = 200_000


# ---------------------------------------------------------------- 요청/응답 스키마


class GenerationParams(BaseModel):
    """생성 옵션. 지정하지 않으면 서버 설정값(config의 generation)을 쓴다."""

    # 오타(max_tokens 등)를 조용히 무시하고 기본값으로 돌아가면 원인을 찾기 어렵다.
    model_config = ConfigDict(extra="forbid")

    max_new_tokens: int | None = Field(default=None, ge=1, le=4096)
    do_sample: bool | None = None
    temperature: float | None = Field(default=None, gt=0, le=5)
    top_p: float | None = Field(default=None, gt=0, le=1)
    repetition_penalty: float | None = Field(default=None, gt=0, le=5)

    def overrides(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.model_dump().items()
            if key in GenerationParams.model_fields and value is not None
        }


class SummarizeRequest(GenerationParams):
    document: str = Field(max_length=MAX_DOCUMENT_CHARS)

    @field_validator("document")
    @classmethod
    def _check_document(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("문서가 비어 있습니다.")
        return value


class BatchRequest(GenerationParams):
    documents: list[str] = Field(min_length=1, max_length=MAX_BATCH_DOCUMENTS)
    batch_size: int = Field(default=1, ge=1, le=16)

    @field_validator("documents")
    @classmethod
    def _check_documents(cls, value: list[str]) -> list[str]:
        for index, document in enumerate(value):
            if not document.strip():
                raise ValueError(f"documents[{index}]가 비어 있습니다.")
            if len(document) > MAX_DOCUMENT_CHARS:
                raise ValueError(f"documents[{index}]가 너무 깁니다({len(document)}자).")
        return value


class SummaryOut(BaseModel):
    summary: str
    input_tokens: int
    document_budget: int
    truncated: bool


class SummarizeResponse(SummaryOut):
    elapsed_ms: int


class BatchResponse(BaseModel):
    results: list[SummaryOut]
    elapsed_ms: int


class HealthResponse(BaseModel):
    status: str
    model: str
    adapter: str | None
    device: str
    max_seq_len: int
    max_new_tokens: int
    document_budget: int


# ---------------------------------------------------------------- 앱


def create_app(
    loader: Callable[[], Summarizer],
    *,
    title: str = "EXAONE-3.5 요약 API",
) -> FastAPI:
    """`loader`가 반환한 Summarizer를 상주시키는 앱을 만든다.

    로딩을 주입받는 이유는 테스트에서 모델 없이 라우팅·검증을 확인하기 위해서다.
    """
    state: dict[str, Summarizer] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("모델 로딩 시작 (수십 초~수 분 걸립니다)")
        state["summarizer"] = loader()
        summarizer = state["summarizer"]
        logger.info(
            "준비 완료 | model=%s adapter=%s device=%s 본문예산=%d토큰",
            summarizer.model_name,
            summarizer.adapter_path,
            summarizer.device,
            summarizer.document_budget(),
        )
        yield
        state.clear()

    app = FastAPI(title=title, version="0.1.0", lifespan=lifespan)

    def get_summarizer() -> Summarizer:
        summarizer = state.get("summarizer")
        if summarizer is None:  # pragma: no cover - lifespan이 보장한다
            raise HTTPException(status_code=503, detail="모델이 아직 로딩되지 않았습니다.")
        return summarizer

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        summarizer = get_summarizer()
        return HealthResponse(
            status="ok",
            model=summarizer.model_name,
            adapter=summarizer.adapter_path,
            device=summarizer.device,
            max_seq_len=summarizer.cfg.data.max_seq_len,
            max_new_tokens=summarizer.cfg.generation.max_new_tokens,
            document_budget=summarizer.document_budget(),
        )

    # 동기 def이므로 FastAPI가 워커 스레드에서 실행한다. 생성이 이벤트 루프를 막지 않는다.
    @app.post("/summarize", response_model=SummarizeResponse)
    def summarize(request: SummarizeRequest) -> SummarizeResponse:
        summarizer = get_summarizer()
        started = time.perf_counter()
        result = _run(summarizer, [request.document], 1, request.overrides())[0]
        return SummarizeResponse(**dataclasses.asdict(result), elapsed_ms=_elapsed_ms(started))

    @app.post("/summarize/batch", response_model=BatchResponse)
    def summarize_batch_endpoint(request: BatchRequest) -> BatchResponse:
        summarizer = get_summarizer()
        started = time.perf_counter()
        results = _run(summarizer, request.documents, request.batch_size, request.overrides())
        return BatchResponse(
            results=[SummaryOut(**dataclasses.asdict(r)) for r in results],
            elapsed_ms=_elapsed_ms(started),
        )

    return app


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _run(summarizer: Summarizer, documents: list[str], batch_size: int, overrides: dict):
    """요약을 실행하고 예외를 HTTP 상태 코드로 옮긴다."""
    try:
        return summarizer.summarize_detailed(
            documents, batch_size=batch_size, **overrides
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except MemoryError as exc:  # pragma: no cover - GPU 필요
        raise HTTPException(
            status_code=503,
            detail=f"메모리가 부족합니다. batch_size를 줄이세요: {exc}",
        ) from exc
    except RuntimeError as exc:  # pragma: no cover - GPU 필요
        message = str(exc)
        if "out of memory" in message.lower():
            raise HTTPException(
                status_code=503,
                detail=f"VRAM이 부족합니다. batch_size를 줄이세요: {message}",
            ) from exc
        logger.exception("요약 실패")
        raise HTTPException(status_code=500, detail=message) from exc


# ---------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EXAONE-3.5 요약 HTTP 서버")
    add_config_args(parser)  # --config / --set
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER_PATH, help="LoRA 어댑터 경로")
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="어댑터 없이 베이스 모델만 서빙 (파인튜닝 효과 비교용)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    return parser


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        logger.warning(
            "%s로 바인딩합니다. 이 서버에는 인증이 없으니 신뢰할 수 있는 네트워크에서만 쓰세요.",
            args.host,
        )

    def loader() -> Summarizer:
        return Summarizer.load(
            config=args.config or DEFAULT_CONFIG_PATH,
            adapter=None if args.no_adapter else args.adapter,
            overrides=args.overrides,
        )

    uvicorn.run(create_app(loader), host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
