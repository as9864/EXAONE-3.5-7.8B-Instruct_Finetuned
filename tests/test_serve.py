"""HTTP 서버 라우팅·검증 검증 (모델 가중치 없이).

가짜 Summarizer를 주입해서 요청 스키마·오류 매핑·배치 분할까지 확인한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

pytest.importorskip("fastapi", reason="pip install fastapi \"uvicorn[standard]\"")

from fastapi.testclient import TestClient  # noqa: E402

from exaone_summarize.api import SummaryResult  # noqa: E402
from exaone_summarize.config import Config  # noqa: E402
from exaone_summarize.serve import MAX_BATCH_DOCUMENTS, create_app  # noqa: E402


@dataclass
class Call:
    documents: list[str]
    batch_size: int
    overrides: dict


class FakeSummarizer:
    """Summarizer의 서버가 쓰는 표면만 흉내낸다."""

    def __init__(self, *, fail_with: Exception | None = None) -> None:
        self.cfg = Config()
        self.cfg.data.max_seq_len = 1536
        self.cfg.generation.max_new_tokens = 512
        self.model_name = "fake/EXAONE-3.5"
        self.adapter_path = "outputs/fake/adapter"
        self.device = "cpu"
        self.calls: list[Call] = []
        self._fail_with = fail_with

    def document_budget(self, max_new_tokens: int | None = None) -> int:
        limit = max_new_tokens or self.cfg.generation.max_new_tokens
        return max(self.cfg.data.max_seq_len - limit, 128)

    def summarize_detailed(self, documents, *, batch_size=1, **overrides):
        if self._fail_with is not None:
            raise self._fail_with
        self.calls.append(Call(list(documents), batch_size, dict(overrides)))
        budget = self.document_budget(overrides.get("max_new_tokens"))
        return [
            SummaryResult(
                summary=f"요약:{doc[:6]}",
                input_tokens=len(doc),
                document_budget=budget,
                truncated=len(doc) > budget,
            )
            for doc in documents
        ]


@pytest.fixture
def fake() -> FakeSummarizer:
    return FakeSummarizer()


@pytest.fixture
def client(fake) -> TestClient:
    # with 블록 안에서만 lifespan(모델 로딩)이 실행된다.
    with TestClient(create_app(lambda: fake)) as test_client:
        yield test_client


# ------------------------------------------------------------------ /health


def test_health_reports_model_and_budget(client, fake):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] == fake.model_name
    assert body["adapter"] == fake.adapter_path
    assert body["document_budget"] == 1536 - 512
    assert body["max_seq_len"] == 1536


def test_loader_runs_once_at_startup():
    calls = []

    def loader():
        calls.append(1)
        return FakeSummarizer()

    with TestClient(create_app(loader)) as client:
        client.get("/health")
        client.post("/summarize", json={"document": "문서"})
    assert len(calls) == 1


# ------------------------------------------------------------------ /summarize


def test_summarize_returns_summary(client):
    response = client.post("/summarize", json={"document": "한국은행이 금리를 인하했다"})
    assert response.status_code == 200
    body = response.json()
    assert body["summary"] == "요약:한국은행이 "
    assert body["truncated"] is False
    assert body["elapsed_ms"] >= 0


def test_summarize_passes_generation_overrides(client, fake):
    client.post("/summarize", json={"document": "문서", "max_new_tokens": 128, "do_sample": True})
    assert fake.calls[0].overrides == {"max_new_tokens": 128, "do_sample": True}


def test_summarize_flags_truncation(client):
    long_document = "가" * 1100  # 예산 1024자(=스텁 토큰) 초과
    body = client.post("/summarize", json={"document": long_document}).json()
    assert body["truncated"] is True
    assert body["input_tokens"] == 1100


@pytest.mark.parametrize(
    "payload",
    [
        {},                                    # document 누락
        {"document": ""},                      # 빈 문자열
        {"document": "   "},                   # 공백만
        {"document": "문서", "max_new_tokens": 0},
        {"document": "문서", "temperature": 0},
        {"document": "문서", "top_p": 2},
        {"document": "문서", "max_tokens": 100},  # 오타는 조용히 무시하지 않는다
    ],
)
def test_summarize_rejects_bad_requests(client, payload):
    assert client.post("/summarize", json=payload).status_code == 422


def test_value_error_becomes_400():
    app = create_app(lambda: FakeSummarizer(fail_with=ValueError("본문 예산이 부족합니다")))
    with TestClient(app) as client:
        response = client.post("/summarize", json={"document": "문서"})
    assert response.status_code == 400
    assert "본문 예산" in response.json()["detail"]


def test_cuda_oom_becomes_503():
    oom = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    with TestClient(create_app(lambda: FakeSummarizer(fail_with=oom))) as client:
        response = client.post("/summarize", json={"document": "문서"})
    assert response.status_code == 503
    assert "batch_size" in response.json()["detail"]


# ------------------------------------------------------------------ /summarize/batch


def test_batch_preserves_order(client):
    documents = [f"문서{i}" for i in range(4)]
    body = client.post("/summarize/batch", json={"documents": documents}).json()
    assert [item["summary"] for item in body["results"]] == [f"요약:문서{i}" for i in range(4)]


def test_batch_forwards_batch_size(client, fake):
    client.post("/summarize/batch", json={"documents": ["가", "나"], "batch_size": 2})
    assert fake.calls[0].batch_size == 2


@pytest.mark.parametrize(
    "payload",
    [
        {"documents": []},
        {"documents": ["정상", ""]},
        {"documents": ["가"] * (MAX_BATCH_DOCUMENTS + 1)},
        {"documents": ["가"], "batch_size": 0},
        {"documents": ["가"], "batch_size": 99},
    ],
)
def test_batch_rejects_bad_requests(client, payload):
    assert client.post("/summarize/batch", json=payload).status_code == 422
