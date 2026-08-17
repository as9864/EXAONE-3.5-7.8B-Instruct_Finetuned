"""다른 프로젝트에서 요약 모델을 재사용하기 위한 고수준 API.

`infer.py`의 CLI는 실행마다 7.8B 모델을 새로 로딩한다(수십 초~수 분). 반복
호출하려면 모델을 프로세스에 상주시켜야 하고, `Summarizer`가 그 역할을 한다.

    from exaone_summarize.api import Summarizer

    summarizer = Summarizer.load()                 # 저장소 기본 config + 어댑터
    print(summarizer.summarize("본문..."))
    print(summarizer.summarize_many([doc1, doc2], batch_size=4))

단일 GPU에 모델이 하나 올라가므로 **프로세스당 하나만** 만든다. 생성 구간은
내부 락으로 직렬화되므로 여러 스레드에서 같은 인스턴스를 호출해도 안전하다
(직렬 실행이라 처리량이 늘지는 않는다).

torch/peft는 `load()` 안에서 지연 임포트한다. 이 모듈을 임포트하는 것만으로는
GPU 스택이 올라오지 않으므로 서버 코드나 테스트에서 가볍게 다룰 수 있다.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import Config, GenerationConfig, apply_overrides, load_config

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "configs/qlora_7.8b.yaml"
DEFAULT_ADAPTER_PATH = "outputs/exaone-3.5-7.8b-summary-qlora/adapter"

# 문서에 남겨줄 최소 토큰 예산. max_new_tokens를 키우면 그만큼 본문 예산이 줄어드는데
# (infer.summarize_batch가 max_seq_len - max_new_tokens로 자른다), 예산이 이보다
# 작아지면 본문이 거의 남지 않아 요약이 의미를 잃는다.
MIN_DOCUMENT_BUDGET = 256

_GENERATION_FIELDS: tuple[str, ...] = tuple(
    f.name for f in dataclasses.fields(GenerationConfig)
)


@dataclass(frozen=True)
class SummaryResult:
    """요약 한 건과, 그 요약이 어떤 조건에서 만들어졌는지."""

    summary: str
    input_tokens: int
    document_budget: int
    truncated: bool


def _package_repo_root() -> Path | None:
    """editable 설치(`src/` 레이아웃)일 때의 저장소 루트를 추정한다.

    site-packages에 복사 설치된 경우에는 `configs/`가 없으므로 None을 돌려주고,
    호출자가 경로를 직접 넘기도록 한다.
    """
    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / DEFAULT_CONFIG_PATH).is_file() else None


def resolve_repo_path(path: str | Path, repo_root: Path | None) -> Path:
    """상대 경로를 저장소 루트 기준으로 푼다. 절대 경로는 그대로 둔다.

    다른 프로젝트에서 호출하면 CWD가 저장소 루트가 아니므로, 기본 경로
    (`configs/...`, `outputs/...`)를 CWD 기준으로 풀면 전부 어긋난다.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if repo_root is None:
        return candidate.resolve()
    return (repo_root / candidate).resolve()


def merge_generation(base: GenerationConfig, **overrides: Any) -> GenerationConfig:
    """`None`인 항목은 base 값을 유지하며 생성 설정을 덮어쓴다."""
    unknown = set(overrides) - set(_GENERATION_FIELDS)
    if unknown:
        raise ValueError(
            f"알 수 없는 생성 옵션: {sorted(unknown)} (가능: {list(_GENERATION_FIELDS)})"
        )

    values = {name: getattr(base, name) for name in _GENERATION_FIELDS}
    values.update({k: v for k, v in overrides.items() if v is not None})

    if int(values["max_new_tokens"]) < 1:
        raise ValueError("max_new_tokens는 1 이상이어야 합니다.")
    if not 0 < float(values["temperature"]) <= 5:
        raise ValueError("temperature는 0 초과 5 이하여야 합니다.")
    if not 0 < float(values["top_p"]) <= 1:
        raise ValueError("top_p는 0 초과 1 이하여야 합니다.")
    if float(values["repetition_penalty"]) <= 0:
        raise ValueError("repetition_penalty는 0보다 커야 합니다.")

    return GenerationConfig(**values)


def _validate_documents(documents: Sequence[str] | Iterable[str]) -> list[str]:
    if isinstance(documents, str):
        raise TypeError("documents는 문자열 리스트여야 합니다. 단건은 summarize()를 쓰세요.")

    cleaned: list[str] = []
    for index, doc in enumerate(documents):
        if not isinstance(doc, str):
            raise TypeError(f"documents[{index}]가 문자열이 아닙니다: {type(doc).__name__}")
        stripped = doc.strip()
        if not stripped:
            raise ValueError(f"documents[{index}]가 비어 있습니다.")
        cleaned.append(stripped)
    return cleaned


class Summarizer:
    """모델을 상주시키고 요약을 반복 생성한다."""

    def __init__(
        self,
        cfg: Config,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        adapter_path: str | Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.adapter_path = str(adapter_path) if adapter_path else None
        # 같은 모델로 동시에 generate하면 VRAM이 겹쳐 터지거나 결과가 섞인다.
        self._lock = threading.Lock()

    @classmethod
    def load(
        cls,
        *,
        config: str | Path | None = DEFAULT_CONFIG_PATH,
        adapter: str | Path | None = DEFAULT_ADAPTER_PATH,
        overrides: Sequence[str] = (),
        repo_root: str | Path | None = None,
    ) -> Summarizer:
        """설정과 어댑터를 읽어 모델을 올린다.

        config/adapter의 상대 경로는 저장소 루트 기준으로 해석하므로, 다른
        프로젝트에서 CWD와 무관하게 `Summarizer.load()`만 호출하면 된다.
        `adapter=None`이면 파인튜닝 전 베이스 모델을 쓴다(비교용).
        """
        root = Path(repo_root).resolve() if repo_root else _package_repo_root()

        config_path: Path | None = None
        if config is not None:
            config_path = resolve_repo_path(config, root)
            if not config_path.is_file():
                raise FileNotFoundError(
                    f"설정 파일을 찾을 수 없습니다: {config_path}\n"
                    "저장소 밖에서 호출한다면 repo_root= 또는 절대 경로를 지정하세요."
                )

        cfg = load_config(config_path)
        if overrides:
            cfg = apply_overrides(cfg, list(overrides))

        adapter_dir: Path | None = None
        if adapter is not None:
            adapter_dir = resolve_repo_path(adapter, root)
            # 모델을 올리는 데 몇 분이 걸리므로 경로 오류는 그 전에 잡는다.
            if not (adapter_dir / "adapter_config.json").is_file():
                raise FileNotFoundError(
                    f"LoRA 어댑터가 없습니다: {adapter_dir}\n"
                    "학습을 먼저 돌리거나, 베이스 모델만 쓰려면 adapter=None을 넘기세요."
                )

        # 여기서부터 torch/bitsandbytes/peft가 필요하다.
        from .modeling import load_for_inference

        logger.info("모델 로딩: %s (adapter=%s)", cfg.model.model_name_or_path, adapter_dir)
        model, tokenizer = load_for_inference(cfg, adapter_dir)
        logger.info("로딩 완료.")
        return cls(cfg, model, tokenizer, adapter_path=adapter_dir)

    # ------------------------------------------------------------------ 메타 정보

    @property
    def model_name(self) -> str:
        return self.cfg.model.model_name_or_path

    @property
    def device(self) -> str:
        return str(getattr(self.model, "device", "unknown"))

    def document_budget(self, max_new_tokens: int | None = None) -> int:
        """본문에 허용되는 토큰 수. `infer.summarize_batch`와 같은 식을 쓴다."""
        limit = max_new_tokens or self.cfg.generation.max_new_tokens
        return max(self.cfg.data.max_seq_len - limit, 128)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    # ------------------------------------------------------------------ 요약

    def summarize(self, document: str, **overrides: Any) -> str:
        """문서 한 건을 요약한다."""
        return self.summarize_detailed([document], **overrides)[0].summary

    def summarize_many(
        self, documents: Sequence[str], *, batch_size: int = 1, **overrides: Any
    ) -> list[str]:
        """문서 여러 건을 요약해 요약문만 돌려준다."""
        results = self.summarize_detailed(documents, batch_size=batch_size, **overrides)
        return [r.summary for r in results]

    def summarize_detailed(
        self, documents: Sequence[str], *, batch_size: int = 1, **overrides: Any
    ) -> list[SummaryResult]:
        """요약과 함께 입력 토큰 수·절단 여부를 돌려준다."""
        if batch_size < 1:
            raise ValueError("batch_size는 1 이상이어야 합니다.")

        docs = _validate_documents(documents)
        if not docs:
            return []

        cfg = self._config_for(**overrides)
        budget = max(cfg.data.max_seq_len - cfg.generation.max_new_tokens, 128)

        # 지연 임포트: torch를 이 모듈 임포트 시점에 끌고 오지 않는다.
        from .infer import summarize_batch

        results: list[SummaryResult] = []
        for start in range(0, len(docs), batch_size):
            chunk = docs[start : start + batch_size]
            with self._lock:
                token_counts = [self.count_tokens(doc) for doc in chunk]
                summaries = summarize_batch(self.model, self.tokenizer, cfg, chunk)
            for n_tokens, summary in zip(token_counts, summaries, strict=True):
                results.append(
                    SummaryResult(
                        summary=summary,
                        input_tokens=n_tokens,
                        document_budget=budget,
                        truncated=n_tokens > budget,
                    )
                )
        return results

    # ------------------------------------------------------------------ 내부

    def _config_for(self, **overrides: Any) -> Config:
        """요청별 생성 옵션을 반영한 Config 사본을 만든다."""
        generation = merge_generation(self.cfg.generation, **overrides)
        remaining = self.cfg.data.max_seq_len - generation.max_new_tokens
        if remaining < MIN_DOCUMENT_BUDGET:
            raise ValueError(
                f"max_new_tokens({generation.max_new_tokens})가 너무 큽니다. "
                f"max_seq_len({self.cfg.data.max_seq_len})에서 본문 예산이 "
                f"{remaining}토큰밖에 남지 않습니다(최소 {MIN_DOCUMENT_BUDGET})."
            )
        if generation == self.cfg.generation:
            return self.cfg
        return dataclasses.replace(self.cfg, generation=generation)
