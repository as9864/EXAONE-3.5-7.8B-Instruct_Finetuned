"""요약 태스크 프롬프트 구성.

EXAONE-3.5-Instruct의 chat template을 그대로 사용하고, 학습 시에는
assistant 응답 구간에만 loss를 걸기 위해 prompt / target을 분리해서 만든다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizerBase

DEFAULT_SYSTEM_PROMPT = (
    "당신은 문서 요약 전문가입니다. 주어진 문서의 핵심 정보만 정확하게 담아 "
    "간결한 한국어로 요약하세요. 문서에 없는 내용은 절대 추가하지 마세요."
)

DEFAULT_USER_TEMPLATE = "다음 문서를 요약해 주세요.\n\n---\n{document}\n---"


def build_messages(
    document: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_template: str = DEFAULT_USER_TEMPLATE,
) -> list[dict[str, str]]:
    """요약 요청 메시지 리스트를 만든다."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_template.format(document=document)},
    ]


def render_prompt(
    tokenizer: "PreTrainedTokenizerBase",
    document: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_template: str = DEFAULT_USER_TEMPLATE,
) -> str:
    """assistant 턴 시작까지 렌더링된 프롬프트 문자열을 반환한다."""
    return tokenizer.apply_chat_template(
        build_messages(document, system_prompt, user_template),
        tokenize=False,
        add_generation_prompt=True,
    )


def truncate_document(
    tokenizer: "PreTrainedTokenizerBase",
    document: str,
    max_document_tokens: int,
) -> str:
    """문서를 토큰 기준으로 앞에서부터 max_document_tokens 만큼만 남긴다.

    chat template을 깨지 않기 위해, 템플릿 적용 *전에* 본문만 잘라낸다.
    요약 태스크에서는 문서 앞부분이 더 중요한 경우가 많아 뒤쪽을 버린다.
    """
    ids = tokenizer(document, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_document_tokens:
        return document
    return tokenizer.decode(ids[:max_document_tokens], skip_special_tokens=True)
