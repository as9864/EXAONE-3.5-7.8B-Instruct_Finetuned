"""문서 중복 / 근사 중복 판정.

한국어 뉴스 코퍼스에는 통신사 기사 재배포, 제목만 바꾼 재게재, 앞뒤가 조금
잘린 판본이 흔하다. 문자열 일치만으로 거르면 이런 것들이 그대로 남아
학습 세트와 평가 세트에 동시에 들어간다(실측: 기존 test.jsonl의 37.6%).

여기서는 두 단계로 본다.

* `exact_key`  : 공백/대소문자 정규화 후 SHA-1 — 완전 일치
* `ShingleIndex`: 어절 5-gram 해시로 만든 역색인 — 근사 중복

근사 중복 점수는 **overlap coefficient** `공유 / min(양쪽 크기)` 다. Jaccard를
쓰면 긴 원문에 짧은 발췌가 들어 있는 관계(부분집합)를 놓친다.

비교량을 줄이려고 shingle을 해시 값으로 표본추출한다(1/SAMPLE_MOD). 내용이
같으면 같은 shingle이 뽑히므로 탐지력은 거의 유지된다. 또 너무 많은 문서에
등장하는 shingle은 상용구(“서울 연합뉴스 기자”)로 보고 조회에서 제외한다.

중요: 학습 세트 **전체**의 shingle 합집합과 비교하면 안 된다. 같은 사건을 다룬
서로 다른 기사들의 조각이 여기저기 맞아떨어져 점수가 부풀려진다
(실측: 합집합 기준 0.50이 실제로는 최근접 문서와 0.17). 반드시 문서 단위로 본다.
"""

from __future__ import annotations

import hashlib
import re
import zlib
from collections import Counter

__all__ = [
    "BOILERPLATE_DF",
    "SAMPLE_MOD",
    "SHINGLE_N",
    "ShingleIndex",
    "exact_key",
    "normalize",
    "shingles",
]

_WORD_RE = re.compile(r"[가-힣]+|[a-zA-Z]+|[0-9]+")

SHINGLE_N = 5  # 어절 5-gram
SAMPLE_MOD = 8  # shingle 8개 중 1개꼴로 표본추출
BOILERPLATE_DF = 100  # 이보다 많은 문서에 나오는 shingle은 조회에서 제외

# 공유 shingle이 이보다 적으면 비율이 높아도 우연으로 본다.
# 짧은 문서에서 특히 중요하다. 판례문 두 건이 "구 상속세 및 증여세법(2002. 12.
# 18. 법률 제6780호로 개정되기 전의 것)" 같은 법령 인용구만 공유해도
# min() 분모 때문에 유사도가 0.6까지 나온다 — 실제로는 서로 다른 사건이다.
MIN_SHARED = 4


def normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def exact_key(text: str) -> str:
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()


def shingles(text: str, sample: bool = True) -> set[int]:
    """어절 5-gram의 crc32 집합. sample=True면 1/SAMPLE_MOD만 남긴다.

    실행마다 결과가 흔들리면 안 되므로 내장 hash() 대신 crc32를 쓴다.
    """
    words = _WORD_RE.findall(text.lower())
    if not words:
        return set()
    if len(words) < SHINGLE_N:
        words = words + [""] * (SHINGLE_N - len(words))
    out: set[int] = set()
    for i in range(len(words) - SHINGLE_N + 1):
        h = zlib.crc32(" ".join(words[i : i + SHINGLE_N]).encode("utf-8"))
        if not sample or h % SAMPLE_MOD == 0:
            out.add(h)
    return out


class ShingleIndex:
    """shingle -> 문서 id 역색인. 문서 단위 최대 유사도를 구한다.

    `add()`로 점진적으로 채울 수 있어서, 병합 중 지금까지 남긴 문서와
    비교하며 걸러내는 용도로도 쓴다.
    """

    def __init__(self, documents: list[str] | None = None) -> None:
        self.postings: dict[int, list[int]] = {}
        self.sizes: list[int] = []
        for text in documents or []:
            self.add(text)

    def __len__(self) -> int:
        return len(self.sizes)

    def add(self, text: str) -> int:
        doc_id = len(self.sizes)
        fingerprint = shingles(text)
        self.sizes.append(len(fingerprint))
        for h in fingerprint:
            self.postings.setdefault(h, []).append(doc_id)
        return doc_id

    def best_match(self, text: str) -> tuple[float, int | None]:
        """(최대 overlap coefficient, 문서 id). 비어 있으면 (0.0, None)."""
        query = shingles(text)
        if not query:
            return 0.0, None

        hits: Counter[int] = Counter()
        for h in query:
            docs = self.postings.get(h)
            if not docs or len(docs) > BOILERPLATE_DF:
                continue  # 상용구 shingle
            for doc_id in docs:
                hits[doc_id] += 1

        best_score, best_id = 0.0, None
        for doc_id, shared in hits.items():
            if shared < MIN_SHARED:
                continue
            denominator = min(len(query), self.sizes[doc_id])
            if not denominator:
                continue
            score = shared / denominator
            if score > best_score:
                best_score, best_id = score, doc_id
        return best_score, best_id
