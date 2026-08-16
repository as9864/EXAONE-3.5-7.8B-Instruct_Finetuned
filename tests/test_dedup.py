"""중복 / 근사 중복 판정 검증.

여기서 쓰는 문장은 실제 데이터에서 관찰된 패턴을 축약한 것이다.
특히 "법령 인용구만 공유하는 서로 다른 판례문"은 실제로 오탐이 났던 사례다.
"""

from __future__ import annotations

from exaone_summarize.dedup import ShingleIndex, exact_key, normalize, shingles

ARTICLE = (
    "수협은행은 최고 연 3.1% 금리 혜택을 받을 수 있는 플러스알파예금과 "
    "최고 연 3.2% 금리의 플러스알파적금 상품을 신규 출시한다고 4일 밝혔다. "
    "개인고객 누구나 가입할 수 있는 플러스알파예금은 1인당 최대 5억원 한도로 "
    "가입할 수 있으며 만기는 12개월이다. 적금은 매월 100만원까지 납입할 수 있다. "
    "수협은행 관계자는 금리 상승기에 맞춘 상품이라고 설명했다."
)

# 같은 기사를 다른 매체가 재게재: 앞머리 바이라인만 다르고 본문은 거의 같다.
REPUBLISHED = (
    "서울 뉴시스 이주혜 기자 사진 수협은행 제공 "
    + ARTICLE
    + " 한편 이번 상품은 비대면 채널에서도 가입할 수 있다."
)

# 주제만 같고 문장은 전혀 다른 기사.
SAME_TOPIC = (
    "국민은행이 연 3.5% 금리를 주는 정기예금을 내놨다. 가입 한도는 1인당 3억원이며 "
    "만기는 6개월과 12개월 두 가지다. 은행 관계자는 예대금리차 논란을 의식한 것은 "
    "아니라고 말했다. 시중은행들의 수신 경쟁은 당분간 이어질 전망이다."
)


def test_normalize_and_exact_key_ignore_whitespace_and_case():
    assert normalize("  A  b\n c ") == "a b c"
    assert exact_key("가나 다라") == exact_key("가나\n\n다라 ")
    assert exact_key("가나 다라") != exact_key("가나 다라마")


def test_shingles_are_stable_across_calls():
    assert shingles(ARTICLE) == shingles(ARTICLE)
    assert shingles(ARTICLE, sample=False) >= shingles(ARTICLE)


def test_shingles_of_short_text_do_not_crash():
    assert shingles("") == set()
    assert len(shingles("한 문장", sample=False)) == 1


def test_republished_article_is_caught():
    index = ShingleIndex([ARTICLE])
    score, doc_id = index.best_match(REPUBLISHED)
    assert doc_id == 0
    assert score >= 0.5, f"재게재 기사를 놓쳤다 (유사도 {score:.2f})"


def test_same_topic_different_article_is_not_flagged():
    index = ShingleIndex([ARTICLE])
    score, _ = index.best_match(SAME_TOPIC)
    assert score < 0.5, f"주제만 같은 기사를 중복으로 봤다 (유사도 {score:.2f})"


def test_shared_legal_boilerplate_is_not_a_duplicate():
    """짧은 판례문 두 건이 법령 인용구만 공유하는 경우 (실제 오탐 사례)."""
    citation = "구 상속세 및 증여세법(2002. 12. 18. 법률 제6780호로 개정되기 전의 것)"
    case_a = (
        f"[1] {citation} 제79조 제1호는 상속재산에 대한 상속회복청구소송의 확정판결로 "
        "말미암아 상속인간 상속재산가액의 변동이 있는 경우에 경정청구를 할 수 있도록 정하고 있다."
    )
    case_b = (
        f"[1] {citation} 제41조의2 제1항의 입법 취지는 명의신탁제도를 이용한 조세회피 "
        "행위를 방지하여 조세정의를 실현하려는 데에 있으므로 실질과세원칙의 예외에 해당한다."
    )
    index = ShingleIndex([case_a])
    score, _ = index.best_match(case_b)
    assert score < 0.5, f"인용구만 공유하는 다른 판례를 중복으로 봤다 (유사도 {score:.2f})"


def test_union_of_all_documents_is_not_used():
    """조각이 여러 문서에 흩어져 있으면 중복이 아니다.

    합집합 기준으로 비교하면 이 케이스가 통째로 오탐이 된다.
    """
    halves = [
        " ".join(ARTICLE.split()[:20]),
        " ".join(ARTICLE.split()[20:]),
    ]
    index = ShingleIndex(halves)
    score, _ = index.best_match(ARTICLE)
    assert score < 1.0


def test_index_grows_incrementally():
    index = ShingleIndex()
    assert len(index) == 0
    assert index.best_match(ARTICLE) == (0.0, None)
    index.add(ARTICLE)
    assert len(index) == 1
    assert index.best_match(ARTICLE)[0] == 1.0
