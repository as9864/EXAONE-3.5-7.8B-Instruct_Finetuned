# EXAONE-3.5-7.8B-Instruct 문서 요약 LoRA 파인튜닝

[LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct)를
LoRA / QLoRA로 파인튜닝해 **한국어 문서 요약**에 특화시키는 프로젝트입니다.

데이터 준비 → 학습 → 추론 → ROUGE 평가 → 어댑터 병합까지 한 흐름으로 묶여 있고,
설정은 YAML 한 곳에서 관리합니다.

## 문서

| 문서 | 내용 |
|---|---|
| **[docs/USAGE.md](docs/USAGE.md)** | 설치 · 데이터 준비 · 학습 · 추론 · 평가 · 병합 · **다른 프로젝트에서 쓰기** · 트러블슈팅 |
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | 시스템 구조도 · 모듈 책임 · 데이터 흐름 · 마스킹 구조 · 확장 포인트 |
| **[docs/WORKLOG.md](docs/WORKLOG.md)** | 작업 기록 · 설계 결정 근거 · 발견한 버그 · 검증 내역 · 미검증 항목 |

---

## 핵심 설계

| 항목 | 내용 |
|---|---|
| 학습 방식 | QLoRA (4-bit NF4 + double quant) 기본 / bf16 LoRA 옵션 |
| loss 범위 | **응답 구간만** — 프롬프트 토큰은 `-100`으로 마스킹 (completion-only) |
| 프롬프트 | EXAONE 공식 chat template (`apply_chat_template`) 사용 |
| `target_modules` | 모델의 `nn.Linear`를 스캔해 **자동 감지** |
| 평가 | ROUGE-1/2/L, 한국어용 분절기 3종 (`word` / `char` / `morph`) + lead-N 베이스라인 · 출처별 분해 · paired bootstrap CI |
| 테스트 | 120개 — 모델 가중치 없이 마스킹·예산·설정·중복판정·서버 스키마 검증 |

### 이 모델에서 특별히 처리한 것

**모듈 이름이 Llama 계열과 다릅니다.**

```
어텐션 : q_proj, k_proj, v_proj, out_proj      (o_proj 아님)
MLP    : c_fc_0, c_fc_1, c_proj                (gate/up/down_proj 아님)
```

PEFT의 Llama 기본값(`q_proj`, `o_proj`, `gate_proj` …)을 그대로 쓰면 대부분
매칭되지 않아 **어댑터가 거의 붙지 않습니다.** `target_modules: null`일 때 모델
그래프를 직접 스캔해 채웁니다.

**16GB VRAM에서는 4-bit가 전제입니다.** 7.8B를 bf16으로 올리면 가중치만 약 16GB라
옵티마이저와 활성값 자리가 없습니다.

**transformers는 5.15.0으로 고정했습니다.** EXAONE-3.5의 `trust_remote_code` 코드는
버전에 민감해서, 학습·추론을 실제로 완주한 버전으로 못박고 전용 `.venv`에서 작업합니다
(전역에 흔한 5.9.0에서는 로딩 실패). 검증된 조합은
[USAGE §2](docs/USAGE.md#전용-venv를-쓰는-이유)에 표로 정리했습니다.

---

## 학습 결과

`outputs/exaone-3.5-7.8b-summary-qlora` 실행 기록입니다.

### 1. 실험 설정

| 항목 | 값 |
|---|---|
| Base model | EXAONE-3.5-7.8B-Instruct (7.86B params) |
| Adaptation | QLoRA — 4-bit NF4 + double quant, compute dtype bf16 |
| LoRA | r=16, α=32, dropout 0.05, bias none |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `out_proj`, `c_fc_0`, `c_fc_1`, `c_proj` (자동 감지) |
| Trainable params | **41.9M / 7.86B (0.53%)** |
| Objective | completion-only CE — 프롬프트 토큰은 `-100`으로 마스킹 |
| Seq. length | 1,536 tokens (target ≤ 256) |
| Optimizer | `adamw_torch_fused`, lr 1e-4, cosine, warmup 3%, wd 0, clip 1.0 |
| Batch | 4 × grad-accum 4 = **effective 16**, `group_by_length`, grad checkpointing, SDPA |
| Schedule | **1 epoch = 1,199 steps** |
| Hardware | 1× RTX 5070 Ti 16GB (sm_120) — **11.1 h** (39,795 s, 0.48 samples/s) |
| Seeds | 1 run, seed 42 (분산 미측정) |

### 2. 데이터

AI Hub 「문서요약 텍스트」 3종 + HuggingFace `daekeun-ml/naver-news-summarization-ko`를
출처가 균형을 이루도록 혼합했습니다.

| Split | 총계 | aihub_editorial | aihub_law | aihub_news | naver_news |
|---|---:|---:|---:|---:|---:|
| train | 19,178 | 4,997 | 4,896 | 4,985 | 4,300 |
| validation | 998 | 249 | 250 | 249 | 250 |
| test | 1,977 | 497 | 498 | 499 | 483 |

train↔test / train↔validation 중복(완전 일치·근사 중복·요약문 일치) **0건**을
`scripts/check_leakage.py`로 확인했습니다. 통합 전 데이터에는 평가 세트의 37.6%가
학습 세트와 겹쳐 있었습니다([WORKLOG §11](docs/WORKLOG.md)).

### 3. 학습 곡선

| step | 200 | 400 | 600 | 800 | 1,000 | 1,199 |
|---|---:|---:|---:|---:|---:|---:|
| eval loss | 0.7625 | 0.7492 | 0.7417 | 0.7359 | 0.7307 | **0.7302** |

최종 train loss 0.7546. eval loss가 끝까지 단조 감소해 best checkpoint가 마지막
스텝(1,199)으로 선택됐습니다 — **1에폭에서는 과적합이 관찰되지 않았습니다.**

### 4. 평가 결과

테스트 세트 1,977건 중 **앞 200건**, ROUGE-1/2/L F1 (%), `word` 분절.
비교 기준은 **lead-3**(본문 앞 세 문장을 그대로 복사한 요약)이며, ΔR-1의
95% 신뢰구간은 문서 단위 paired bootstrap(10,000회, seed 0)입니다.

| 데이터 | n | R-1 | R-2 | R-L | lead-3 R-1 | ΔR-1 | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| **전체** | 200 | 39.35 | 28.17 | 35.68 | 38.86 | +0.49 | [−1.90, +2.95] |
| aihub_editorial | 49 | 24.23 | 14.09 | 23.02 | 19.61 | **+4.62** | [+0.55, +8.64] |
| aihub_law | 47 | 42.15 | 29.88 | 40.00 | 44.05 | −1.91 | [−7.21, +3.40] |
| aihub_news | 53 | 36.90 | 23.04 | 31.02 | 42.17 | **−5.28** | [−9.23, −1.41] |
| naver_news | 51 | 53.86 | 45.44 | 48.71 | 49.15 | +4.71 | [−0.93, +10.25] |

**전체 평균 +0.49는 서로 반대 방향인 도메인이 상쇄된 결과입니다.** n=200에서
신뢰구간이 0을 벗어나는 것은 두 개뿐입니다 — 사설에서는 lead-3보다 **유의하게
낫고**, AI Hub 신문기사에서는 유의하게 **나쁩니다.** naver 뉴스의 +4.71과 법률의
−1.91은 이 표본 크기에서 잡음과 구별되지 않습니다.

### 5. 추상성과 생성 안정성

요약의 **신규 4-gram 비율**(원문에 없는 4-gram의 비율)이 높을수록 복사가 아니라
재구성입니다. lead-3은 정의상 0%입니다.

| 데이터 | 예측 | 정답 | 길이비(예측/정답) | 빈 출력 | 문장 미완결 | 5-gram 반복 |
|---|---:|---:|---:|---:|---:|---:|
| 전체 | 43.9% | 62.7% | 1.22 | 0 | 0 | 13 / 200 |
| aihub_editorial | 54.4% | 82.9% | 1.21 | 0 | 0 | 0 / 49 |
| aihub_law | 43.4% | 59.1% | 1.33 | 0 | 0 | 9 / 47 |
| aihub_news | 49.7% | 70.1% | 1.20 | 0 | 0 | 0 / 53 |
| naver_news | 28.4% | 38.9% | 1.13 | 0 | 0 | 4 / 51 |

- **모델은 복사기가 아닙니다** (43.9% vs lead-3의 0%). 다만 정답 요약(62.7%)보다는
  원문 표현에 더 의존합니다.
- lead-3에 지는 두 도메인(법률·AI Hub 신문기사)은 정답 요약의 어휘가 기사 앞부분과
  많이 겹치는 쪽이라, **ROUGE 관점에서는 "복사"가 정답에 가깝습니다.** 모델이 더
  나쁘다는 증거가 아니라, ROUGE로 가치를 증명할 수 없는 구간이라는 뜻입니다.
- 생성 붕괴는 없습니다(빈 출력 0, 문장 중간 잘림 0). 단 **6.5%에서 같은 5-gram이
  반복**되고, 출력이 정답보다 22% 깁니다. ROUGE-1 F1은 길이 초과를 감점하므로
  길이만 맞춰도 점수가 오릅니다.

### 6. 한계

1. **표본 크기.** 200건으로는 도메인당 n≈50이라 CI가 ±5~10점입니다. 1,977건 전체로
   재평가해야 도메인별 결론이 확정됩니다.
2. **가장 중요한 ablation이 없습니다.** 파인튜닝 전 베이스 모델의 zero-shot 점수를
   측정하지 않았습니다. EXAONE-3.5-Instruct는 그 자체로 한국어 요약을 잘하므로,
   **어댑터가 zero-shot을 넘는다는 증거는 아직 없습니다.** lead-3은 하한선일 뿐입니다.
3. **사람 평가 없음.** ROUGE는 사실 왜곡을 잡지 못합니다. 실제로 금리 *동결* 기사에
   "금리 인상을 결정할 것으로 보인다"가, 반도체 수출 기사에 원문에 없는 수요 원인
   분석이 붙는 사례를 확인했습니다.
4. **단일 실행.** seed 1개, 1에폭. 하이퍼파라미터 탐색과 분산 측정을 하지 않았습니다.
5. ROUGE 절대값은 분절기에 따라 10점 이상 움직입니다. `word` 기준끼리만 비교하세요.

### 7. 재현

```powershell
# 추론 + 평가 (표 4의 원본 수치)
python -m exaone_summarize.evaluate -c configs\qlora_7.8b.yaml `
    --input-jsonl data\processed\test.jsonl `
    --adapter outputs\exaone-3.5-7.8b-summary-qlora\adapter `
    --limit 200 --batch-size 4 `
    --save-predictions outputs\exaone-3.5-7.8b-summary-qlora\predictions.jsonl `
    --output-json outputs\exaone-3.5-7.8b-summary-qlora\metrics.json

# 출처별 lead-3 · 신뢰구간 · 추상성 (표 4·5)
python scripts\report_predictions.py `
    --predictions outputs\exaone-3.5-7.8b-summary-qlora\predictions.jsonl --markdown

# 누수 검사 (표 2)
python scripts\check_leakage.py --train data\processed\train.jsonl `
    --eval data\processed\test.jsonl --eval data\processed\validation.jsonl

# 위 1번 한계를 해소하려면 --limit 을 빼세요 (1,977건, GPU 약 45분)
# 위 2번 한계를 해소하려면 --adapter 를 빼세요 (베이스 zero-shot)
```

---

## 요구사항

- Python 3.10+, NVIDIA GPU (QLoRA 16GB+ / bf16 LoRA 40GB+), 디스크 약 20GB
- RTX 50xx(sm_120)는 **cu128+ PyTorch**와 **bitsandbytes 0.46+** 필요

---

## 빠른 시작

```powershell
.\scripts\setup.ps1                 # venv + cu128 torch + 의존성 + 환경 점검
.\.venv\Scripts\Activate.ps1
huggingface-cli login               # EXAONE-3.5는 라이선스 동의 필요

# 번들 샘플로 4스텝 스모크 테스트
python scripts\prepare_data.py --from-sample
python -m exaone_summarize.train -c configs\smoke.yaml
```

실제 데이터로 학습·평가까지 한 번에:

```powershell
.\scripts\run_pipeline.ps1

# AI Hub 「문서요약 텍스트」(신문기사 · 사설 · 법률)를 HF 뉴스 데이터와 혼합
.\scripts\run_pipeline.ps1 -WithAihub
```

상세는 [docs/USAGE.md](docs/USAGE.md)를 보세요.

---

## 학습한 모델을 다른 프로젝트에서 쓰기

CLI(`infer.py`)는 실행마다 모델을 새로 올립니다(약 20초). 반복 호출하려면 모델을
상주시키세요. 상세는 [docs/USAGE.md §9](docs/USAGE.md#9-다른-프로젝트에서-사용하기).

**HTTP 서버 (권장)** — 상대 프로젝트에 무거운 의존성을 심지 않아도 됩니다.

```powershell
pip install -e ".[serve]"
python -m exaone_summarize.serve --port 8000
```

```python
# 상대 프로젝트: scripts/client_example.py를 복사해서 쓰면 설치할 게 없습니다
from client_example import SummarizeClient
print(SummarizeClient().summarize("요약할 본문..."))
```

**같은 venv를 쓸 수 있다면** 직접 임포트가 더 빠릅니다.

```python
from exaone_summarize.api import Summarizer

summarizer = Summarizer.load()           # 프로세스당 하나만
print(summarizer.summarize("요약할 본문..."))
```

> 본문은 **1024토큰**(약 2,000자)까지만 반영됩니다(`max_seq_len - max_new_tokens`).
> 넘으면 뒷부분이 잘리고, 응답의 `truncated`로 알려줍니다.

---

## 프로젝트 구조

```
├── configs/                  qlora_7.8b · lora_bf16_7.8b · smoke
├── data/sample/              번들 샘플 8건/3건 (오프라인 검증용)
├── docs/                     USAGE · ARCHITECTURE · WORKLOG
├── scripts/                  setup.ps1 · check_env.py · prepare_data.py · prepare_aihub.py
│                             merge_datasets.py · check_leakage.py · run_pipeline.ps1
│                             report_predictions.py (출처별 · 신뢰구간 리포트)
│                             client_example.py (다른 프로젝트로 복사해 쓰는 클라이언트)
├── src/exaone_summarize/
│   ├── config.py             설정 스키마 · --set 오버라이드 · 정합성 검증
│   ├── prompt.py             chat template 구성 · 문서 토큰 절단
│   ├── jsonl.py              JSONL 입출력 (torch 비의존)
│   ├── dedup.py              완전/근사 중복 판정 (누수 차단)
│   ├── data.py               completion-only 마스킹 · 동적 패딩 콜레이터
│   ├── modeling.py           4bit 양자화 · target_modules 자동 감지 · LoRA 부착
│   ├── train.py              학습 진입점
│   ├── infer.py              요약 생성 (CLI · 배치)
│   ├── api.py                Summarizer — 모델 상주 + 요청별 생성 옵션
│   ├── serve.py              로컬 HTTP 서버 (FastAPI)
│   ├── evaluate.py           ROUGE 평가
│   └── merge_lora.py         어댑터 병합
└── tests/                    120개 (모델 다운로드 불필요)
```

```powershell
$env:PYTHONPATH="src"; python -m pytest tests -q
```

---

## 알아둘 점

**ROUGE 절대값을 믿지 마세요.** 한국어 요약 데이터는 정답 요약이 본문 복붙에
가까운 경우가 많아(실측 84.8%), 본문 앞 세 문장을 복사만 해도 char ROUGE-1이
63점 나옵니다. **항상 lead-N 베이스라인과의 차이를 보세요** — `evaluate.py`가 함께
출력하고, `scripts/report_predictions.py`가 출처별 차이와 신뢰구간까지 냅니다.
점수가 의심스러우면 `scripts/check_leakage.py`로 누수부터 확인하세요. 실제로 통합 전
평가 세트의 37.6%가 학습 세트와 겹쳐 있었습니다(현재 세트는 0건).
위 [학습 결과](#학습-결과)와
[docs/WORKLOG.md §11](docs/WORKLOG.md#11-평가-점수-검증과-누수-제거-2026-08-16)을 보세요.

**라이선스.** EXAONE-3.5 가중치는 `EXAONE AI Model License Agreement 1.1 - NC`로
**비상업적 연구 목적**에 제한되며, 이 제약은 학습한 LoRA 어댑터와 병합 모델 같은
파생물에도 이어집니다. 상업적 활용을 검토한다면 LG AI Research의 라이선스 원문을
직접 확인하세요. 저장소의 코드 자체는 자유롭게 사용하세요.
