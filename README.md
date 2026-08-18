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
| 평가 | ROUGE-1/2/L, 한국어용 분절기 3종 (`word` / `char` / `morph`) + **베이스 zero-shot · lead-N** 베이스라인 · 출처별 분해 · paired bootstrap CI |
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

`outputs/exaone-3.5-7.8b-summary-qlora` 실행 기록입니다(이 저장소의 실제 산출물은
`outputs/exaone-3.5-7.8b-summary-qlora_task2/`, 평가 결과는 [§7 재현](#7-재현) 표 참조).

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

테스트 세트 1,977건 중 **앞 200건**, ROUGE-1/2/L F1 (%), `word` 분절, batch 4 greedy.
네 가지를 같은 문서·같은 생성 설정으로 비교합니다.

| 시스템 | 내용 |
|---|---|
| **lead-3** | 본문 앞 세 문장 복사 — 학습이 필요 없는 하한선 |
| **base-0shot** | 파인튜닝 전 베이스 모델. 어댑터와 **완전히 같은** 프롬프트·생성 설정 |
| **base-prompted** | 파인튜닝 전 베이스 모델. 프롬프트로 "2~3문장·150자 내외·요약문만" 지시 (`configs/zeroshot_prompted.yaml`) |
| **qlora** | 학습한 어댑터 |

> **이 표는 remote code 패치 후 재측정한 값입니다.** 캐시된 `modeling_exaone.py`가
> attention mask를 버리고 있어서, **left padding을 쓰는 배치 생성**이 조용히
> 망가져 있었습니다(§6-6, [WORKLOG §12](docs/WORKLOG.md)). 어댑터는 영향이
> 작았지만(R-1 39.35 → 39.09) 베이스 모델은 요약이 반복 루프로 붕괴해 R-1 8.83까지
> 떨어졌습니다. 평가 전에 `python scripts\patch_remote_code.py --check`를 돌리세요.

**ROUGE-1 F1 (%)** — 각 행의 최고값을 굵게 표시했습니다.

| 데이터 | n | lead-3 | base-0shot | base-prompted | qlora |
|---|---:|---:|---:|---:|---:|
| 전체 | 200 | 38.86 | 18.72 | 19.89 | **39.09** |
| aihub_editorial | 49 | 19.61 | 11.49 | 14.06 | **24.89** |
| aihub_law | 47 | **44.05** | 18.04 | 17.17 | 42.39 |
| aihub_news | 53 | **42.17** | 19.32 | 21.51 | 35.46 |
| naver_news | 51 | 49.15 | 25.65 | 26.31 | **53.48** |

전체 200건 기준 **R-2 / R-L**: lead-3 `29.50 / 34.04` · base-0shot `7.32 / 14.64` ·
base-prompted `7.48 / 15.79` · qlora `28.01 / 35.58`.
출처별 R-2/R-L은 `outputs/*/metrics.json`에 있습니다.

**ΔR-1과 95% 신뢰구간** — 문서 단위 paired bootstrap(10,000회, seed 0).
CI가 0을 포함하면 그 차이는 표본 오차와 구별되지 않습니다.

| 데이터 | vs base-0shot | 95% CI | vs base-prompted | 95% CI | vs lead-3 | 95% CI |
|---|---:|---|---:|---|---:|---|
| 전체 | **+20.38** | [+18.07, +22.75] | **+19.20** | [+16.85, +21.66] | +0.23 | [−2.21, +2.74] |
| aihub_editorial | **+13.40** | [+9.48, +17.52] | **+10.83** | [+7.02, +14.78] | **+5.28** | [+1.14, +9.36] |
| aihub_law | **+24.35** | [+19.19, +29.55] | **+25.21** | [+19.61, +30.81] | −1.66 | [−6.81, +3.55] |
| aihub_news | **+16.14** | [+12.31, +20.11] | **+13.95** | [+10.18, +17.61] | **−6.71** | [−10.60, −2.99] |
| naver_news | **+27.83** | [+23.43, +32.18] | **+27.17** | [+23.05, +31.13] | +4.34 | [−1.45, +9.95] |

**파인튜닝 효과는 분명합니다.** 베이스 zero-shot 대비 전체 +20.38, 네 도메인 모두
신뢰구간이 0 밖입니다. 이 격차는 **길이 아티팩트가 아닙니다** — 베이스는 같은
프롬프트에서 정답보다 2.12배 길게 쓰는데(§5), 프롬프트로 길이를 지시해 1.29배까지
줄여도 격차는 +19.20으로 거의 그대로입니다.

**반면 lead-3(복사) 대비는 도메인별로 갈립니다.** 전체 +0.23은 서로 반대 방향인
도메인이 상쇄된 값입니다. 사설에서는 유의하게 낫고(+5.28), AI Hub 신문기사에서는
유의하게 나쁩니다(−6.71). naver 뉴스 +4.34와 법률 −1.66은 이 표본 크기에서 잡음과
구별되지 않습니다.

**두 결론이 충돌하는 게 아닙니다.** 어댑터가 배운 것의 큰 부분은 "정답 요약의
어휘·길이·문체에 맞추는 것"이고, 정답 요약 자체가 본문 어휘를 많이 재사용하는
도메인(법률·AI Hub 신문기사)에서는 그 상한이 복사 베이스라인입니다.

### 5. 추상성과 생성 안정성

요약의 **신규 4-gram 비율**(원문에 없는 4-gram의 비율)이 높을수록 복사가 아니라
재구성입니다. lead-3은 정의상 0%입니다.

**qlora** (신규 4-gram 비율 · 길이비 · 붕괴 지표)

| 데이터 | 예측 | 정답 | 길이비(예측/정답) | 빈 출력 | 문장 미완결 | 5-gram 반복 |
|---|---:|---:|---:|---:|---:|---:|
| 전체 | 42.9% | 62.7% | 1.22 | 0 | 0 | 14 / 200 |
| aihub_editorial | 53.4% | 82.9% | 1.21 | 0 | 0 | 0 / 49 |
| aihub_law | 43.3% | 59.1% | 1.31 | 0 | 0 | 8 / 47 |
| aihub_news | 47.7% | 70.1% | 1.23 | 0 | 0 | 1 / 53 |
| naver_news | 27.6% | 38.9% | 1.14 | 0 | 0 | 5 / 51 |

**베이스 zero-shot** (전체 200건)

| 시스템 | 신규 4-gram | 길이비 | 빈 출력 | 문장 미완결 | 5-gram 반복 |
|---|---:|---:|---:|---:|---:|
| base-0shot | 95.9% | 2.12 | 0 | 0 | 4 / 200 |
| base-prompted | 97.0% | 1.29 | 0 | 0 | 0 / 200 |
| qlora | 42.9% | 1.22 | 0 | 0 | 14 / 200 |

- **모델은 복사기가 아닙니다** (42.9% vs lead-3의 0%). 다만 정답 요약(62.7%)보다는
  원문 표현에 더 의존합니다.
- **베이스 모델은 반대쪽 극단입니다** (95~97%). 원문 표현을 거의 쓰지 않고 새로
  씁니다. 요약을 못 하는 게 아니라 **정답과 다른 어휘로** 요약하며, ROUGE는 그걸
  구분하지 못하고 감점만 합니다. 어댑터가 얻은 +20점의 상당 부분이 이 정렬입니다.
- lead-3에 지는 두 도메인(법률·AI Hub 신문기사)은 정답 요약의 어휘가 기사 앞부분과
  많이 겹치는 쪽이라, **ROUGE 관점에서는 "복사"가 정답에 가깝습니다.** 모델이 더
  나쁘다는 증거가 아니라, ROUGE로 가치를 증명할 수 없는 구간이라는 뜻입니다.
- 생성 붕괴는 없습니다(빈 출력 0, 문장 중간 잘림 0). 단 **7%에서 같은 5-gram이
  반복**되고(법률 8건이 대부분), 출력이 정답보다 22% 깁니다. ROUGE-1 F1은 길이
  초과를 감점하므로 길이만 맞춰도 점수가 오릅니다.

**ΔR-1이 +60인 샘플이 실제로 어떻게 다른지** (naver_news, 발췌):

```
정답          7월 서울 강동구에서 오피스텔 '디유니크 강동 투웨니퍼스트'가 분양을
              앞두고 있고, 이 단지는 ... 지하 3층 지상 20층 전용 28 84m2의 오피스텔
              63실과 라이브 오피스 7실 근린생활시설 등이 함께 조성되며 ...

base-prompted 강동구 길동에 위치한 '디유니크 강동 투웨니퍼스트' 오피스텔은 63실의
              다양한 크기의 오피스텔과 ... 더블역세권과 풍부한 생활 인프라를 갖춘
              이 단지는 1·2인 가구 수요에 맞춘 특화 설계와 하이엔드 컨시어지 ...

qlora         7월 서울 강동구에서 오피스텔 '디유니크 강동 투웨니퍼스트'가 분양을
              앞두고 있는데, 이 단지는 지하 3층 지상 20층 전용 28 84m2의 오피스텔
              63실과 라이브 오피스 7실 근린생활시설 등이 함께 조성되며 ...
```

베이스 요약도 **틀린 요약이 아닙니다.** 다르게 고른 정보를 다른 어휘로 씁니다.
어댑터는 정답의 어휘·순서·문체를 따라가고, ROUGE는 그것만 보상합니다. 이게 +19~20점의
정체이고, 동시에 그 숫자로 "더 좋은 요약"이라고 말할 수 없는 이유입니다.

### 6. 한계

1. **표본 크기.** 200건으로는 도메인당 n≈50이라 CI가 ±5~10점입니다. 1,977건 전체로
   재평가해야 도메인별 결론이 확정됩니다.
2. **베이스 zero-shot ablation은 측정했지만, ROUGE로는 절반만 말할 수 있습니다.**
   어댑터는 두 zero-shot 베이스라인을 크게 앞섭니다(+20.38 / +19.20 R-1, 네 도메인
   모두 CI가 0 밖). 길이를 맞춰도 격차가 유지되므로 형식 아티팩트도 아닙니다.
   다만 베이스 모델의 요약은 신규 4-gram이 97%라 정답과 어휘가 거의 겹치지 않습니다.
   **격차의 상당 부분은 "정답 스타일과의 정렬"이고, 내용 충실도까지 앞선다는 증거는
   아닙니다**(3번과 같은 이유). 두 요약을 사람이 읽고 고르는 평가가 필요합니다.
3. **사람 평가 없음.** ROUGE는 사실 왜곡을 잡지 못합니다. 실제로 금리 *동결* 기사에
   "금리 인상을 결정할 것으로 보인다"가, 반도체 수출 기사에 원문에 없는 수요 원인
   분석이 붙는 사례를 확인했습니다.
4. **단일 실행.** seed 1개, 1에폭. 하이퍼파라미터 탐색과 분산 측정을 하지 않았습니다.
5. ROUGE 절대값은 분절기에 따라 10점 이상 움직입니다. `word` 기준끼리만 비교하세요.
6. **remote code 의존.** 위 수치는 캐시된 `modeling_exaone.py`를
   `scripts/patch_remote_code.py`로 고친 상태에서 측정했습니다. 모델 캐시를 지우고
   다시 받으면 원래 코드(= transformers 5.15에서 마스크 생성 실패)로 돌아가므로,
   **평가 전에 `--check`를 돌려야 합니다.** 학습은 right padding이라 영향이 작지만
   배치 생성은 크게 망가집니다([WORKLOG §12](docs/WORKLOG.md)).

### 7. 재현

```powershell
# 0. remote code 상태 확인 — 이걸 빼먹으면 배치 생성이 망가진 값이 나옵니다 (§6-6)
python scripts\patch_remote_code.py --check    # 종료코드 1이면: 옵션 없이 다시 실행

# 1. 어댑터 추론 + 평가 (표 4의 qlora 열)
python -m exaone_summarize.evaluate -c configs\qlora_7.8b.yaml `
    --input-jsonl data\processed\test.jsonl `
    --adapter outputs\exaone-3.5-7.8b-summary-qlora\adapter `
    --limit 200 --batch-size 4 `
    --save-predictions outputs\qlora-eval-fixed\predictions.jsonl `
    --output-json outputs\qlora-eval-fixed\metrics.json

# 2. 베이스 zero-shot — 같은 프롬프트 (표 4의 base-0shot 열). --adapter 를 빼면 됩니다
python -m exaone_summarize.evaluate -c configs\qlora_7.8b.yaml `
    --input-jsonl data\processed\test.jsonl --limit 200 --batch-size 4 `
    --save-predictions outputs\baseline-zeroshot\predictions.jsonl `
    --output-json outputs\baseline-zeroshot\metrics.json

# 3. 베이스 zero-shot — 길이·형식을 지시한 프롬프트 (표 4의 base-prompted 열)
python -m exaone_summarize.evaluate -c configs\zeroshot_prompted.yaml `
    --input-jsonl data\processed\test.jsonl --limit 200 --batch-size 4 `
    --save-predictions outputs\baseline-zeroshot-prompted\predictions.jsonl `
    --output-json outputs\baseline-zeroshot-prompted\metrics.json

# 4. 어댑터 vs 베이스 — 같은 문서끼리 짝지어 ΔR-1 + 신뢰구간 (표 4의 Δ 표)
python scripts\compare_runs.py `
    --a outputs\baseline-zeroshot\predictions.jsonl --a-label base-0shot `
    --b outputs\qlora-eval-fixed\predictions.jsonl --b-label qlora --markdown

# 5. 출처별 lead-3 · 신뢰구간 · 추상성 (표 4의 lead-3 열, 표 5)
python scripts\report_predictions.py `
    --predictions outputs\qlora-eval-fixed\predictions.jsonl --markdown

# 6. 누수 검사 (표 2)
python scripts\check_leakage.py --train data\processed\train.jsonl `
    --eval data\processed\test.jsonl --eval data\processed\validation.jsonl

# 위 1번 한계를 해소하려면 --limit 을 빼세요 (1,977건, 시스템당 GPU 약 45분)
```

이 저장소에 들어 있는 실제 산출물 경로:

| 표 | 파일 |
|---|---|
| qlora | `outputs/qlora-eval-fixed/{predictions.jsonl,metrics.json,report.json}` |
| base-0shot | `outputs/baseline-zeroshot/{predictions.jsonl,metrics.json}` |
| base-prompted | `outputs/baseline-zeroshot-prompted/{predictions.jsonl,metrics.json}` |
| Δ 표 | `outputs/qlora-eval-fixed/comparison.json`, `outputs/baseline-zeroshot-prompted/comparison.json` |
| 학습 체크포인트·어댑터 | `outputs/exaone-3.5-7.8b-summary-qlora_task2/` |
| 마스크 버그 상태의 옛 측정 (참고용) | `outputs/baseline-zeroshot-brokenmask/` |

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
│                             zeroshot_prompted (베이스 zero-shot 베이스라인용 프롬프트)
├── data/sample/              번들 샘플 8건/3건 (오프라인 검증용)
├── docs/                     USAGE · ARCHITECTURE · WORKLOG
├── scripts/                  setup.ps1 · check_env.py · prepare_data.py · prepare_aihub.py
│                             merge_datasets.py · check_leakage.py · run_pipeline.ps1
│                             report_predictions.py (출처별 · 신뢰구간 리포트)
│                             compare_runs.py (두 실행 비교 — 어댑터 vs 베이스)
│                             patch_remote_code.py (EXAONE remote code 마스크 수정)
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

**베이스라인 없는 점수는 해석할 수 없습니다.** lead-N은 하한선일 뿐이라, 파인튜닝의
효과는 **같은 프롬프트·같은 설정으로 돌린 베이스 모델**과 비교해야 나옵니다
(`scripts/compare_runs.py`). 이 저장소 실측으로는 어댑터가 베이스 zero-shot 대비
+20.4 R-1인데 lead-3 대비로는 +0.2뿐입니다 — 같은 모델에 대한 두 문장이 모두 사실입니다.

**배치 생성이 조용히 망가질 수 있습니다.** EXAONE remote code가 transformers 5.15에서
attention mask를 만들지 못하면 패딩 토큰까지 문맥으로 읽습니다. 학습(right padding)은
거의 무해하지만 생성(left padding)은 붕괴합니다. `python scripts\check_env.py` 또는
`python scripts\patch_remote_code.py --check`로 상태를 확인하세요
([WORKLOG §12](docs/WORKLOG.md)).

**라이선스.** EXAONE-3.5 가중치는 `EXAONE AI Model License Agreement 1.1 - NC`로
**비상업적 연구 목적**에 제한되며, 이 제약은 학습한 LoRA 어댑터와 병합 모델 같은
파생물에도 이어집니다. 상업적 활용을 검토한다면 LG AI Research의 라이선스 원문을
직접 확인하세요. 저장소의 코드 자체는 자유롭게 사용하세요.
