# EXAONE-3.5-7.8B-Instruct 문서 요약 LoRA 파인튜닝

[LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct)를
LoRA / QLoRA로 파인튜닝해 **한국어 문서 요약**에 특화시키는 프로젝트입니다.

데이터 준비 → 학습 → 추론 → ROUGE 평가 → 어댑터 병합까지 한 흐름으로 묶여 있고,
설정은 YAML 한 곳에서 관리합니다.

## 문서

| 문서 | 내용 |
|---|---|
| **[docs/USAGE.md](docs/USAGE.md)** | 설치 · 데이터 준비 · 학습 · 추론 · 평가 · 병합 · 트러블슈팅 |
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
| 평가 | ROUGE-1/2/L, 한국어용 분절기 3종 (`word` / `char` / `morph`) + lead-N 베이스라인 · 출처별 분해 |
| 테스트 | 72개 — 모델 가중치 없이 마스킹·예산·설정·중복판정 검증 |

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

**transformers는 4.48.3으로 고정했습니다.** EXAONE-3.5의 `trust_remote_code` 코드가
4.4x API를 전제하므로, 전용 `.venv`에서 작업합니다.

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

## 프로젝트 구조

```
├── configs/                  qlora_7.8b · lora_bf16_7.8b · smoke
├── data/sample/              번들 샘플 8건/3건 (오프라인 검증용)
├── docs/                     USAGE · ARCHITECTURE · WORKLOG
├── scripts/                  setup.ps1 · check_env.py · prepare_data.py · prepare_aihub.py
│                             merge_datasets.py · check_leakage.py · run_pipeline.ps1
├── src/exaone_summarize/
│   ├── config.py             설정 스키마 · --set 오버라이드 · 정합성 검증
│   ├── prompt.py             chat template 구성 · 문서 토큰 절단
│   ├── jsonl.py              JSONL 입출력 (torch 비의존)
│   ├── dedup.py              완전/근사 중복 판정 (누수 차단)
│   ├── data.py               completion-only 마스킹 · 동적 패딩 콜레이터
│   ├── modeling.py           4bit 양자화 · target_modules 자동 감지 · LoRA 부착
│   ├── train.py              학습 진입점
│   ├── infer.py              요약 생성
│   ├── evaluate.py           ROUGE 평가
│   └── merge_lora.py         어댑터 병합
└── tests/                    72개 (모델 다운로드 불필요)
```

```powershell
$env:PYTHONPATH="src"; python -m pytest tests -q
```

---

## 알아둘 점

**ROUGE 절대값을 믿지 마세요.** 한국어 요약 데이터는 정답 요약이 본문 복붙에
가까운 경우가 많아(실측 84.8%), 본문 앞 세 문장을 복사만 해도 char ROUGE-1이
63점 나옵니다. `evaluate.py`가 **lead-N 베이스라인**을 함께 출력하니 그 차이를
보세요. 점수가 의심스러우면 `scripts/check_leakage.py`로 누수부터 확인하세요 —
실제로 통합 전 평가 세트의 37.6%가 학습 세트와 겹쳐 있었습니다.
자세한 내용은 [docs/WORKLOG.md §11](docs/WORKLOG.md#11-평가-점수-검증과-누수-제거-2026-08-16).

**라이선스.** EXAONE-3.5 가중치는 `EXAONE AI Model License Agreement 1.1 - NC`로
**비상업적 연구 목적**에 제한되며, 이 제약은 학습한 LoRA 어댑터와 병합 모델 같은
파생물에도 이어집니다. 상업적 활용을 검토한다면 LG AI Research의 라이선스 원문을
직접 확인하세요. 저장소의 코드 자체는 자유롭게 사용하세요.
