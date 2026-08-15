# 사용 가이드

설치부터 학습·추론·평가·병합·트러블슈팅까지의 상세 안내입니다.

- 시스템 구조는 [ARCHITECTURE.md](ARCHITECTURE.md)
- 작업 기록과 설계 근거는 [WORKLOG.md](WORKLOG.md)

---

## 1. 요구사항

- Python 3.10+
- NVIDIA GPU
  - **QLoRA (기본)**: VRAM 16GB 이상
  - **bf16 LoRA**: VRAM 40GB 이상
- 디스크 약 20GB (모델 가중치 캐시 + 체크포인트)
- 어댑터 병합 시 시스템 RAM 약 20GB

### RTX 50xx (Blackwell) 사용자

`sm_120`은 **CUDA 12.8 이상으로 빌드된 PyTorch**가 필요합니다. 구버전 휠은
`no kernel image is available for execution on the device`로 죽습니다.

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

`bitsandbytes`도 4-bit 커널이 `sm_120`을 커버해야 하므로 **0.46 이상**을 쓰세요.
`python scripts\check_env.py`가 실제 4-bit 커널을 한 번 돌려서 확인해 줍니다.

---

## 2. 설치

```powershell
.\scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
```

`setup.ps1`은 `.venv` 생성 → PyTorch(cu128) → `requirements.txt` → editable 설치 →
환경 점검까지 수행합니다. 다른 CUDA 빌드가 필요하면 `-CudaTag cu126`처럼 넘기세요.

### 전용 venv를 쓰는 이유

EXAONE-3.5는 `trust_remote_code` 기반 커스텀 모델링 코드를 쓰고, 그 코드는
transformers 4.4x API를 전제로 작성돼 있습니다. 그래서 `requirements.txt`가
`transformers==4.48.3`으로 고정돼 있습니다. transformers 5.x가 깔린 환경에 그대로
섞으면 로딩 단계에서 깨지고, 반대로 전역을 다운그레이드하면 다른 작업이 깨집니다.

### 수동 설치

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e . --no-deps
python scripts\check_env.py
```

### 모델 접근 권한

EXAONE-3.5는 HuggingFace에서 라이선스 동의가 필요한 모델입니다. 모델 페이지에서
약관에 동의한 뒤 로그인하세요.

```powershell
huggingface-cli login
```

---

## 3. 빠른 시작 (스모크 테스트)

번들 샘플 데이터로 4 스텝만 돌려서 파이프라인이 끝까지 도는지 확인합니다.
모델 가중치(약 16GB) 최초 다운로드 시간은 별도입니다.

```powershell
python scripts\prepare_data.py --from-sample
python -m exaone_summarize.train -c configs\smoke.yaml
python -m exaone_summarize.infer -c configs\smoke.yaml `
    --adapter outputs\smoke\adapter `
    --input-jsonl data\sample\validation.jsonl `
    --output-jsonl outputs\smoke\preds.jsonl
python -m exaone_summarize.evaluate --predictions outputs\smoke\preds.jsonl --tokenizer char
```

전체 파이프라인을 한 번에:

```powershell
.\scripts\run_pipeline.ps1 -Config configs\smoke.yaml -Sample
```

`run_pipeline.ps1` 주요 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `-Config` | `configs\qlora_7.8b.yaml` | 설정 파일 |
| `-Sample` | off | HF 다운로드 없이 번들 샘플 사용 |
| `-HfDataset` | `daekeun-ml/naver-news-summarization-ko` | HF 데이터셋 |
| `-MaxTrain` | 0 (전체) | 학습 데이터 상한 |
| `-EvalLimit` | 200 | 평가 샘플 수 |
| `-RougeTokenizer` | `char` | ROUGE 분절기 |
| `-SkipData` / `-SkipTrain` / `-SkipEval` | off | 단계 생략 |

---

## 4. 데이터 준비

학습 데이터 형식은 JSONL 한 줄에 객체 하나입니다.

```json
{"document": "요약할 본문 전체...", "summary": "정답 요약문..."}
```

`scripts/prepare_data.py`가 정규화와 스플릿을 처리하고
`data/processed/{train,validation,test}.jsonl`에 씁니다.

```powershell
# 번들 샘플 (오프라인)
python scripts\prepare_data.py --from-sample

# HuggingFace 한국어 뉴스 요약 데이터셋
python scripts\prepare_data.py --hf-dataset daekeun-ml/naver-news-summarization-ko

# 학습 데이터를 2만건으로 제한
python scripts\prepare_data.py --hf-dataset daekeun-ml/naver-news-summarization-ko --max-train 20000

# 내 로컬 파일 (jsonl / json / csv) — 컬럼명 매핑
python scripts\prepare_data.py --input-file data\raw\mydata.csv `
    --document-column body --summary-column abstract
```

### 자동 필터

| 조건 | 기본값 | 이유 |
|---|---|---|
| 본문이 너무 짧음 | `--min-doc-chars 100` | 요약할 내용이 없는 샘플 |
| 요약이 너무 짧음 | `--min-summary-chars 10` | 빈 라벨 |
| **요약 길이 ≥ 본문 길이** | 항상 | 라벨이 뒤바뀐 데이터일 가능성이 높음 |

그 외 옵션: `--val-ratio`, `--test-ratio`(스플릿이 없는 입력에 적용),
`--max-doc-chars`, `--seed`.

> 번들 샘플은 8건/3건짜리 **파이프라인 검증용**입니다. 실제 품질을 얻으려면
> 최소 수천 건 규모의 도메인 데이터가 필요합니다.

---

## 5. 학습

```powershell
# QLoRA (16GB GPU)
python -m exaone_summarize.train -c configs\qlora_7.8b.yaml

# bf16 LoRA (40GB+)
python -m exaone_summarize.train -c configs\lora_bf16_7.8b.yaml
```

설정은 YAML을 고쳐도 되고 CLI로 덮어써도 됩니다. `--set`은 필드의 **선언 타입**에
맞춰 값을 변환하고, 알 수 없는 키나 타입 불일치는 즉시 에러를 냅니다.

```powershell
python -m exaone_summarize.train -c configs\qlora_7.8b.yaml `
    --set train.num_train_epochs=2 `
    --set train.learning_rate=5e-5 `
    --set data.max_train_samples=5000 `
    --set lora.r=32 --set lora.lora_alpha=64

# 평가 비활성화
python -m exaone_summarize.train -c configs\qlora_7.8b.yaml --set data.eval_file=none

# 체크포인트에서 재개
python -m exaone_summarize.train -c configs\qlora_7.8b.yaml `
    --set train.resume_from_checkpoint=outputs\exaone-3.5-7.8b-summary-qlora\checkpoint-200
```

### 산출물

```
outputs/<run>/
├── adapter/              # LoRA 어댑터 (+ 토크나이저) — 추론에 이걸 쓰세요
├── checkpoint-*/         # 중간 체크포인트
├── run_config.json       # 실제로 사용된 전체 설정 (--set 반영 후)
├── train_results.json
└── eval_results.json
```

### VRAM이 부족할 때 (조절 순서)

1. `data.max_seq_len` 축소 (3072 → 2048 → 1536) — **효과가 가장 큼**
2. `train.gradient_accumulation_steps` 늘리고 `per_device_train_batch_size=1` 유지
3. `lora.r` 축소 (16 → 8)
4. `model.gradient_checkpointing: true` 확인 (기본 켜짐)

반대로 VRAM이 남으면 `per_device_train_batch_size`를 먼저 올리는 게 가장 빠릅니다.

### 하이퍼파라미터 감각

| 설정 | 권장 |
|---|---|
| `learning_rate` | LoRA는 1e-4 근방. 3e-4 이상은 요약이 장황해지거나 붕괴 위험 |
| `lora.r` / `lora_alpha` | `alpha = 2 * r` 유지. 요약은 r=16으로 충분한 경우가 많음 |
| `num_train_epochs` | 2~3. 데이터가 수천 건 규모면 3 이상에서 과적합(원문 복사) 시작 |
| 유효 배치 | 16~32 (`per_device_train_batch_size × gradient_accumulation_steps`) |
| `generation.do_sample` | 요약은 `false`(greedy). 샘플링은 사실 왜곡을 늘림 |

### `target_modules` 지정

기본값 `null`이면 모델의 `nn.Linear`를 스캔해 자동으로 채웁니다. 명시하려면:

```yaml
lora:
  target_modules: [q_proj, k_proj, v_proj, out_proj, c_fc_0, c_fc_1, c_proj]
```

EXAONE-3.5는 Llama 계열과 모듈 이름이 달라서, PEFT 기본값(`o_proj`, `gate_proj` …)을
그대로 쓰면 어댑터가 거의 붙지 않습니다. 상세는
[ARCHITECTURE.md §6](ARCHITECTURE.md#exaone-35-모듈-이름-자동-감지가-필요한-이유)을 보세요.

---

## 6. 추론

```powershell
# 단일 문서
python -m exaone_summarize.infer -c configs\qlora_7.8b.yaml `
    --adapter outputs\exaone-3.5-7.8b-summary-qlora\adapter `
    --text "요약할 문서 본문..."

# 텍스트 파일
python -m exaone_summarize.infer -c configs\qlora_7.8b.yaml `
    --adapter outputs\exaone-3.5-7.8b-summary-qlora\adapter `
    --input-file data\raw\article.txt

# JSONL 배치
python -m exaone_summarize.infer -c configs\qlora_7.8b.yaml `
    --adapter outputs\exaone-3.5-7.8b-summary-qlora\adapter `
    --input-jsonl data\processed\test.jsonl `
    --output-jsonl outputs\preds.jsonl `
    --batch-size 4
```

`--adapter`를 생략하면 파인튜닝 전 베이스 모델로 추론합니다. **개선 폭을 보려면
베이스와 어댑터를 각각 돌려 ROUGE를 비교하세요.**

---

## 7. 평가

```powershell
# 원본 데이터로 추론 + 평가를 한 번에
python -m exaone_summarize.evaluate -c configs\qlora_7.8b.yaml `
    --input-jsonl data\processed\test.jsonl `
    --adapter outputs\exaone-3.5-7.8b-summary-qlora\adapter `
    --tokenizer char `
    --save-predictions outputs\preds.jsonl `
    --output-json outputs\metrics.json

# 이미 만든 예측 파일만 채점 (torch·peft 불필요)
python -m exaone_summarize.evaluate --predictions outputs\preds.jsonl --tokenizer char
```

### 한국어 ROUGE 분절기 선택

`rouge-score`의 기본 토크나이저는 영문 기준이라 한국어 점수가 왜곡됩니다.
`--tokenizer`로 고르세요.

| 값 | 특징 |
|---|---|
| `word` | 정규식 어절 단위. 의존성 없음. 조사가 다르면 불일치 처리돼 점수가 낮게 나옴 |
| `char` | 음절 단위. 조사 변화에 관대해 **한국어에서 보통 가장 안정적** |
| `morph` | konlpy 형태소. 가장 정확하지만 `pip install konlpy` + JDK 필요 |

> 절대 점수는 분절기에 따라 크게 달라집니다. **같은 분절기로 측정한 값끼리만
> 비교하세요.** 점수는 방향 지표일 뿐이니 실제 요약문 몇 개는 직접 읽어보는 게 좋습니다.

---

## 8. 어댑터 병합 (서빙용)

vLLM / TGI 등으로 서빙하려면 어댑터를 베이스에 병합해 단일 모델로 만듭니다.

```powershell
python -m exaone_summarize.merge_lora `
    --adapter outputs\exaone-3.5-7.8b-summary-qlora\adapter `
    --output merged\exaone-3.5-7.8b-summary
```

> **병합은 반드시 비양자화(bf16) 상태에서** 해야 합니다. 4-bit 모델에 병합하면
> 양자화 오차가 가중치에 그대로 굳어 품질이 크게 떨어집니다. 기본값은 CPU에서
> 병합하므로 **시스템 RAM 약 20GB**가 필요합니다.

---

## 9. 테스트

스텁 토크나이저를 써서 모델 가중치 없이 마스킹·토큰 예산·설정 검증 로직을 확인합니다.

```powershell
pip install pytest
$env:PYTHONPATH="src"; python -m pytest tests -q
```

46개 중 37개는 GPU와 모델 가중치 없이 동작합니다.

---

## 10. 트러블슈팅

**`KeyError: 'exaone'` / remote code 로딩 실패**
transformers 버전 문제입니다. `pip install transformers==4.48.3`으로 맞추고,
`model.trust_remote_code: true`인지 확인하세요.

**`Target modules ... not found in the base model`**
`lora.target_modules`를 Llama 기준으로 적었을 때 발생합니다. `null`로 두어
자동 감지에 맡기거나 §5의 EXAONE 모듈 이름을 쓰세요.

**`no kernel image is available for execution on the device`**
PyTorch가 GPU 아키텍처를 지원하지 않습니다. RTX 50xx는 cu128+ 휠이 필요합니다(§1).

**`CUDA out of memory`**
§5의 "VRAM이 부족할 때" 순서를 따르세요. `max_seq_len`이 가장 효과적입니다.

**bitsandbytes 4-bit 커널 실행 실패**
`python scripts\check_env.py`로 확인하세요. 해결이 안 되면
`configs/lora_bf16_7.8b.yaml` 경로(양자화 없음)를 쓰되 VRAM 40GB+가 필요합니다.

**`save_steps는 eval_steps의 배수여야 합니다`**
`load_best_model_at_end: true`의 제약입니다. 두 값을 맞추거나
`--set train.load_best_model_at_end=false`로 끄세요. 학습 시작 직후에 잡히도록
의도적으로 미리 검증합니다.

**`slice indices must be integers`**
`--set`으로 넘긴 값의 타입 문제입니다. 이 프로젝트는 선언 타입 기준으로 변환하므로
정상적으로는 발생하지 않습니다. 발생하면 오버라이드 키 이름을 확인하세요.

**요약이 원문을 그대로 복사함**
과적합입니다. 에폭 축소(2), `lora.r` 축소, 학습률 하향(5e-5)을 시도하세요.

**요약이 끝나지 않고 이어짐**
학습 데이터의 `summary` 끝에 EOS가 붙는지 확인하세요(인코더가 자동으로 붙입니다).
`generation.max_new_tokens`도 확인하세요.

**`.ps1` 실행 시 `The string is missing the terminator: "` 오류**
Windows PowerShell 5.1은 BOM이 없는 `.ps1`을 시스템 ANSI 코드페이지로 읽기 때문에,
한글 주석·메시지가 있으면 문자열이 깨져 파싱에 실패합니다. 이 저장소의 `.ps1`은
**UTF-8 with BOM**으로 저장돼 있습니다. 에디터에서 수정한 뒤 이 오류가 나면 BOM이
사라진 것이니 다시 넣으세요.

```powershell
$p = (Resolve-Path scripts\run_pipeline.ps1).Path
$text = [System.IO.File]::ReadAllText($p, (New-Object System.Text.UTF8Encoding $false))
[System.IO.File]::WriteAllText($p, $text, (New-Object System.Text.UTF8Encoding $true))
```

---

## 11. 라이선스

- **이 저장소의 코드**: 자유롭게 사용하세요.
- **EXAONE-3.5 모델 가중치**: `EXAONE AI Model License Agreement 1.1 - NC`가 적용됩니다.
  **비상업적 연구 목적**으로 제한되며, 파생 모델(학습한 LoRA 어댑터와 병합 모델
  포함)에도 동일한 제약이 이어집니다. 상업적 활용을 검토한다면 LG AI Research의
  라이선스 원문을 반드시 직접 확인하세요.
- 학습에 사용하는 데이터셋의 라이선스는 별도로 확인해야 합니다.
