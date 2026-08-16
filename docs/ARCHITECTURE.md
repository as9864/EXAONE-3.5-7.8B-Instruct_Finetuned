# 시스템 구조

EXAONE-3.5-7.8B-Instruct 문서 요약 LoRA 파인튜닝 프로젝트의 상세 구조 문서입니다.

- 사용법은 [USAGE.md](USAGE.md)
- 만들면서 수행한 작업과 설계 근거는 [WORKLOG.md](WORKLOG.md)

---

## 1. 전체 구성도

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 입력                                                                     │
│   HuggingFace 데이터셋  │  로컬 jsonl/json/csv  │  data/sample (번들)     │
│   AI Hub 문서요약 zip (data/AIHUB_DocSummaryData)                        │
└────────────┬────────────────────┬────────────────────────┬───────────────┘
             └────────────────────┴────────────────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │  scripts/prepare_data.py   │  정규화 · 필터 · 스플릿
                    │  scripts/prepare_aihub.py  │  zip 스트리밍 파싱
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │ scripts/merge_datasets.py  │  중복 제거 · 샘플링 · 누수 차단
                    └─────────────┬──────────────┘
                                  │
                    data/processed/{train,validation,test}.jsonl
                        {"document": "...", "summary": "..."}
                                  │
   ┌──────────────────────────────┼──────────────────────────────┐
   │                              │                              │
   ▼                              ▼                              ▼
┌────────────────┐   ┌─────────────────────────┐   ┌──────────────────────┐
│ configs/*.yaml │──▶│  train.py               │   │  evaluate.py         │
│  + --set       │   │   data.py   (인코딩)     │   │   infer.py (생성)     │
│  = config.py   │   │   modeling.py (LoRA)    │   │   ROUGE-1/2/L        │
└────────────────┘   │   HF Trainer            │   └──────────┬───────────┘
                     └────────────┬────────────┘              │
                                  │                           │
                    outputs/<run>/adapter/  ──────────────────▶│
                    (LoRA 어댑터 + 토크나이저)                   │
                                  │                           ▼
                    ┌─────────────▼──────────────┐   outputs/<run>/
                    │  merge_lora.py (bf16/CPU)  │     ├ predictions.jsonl
                    └─────────────┬──────────────┘     └ metrics.json
                                  │
                    merged/<name>/  (vLLM · TGI 서빙용 단일 모델)
```

---

## 2. 디렉터리 구조와 책임

```
.
├── configs/                     설정 프로파일 (코드 수정 없이 실험 전환)
│   ├── qlora_7.8b.yaml          기본: 4-bit QLoRA, 16GB GPU 타깃
│   ├── lora_bf16_7.8b.yaml      비양자화 bf16 LoRA, 40GB+ 타깃
│   └── smoke.yaml               4스텝 파이프라인 검증용
│
├── data/
│   ├── sample/                  번들 샘플 8건/3건 (커밋됨, 오프라인 검증)
│   ├── raw/                     원본 입력 (gitignore)
│   ├── AIHUB_DocSummaryData/    AI Hub 배포 zip 원본 (gitignore)
│   └── processed/               학습에 쓰는 JSONL (gitignore)
│       ├── {train,validation,test}.jsonl   최종 병합 결과
│       ├── naver_news/          HF 데이터셋 변환본
│       └── aihub/               AI Hub 도메인별 변환본
│
├── docs/                        이 문서들
│
├── scripts/                     사람이 직접 실행하는 진입점
│   ├── setup.ps1                venv + cu128 torch + 의존성 + 점검
│   ├── check_env.py             CUDA / sm_120 / bitsandbytes 4bit 커널 실측
│   ├── prepare_data.py          데이터 정규화 및 스플릿
│   ├── prepare_aihub.py         AI Hub 문서요약 zip → 도메인별 JSONL
│   ├── merge_datasets.py        여러 JSONL 병합 (중복 제거 · 샘플링 · 누수 차단)
│   ├── check_leakage.py         학습/평가 세트 누수 검사
│   └── run_pipeline.ps1         데이터 → 학습 → 평가 일괄
│
├── src/exaone_summarize/        라이브러리 + CLI 모듈
└── tests/                       모델 가중치 없이 도는 단위 테스트
```

### 모듈별 책임

| 모듈 | 줄수 | 책임 | 핵심 심볼 |
|---|---:|---|---|
| `config.py` | 245 | 설정 스키마·YAML 로딩·`--set` 오버라이드·정합성 검증 | `Config`, `load_config`, `apply_overrides`, `validate` |
| `prompt.py` | 47 | chat template 메시지 구성, 문서 토큰 단위 절단 | `build_messages`, `render_prompt`, `truncate_document` |
| `jsonl.py` | 33 | JSONL 입출력 (stdlib만 사용) | `read_jsonl`, `write_jsonl` |
| `dedup.py` | 121 | 완전/근사 중복 판정 — shingle 역색인, 누수 차단 | `exact_key`, `shingles`, `ShingleIndex` |
| `data.py` | 141 | completion-only 마스킹 인코딩, 동적 패딩 콜레이터 | `SummarizationEncoder`, `DataCollatorForCausalSummarization`, `build_dataset` |
| `modeling.py` | 130 | 모델/토크나이저 로딩, 4bit 양자화, LoRA 부착 | `load_for_training`, `load_for_inference`, `find_linear_module_names` |
| `train.py` | 120 | 학습 오케스트레이션, `TrainingArguments` 구성 | `main`, `build_training_args` |
| `infer.py` | 108 | 배치 요약 생성 (left padding) | `summarize_batch` |
| `evaluate.py` | 190 | ROUGE-1/2/L, 한국어 분절기 3종, lead-N 베이스라인, 출처별 분해 | `compute_rouge`, `lead_sentences`, `WordTokenizer`, `CharTokenizer`, `MorphTokenizer` |
| `merge_lora.py` | 50 | 어댑터를 bf16 베이스에 병합 | `main` |

---

## 3. 의존 계층

무거운 의존성을 아래쪽으로 몰아서, 가벼운 작업이 무거운 임포트를 물지 않게 했습니다.

```
계층 0 — stdlib만
    jsonl.py, dedup.py

계층 1 — yaml + (transformers는 TYPE_CHECKING 전용)
    prompt.py ◀── config.py
    ▸ 모델 없이 설정·프롬프트 로직 테스트 가능

계층 2 — torch · datasets · transformers
    data.py ◀── config.py, jsonl.py, prompt.py

계층 3 — + peft · bitsandbytes
    modeling.py ◀── config.py

계층 4 — CLI 진입점
    train.py      ◀── config, data, modeling          (계층 3 필요)
    infer.py      ◀── config, jsonl, modeling, prompt (계층 3 필요)
    merge_lora.py ◀── 독립                             (계층 3 필요)
    evaluate.py   ◀── config, jsonl  +  infer/modeling 은 *지연 임포트*
```

`evaluate.py`의 지연 임포트가 중요합니다. `--predictions`로 이미 만들어진 예측
파일만 채점할 때는 `torch`/`peft`를 전혀 건드리지 않습니다.

```python
# evaluate.py — 추론이 필요한 분기 안에서만 임포트
from .infer import summarize_batch
from .modeling import load_for_inference
```

덕분에 `peft`가 없는 환경에서도 `python -m exaone_summarize.evaluate --help`와
순수 채점이 동작하고, 테스트 72개 중 63개가 GPU·모델 없이 돕니다.

---

## 4. 학습 데이터 인코딩 (핵심)

### 4.1 completion-only 마스킹

프롬프트 토큰의 label을 `-100`(`IGNORE_INDEX`)으로 덮어서 **요약문 구간에만
loss가 걸리도록** 합니다. 이걸 안 하면 모델이 "문서를 그대로 다시 쓰는 법"을
같이 학습해 요약이 원문 복사로 붕괴합니다.

```
                apply_chat_template(add_generation_prompt=True)      summary + eos
        ┌──────────────────────────────────────────────────────┐ ┌────────────────┐
텍스트   [|system|]…[|endofturn|] [|user|]…[|endofturn|] [|assistant|] 요약문입니다. <eos>

input_ids   [ p₀  p₁  p₂  …  pₙ ][ t₀  t₁  …  tₘ ][ eos ]
labels      [-100 -100 -100 … -100][ t₀  t₁  …  tₘ ][ eos ]     ← 앞부분만 마스킹
attn_mask   [  1   1   1  …   1  ][  1   1  …   1 ][  1  ]
                    ↑                      ↑
              loss 계산 제외          loss 계산 대상
```

구현은 `data.py:SummarizationEncoder.__call__`입니다. 템플릿 문자열을 정규식으로
찾아 자르는 방식(`DataCollatorForCompletionOnlyLM` 등) 대신, **프롬프트와 타깃을
따로 토크나이즈해 길이로 경계를 정합니다.** 템플릿이 바뀌어도 경계가 어긋나지
않습니다.

```python
prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True,
                                           add_generation_prompt=True)
target_ids = tokenizer(summary, add_special_tokens=False)["input_ids"]
target_ids = target_ids[: max_target_tokens - 1] + [tokenizer.eos_token_id]

input_ids = prompt_ids + target_ids
labels    = [-100] * len(prompt_ids) + target_ids
```

EOS를 항상 붙이는 이유: 붙이지 않으면 모델이 요약을 끝내는 법을 배우지 못해
생성이 `max_new_tokens`까지 계속 이어집니다.

### 4.2 토큰 예산 배분

`max_seq_len`을 세 조각으로 나눠 쓰고, 문서만 절단합니다.

```
max_seq_len (예: 3072)
├── overhead          템플릿 자체 (system/user/assistant 마커) — 시작 시 1회 측정
├── document_budget   = max_seq_len - max_target_tokens - overhead
└── max_target_tokens 요약문 상한 (예: 512)
```

문서 절단은 **템플릿 적용 전에 본문만** 대상으로 합니다
(`prompt.py:truncate_document`). 템플릿 적용 후에 자르면 `[|assistant|]` 같은
마커가 잘려나가 프롬프트 구조가 깨집니다.

```
document ──▶ tokenize ──▶ ids[:document_budget] ──▶ decode ──▶ 절단된 document
                                                                    │
                                            apply_chat_template ◀───┘
```

요약 태스크에서는 문서 앞부분이 대체로 더 중요하므로 뒤쪽을 버립니다.

### 4.3 배치 콜레이트

가변 길이를 오른쪽 패딩하되, 세 텐서를 각각 다른 값으로 채웁니다.

| 텐서 | 패딩 값 | 이유 |
|---|---|---|
| `input_ids` | `pad_token_id` | 정상 토큰 |
| `labels` | `-100` | 패딩에 loss가 걸리면 안 됨 |
| `attention_mask` | `0` | 패딩을 어텐션에서 제외 |

`pad_to_multiple_of=8`로 텐서 코어 정렬을 맞춥니다.

```
샘플 A (길이 3) ▸ [1 2 3 P P P P P]   labels [-100 2 3 -100×5]   mask [1 1 1 0 0 0 0 0]
샘플 B (길이 2) ▸ [4 5 P P P P P P]   labels [-100 5 -100×6]     mask [1 1 0 0 0 0 0 0]
                                    └─ max_len 3 → 8의 배수로 정렬
```

---

## 5. 설정 해석 흐름

```
configs/*.yaml
     │  yaml.safe_load
     ▼
_build_section()  ── 알 수 없는 키/섹션 → 즉시 ValueError
     │
     ▼
Config(model, lora, data, train, generation)   dataclass 5개 섹션
     │
     │  --set section.key=value
     ▼
apply_overrides()
     │   _field_types()  = get_type_hints(dataclass)   ← 캐시
     │   _unwrap_optional()  X | None → (X, True)
     │   _coerce()  선언 타입 기준 변환 (bool/int/float/list/str/None)
     ▼
validate()   bf16⊕fp16 · max_seq_len>max_target_tokens · lora.r>0 ·
     │       load_best_model_at_end 제약 · paged optimizer 경고
     ▼
build_training_args() ──▶ transformers.TrainingArguments
     └─ run_config.json 으로 산출물 디렉터리에 기록
```

**타입 변환은 현재 값이 아니라 선언 타입을 봅니다.** 기본값이 `None`인 필드
(`data.max_train_samples: int | None`, `lora.target_modules: list[str] | None`)에
현재 값 기준으로 추론하면 문자열이 그대로 새어 들어갑니다. 실제로 이 버그를 만들고
테스트로 잡았습니다 — 경위는 [WORKLOG.md](WORKLOG.md) §5.1에 있습니다.

### 시작 시점 검증을 두는 이유

`load_best_model_at_end`는 Trainer 내부 제약이 많습니다. 몇 시간 학습한 뒤
저장 단계에서 터지면 손실이 크므로, `config.py:resolve_best_model_setting`이
시작 직후에 판정합니다.

| 상황 | 처리 |
|---|---|
| 평가 없음 (`eval_file: null`) | 경고 후 자동 비활성화 |
| `save_strategy: "no"` | 경고 후 자동 비활성화 |
| `eval_strategy` ≠ `save_strategy` | **ValueError** |
| `save_steps % eval_steps != 0` | **ValueError** |

---

## 6. 모델 로딩과 LoRA 부착

```
ModelConfig
     │
     ▼
BitsAndBytesConfig(load_in_4bit, nf4, bf16 compute, double_quant)
     │
     ▼
AutoModelForCausalLM.from_pretrained(trust_remote_code=True, device_map={"":0})
     │   ▸ EXAONE-3.5는 커스텀 모델링 코드를 원격에서 받아 실행
     │   ▸ use_cache=False (학습 시 gradient checkpointing과 충돌)
     ▼
prepare_model_for_kbit_training(use_gradient_checkpointing=True)
     │   ▸ LayerNorm을 fp32로 승격, 입력 grad 활성화
     ▼
find_linear_module_names(model)          ← target_modules 가 null 일 때만
     │   ▸ nn.Linear / Linear4bit / Linear8bitLt 스캔
     │   ▸ lm_head, wte, embed_tokens, embed_out 제외
     ▼
LoraConfig(r, alpha, dropout, task_type="CAUSAL_LM", target_modules=[...])
     │
     ▼
get_peft_model()  ──▶  print_trainable_parameters()
```

### EXAONE-3.5 모듈 이름 (자동 감지가 필요한 이유)

| 역할 | Llama 계열 | **EXAONE-3.5** |
|---|---|---|
| Query / Key / Value | `q_proj` `k_proj` `v_proj` | `q_proj` `k_proj` `v_proj` |
| 어텐션 출력 | `o_proj` | **`out_proj`** |
| MLP gate | `gate_proj` | **`c_fc_0`** |
| MLP up | `up_proj` | **`c_fc_1`** |
| MLP down | `down_proj` | **`c_proj`** |

PEFT의 Llama 기본값을 그대로 쓰면 5개 중 4개가 매칭되지 않아 어댑터가 거의
붙지 않습니다. 모델 그래프를 직접 스캔하면 모델 버전이 바뀌어도 따라갑니다.

### 학습 vs 추론 로딩 차이

| | 학습 (`load_for_training`) | 추론 (`load_for_inference`) |
|---|---|---|
| `use_cache` | `False` | `True` |
| `padding_side` | `right` | **`left`** |
| LoRA | `get_peft_model` (신규) | `PeftModel.from_pretrained` (기존) |
| gradient checkpointing | 활성화 | — |
| 토크나이저 출처 | 베이스 모델 | 어댑터 디렉터리 우선 |

`padding_side`가 갈리는 이유: causal LM 학습은 오른쪽 패딩이 맞지만, 배치 생성
시 오른쪽 패딩을 쓰면 패딩이 프롬프트와 생성 시작점 사이에 끼어 출력이 망가집니다.

---

## 7. 추론 · 평가 흐름

```
documents[]
     │
     │  document_budget = max_seq_len - max_new_tokens
     ▼
truncate_document ──▶ apply_chat_template(add_generation_prompt=True)
     │
     ▼
tokenizer(prompts, padding=True, add_special_tokens=False)   ← left padding
     │      ▸ 템플릿이 이미 특수 토큰을 포함하므로 중복 추가 방지
     ▼
model.generate(eos_token_id, pad_token_id, greedy 기본)
     │
     ▼
outputs[:, input_len:]  ──▶  프롬프트 구간을 잘라내고 생성분만 디코드
     │
     ▼
predictions.jsonl  {document, reference, prediction}
     │
     ▼
compute_rouge(predictions, references, tokenizer)
     │   rouge_scorer.RougeScorer(tokenizer=<분절기>)
     ▼
metrics.json  {rouge1, rouge2, rougeL, pred_len_mean, ref_len_mean, n_samples}
```

### 한국어 ROUGE 분절기

`rouge-score` 기본 토크나이저는 영문 전제라 한국어 점수가 왜곡됩니다.
`RougeScorer(tokenizer=...)` 훅에 직접 주입합니다.

| 분절기 | 구현 | 특징 |
|---|---|---|
| `word` | 정규식 `[가-힣]+|[a-zA-Z]+|[0-9]+` | 의존성 없음. 조사 차이를 불일치로 처리 |
| `char` | 공백 제외 음절 단위 | 조사 변화에 관대, 한국어에서 보통 가장 안정적 |
| `morph` | konlpy `Okt.morphs` | 가장 정확, JDK 필요 |

절대 점수는 분절기에 따라 크게 달라지므로 **같은 분절기끼리만 비교**해야 합니다.

---

## 8. 산출물 레이아웃

```
outputs/<run-name>/
├── adapter/                  ◀ 추론·병합에 사용
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer.json, tokenizer_config.json
├── checkpoint-100/           중간 체크포인트 (save_total_limit 만큼 유지)
├── checkpoint-200/
├── run_config.json           실제 적용된 전체 설정 (--set 반영 후)
├── train_results.json
├── eval_results.json
├── trainer_state.json
├── predictions.jsonl         run_pipeline.ps1 평가 단계 산출
└── metrics.json

merged/<name>/                merge_lora.py 산출 — 서빙용 단일 모델
```

`run_config.json`은 실험 재현의 기준점입니다. `--set`으로 덮어쓴 값까지 반영된
최종 설정이 기록되므로, 나중에 "이 체크포인트를 어떤 설정으로 뽑았는지"를
추적할 수 있습니다.

---

## 9. 테스트 구조

모델 가중치(약 16GB)를 내려받지 않고 핵심 로직을 검증합니다.

```
tests/
├── conftest.py         StubTokenizer — 문자 1개 = 토큰 1개, chat template 골격 모방
├── test_config.py  28  스키마 · YAML · --set 타입 변환 · validate 제약
├── test_data.py     9  마스킹 경계 · 토큰 예산 · 절단 · 콜레이트 패딩
└── test_evaluate.py 9  ROUGE 상·하한 · 분절기 · JSONL 왕복 · 샘플 데이터 정합성
                   ─────
                    46
```

`StubTokenizer`가 있어서 `transformers`의 실제 토크나이저 없이도 마스킹 경계와
토큰 예산 계산을 검증할 수 있습니다. 검증하는 성질의 예:

```python
# 마스킹은 앞쪽에 연속으로만 존재해야 한다 (중간에 다시 나오면 경계 계산 오류)
n_masked = sum(1 for label in out["labels"] if label == IGNORE_INDEX)
assert IGNORE_INDEX not in out["labels"][n_masked:]
assert out["labels"][n_masked:] == out["input_ids"][n_masked:]
assert out["labels"][-1] == tokenizer.eos_token_id
```

```powershell
$env:PYTHONPATH="src"; python -m pytest tests -q
```

---

## 10. 확장 포인트

| 하고 싶은 것 | 손댈 곳 |
|---|---|
| 프롬프트 문구 변경 | `configs/*.yaml`의 `data.system_prompt` / `data.user_template` |
| 다른 태스크(번역·QA)로 전환 | 위 템플릿 + `data.document_key` / `summary_key` |
| 요약 길이 제어 | `data.max_target_tokens`, `generation.max_new_tokens` |
| LoRA 부착 위치 한정 | `lora.target_modules`에 명시 (자동 감지 무시) |
| 임베딩·헤드까지 학습 | `lora.modules_to_save` |
| 다른 베이스 모델 | `model.model_name_or_path` — 모듈 이름은 자동 감지됨 |
| 평가 지표 추가 | `evaluate.py:compute_rouge` 옆에 함수 추가 |
| 새 데이터 소스 | `scripts/prepare_data.py`의 `_load_local` / `_load_hf` |
| 데이터 혼합 비율 조정 | `scripts/merge_datasets.py --input 경로:건수` (재변환 불필요) |
| 문서 앞이 아니라 뒤를 남기기 | `prompt.py:truncate_document`의 슬라이스 방향 |

---

## 11. 설계 제약 요약

| 제약 | 근거 |
|---|---|
| QLoRA 4-bit 기본 | 7.8B는 16GB VRAM에서 bf16 LoRA 불가 (가중치만 ~16GB) |
| `transformers==4.48.3` 고정 | EXAONE-3.5 remote code가 4.4x API 전제 |
| 전용 `.venv` 필수 | 전역 다운그레이드로 다른 작업을 깨뜨리지 않기 위해 |
| PyTorch cu128+ | RTX 50xx(sm_120) 커널 요구사항 |
| `bitsandbytes>=0.46` | sm_120 4-bit 커널 |
| 병합은 bf16 + CPU | 4-bit에 병합하면 양자화 오차가 가중치에 굳음, RAM ~20GB 필요 |
| `.ps1`은 UTF-8 **with BOM** | PowerShell 5.1은 BOM 없는 파일을 ANSI로 읽어 한글에서 파싱 실패 |
| `dataloader_num_workers=0` | Windows 멀티프로세싱 안정성 |
