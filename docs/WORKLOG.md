# 작업 기록

이 프로젝트를 만들면서 수행한 작업, 내린 설계 결정과 근거, 발견하고 고친 문제,
검증 내역, 그리고 **하지 않은 것**을 남깁니다.

- 시스템 구조는 [ARCHITECTURE.md](ARCHITECTURE.md)
- 사용법은 [USAGE.md](USAGE.md)

---

## 1. 요청과 결과

**요청:** `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct`로 문서 요약 파인튜닝을 수행하는
LoRA 프로젝트를 이 폴더에 생성.

**결과:** 빈 디렉터리에서 시작해 데이터 준비 → 학습 → 추론 → ROUGE 평가 →
어댑터 병합까지 도는 프로젝트를 구성했습니다.

| 항목 | 수량 |
|---|---:|
| Python 모듈 (`src/`) | 10개 / 1,004줄 |
| 실행 스크립트 (`scripts/`) | 4개 / 359줄 |
| 설정 프로파일 (`configs/`) | 3개 / 151줄 |
| 단위 테스트 (`tests/`) | 46개 / 321줄 |
| 번들 샘플 데이터 | 학습 8건 / 검증 3건 |

---

## 2. 착수 전 환경 조사

코드를 쓰기 전에 대상 환경을 먼저 확인했습니다. 이 결과가 이후 설계를 대부분
결정했습니다.

```
디렉터리 : 빈 폴더 (git 저장소 아님)
Python   : 3.12.7
GPU      : NVIDIA GeForce RTX 5070 Ti — 16,303 MiB (15.9 GiB), sm_120 (Blackwell)
전역 패키지 : torch 2.12.0+cu132, transformers 5.9.0, datasets 5.0.0
             (peft · bitsandbytes · rouge-score 없음)
```

여기서 나온 세 가지 제약:

1. **VRAM 15.9GiB** — 7.8B를 bf16으로 올리면 가중치만 약 16GB입니다. 옵티마이저
   상태와 활성값을 더할 자리가 없으므로 **4-bit QLoRA가 선택이 아니라 전제**입니다.
2. **sm_120 (Blackwell)** — CUDA 12.8 이상으로 빌드된 PyTorch가 필요합니다.
   구버전 휠은 `no kernel image is available for execution on the device`로 죽습니다.
3. **전역에 transformers 5.9.0** — EXAONE-3.5는 `trust_remote_code` 커스텀 모델링
   코드를 쓰고, 그 코드는 transformers 4.4x API를 전제합니다. 전역을 4.48.3으로
   내리면 사용자의 다른 작업이 깨집니다.

---

## 3. 설계 결정과 근거

| # | 결정 | 검토한 대안 | 선택 근거 |
|---|---|---|---|
| D1 | QLoRA(NF4 + double quant)를 기본 프로파일로 | 전체 파인튜닝 / bf16 LoRA | 16GB VRAM에서 7.8B는 4-bit가 유일한 현실적 경로. bf16 LoRA는 40GB+ 별도 프로파일로 분리 |
| D2 | `target_modules` 자동 감지 | PEFT 기본값 / 하드코딩 | EXAONE-3.5는 `out_proj`·`c_fc_0`·`c_fc_1`·`c_proj`를 써서 Llama 기본값이 대부분 매칭 실패. 모델 그래프 스캔은 버전 변화에도 따라감 |
| D3 | 프롬프트·타깃 분리 토크나이즈로 마스킹 | `DataCollatorForCompletionOnlyLM` | 템플릿 문자열을 정규식으로 찾는 방식은 템플릿이 바뀌면 조용히 어긋남. 길이로 경계를 정하면 깨지지 않음 |
| D4 | `transformers==4.48.3` 고정 + 전용 `.venv` | 전역 사용 / 최신 버전 | 전역 5.9.0으로는 remote code 로딩 실패, 전역 다운그레이드는 다른 작업 파괴. 격리가 유일한 답 |
| D5 | 문서 절단을 템플릿 적용 **전에** 수행 | 토크나이즈 후 `truncation=True` | 후자는 `[|assistant|]` 생성 마커까지 잘라내 프롬프트 구조를 파괴 |
| D6 | `jsonl.py`를 별도 모듈로 분리 | `data.py`에 포함 | 데이터 전처리만 돌릴 때 `torch`/`datasets` 임포트 비용을 물지 않게 |
| D7 | 설정 검증을 `config.py`에 배치 | `train.py`에 배치 | `train.py`는 `peft`를 임포트해서 테스트 불가. `config.py`는 `yaml`만 필요해 검증 로직을 단위 테스트할 수 있음 |
| D8 | ROUGE 분절기 3종 주입 | `rouge-score` 기본값 | 기본 토크나이저는 영문 전제라 한국어 점수가 왜곡됨. `char`가 조사 변화에 관대해 보통 가장 안정적 |
| D9 | 병합은 bf16 + CPU 기본 | 4-bit 상태로 병합 | 4-bit에 병합하면 양자화 오차가 가중치에 그대로 굳어 품질 손상 |
| D10 | `StubTokenizer`로 모델 없는 테스트 | 실제 토크나이저 다운로드 | 16GB 다운로드 없이 마스킹 경계·토큰 예산을 검증. 46개 중 37개가 GPU·모델 없이 동작 |
| D11 | 요약 생성 기본을 greedy | 샘플링 | 요약에서 샘플링은 원문에 없는 내용(환각)을 늘림 |
| D12 | 번들 샘플 데이터 8건/3건 커밋 | 없음 / 대용량 | 네트워크·라이선스 동의 없이 파이프라인 전체를 검증할 수 있는 경로 확보 |

---

## 4. 구현 순서

| 단계 | 산출물 |
|---|---|
| 1 | 환경 조사 (§2) |
| 2 | `requirements.txt`, `.gitignore`, `pyproject.toml` |
| 3 | `prompt.py` — chat template 구성, 문서 절단 |
| 4 | `config.py` — 5개 섹션 dataclass, YAML 로딩, `--set` 오버라이드 |
| 5 | `modeling.py` — 4bit 양자화, `find_linear_module_names`, LoRA 부착 |
| 6 | `data.py` — completion-only 마스킹 인코더, 콜레이터 |
| 7 | `train.py`, `infer.py` |
| 8 | `evaluate.py`, `merge_lora.py` |
| 9 | `configs/` 3종 (qlora / bf16 / smoke) |
| 10 | `scripts/prepare_data.py`, `scripts/check_env.py` |
| 11 | `data/sample/` 한국어 뉴스 요약 샘플 작성 |
| 12 | `jsonl.py` 분리 및 임포트 정리 (D6) |
| 13 | `scripts/setup.ps1`, `scripts/run_pipeline.ps1` |
| 14 | `tests/` — `StubTokenizer` + 46개 테스트 |
| 15 | 실행 검증 및 문제 수정 (§5) |
| 16 | `README.md`, `docs/` |

---

## 5. 발견하고 고친 문제

### 5.1 `--set` 오버라이드 타입 누락

**발견 경로:** `tests/test_config.py::test_overrides_coerce_types` 실패.

```
AssertionError: assert '500' == 500
  where '500' = DataConfig(... max_train_samples='500' ...)
```

**원인:** `_coerce`가 **현재 값**의 타입으로 변환 대상을 추론했습니다.

```python
# 수정 전
def _coerce(current, value):
    if isinstance(current, int): return int(value)
    ...
    return value          # ← 현재 값이 None이면 문자열이 그대로 통과
```

기본값이 `None`인 필드는 타입 정보가 없어 문자열이 그대로 들어갔습니다.
영향받는 필드: `data.max_train_samples`, `data.max_eval_samples`,
`lora.target_modules`, `lora.modules_to_save`, `train.resume_from_checkpoint`.

**터지는 지점:** `--set data.max_train_samples=500` → `data.py`에서
`rows[:'500']` → `TypeError: slice indices must be integers`. 모델 로딩(수 분)을
지나 데이터 처리 단계에서 죽습니다.

**수정:** `get_type_hints()`로 dataclass **선언 타입**을 읽어 변환합니다.
`X | None`은 `_unwrap_optional`로 벗기고, 결과는 `functools.lru_cache`로 캐시합니다.

부수적으로 잡힌 것들:
- `bool` 필드에 `maybe` 같은 값 → 조용히 `False`가 되던 것을 `ValueError`로
- `int` 필드에 `1e3` → 지수 표기 허용 (`1000`)
- `int` 필드에 `1.5` → `ValueError`
- Optional이 아닌 필드에 `none` → `ValueError` (기존에는 `None`이 들어가 나중에 터짐)

**추가한 테스트 4개** + 잘못된 오버라이드 8종 파라미터화.

### 5.2 `load_best_model_at_end` 지연 폭발 (예방)

`transformers.Trainer`는 `load_best_model_at_end=True`일 때
`save_steps`가 `eval_steps`의 배수가 아니거나 전략이 어긋나면 예외를 던집니다.
문제는 이게 **첫 저장 시점**에 터진다는 것으로, 학습을 몇 시간 돌린 뒤입니다.

`config.py:resolve_best_model_setting`으로 **시작 직후** 판정하게 했습니다.
평가·저장이 아예 없는 경우는 경고 후 자동 비활성화, 전략 불일치와 배수 위반은
즉시 `ValueError`입니다.

처음에는 `train.py`에 넣었는데, `train.py`는 `peft`를 임포트하므로 테스트할 수
없었습니다. `config.py`로 옮기고(D7) `validate()`에 편입해 테스트 7개를 붙였습니다.
같은 김에 `bf16`⊕`fp16`, `max_seq_len > max_target_tokens`, `lora.r > 0`,
비양자화 경로의 paged optimizer 경고도 `validate()`에 넣었습니다.

### 5.3 `.ps1` 파일 BOM 누락

**발견 경로:** PowerShell 파서로 직접 검증.

```
scripts\setup.ps1 : PARSE ERROR
   The string is missing the terminator: ".
scripts\run_pipeline.ps1 : PARSE ERROR
   The string is missing the terminator: ".
   Missing closing '}' in statement block or type definition.
```

**원인:** Windows PowerShell 5.1은 BOM이 없는 `.ps1`을 **시스템 ANSI 코드페이지**로
읽습니다. 파일이 UTF-8(BOM 없음)로 저장돼 한글 주석·메시지의 바이트가 잘못
해석되면서 문자열 종결자를 찾지 못했습니다. 두 스크립트 모두 실행 불가 상태였습니다.

**수정:** 두 파일을 UTF-8 **with BOM**으로 재저장. 파서 재검증 통과.

에디터에서 수정하면 BOM이 사라져 재발할 수 있으므로 [USAGE.md](USAGE.md)
트러블슈팅에 복구 방법을 남겼습니다.

### 5.4 그 외 예방적 수정

| 항목 | 내용 |
|---|---|
| `run_pipeline.ps1` here-string | `python -c @'...'@ $Config` 형태에서 닫는 `'@` 뒤에 인자를 두는 것이 PS 5.1에서 불확실해, 단일 행 문자열 + 배열 스플래팅으로 재작성 |
| `check_env.py` 버전 비교 | `torch.version.cuda < "12.8"` 문자열 비교는 `"12.10" < "12.8"`이 참이 되는 오탐. 튜플 비교로 교체 |
| `requirements.txt` | `tokenizers>=0.21.0` 제거 — transformers 4.48.3의 `<0.22` 제약과 충돌해 리졸버가 꼬일 수 있어 transformers가 끌어오게 위임 |
| `check_env.py` VRAM 경고 | 15GB 미만이면 `max_seq_len` 축소를 안내하도록 추가 |

### 5.5 내 테스트 기대값 오류

`test_word_tokenizer_splits_korean_and_latin`에서 기대 리스트에 `발표`를
빠뜨렸습니다. 코드가 아니라 테스트가 틀렸고, 기대값을 고쳤습니다.

---

## 6. 검증 내역

실제로 실행하고 결과를 확인한 것만 적습니다.

| 검증 | 명령 | 결과 |
|---|---|---|
| 단위 테스트 | `python -m pytest tests -q` | **46개 통과** (config 28 / data 9 / evaluate 9) |
| 문법 | `python -m py_compile <16 files>` | 전체 통과 |
| PowerShell 파싱 | `[Parser]::ParseFile()` × 2 | BOM 수정 후 통과 |
| 데이터 준비 | `python scripts\prepare_data.py --from-sample` | train 8건 / validation 3건 생성, 본문 평균 316자 / 요약 140자 |
| 파이프라인 오케스트레이션 | `.\scripts\run_pipeline.ps1 -Config configs\smoke.yaml -Sample -SkipTrain -SkipEval` | YAML에서 `output_dir` 추출 → 데이터 단계 정상 완료 |
| 환경 점검 | `python scripts\check_env.py` | sm_120 / CUDA 13.2 / 15.9GiB 정확히 리포트, 미설치 패키지 4건 정확히 FAIL |
| 지연 임포트 | `python -m exaone_summarize.evaluate --help` | `peft` 없이 동작 확인 |

`check_env.py` 실제 출력:

```
Python  : 3.12.7  (Windows 11)
[ OK ] torch          2.12.0+cu132
[ OK ] transformers   5.9.0
[FAIL] peft           임포트 실패: No module named 'peft'
[FAIL] accelerate     임포트 실패: No module named 'accelerate'
[ OK ] datasets       5.0.0
[FAIL] bitsandbytes   임포트 실패: No module named 'bitsandbytes'
[ OK ] rouge_score    ?
[ OK ] yaml           6.0.3
CUDA available : True
torch CUDA ver : 13.2
  GPU 0: NVIDIA GeForce RTX 5070 Ti  15.9GiB  sm_120
bf16 지원      : True
```

---

## 7. 하지 않은 것

정직하게 남깁니다.

### 학습을 실제로 실행하지 않았습니다

`peft`와 `bitsandbytes`가 설치돼 있지 않고, 설치하면 수 GB 다운로드와 전역 패키지
변경(특히 `transformers` 다운그레이드)이 따릅니다. 그건 사용자 판단이 필요한
변경이라 남겨뒀습니다.

**따라서 검증되지 않은 것:**

- `AutoModelForCausalLM.from_pretrained(trust_remote_code=True)`로 EXAONE-3.5가
  실제로 로딩되는지
- `find_linear_module_names`가 실제 모델에서 반환하는 이름 목록
  (예상: `q_proj`, `k_proj`, `v_proj`, `out_proj`, `c_fc_0`, `c_fc_1`, `c_proj`)
- 4-bit + LoRA 조합의 실제 VRAM 사용량
- `paged_adamw_8bit` 옵티마이저 동작
- 실제 EXAONE 토크나이저의 `apply_chat_template` 출력과 `eos_token_id`
- 학습 손실 수렴 여부 및 요약 품질

마스킹·예산 로직은 `StubTokenizer`로 검증했지만, 이는 **실제 템플릿이 아니라
골격을 모방한 것**입니다. 첫 실행 시 `configs/smoke.yaml`로 4스텝만 돌려보는 것을
권합니다.

### 전역 환경에 가한 변경

`rouge-score` 1개만 설치했습니다 (`absl-py` 의존성 동반). 순수 추가이며
다운그레이드는 없습니다. `evaluate.py`를 실제로 검증하기 위해 필요했습니다.

`peft`, `bitsandbytes`, `transformers==4.48.3`은 전역에 설치하지 않았습니다.

### 데이터 규모

번들 샘플은 8건/3건짜리 **파이프라인 검증용**입니다. 실제 요약 품질을 얻으려면
최소 수천 건 규모의 도메인 데이터가 필요합니다.

---

## 8. 권장 다음 단계

```powershell
# 1. 격리 환경 구성 (cu128 torch + transformers 4.48.3 + peft + bitsandbytes)
.\scripts\setup.ps1
.\.venv\Scripts\Activate.ps1

# 2. 모델 라이선스 동의 후 로그인
huggingface-cli login

# 3. 4스텝 스모크 — 여기서 §7의 미검증 항목들이 한 번에 드러납니다
python -m exaone_summarize.train -c configs\smoke.yaml
```

스모크가 통과하면 `print_trainable_parameters()` 출력과 자동 감지된
`target_modules` 로그를 확인하세요. 어댑터가 제대로 붙었는지 판단하는 지점입니다.

그 다음:

```powershell
# 4. 실제 데이터로 학습
python scripts\prepare_data.py --hf-dataset daekeun-ml/naver-news-summarization-ko --max-train 20000
python -m exaone_summarize.train -c configs\qlora_7.8b.yaml

# 5. 베이스 대비 개선 폭 측정 — --adapter 유무로 두 번 돌려 비교
python -m exaone_summarize.evaluate -c configs\qlora_7.8b.yaml `
    --input-jsonl data\processed\test.jsonl --tokenizer char --limit 200
python -m exaone_summarize.evaluate -c configs\qlora_7.8b.yaml `
    --input-jsonl data\processed\test.jsonl --tokenizer char --limit 200 `
    --adapter outputs\exaone-3.5-7.8b-summary-qlora\adapter
```

ROUGE는 방향 지표일 뿐이니 실제 요약문 몇 개는 직접 읽어보는 것을 권합니다.

---

## 9. 라이선스 주의

EXAONE-3.5 가중치는 `EXAONE AI Model License Agreement 1.1 - NC`로
**비상업적 연구 목적**에 제한됩니다. 이 제약은 학습한 LoRA 어댑터와 병합 모델
같은 **파생물에도 이어집니다.** 상업적 활용을 검토한다면 LG AI Research의 라이선스
원문을 직접 확인해야 합니다.

---

## 10. AI Hub 「문서요약 텍스트」 통합 (2026-08-16)

**요청:** `data/AIHUB_DocSummaryData`에 받아 둔 AI Hub 자료를 기존 데이터와
융합해 학습에 쓸 수 있게 할 것.

### 10.1 원본 구조

zip 하나당 JSON 하나이고, 스키마는 세 도메인(신문기사 · 사설 · 법률) 모두 같습니다.

```json
{"documents": [{"id": "...", "category": "...", "title": "...",
  "text": [[{"index": 0, "sentence": "...", "highlight_indices": "20,21"}], ...],
  "extractive": [0, 4, 6],
  "abstractive": ["사람이 쓴 생성 요약문"]}]}
```

`신문기사_train_original.json`은 **압축 해제 시 1.1GB**입니다.

### 10.2 설계 결정

| # | 결정 | 근거 |
|---|---|---|
| D8 | zip을 풀지 않고 `zipfile` + `TextIOWrapper`로 직접 스트리밍 | 6개 zip 합 420MB → 해제하면 1.7GB. 디스크에 사본을 두 벌 만들 이유가 없음 |
| D9 | `documents` 배열을 `raw_decode`로 객체 단위 파싱 | `json.load()`는 1.1GB 파일에서 수 GB의 파이썬 객체를 한 번에 만듦. 상수 메모리로 훑도록 커스텀 스트리밍 파서 작성 |
| D10 | 요약 라벨은 `abstractive` 기본 | 생성 요약 학습이 목표. `extractive`(핵심문장 3개)는 `--summary-type`으로 선택 가능하게만 남김 |
| D11 | 변환(`prepare_aihub.py`)과 병합(`merge_datasets.py`)을 분리 | 변환은 무겁고 한 번만 하면 되지만, 혼합 비율은 실험할 때마다 바뀜. 도메인별 파일로 떨궈 두면 재변환 없이 비율만 조정 가능 |
| D12 | 기존 HF 데이터를 `data/processed/naver_news/`로 옮기고 원본 보존 | 병합 결과를 `data/processed/*.jsonl`에 쓰므로, 원본을 그 자리에 두면 재실행 때마다 자기 자신을 다시 먹음 |
| D13 | Validation zip을 valid/test로 반씩 분할 | AI Hub에는 test 스플릿이 없음. 시드 고정 셔플 후 분할해 두 세트가 겹치지 않도록 보장 |
| D14 | 병합 시 `--exclude`로 평가 본문을 학습에서 제거 | 실제로 누수가 있었음 — 아래 10.4 |

### 10.3 결과

변환(`data/processed/aihub/`, 필터 후):

| 도메인 | train | valid | test | 본문 평균 | 요약 평균 |
|---|---:|---:|---:|---:|---:|
| 신문기사 | 243,428 | 14,863 | 14,862 | 1,007자 | 129자 |
| 사설 | 53,279 | 3,162 | 3,162 | 1,178자 | 122자 |
| 법률 | 24,038 | 1,489 | 1,488 | 671자 | 199자 |

병합 결과(도메인별 2만건 상한):

| 스플릿 | 건수 | 구성 |
|---|---:|---|
| train | 79,219 | aihub 신문기사/사설/법률 각 25.2% + naver_news 24.3% |
| validation | 1,000 | 300 / 200 / 200 + naver_news 300 |
| test | 1,000 | 300 / 200 / 200 + naver_news 300 |

기존 20,425건 → **79,219건 (3.9배)**, 뉴스 단일 도메인 → 4개 출처.

### 10.4 발견한 문제

**기존 `data/processed`에 중복과 누수가 있었습니다.** 병합 단계의 중복 제거가
드러냈습니다.

- `naver_news/train.jsonl` 내부 중복 **1,071건** (HF 원본 자체의 중복)
- train에 들어 있던 본문이 기존 validation/test에도 있던 것 **126건**
- AI Hub `법률` train ∩ 새 평가 세트 **9건** — AI Hub의 Training/Validation 배포본에도
  같은 문서가 일부 들어 있습니다

셋 다 새 학습 세트에서 제외했습니다. 이전 학습(`Fine tuning With LORA` 커밋)의
평가 점수는 이 누수를 포함한 값이므로 낙관 편향이 있습니다.

### 10.5 검증

- 단위 테스트 13개 추가 (`tests/test_prepare_aihub.py`) — 통과
  - 스트리밍 파서: 정상 파싱 / **청크 경계가 객체 중간을 자르는 경우**(`_CHUNK=7`) / 포맷 불일치 시 에러
  - 변환: 본문 평탄화, abstractive·extractive 선택, 제목 포함, 요약 없음 → 스킵
  - 필터: 짧은 본문 · 짧은 요약 · 중복 제거, zip 직접 읽기
  - 탐색: 파일명에서 도메인/스플릿 판별
  - 병합: `경로:N` 파싱(Windows 드라이브 문자 오인 방지), 중복·제외·백업·상한
- 실제 6개 zip 전량 변환 성공 (32만건, 최대 메모리 상수)
- `run_pipeline.ps1` 구문 검사 통과

### 10.6 하지 않은 것

- **학습 재실행 안 함.** 데이터만 준비했습니다. 79,219건은 유효 배치 16 기준
  1에폭 약 5,000스텝입니다.
- 도메인 비율 튜닝 안 함. 지금은 4개 출처를 거의 균등하게 뒀습니다. 뉴스 요약이
  주 목적이라면 `--input ...\news_train.jsonl:50000` 식으로 비율을 올리세요.
- `highlight_indices`(문장 내 핵심 어구 위치)와 `document_quality_scores`는
  버렸습니다. 품질 점수로 저품질 샘플을 거르는 것은 시도해 볼 만합니다.

---

## 11. 평가 점수 검증과 누수 제거 (2026-08-16)

**요청:** train/validation/test 건수를 다시 조정할 것. 그리고 ROUGE가
`rouge1 67.69 / rouge2 56.12 / rougeL 54.25`로 너무 높게 나와 걱정되니
데이터 누수가 있었는지 확인할 것.

**결론:** 누수는 **있었다**(평가 세트의 37.6%). 다만 그것만으로 67.69가 설명되지는
않는다. 원인은 세 가지가 겹친 것이고, 기여도는 아래 순서다.

| 원인 | 기여 |
|---|---|
| ① 정답 요약이 본문 복붙 (naver 뉴스 데이터의 성질) | 가장 큼 — lead-3 복사만으로 63.32 |
| ② char 분절 ROUGE | +14.9 (word 분절로는 52.82) |
| ③ 학습/평가 누수 37.6% | +2.31 |

### 11.1 ① 정답 요약이 본문 복붙

정답 요약의 문자 4-gram 중 **84.8%가 본문에 그대로 있다**(중앙값 86.2%).
추출 요약에 가까운 라벨이다. 그래서 아무것도 학습하지 않고 본문 앞 세 문장을
복사하는 것만으로 높은 점수가 나온다.

| 예측 | char R-1 | word R-1 |
|---|---:|---:|
| 학습한 모델 | 67.69 | 52.82 |
| **lead-3 복사 (모델 없음)** | **63.32** | **49.14** |
| 본문 전체 복사 | 34.79 | 32.00 |

즉 67.69의 실질은 "복사 베이스라인 대비 **+4.37**"이다. 낮은 값은 아니지만
67.69라는 절대값이 주는 인상과는 다르다.

### 11.2 ② char 분절

`--tokenizer char`는 음절 단위라 조사·어미가 우연히 겹치는 것까지 점수가 된다.
같은 예측 파일에서 char 67.69 / word 52.82로 **14.9점** 차이가 났다.
`run_pipeline.ps1`의 기본 분절기를 `char` → `word`로 바꿨다.

### 11.3 ③ 누수 — 실제로 있었다

`scripts/check_leakage.py`를 새로 만들어 측정했다. 학습에 쓴 파일과 평가 파일을
직접 대조한다.

| 통합 전 평가 세트 | 완전 일치 | 근사 중복 | 합계 |
|---|---:|---:|---:|
| test (933건) | 73 | 278 | **351 (37.6%)** |
| validation (916건) | 71 | 287 | **358 (39.1%)** |

실제 채점된 200건 중 77건이 누수였고, 그 부분집합의 점수가 더 높다.

| 구분 | n | R-1 | R-2 | R-L |
|---|---:|---:|---:|---:|
| 전체 | 200 | 67.69 | 56.12 | 54.25 |
| 누수 샘플 | 77 | 69.11 | 58.97 | 57.70 |
| 깨끗한 샘플 | 123 | 66.80 | 54.34 | 52.09 |

누수 제거만으로는 **-2.31점**이다. 예상보다 작은 이유는 ①번 때문이다 — 어차피
복사로 풀리는 과제라 원문을 봤든 안 봤든 점수 차이가 크게 나지 않는다.

### 11.4 근사 중복 판정을 어떻게 짰나

완전 일치만 보면 73건, 실제로는 351건이었다. 한국어 뉴스는 통신사 기사 재배포와
재게재가 많아 문자열 일치로는 걸러지지 않는다. `src/exaone_summarize/dedup.py`에
어절 5-gram shingle 역색인을 만들었고, 다음 두 번의 오탐을 잡아 고쳤다.

**오탐 1 — 합집합과 비교하면 안 된다.** 처음엔 학습 세트 전체 shingle의 합집합과
비교했다. 같은 사건을 다룬 서로 다른 기사들의 조각이 여기저기 맞아떨어져
포함률이 0.50까지 나왔지만, **최근접 문서 한 건과는 0.17**이었다. 문서 단위
역색인으로 바꿨다(전체 누수 403 → 351건).

**오탐 2 — 짧은 문서 + 상용구.** 판례문 두 건이 `구 상속세 및 증여세법(2002. 12.
18. 법률 제6780호로 개정되기 전의 것)` 같은 법령 인용구만 공유해도, overlap
coefficient의 `min()` 분모 때문에 유사도가 0.60이 나왔다. 서로 다른 사건이다.
공유 shingle 최소 4개(`MIN_SHARED`) 조건을 추가하고 표본 비율을 1/16 → 1/8로
높여 해결했다. 두 사례 모두 `tests/test_dedup.py`에 회귀 테스트로 남겼다.

### 11.5 건수 재조정

기준을 **실측 처리량**에 뒀다. 이전 학습 로그: 20,425건 1에폭 = 53,655초
(**0.381 샘플/초**, `max_seq_len=1536`). 이전과 같은 시간(약 15시간)을 유지하면서
데이터의 다양성만 4배로 올리는 구성을 골랐다.

| 스플릿 | 이전 | 지금 | 구성 |
|---|---:|---:|---|
| train | 20,425 | **20,130** | 출처 4종 × 약 5,000 (각 25%) |
| validation | 916 | **998** | 출처별 249~250 |
| test | 933 | **1,977** | 출처별 483~499 |

- **train 2만건 유지**: AI Hub 전체(32만건)를 쓰면 1에폭에 230시간이다. 같은
  시간에 4개 도메인을 균등하게 배우는 쪽이 낫다.
- **출처 균등**: 단일 평균이 도메인별 실력 차를 가리지 않도록 했고,
  `evaluate.py`에 출처별 분해 출력을 추가했다.
- **test를 두 배로**: 도메인별로 500건씩 있어야 출처별 점수가 흔들리지 않는다.
  (200건 전체 평균 하나만으로는 도메인별 판단이 불가능했다.)

병합 순서를 validation → test → train으로 고정하고, 뒤 단계에서 앞 단계 본문을
`--exclude`로 뺀다. 재검사 결과 **완전 일치 0 / 근사 중복 0 / 요약문 일치 0**.

naver 쪽은 원본 자체에 중복이 많아 5,000건을 남기려면 상한 6,000이 필요했다
(내부 중복 125 + 근사 중복 486 + 평가셋과 겹침 285).

### 11.6 검증

- 단위 테스트 26개 추가 (`tests/test_dedup.py` 8, 병합 근사중복 2,
  lead 베이스라인 3, 기존 AI Hub 13) — 전체 72개 통과
- `tests/test_data.py`의 낡은 기대값 1건 수정 (`length` 컬럼 추가 반영)
- 재검사로 새 스플릿 누수 0건 확인

### 11.7 남은 한계

- **이전 학습의 점수는 폐기해야 한다.** 누수 세트에서 측정한 값이다. 새 데이터로
  다시 학습하고 새 test로 재측정해야 비교가 가능하다.
- ①번(정답 요약이 복붙)은 데이터의 성질이라 고칠 수 없다. 다만 AI Hub
  `abstractive`는 사람이 새로 쓴 요약이라 복사율이 낮다. 새 test는 4개 출처가
  균등해서 이전보다 점수가 **낮게** 나올 것이고, 그게 정상이다.
- ROUGE 자체의 한계(사실 오류를 못 잡음)는 그대로다. 요약문을 직접 읽는 절차를
  대체하지 못한다.

---

## 12. 베이스 zero-shot 베이스라인 측정과 attention mask 버그 (2026-08-18)

**요청:** 파인튜닝한 모델이 아니라 EXAONE-3.5-7.8B-Instruct 베이스 모델로 평가해서
결과를 비교할 것. (README §6 한계 2번으로 남겨 뒀던 ablation)

**결론:** 어댑터는 베이스 zero-shot 대비 **+20.4 R-1**로 유의하게 낫다. 다만 이 값을
얻는 과정에서 **평가 파이프라인이 조용히 망가져 있었다는 것**을 발견했다. 캐시된
remote code가 attention mask를 버리고 있어서, left padding을 쓰는 배치 생성이
붕괴하고 있었다. 베이스 모델의 첫 측정값 R-1 8.83은 이 버그의 산물이다.

### 12.1 발견 경로

베이스 모델(`--adapter` 없이) 200건을 평가하니 R-1 8.83이 나왔다. 출력 길이는 정답의
5.2배(817자 vs 157자), 200건 중 113건에 5-gram 반복, 120건이 문장 중간에서 끊겼다.
표본을 직접 읽으니 문서와 무관한 환각(문서에 없는 "2017 FOOD KOREA CONFERENCE")과
같은 구절의 무한 반복이었다. EXAONE-3.5-Instruct가 이 정도로 못하는 것은 이상했다.

가설을 순서대로 지웠다.

| 가설 | 확인 방법 | 결과 |
|---|---|---|
| 문서 절단(1,024토큰 예산 초과) | 토큰 길이 측정 | 200건 중 7건만 절단, 반복은 그 7건에서 안 나왔다 |
| left padding 설정 누락 | `modeling.load_for_inference` 확인 | `padding_side="left"` 정상 |
| 4-bit 양자화 열화 | — | 아래에서 배제됨 |
| **배치 생성** | 같은 12건을 `--batch-size 1`로 재실행 | **정상 요약이 나왔다** |

bs=1은 정상, bs=4는 붕괴. 그리고 bs=4에서도 **배치 내 최장 문서(=pad 0개)만** 정상
이었다. 패딩 처리 문제로 좁혀졌다.

### 12.2 원인

`~/.cache/huggingface/modules/.../modeling_exaone.py`의 mask 생성부가 손으로
수정돼 있었다.

```python
# 원본 (hub 스냅샷) — transformers 5.15에 없는 인자명을 쓴다
causal_mask = create_causal_mask(
    config=self.config, input_embeds=inputs_embeds,      # 5.15는 inputs_embeds
    attention_mask=attention_mask, cache_position=cache_position,   # 5.15에는 없는 인자
    past_key_values=past_key_values, position_ids=position_ids)

# 수정돼 있던 코드 — TypeError를 try/except로 덮고 mask를 버린다
try:
    from transformers.masking_utils import _prepare_4d_causal_attention_mask
    causal_mask = _prepare_4d_causal_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, ...)  # 둘 다 미정의 → NameError
except Exception:
    causal_mask = None
```

`batch_size` / `seq_length`가 그 함수 스코프에 없으므로 `NameError`가 **매 forward마다**
발생하고, 그때마다 `causal_mask = None`이 된다. SDPA는 mask가 `None`이면 `is_causal=True`로
동작하므로 **패딩 토큰을 실제 토큰처럼 attend한다.**

왜 학습에서는 안 드러났나: 학습은 right padding이라 pad가 뒤에 있고, causal 어텐션에서
각 실토큰의 앞쪽은 전부 실토큰이다. 그래서 loss가 거의 오염되지 않는다. 반대로 **생성은
left padding**이라 짧은 문서일수록 앞에 pad가 잔뜩 붙고, 모델은 그 pad를 문맥으로 읽는다.

### 12.3 수정

`scripts/patch_remote_code.py` — 5.15 시그니처에 맞춘 호출로 교체한다(인자명 2개,
단일 hunk). `--check`(종료코드로 알림) / `--restore`(`.orig` 백업 복원) 지원.
`scripts/check_env.py`에도 같은 판정을 붙여 설치 점검에서 걸리게 했다.

만드는 과정에서 **스크립트가 파일을 크게 훼손하는 실수**를 한 번 했다. `causal_mask = `
문자열을 인덱스로 찾아 그 뒤 `hidden_states = inputs_embeds`까지를 교체 범위로 잡았는데,
손수정 버전에는 그 패턴이 없어 앞쪽의 엉뚱한 `try:`가 잡히고 300줄이 지워졌다(문법은
유효해서 `ast.parse`도 통과했다). hub 스냅샷에서 복구한 뒤, **알려진 블록 전체를 문자열
그대로 매칭**해 치환하고 줄 수 변화가 12줄을 넘으면 거부하도록 바꿨다. 인덱스 탐색으로
코드 범위를 자르지 말 것.

Windows에서 `write_text`가 줄바꿈을 CRLF로 바꿔 파일 전체가 변경돼 보이는 문제도 함께
고쳤다(바이트 단위로 읽고 원래 줄바꿈으로 되돌려 쓴다).

### 12.4 재측정 결과

테스트 200건, `word` 분절, batch 4 greedy. 마스크 수정 전/후.

| 시스템 | 수정 전 R-1 | 수정 후 R-1 | 길이비 | 5-gram 반복 |
|---|---:|---:|---:|---:|
| base-0shot (어댑터와 같은 프롬프트) | 8.83 | **18.72** | 5.19 → 2.12 | 113 → 4 |
| qlora 어댑터 | 39.35 | **39.09** | 1.22 → 1.22 | 13 → 14 |

**어댑터는 이 버그에 거의 영향을 받지 않았다**(±1.4점 이내, 도메인별로도). 태스크에
고정돼 있어 pad 노이즈를 견딘 것으로 보인다. 그래서 README의 기존 파인튜닝 수치는
사실상 유효했고, 붕괴한 쪽은 베이스 모델뿐이었다.

### 12.5 베이스라인을 두 개 낸 이유

같은 프롬프트로 비교하면 베이스는 정답보다 **2.12배** 길게 쓴다(목록·머리말 포함).
ROUGE F1은 길이 초과를 감점하므로, 그 상태의 격차에는 "형식을 모른다"가 섞인다.
그래서 `configs/zeroshot_prompted.yaml`로 "2~3문장·150자 내외·요약문만"을 지시한
베이스라인을 하나 더 만들었다. 길이비가 1.29(어댑터 1.22)까지 내려온다.

| 데이터 | base-0shot | base-prompted | qlora | Δ vs 0shot | Δ vs prompted |
|---|---:|---:|---:|---:|---:|
| 전체 | 18.72 | 19.89 | 39.09 | +20.38 | +19.20 |
| aihub_editorial | 11.49 | 14.06 | 24.89 | +13.40 | +10.83 |
| aihub_law | 18.04 | 17.17 | 42.39 | +24.35 | +25.21 |
| aihub_news | 19.32 | 21.51 | 35.46 | +16.14 | +13.95 |
| naver_news | 25.65 | 26.31 | 53.48 | +27.83 | +27.17 |

**길이를 맞춰도 격차는 유지된다**(+19.20, 네 도메인 모두 95% CI가 0 밖). 파인튜닝
효과가 형식 아티팩트가 아니라는 뜻이다.

비교는 `scripts/compare_runs.py`로 한다. `report_predictions.py`는 한 실행을 lead-N과
비교하는 도구여서, 두 실행을 같은 문서끼리 짝지어 paired bootstrap하는 코드가 없었다.

### 12.6 그래도 ROUGE로는 절반만 말할 수 있다

베이스 모델 요약의 **신규 4-gram 비율은 97%**다. 원문 표현을 거의 쓰지 않고 새로 쓴다.
정답 요약은 62.7%, 어댑터는 42.9%다. 즉 베이스는 "요약을 못 한다"기보다 **정답과 다른
어휘로 요약한다**. +20점의 상당 부분은 정답 스타일(어휘·길이·문체) 정렬이고, 내용
충실도까지 앞선다는 증거는 아니다. 이 구분은 사람 평가로만 확정된다(README §6-3).

### 12.7 검증

- 같은 12건 bs=1 / bs=4(수정 후) 비교 — 붕괴 사라짐, 반복 5-gram 최대 83회 → 1회.
  bs=1과 bit 단위로 같지는 않다(배치 패딩의 부동소수점 차이가 greedy에서 갈린다).
- `patch_remote_code.py` 멱등성: 두 번 돌려도 `fixed`, hub 원본과의 diff는 단일 hunk.
- 테스트 119 passed / 1 failed. 실패한 `tests/test_api.py::test_default_paths_exist_in_repo`는
  이 작업과 무관하다 — `DEFAULT_ADAPTER_PATH`(`outputs/exaone-3.5-7.8b-summary-qlora/adapter`)가
  비어 있고 실제 어댑터가 `outputs/exaone-3.5-7.8b-summary-qlora_task2/`에 있어서다.
  경로를 맞추거나 기본값을 바꿔야 한다(미해결).

### 12.8 남은 것

- 200건 → 1,977건 전체 재평가 (도메인당 n≈50이라 CI가 ±5~10점)
- 사람 평가 — 베이스 요약과 어댑터 요약을 나란히 놓고 사실성·충실도 비교
- `patch_remote_code.py`는 캐시를 지우면 다시 돌려야 한다. `setup.ps1`에 넣을지는
  보류했다(모델을 한 번 로딩한 뒤에야 캐시가 생기므로 순서가 어긋난다)
