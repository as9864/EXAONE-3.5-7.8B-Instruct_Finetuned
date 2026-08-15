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
