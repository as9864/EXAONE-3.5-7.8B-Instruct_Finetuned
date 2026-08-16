# 데이터 준비 -> 학습 -> 평가를 한 번에 실행합니다.
#
#   .\scripts\run_pipeline.ps1                                  # 기본 QLoRA 설정
#   .\scripts\run_pipeline.ps1 -Config configs\smoke.yaml -Sample
#   .\scripts\run_pipeline.ps1 -WithAihub                       # HF 뉴스 + AI Hub 문서요약 혼합
#   .\scripts\run_pipeline.ps1 -SkipData -EvalLimit 100

param(
    [string]$Config = "configs\qlora_7.8b.yaml",
    [string]$HfDataset = "daekeun-ml/naver-news-summarization-ko",
    [switch]$Sample,        # HF 다운로드 없이 번들 샘플로 진행
    [switch]$WithAihub,     # AI Hub 문서요약 데이터를 기존 데이터와 혼합
    [switch]$SkipData,
    [switch]$SkipTrain,
    [switch]$SkipEval,
    [int]$MaxTrain = 0,     # 0 = 전체
    [int]$PerSourceTrain = 5000,    # 혼합 시 출처별 학습 건수 (0 = 전체)
    [int]$PerSourceValid = 250,     # 혼합 시 출처별 검증 건수
    [int]$PerSourceTest = 500,      # 혼합 시 출처별 테스트 건수
    [switch]$RebuildAihub,  # data\processed\aihub 가 있어도 zip에서 다시 변환
    [int]$EvalLimit = 200,
    # char 분절은 한국어에서 점수를 크게 부풀린다(실측 +15점). evaluate.py와 같은
    # 기본값(word)을 쓴다.
    [ValidateSet("word", "char", "morph")]
    [string]$RougeTokenizer = "word"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = "src"
$env:TOKENIZERS_PARALLELISM = "false"

if (-not (Test-Path $Config)) { throw "설정 파일을 찾을 수 없습니다: $Config" }

# 설정 파일에서 output_dir을 읽어 어댑터 경로를 만든다.
$reader = "import sys,yaml;c=yaml.safe_load(open(sys.argv[1],encoding='utf-8')) or {};print((c.get('train') or {}).get('output_dir','outputs/run'))"
$outputDir = & python -c $reader $Config
if ($LASTEXITCODE -ne 0) { throw "설정 파일을 읽지 못했습니다: $Config" }
$adapter = Join-Path $outputDir "adapter"
Write-Host "==> output_dir = $outputDir"

if (-not $SkipData) {
    Write-Host ""
    Write-Host "==> [1/3] 데이터 준비"

    if ($Sample -and $WithAihub) { throw "-Sample 과 -WithAihub 는 함께 쓸 수 없습니다." }

    if (-not $WithAihub) {
        $dataArgs = @("scripts\prepare_data.py")
        if ($Sample) {
            $dataArgs += "--from-sample"
        } else {
            $dataArgs += @("--hf-dataset", $HfDataset)
            if ($MaxTrain -gt 0) { $dataArgs += @("--max-train", "$MaxTrain") }
        }
        & python $dataArgs
        if ($LASTEXITCODE -ne 0) { throw "데이터 준비 실패" }
    } else {
        # (1) HF 뉴스 요약: data\processed\naver_news 에 따로 받아 둔다.
        #     이미 있으면 재다운로드하지 않는다.
        $hfDir = "data\processed\naver_news"
        if (Test-Path (Join-Path $hfDir "train.jsonl")) {
            Write-Host "    HF 데이터 재사용: $hfDir"
        } else {
            $dataArgs = @("scripts\prepare_data.py", "--hf-dataset", $HfDataset, "--output-dir", $hfDir)
            if ($MaxTrain -gt 0) { $dataArgs += @("--max-train", "$MaxTrain") }
            & python $dataArgs
            if ($LASTEXITCODE -ne 0) { throw "HF 데이터 준비 실패" }
        }

        # (2) AI Hub zip -> data\processed\aihub\{도메인}_{스플릿}.jsonl
        $aihubDir = "data\processed\aihub"
        if ((Test-Path (Join-Path $aihubDir "news_train.jsonl")) -and (-not $RebuildAihub)) {
            Write-Host "    AI Hub 변환 결과 재사용: $aihubDir (다시 만들려면 -RebuildAihub)"
        } else {
            & python scripts\prepare_aihub.py
            if ($LASTEXITCODE -ne 0) { throw "AI Hub 변환 실패" }
        }

        # (3) 병합. 평가 세트를 먼저 만들고, 학습 세트에서 그 본문을 빼서 누수를 막는다.
        #     merge_datasets.py는 근사 중복(재게재 기사)까지 제거하므로
        #     실제 결과는 상한보다 조금 적게 나온다.
        # 출처 4종의 파일명 규칙: HF는 train/validation/test, AI Hub는 {도메인}_{train,valid,test}
        function Get-Inputs([string]$hfName, [string]$aihubName, [int]$cap) {
            $suffix = if ($cap -gt 0) { ":$cap" } else { "" }
            return @(
                "$hfDir\$hfName.jsonl$suffix",
                "$aihubDir\news_$aihubName.jsonl$suffix",
                "$aihubDir\editorial_$aihubName.jsonl$suffix",
                "$aihubDir\law_$aihubName.jsonl$suffix"
            )
        }
        $merges = @(
            @("data\processed\validation.jsonl",
              (Get-Inputs "validation" "valid" $PerSourceValid),
              @()),
            @("data\processed\test.jsonl",
              (Get-Inputs "test" "test" $PerSourceTest),
              @("data\processed\validation.jsonl")),
            @("data\processed\train.jsonl",
              (Get-Inputs "train" "train" $PerSourceTrain),
              @("data\processed\validation.jsonl", "data\processed\test.jsonl"))
        )

        foreach ($merge in $merges) {
            $mergeArgs = @("scripts\merge_datasets.py", "--output", $merge[0])
            foreach ($src in $merge[1]) { $mergeArgs += @("--input", $src) }
            foreach ($ex in $merge[2]) { $mergeArgs += @("--exclude", $ex) }
            & python $mergeArgs
            if ($LASTEXITCODE -ne 0) { throw "병합 실패: $($merge[0])" }
        }

        # (4) 누수 검사. 여기서 0건이 아니면 평가 점수를 믿을 수 없다.
        & python scripts\check_leakage.py `
            --train data\processed\train.jsonl `
            --eval data\processed\validation.jsonl `
            --eval data\processed\test.jsonl --show 0
        if ($LASTEXITCODE -ne 0) { throw "누수 검사 실패" }
    }
}

if (-not $SkipTrain) {
    Write-Host ""
    Write-Host "==> [2/3] 학습"
    & python -m exaone_summarize.train -c $Config
    if ($LASTEXITCODE -ne 0) { throw "학습 실패" }
}

if (-not $SkipEval) {
    Write-Host ""
    Write-Host "==> [3/3] 평가"
    $evalFile = @(
        "data\processed\test.jsonl",
        "data\processed\validation.jsonl",
        "data\sample\validation.jsonl"
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $evalFile) { throw "평가에 쓸 JSONL을 찾을 수 없습니다." }
    Write-Host "    평가 파일: $evalFile"

    if (-not (Test-Path $adapter)) {
        Write-Host "    경고: 어댑터가 없어 베이스 모델로 평가합니다 ($adapter)"
        $evalArgs = @()
    } else {
        $evalArgs = @("--adapter", $adapter)
    }

    $evalArgs = @(
        "-m", "exaone_summarize.evaluate",
        "-c", $Config,
        "--input-jsonl", $evalFile,
        "--limit", "$EvalLimit",
        "--tokenizer", $RougeTokenizer,
        "--save-predictions", (Join-Path $outputDir "predictions.jsonl"),
        "--output-json", (Join-Path $outputDir "metrics.json")
    ) + $evalArgs

    & python $evalArgs
    if ($LASTEXITCODE -ne 0) { throw "평가 실패" }
}

Write-Host ""
Write-Host "완료. 산출물: $outputDir"
