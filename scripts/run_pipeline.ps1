# 데이터 준비 -> 학습 -> 평가를 한 번에 실행합니다.
#
#   .\scripts\run_pipeline.ps1                                  # 기본 QLoRA 설정
#   .\scripts\run_pipeline.ps1 -Config configs\smoke.yaml -Sample
#   .\scripts\run_pipeline.ps1 -SkipData -EvalLimit 100

param(
    [string]$Config = "configs\qlora_7.8b.yaml",
    [string]$HfDataset = "daekeun-ml/naver-news-summarization-ko",
    [switch]$Sample,        # HF 다운로드 없이 번들 샘플로 진행
    [switch]$SkipData,
    [switch]$SkipTrain,
    [switch]$SkipEval,
    [int]$MaxTrain = 0,     # 0 = 전체
    [int]$EvalLimit = 200,
    [ValidateSet("word", "char", "morph")]
    [string]$RougeTokenizer = "char"
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
    $dataArgs = @("scripts\prepare_data.py")
    if ($Sample) {
        $dataArgs += "--from-sample"
    } else {
        $dataArgs += @("--hf-dataset", $HfDataset)
        if ($MaxTrain -gt 0) { $dataArgs += @("--max-train", "$MaxTrain") }
    }
    & python $dataArgs
    if ($LASTEXITCODE -ne 0) { throw "데이터 준비 실패" }
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
