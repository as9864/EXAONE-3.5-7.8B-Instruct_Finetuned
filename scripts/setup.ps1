# 가상환경 생성 + 의존성 설치 (Windows PowerShell)
#   .\scripts\setup.ps1
#   .\scripts\setup.ps1 -CudaTag cu126   # 다른 CUDA 빌드가 필요한 경우

param(
    # RTX 50xx(Blackwell, sm_120)는 cu128 이상 빌드가 필요합니다.
    [string]$CudaTag = "cu128",
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path $VenvPath)) {
    Write-Host "==> 가상환경 생성: $VenvPath"
    python -m venv $VenvPath
} else {
    Write-Host "==> 기존 가상환경 사용: $VenvPath"
}

$python = Join-Path $root "$VenvPath\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "가상환경 python을 찾을 수 없습니다: $python" }

Write-Host "==> pip 업그레이드"
& $python -m pip install --upgrade pip setuptools wheel

Write-Host "==> PyTorch 설치 ($CudaTag)"
& $python -m pip install torch --index-url "https://download.pytorch.org/whl/$CudaTag"

Write-Host "==> 프로젝트 의존성 설치"
& $python -m pip install -r requirements.txt

Write-Host "==> 프로젝트 editable 설치"
& $python -m pip install -e . --no-deps

Write-Host "==> 환경 점검"
& $python scripts\check_env.py

Write-Host ""
Write-Host "완료. 다음 단계:"
Write-Host "  .\$VenvPath\Scripts\Activate.ps1"
Write-Host "  python scripts\prepare_data.py --from-sample"
Write-Host "  python -m exaone_summarize.train -c configs\smoke.yaml"
