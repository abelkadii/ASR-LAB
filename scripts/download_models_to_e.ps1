param(
    [string]$CacheRoot = "E:\speechRecon-cache",
    [double]$DownloadMbps = 100,
    [switch]$IncludeDatasets,
    [switch]$IncludeDiscoveredDatasets,
    [switch]$Run,
    [switch]$WarmLoad
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}

$hfHome = Join-Path $CacheRoot "huggingface"
$env:HF_HOME = $hfHome
$env:HF_HUB_CACHE = Join-Path $hfHome "hub"
$env:HF_DATASETS_CACHE = Join-Path $hfHome "datasets"
$env:TRANSFORMERS_CACHE = Join-Path $hfHome "hub"
$env:SPEECHRECON_CACHE_ROOT = $CacheRoot
$env:SPEECHRECON_MODEL_DIR = Join-Path $CacheRoot "models"
$env:SPEECHRECON_DATASET_DIR = Join-Path $CacheRoot "datasets"

New-Item -ItemType Directory -Force -Path $env:HF_HOME, $env:HF_HUB_CACHE, $env:HF_DATASETS_CACHE, $env:SPEECHRECON_MODEL_DIR, $env:SPEECHRECON_DATASET_DIR | Out-Null

$downloadArgs = @(
    "scripts\download_hf_assets.py",
    "--scope", "all",
    "--include-experimental-models",
    "--cache-dir", $env:HF_HUB_CACHE,
    "--dataset-cache-dir", $env:HF_DATASETS_CACHE,
    "--download-mbps", "$DownloadMbps"
)

if (-not $IncludeDatasets) {
    $downloadArgs += "--models-only"
}

if ($IncludeDiscoveredDatasets) {
    $downloadArgs += "--include-discovered-datasets"
}

if (-not $Run) {
    $downloadArgs += "--dry-run"
    Write-Host "[download] Plan only. Add -Run to actually download."
}

Push-Location $repoRoot
try {
    & $python @downloadArgs
    if ($Run -and $WarmLoad) {
        $warmArgs = @("scripts\warm_load_model_catalog.py", "--include-experimental-models")
        & $python @warmArgs
    }
}
finally {
    Pop-Location
}
