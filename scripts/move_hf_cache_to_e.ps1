param(
    [string]$TargetRoot = "E:\speechRecon-cache",
    [string]$SourceHfCache = "$env:USERPROFILE\.cache\huggingface",
    [switch]$WhatIfOnly
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$targetRootPath = [System.IO.Path]::GetFullPath($TargetRoot)
$targetHf = Join-Path $targetRootPath "huggingface"
$targetModels = Join-Path $targetRootPath "models"
$targetDatasets = Join-Path $targetRootPath "datasets"
$envPath = Join-Path $repoRoot ".env"

function Get-DirectorySizeBytes([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return 0
    }
    $sum = (Get-ChildItem -LiteralPath $Path -Recurse -Force -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    if ($null -eq $sum) {
        return 0
    }
    return [int64]$sum
}

function Format-Size([int64]$Bytes) {
    if ($Bytes -ge 1GB) {
        return "{0:N2} GB" -f ($Bytes / 1GB)
    }
    if ($Bytes -ge 1MB) {
        return "{0:N1} MB" -f ($Bytes / 1MB)
    }
    return "$Bytes B"
}

function Move-DirectoryContents([string]$Source, [string]$Destination) {
    if (-not (Test-Path -LiteralPath $Source)) {
        Write-Host "[move] Missing source, skipping: $Source"
        return
    }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    $items = Get-ChildItem -LiteralPath $Source -Force
    foreach ($item in $items) {
        $dest = Join-Path $Destination $item.Name
        if (Test-Path -LiteralPath $dest) {
            Write-Host "[move] Exists at target, leaving source item in place: $dest"
            continue
        }
        Write-Host "[move] $($item.FullName) -> $dest"
        if (-not $WhatIfOnly) {
            Move-Item -LiteralPath $item.FullName -Destination $dest
        }
    }
}

function Set-EnvValue([string[]]$Lines, [string]$Key, [string]$Value) {
    $pattern = "^\s*$([regex]::Escape($Key))\s*="
    $updated = $false
    $result = foreach ($line in $Lines) {
        if ($line -match $pattern) {
            $updated = $true
            "$Key=$Value"
        } else {
            $line
        }
    }
    if (-not $updated) {
        $result += "$Key=$Value"
    }
    return $result
}

Write-Host "[move] Target root: $targetRootPath"
Write-Host "[move] Source Hugging Face cache: $SourceHfCache"

$sourceSize = Get-DirectorySizeBytes $SourceHfCache
Write-Host "[move] Source HF cache size: $(Format-Size $sourceSize)"

if (-not $WhatIfOnly) {
    New-Item -ItemType Directory -Force -Path $targetHf, $targetModels, $targetDatasets | Out-Null
}

Move-DirectoryContents -Source $SourceHfCache -Destination $targetHf
Move-DirectoryContents -Source (Join-Path $repoRoot "models") -Destination $targetModels
Move-DirectoryContents -Source (Join-Path $repoRoot "datasets") -Destination $targetDatasets

$envLines = @()
if (Test-Path -LiteralPath $envPath) {
    $envLines = Get-Content -LiteralPath $envPath
}

$envLines = Set-EnvValue $envLines "SPEECHRECON_CACHE_ROOT" $targetRootPath
$envLines = Set-EnvValue $envLines "SPEECHRECON_MODEL_DIR" $targetModels
$envLines = Set-EnvValue $envLines "SPEECHRECON_DATASET_DIR" $targetDatasets
$envLines = Set-EnvValue $envLines "HF_HOME" $targetHf
$envLines = Set-EnvValue $envLines "HF_HUB_CACHE" (Join-Path $targetHf "hub")
$envLines = Set-EnvValue $envLines "HF_DATASETS_CACHE" (Join-Path $targetHf "datasets")
$envLines = Set-EnvValue $envLines "TRANSFORMERS_CACHE" (Join-Path $targetHf "hub")

Write-Host "[move] Updating .env cache paths"
if (-not $WhatIfOnly) {
    Set-Content -LiteralPath $envPath -Value $envLines -Encoding UTF8
}

$targetSize = Get-DirectorySizeBytes $targetRootPath
Write-Host "[move] Target cache size now: $(Format-Size $targetSize)"
Write-Host "[move] Done. Restart Flask/Vite so the new cache paths are picked up."
