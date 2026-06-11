param(
    [string]$HfRoot = "E:\speechRecon-cache\huggingface"
)

$ErrorActionPreference = "Stop"

$hfRootPath = [System.IO.Path]::GetFullPath($HfRoot)
$hubPath = Join-Path $hfRootPath "hub"
$datasetsPath = Join-Path $hfRootPath "datasets"

if (-not (Test-Path -LiteralPath $hfRootPath)) {
    throw "Hugging Face cache root does not exist: $hfRootPath"
}

New-Item -ItemType Directory -Force -Path $hubPath, $datasetsPath | Out-Null

function Move-IfMissing([System.IO.FileSystemInfo]$Item, [string]$DestinationRoot) {
    $destination = Join-Path $DestinationRoot $Item.Name
    if (Test-Path -LiteralPath $destination) {
        Write-Host "[normalize] target exists, leaving source in place: $destination"
        return
    }
    Write-Host "[normalize] $($Item.FullName) -> $destination"
    Move-Item -LiteralPath $Item.FullName -Destination $destination
}

Get-ChildItem -LiteralPath $hfRootPath -Force -Directory |
    Where-Object { $_.Name -like "models--*" -or $_.Name -eq ".locks" -or $_.Name -eq "downloads" } |
    ForEach-Object { Move-IfMissing $_ $hubPath }

Get-ChildItem -LiteralPath $hfRootPath -Force -Directory |
    Where-Object { $_.Name -like "*___*" } |
    ForEach-Object { Move-IfMissing $_ $datasetsPath }

Write-Host "[normalize] Done."
Write-Host "[normalize] Hub cache: $hubPath"
Write-Host "[normalize] Dataset cache: $datasetsPath"
