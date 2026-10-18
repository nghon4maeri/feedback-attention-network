param(
    [string]$Kernel = "namnguynnnn/fanet-setup",
    [string]$OutDir = "kaggle\outputs"
)

$ErrorActionPreference = "Stop"

Write-Host "Downloading Kaggle kernel output..."
Write-Host "  Kernel : $Kernel"
Write-Host "  Dest   : $OutDir"
Write-Host ""

if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
}

kaggle kernels output $Kernel -p $OutDir

Write-Host ""
Write-Host "Downloaded files:"
Get-ChildItem -Recurse $OutDir -File | ForEach-Object {
    $size = [math]::Round($_.Length / 1MB, 2)
    Write-Host ("  {0}  ({1} MB)" -f $_.FullName, $size)
}
