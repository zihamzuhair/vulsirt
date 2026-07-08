$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $RepoRoot

try {
    Write-Host "1. Preprocessing PrimeVul raw dataset"
    python preprocess.py

    Write-Host "2. Compiling processed datasets to LLVM"
    python compiler.py

    Write-Host "Pipeline complete"
}
finally {
    Pop-Location
}
