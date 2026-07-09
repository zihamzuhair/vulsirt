param(
    [switch]$FullRebuild,
    [switch]$Overwrite,
    [switch]$SkipPrepare,
    [switch]$SkipTrain,
    [switch]$SkipEvaluate,
    [switch]$SkipInitializedB4,
    [switch]$SkipScanner
)

$ErrorActionPreference = "Stop"

if (-not $SkipPrepare) {
    if ($FullRebuild) {
        & "$PSScriptRoot\test_prepare_data.ps1" -FullRebuild
    }
    else {
        & "$PSScriptRoot\test_prepare_data.ps1"
    }
}

if (-not $SkipTrain) {
    & "$PSScriptRoot\test_train_baselines.ps1"
}

if (-not $SkipEvaluate) {
    if ($Overwrite) {
        & "$PSScriptRoot\test_evaluate_baselines.ps1" -Overwrite
    }
    else {
        & "$PSScriptRoot\test_evaluate_baselines.ps1"
    }
}

if (-not $SkipInitializedB4) {
    if ($Overwrite) {
        & "$PSScriptRoot\test_b4_initialized.ps1" -Overwrite
    }
    else {
        & "$PSScriptRoot\test_b4_initialized.ps1"
    }
}

if (-not $SkipScanner) {
    & "$PSScriptRoot\test_scanner_smoke.ps1"
}
