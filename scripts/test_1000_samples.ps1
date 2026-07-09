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
$Config = "configs/1000_samples.yaml"

if (-not $SkipPrepare) {
    if ($FullRebuild) {
        & "$PSScriptRoot\test_prepare_data.ps1" -Configs $Config -FullRebuild
    }
    else {
        & "$PSScriptRoot\test_prepare_data.ps1" -Configs $Config
    }
}

if (-not $SkipTrain) {
    & "$PSScriptRoot\test_train_baselines.ps1" -Configs $Config
}

if (-not $SkipEvaluate) {
    if ($Overwrite) {
        & "$PSScriptRoot\test_evaluate_baselines.ps1" -Configs $Config -Overwrite
    }
    else {
        & "$PSScriptRoot\test_evaluate_baselines.ps1" -Configs $Config
    }
}

if (-not $SkipInitializedB4) {
    if ($Overwrite) {
        & "$PSScriptRoot\test_b4_initialized.ps1" -Configs $Config -Overwrite
    }
    else {
        & "$PSScriptRoot\test_b4_initialized.ps1" -Configs $Config
    }
}

if (-not $SkipScanner) {
    & "$PSScriptRoot\test_scanner_smoke.ps1" -Configs $Config
}
