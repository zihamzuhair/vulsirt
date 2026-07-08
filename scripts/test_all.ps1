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
    $prepareArgs = @()
    if ($FullRebuild) {
        $prepareArgs += "-FullRebuild"
    }
    & "$PSScriptRoot\test_prepare_data.ps1" @prepareArgs
}

if (-not $SkipTrain) {
    & "$PSScriptRoot\test_train_baselines.ps1"
}

if (-not $SkipEvaluate) {
    $evaluateArgs = @()
    if ($Overwrite) {
        $evaluateArgs += "-Overwrite"
    }
    & "$PSScriptRoot\test_evaluate_baselines.ps1" @evaluateArgs
}

if (-not $SkipInitializedB4) {
    $initializedArgs = @()
    if ($Overwrite) {
        $initializedArgs += "-Overwrite"
    }
    & "$PSScriptRoot\test_b4_initialized.ps1" @initializedArgs
}

if (-not $SkipScanner) {
    & "$PSScriptRoot\test_scanner_smoke.ps1"
}
