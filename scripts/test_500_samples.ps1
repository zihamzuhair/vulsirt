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
$Config = "configs/500_samples.yaml"

if (-not $SkipPrepare) {
    $prepareArgs = @("-Configs", $Config)
    if ($FullRebuild) {
        $prepareArgs += "-FullRebuild"
    }
    & "$PSScriptRoot\test_prepare_data.ps1" @prepareArgs
}

if (-not $SkipTrain) {
    & "$PSScriptRoot\test_train_baselines.ps1" -Configs $Config
}

if (-not $SkipEvaluate) {
    $evaluateArgs = @("-Configs", $Config)
    if ($Overwrite) {
        $evaluateArgs += "-Overwrite"
    }
    & "$PSScriptRoot\test_evaluate_baselines.ps1" @evaluateArgs
}

if (-not $SkipInitializedB4) {
    $initializedArgs = @("-Configs", $Config)
    if ($Overwrite) {
        $initializedArgs += "-Overwrite"
    }
    & "$PSScriptRoot\test_b4_initialized.ps1" @initializedArgs
}

if (-not $SkipScanner) {
    & "$PSScriptRoot\test_scanner_smoke.ps1" -Configs $Config
}
