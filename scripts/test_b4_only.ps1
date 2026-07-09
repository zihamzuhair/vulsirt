param(
    [string[]]$Configs = @("configs/1000_samples.yaml"),
    [ValidateSet("primevul", "rust", "all")]
    [string]$Dataset = "all",
    [switch]$FullRebuild,
    [switch]$Overwrite,
    [switch]$SkipPrepare,
    [switch]$SkipTrain,
    [switch]$SkipEvaluate,
    [switch]$SkipScanner
)

$ErrorActionPreference = "Stop"

. "$PSScriptRoot\test_common.ps1"

Enter-ProjectRoot
$RunName = "b4_only_" + ((Resolve-ConfigList $Configs | Select-Object -First 1 | ForEach-Object { Get-RunName $_ }))
$Timing = Start-TestTiming -RunName $RunName
$Status = "completed"
try {
    $configList = Resolve-ConfigList $Configs

    if (-not $SkipPrepare) {
        if ($FullRebuild) {
            & "$PSScriptRoot\test_prepare_data.ps1" -Configs $configList -FullRebuild
        }
        else {
            & "$PSScriptRoot\test_prepare_data.ps1" -Configs $configList
        }
    }

    if (-not $SkipTrain) {
        & "$PSScriptRoot\test_train_baselines.ps1" -Configs $configList -Baselines b4
    }

    if (-not $SkipEvaluate) {
        if ($Overwrite) {
            & "$PSScriptRoot\test_evaluate_baselines.ps1" -Configs $configList -Baselines b4 -Dataset $Dataset -Overwrite
        }
        else {
            & "$PSScriptRoot\test_evaluate_baselines.ps1" -Configs $configList -Baselines b4 -Dataset $Dataset
        }
    }

    if (-not $SkipScanner) {
        & "$PSScriptRoot\test_scanner_smoke.ps1" -Configs $configList
    }
}
catch {
    $Status = "failed"
    throw
}
finally {
    Stop-TestTiming -Timing $Timing -Status $Status
    Pop-Location
}
