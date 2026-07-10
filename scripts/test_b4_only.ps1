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
$Timing = $null
$Status = "completed"
try {
    $baseConfigList = Resolve-ConfigList $Configs
    $configList = @()
    foreach ($config in $baseConfigList) {
        Assert-ProjectPath $config "config"
        $generatedConfig = & python -m helpers.experiment_naming --config $config --output-dir temp/generated_configs
        if ($LASTEXITCODE -ne 0) {
            throw "failed to generate experiment config for $config"
        }
        $configList += $generatedConfig.Trim()
    }

    $RunName = if ($configList.Count -eq 1) {
        Get-RunName $configList[0]
    }
    else {
        "b4_only_multi_config"
    }
    $Timing = Start-TestTiming -RunName $RunName

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
    if ($Timing) {
        Stop-TestTiming -Timing $Timing -Status $Status
    }
    Pop-Location
}
