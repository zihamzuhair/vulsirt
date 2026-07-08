param(
    [string[]]$Configs,
    [ValidateSet("b1", "b2", "b3", "b4")]
    [string]$Baseline = "b4",
    [string]$SourceFile = "test/sample.c"
)

. "$PSScriptRoot\test_common.ps1"

function Get-ConfigCheckpointDir {
    param([string]$ConfigPath)

    $script = @"
from helpers.config_loader import load_config
cfg = load_config(r'$ConfigPath')
print(cfg['paths']['checkpoints'])
"@
    $value = & python -c $script
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to read checkpoint directory from $ConfigPath"
    }
    return ($value | Select-Object -Last 1).Trim()
}

Enter-ProjectRoot
try {
    $configList = Resolve-ConfigList $Configs
    Assert-ProjectPath $SourceFile "scanner source file"

    foreach ($config in $configList) {
        Assert-ProjectPath $config "config"
        $checkpointDir = Get-ConfigCheckpointDir $config
        $checkpointPath = Join-Path $checkpointDir "$($Baseline)_best.pt"

        if (-not (Test-Path -LiteralPath $checkpointPath)) {
            Write-Host "Skipping scanner smoke test for $config because checkpoint is missing: $checkpointPath"
            continue
        }

        Write-Host "Scanner smoke test for $($Baseline.ToUpper()) with $config"
        Invoke-ProjectPython -Arguments @("scanner.py", "--config", $config, "--baseline", $Baseline, "--source-file", $SourceFile)
    }
}
finally {
    Pop-Location
}
