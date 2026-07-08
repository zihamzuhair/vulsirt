param(
    [string[]]$Configs,
    [ValidateSet("b1", "b2", "b3", "b4")]
    [string[]]$Baselines
)

. "$PSScriptRoot\test_common.ps1"

Enter-ProjectRoot
try {
    $configList = Resolve-ConfigList $Configs
    $baselineList = Resolve-BaselineList $Baselines

    foreach ($config in $configList) {
        Assert-ProjectPath $config "config"
        foreach ($baseline in $baselineList) {
            Write-Host "Training $($baseline.ToUpper()) with $config"
            Invoke-ProjectPython -Arguments @("train.py", "--config", $config, "--baseline", $baseline)
        }
    }
}
finally {
    Pop-Location
}
