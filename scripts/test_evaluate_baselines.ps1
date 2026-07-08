param(
    [string[]]$Configs,
    [ValidateSet("b1", "b2", "b3", "b4")]
    [string[]]$Baselines,
    [ValidateSet("primevul", "rust", "all")]
    [string]$Dataset = "all",
    [switch]$Overwrite
)

. "$PSScriptRoot\test_common.ps1"

Enter-ProjectRoot
try {
    $configList = Resolve-ConfigList $Configs
    $hasExplicitBaselines = $PSBoundParameters.ContainsKey("Baselines") -and $Baselines -and $Baselines.Count -gt 0
    $baselineList = Resolve-BaselineList $Baselines

    foreach ($config in $configList) {
        Assert-ProjectPath $config "config"
        if (-not $hasExplicitBaselines) {
            Write-Host "Evaluating all baselines with $config on $Dataset"
            $args = @("evaluate.py", "--config", $config, "--dataset", $Dataset)
            if ($Overwrite) {
                $args += "--overwrite"
            }
            Invoke-ProjectPython -Arguments $args
        }
        else {
            foreach ($baseline in $baselineList) {
                Write-Host "Evaluating $($baseline.ToUpper()) with $config on $Dataset"
                $args = @("evaluate.py", "--config", $config, "--baseline", $baseline, "--dataset", $Dataset)
                if ($Overwrite) {
                    $args += "--overwrite"
                }
                Invoke-ProjectPython -Arguments $args
            }
        }
    }
}
finally {
    Pop-Location
}
