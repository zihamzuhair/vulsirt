param(
    [string[]]$Configs,
    [switch]$FullRebuild
)

. "$PSScriptRoot\test_common.ps1"

Enter-ProjectRoot
try {
    $configList = Resolve-ConfigList $Configs

    if ($FullRebuild) {
        Write-Host "1. Preprocessing PrimeVul raw dataset"
        Invoke-ProjectPython -Arguments @("-u", "preprocess.py")

        Write-Host "2. Compiling processed datasets to LLVM"
        Invoke-ProjectPython -Arguments @("-u", "compiler.py")
    }

    Write-Host "3. Building experiment datasets"
    $args = @("-m", "helpers.build_experiment_dataset", "--config") + $configList
    Invoke-ProjectPython -Arguments $args
}
finally {
    Pop-Location
}
