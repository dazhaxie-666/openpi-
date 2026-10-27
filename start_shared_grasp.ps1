param([switch]$NoDataset, [string]$Python)
$ErrorActionPreference = 'Stop'
$sharedArgs = @('--close-on-finish')
if ($NoDataset) { $sharedArgs += '--no-collect-shared-dataset' }
$launch = @{ Mode = 'shared-grasp'; ScriptArgs = $sharedArgs }
if ($Python) { $launch.Python = $Python }
& (Join-Path $PSScriptRoot 'run.ps1') @launch
exit $LASTEXITCODE
