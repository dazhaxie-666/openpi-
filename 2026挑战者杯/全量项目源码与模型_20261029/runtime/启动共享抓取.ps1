param([switch]$NoDataset)
$ErrorActionPreference='Stop'
$sharedProject=Join-Path $PSScriptRoot 'latest\openpi--main'
Push-Location -LiteralPath $sharedProject
try {
    $sharedLaunchArgs=@('--close-on-finish')
    if($NoDataset){$sharedLaunchArgs+='--no-collect-shared-dataset'}
    & .\run.ps1 -Mode shared-grasp -ScriptArgs $sharedLaunchArgs
    $sharedExit=$LASTEXITCODE
} finally {Pop-Location}
exit $sharedExit
