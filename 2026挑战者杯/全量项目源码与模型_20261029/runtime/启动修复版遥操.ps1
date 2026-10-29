param([switch]$PreviousMapping,[switch]$NoGraspAssist,[switch]$PreviousControl,[switch]$Direct,[switch]$NoDataset)
$ErrorActionPreference = 'Stop'
$teleopProject = Join-Path $PSScriptRoot 'latest\openpi--main'
Push-Location -LiteralPath $teleopProject
try {
    if ($PreviousMapping) {
        & .\run.ps1 -Mode single-arm -ScriptArgs '--position-frame','upper-body','--depth-source','palm-relative','--direct-wrist-policy','position','--no-auto-grasp-assist','--control-hz','20','--robot-preview-hz','10','--close-on-finish'
    } elseif ($PreviousControl) {
        & .\run.ps1 -Mode single-arm -ScriptArgs '--direct-wrist-policy','position','--no-visual-grasp-align','--control-hz','20','--robot-preview-hz','10','--close-on-finish'
    } elseif ($NoGraspAssist) {
        & .\run.ps1 -Mode single-arm -ScriptArgs '--no-auto-grasp-assist','--close-on-finish'
    } elseif ($Direct) {
        & .\run.ps1 -Mode single-arm -ScriptArgs '--close-on-finish'
    } else {
        $sharedLaunchArgs=@('--close-on-finish')
        if($NoDataset){$sharedLaunchArgs+='--no-collect-shared-dataset'}
        & .\run.ps1 -Mode shared-grasp -ScriptArgs $sharedLaunchArgs
    }
    $teleopExit = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $teleopExit
