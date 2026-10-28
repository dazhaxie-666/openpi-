[CmdletBinding()]
param(
    [ValidateSet('Check','Camera','Test','Smoke','Pilot','Collect','Coverage','Recovery','Eval','Audit','Replay','Review','Summary')]
    [string]$Mode = 'Check',
    [ValidateSet('PickPlace','Hold')]
    [string]$Task = 'PickPlace',
    [string]$Session,
    [ValidateSet('center','left','right','near','far','yaw0','yaw60','random')]
    [string]$Pose = 'center',
    [int]$Camera = 0,
    [string]$Operator = 'operator01',
    [ValidateRange(1,999999)]
    [int]$Batch = 1,
    [int]$Trials = 1,
    [ValidateRange(0,3600)]
    [double]$ResetSeconds = 3,
    [double]$Seconds = 180,
    [int]$Steps = 1500,
    [Nullable[int]]$Seed,
    [int]$Trial = 0,
    [ValidateSet('accept','reject')]
    [string]$Decision,
    [string]$Note = '',
    [switch]$Visible,
    [string]$Python
)
$ErrorActionPreference = 'Stop'
$datasetParent = Split-Path $PSScriptRoot -Parent
$externalProject = Join-Path $datasetParent 'latest\openpi--main'
$datasetProject = if (Test-Path -LiteralPath (Join-Path $externalProject 'run.ps1') -PathType Leaf) {
    $externalProject
} elseif (Test-Path -LiteralPath (Join-Path $datasetParent 'run.ps1') -PathType Leaf) {
    $datasetParent
} else {
    throw 'Could not locate the OpenPI project beside dataset/ or at ../latest/openpi--main.'
}
$datasetRunner = Join-Path $datasetProject 'run.ps1'
if (!(Test-Path -LiteralPath $datasetRunner -PathType Leaf)) { throw 'Production run.ps1 is missing.' }
$datasetCli = Join-Path $PSScriptRoot 'scripts\dataset_cli.py'
$datasetModeMap = @{Pilot='pilot'; Collect='core'; Coverage='coverage'; Recovery='recovery'; Eval='eval'}
if ($Mode -eq 'Test') {
    $datasetCliArgs = @((Join-Path $PSScriptRoot 'tests\run_tests.py'))
} else {
    $datasetOperation = if ($datasetModeMap.ContainsKey($Mode)) { 'collect' } else { $Mode.ToLowerInvariant() }
    $datasetCliArgs = @($datasetCli, $datasetOperation, '--camera', "$Camera", '--operator', $Operator,
                        '--task', $Task.ToLowerInvariant(), '--pose', $Pose, '--batch', "$Batch", '--trials', "$Trials", '--reset-seconds', "$ResetSeconds",
                        '--seconds', "$Seconds", '--steps', "$Steps")
    if ($datasetModeMap.ContainsKey($Mode)) { $datasetCliArgs += @('--stage', $datasetModeMap[$Mode]) }
    if ($Session) { $datasetCliArgs += @('--session', $Session) }
    if ($null -ne $Seed) { $datasetCliArgs += @('--seed', "$Seed") }
    if ($Visible) { $datasetCliArgs += '--visible' }
    if ($Mode -eq 'Review') {
        if ($Trial -lt 1 -or !$Decision -or !$Note.Trim()) { throw 'Review needs -Trial, -Decision and -Note.' }
        $datasetCliArgs += @('--trial', "$Trial", '--decision', $Decision, '--note', $Note)
    }
}
$datasetPreviousBytecode = $env:PYTHONDONTWRITEBYTECODE
$env:PYTHONDONTWRITEBYTECODE = '1'
try {
    $datasetRunParameters = @{Mode='python'; ScriptArgs=$datasetCliArgs}
    if ($Python) { $datasetRunParameters['Python'] = $Python }
    & $datasetRunner @datasetRunParameters
    $datasetExitCode = $LASTEXITCODE
} finally {
    $env:PYTHONDONTWRITEBYTECODE = $datasetPreviousBytecode
}
exit $datasetExitCode
