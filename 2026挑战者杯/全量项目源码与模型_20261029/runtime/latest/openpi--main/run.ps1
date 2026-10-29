param(
    [ValidateSet('single-arm', 'shared-grasp', 'depth-diagnose', 'live', 'human-task', 'human-task-replay', 'human-task-validate', 'replay', 'grasp', 'tabletop', 'tabletop-ui', 'tabletop-check', 'fusion', 'landmark-fusion', 'fusion-replay', 'video-skill', 'check', 'test', 'python')]
    [string]$Mode = 'fusion',
    [string]$Python,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs = @()
)

$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskB = Join-Path $taskRoot 'openarm_cloth'
$taskConfig = Join-Path $taskRoot '.runtime.local.json'
if (!$Python) { $Python = $env:OPENPI_PYTHON }
if (!$Python -and (Test-Path -LiteralPath $taskConfig)) {
    $Python = (Get-Content -Raw -LiteralPath $taskConfig | ConvertFrom-Json).python
}
if (!$Python) {
    $taskCandidates = @((Join-Path $taskB '.venv\Scripts\python.exe'))
    if ($env:CONDA_PREFIX -and (Split-Path $env:CONDA_PREFIX -Leaf) -eq 'openpi') {
        $taskCandidates += Join-Path $env:CONDA_PREFIX 'python.exe'
    }
    $taskConda = Get-Command conda -ErrorAction SilentlyContinue
    if ($taskConda) {
        $taskEnvs = (& conda env list --json | ConvertFrom-Json).envs
        $taskCandidates += @($taskEnvs | Where-Object { (Split-Path $_ -Leaf) -eq 'openpi' } |
            ForEach-Object { Join-Path $_ 'python.exe' })
    }
    $Python = $taskCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
        Select-Object -First 1
}
if (!$Python -or !(Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'B Python not found. Use -Python <openpi python.exe> or set OPENPI_PYTHON. See RUNNING.md.'
}

# Windows MuJoCo cannot reliably open non-ASCII XML/mesh paths.
if ($taskRoot -match '[^\x00-\x7F]') {
    throw 'Move this checkout to an ASCII-only path, for example D:\openpi\latest\openpi-main.'
}
$taskTmp = Join-Path $taskB 'outputs\runtime_tmp'
New-Item -ItemType Directory -Force -Path $taskTmp | Out-Null
$taskPreviousEnvironment = @{}
foreach ($taskName in @('TEMP', 'TMP', 'PYTHONUTF8', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')) {
    $taskPreviousEnvironment[$taskName] = [Environment]::GetEnvironmentVariable($taskName, 'Process')
}
$env:TEMP = $taskTmp
$env:TMP = $taskTmp
$env:PYTHONUTF8 = '1'
if (!$env:OPENBLAS_NUM_THREADS) { $env:OPENBLAS_NUM_THREADS = '1' }
if (!$env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS = '1' }
Write-Host "B Python: $Python"
Push-Location -LiteralPath $taskB
try {
    switch ($Mode) {
        'live'   { & $Python -u (Join-Path $taskB 'live_bimanual_human.py') @ScriptArgs }
        'human-task' { & $Python -u (Join-Path $taskB 'run_human_task.py') @ScriptArgs }
        'depth-diagnose' { & $Python -u (Join-Path $taskB 'diagnose_teleop_depth.py') @ScriptArgs }
        'single-arm' {
            & $Python -u (Join-Path $taskB 'run_human_task.py') --control direct --mapping wrist --position-frame fixed-image --body-position-source mediapipe --depth-source palm-planar --depth-gain 1.0 --direct-wrist-policy palm --auto-grasp-assist --arms right --gripper-mode continuous --no-images --realtime --control-hz 30 --preview-hz 30 --robot-preview-hz 20 @ScriptArgs
        }
        'shared-grasp' {
            & $Python -u (Join-Path $taskB 'run_human_task.py') --control direct --mapping wrist --position-frame fixed-image --body-position-source mediapipe --depth-source palm-planar --depth-gain 1.0 --direct-wrist-policy palm --shared-grasp --collect-shared-dataset --arms right --gripper-mode continuous --no-images --realtime --control-hz 30 --preview-hz 30 --robot-preview-hz 20 @ScriptArgs
        }
        'human-task-replay' { & $Python -u (Join-Path $taskB 'replay_human_task.py') @ScriptArgs }
        'human-task-validate' { & $Python -u (Join-Path $taskB 'validate_human_task.py') @ScriptArgs }
        'replay' { & $Python -u (Join-Path $taskB 'demo_member_ab.py') @ScriptArgs }
        'fusion' { & $Python -u (Join-Path $taskB 'run_a_reference_on_b.py') @ScriptArgs }
        'landmark-fusion' { & $Python -u (Join-Path $taskB 'run_fused_motion.py') @ScriptArgs }
        'video-skill' { & $Python -u (Join-Path $taskB 'demo_video_grasp.py') @ScriptArgs }
        'fusion-replay' { & $Python -u (Join-Path $taskB 'replay_video_fusion.py') @ScriptArgs }
        'tabletop' {
            # Fixture options only construct the test world; they are not vision hints.
            $taskFixtureGiven = @($ScriptArgs | Where-Object {
                $_ -match '^--(shape|color|size|random-fixture)(=|$)'
            }).Count -gt 0
            $taskTabletopDefaults = @()
            if (!$taskFixtureGiven) { $taskTabletopDefaults += '--random-fixture' }
            & $Python -u (Join-Path $taskB 'demo_tabletop_objects.py') @taskTabletopDefaults @ScriptArgs
        }
        'tabletop-ui' { & $Python (Join-Path $taskB 'launch_tabletop_grasp.pyw') @ScriptArgs }
        'tabletop-check' { & $Python -u (Join-Path $taskB 'benchmark_tabletop_objects.py') @ScriptArgs }
        'grasp'  {
            $taskGraspDefaults = @('--preview-initial-s', '5', '--post-lift-s', '1.5')
            $taskPlacementGiven = @($ScriptArgs | Where-Object {
                $_ -match '^--(check|cube-test-pose|random-cube|interactive-place-s)(=|$)'
            }).Count -gt 0
            if (!$taskPlacementGiven) { $taskGraspDefaults += @('--interactive-place-s', '86400') }
            & $Python -u (Join-Path $taskB 'demo_grasp_viewer.py') @taskGraspDefaults @ScriptArgs
        }
        'check'  { & $Python (Join-Path $taskB 'check_runtime.py') @ScriptArgs }
        'test'   {
            & $Python -m unittest discover @ScriptArgs
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
            Push-Location -LiteralPath (Join-Path $taskRoot 'pose2arm-main')
            try { & $Python -m unittest discover -s tests @ScriptArgs }
            finally { Pop-Location }
        }
        'python' {
            if (!$ScriptArgs.Count) { throw 'Use -ScriptArgs with a Python script or -c expression.' }
            & $Python @ScriptArgs
        }
    }
    $taskExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
    foreach ($taskName in @('TEMP', 'TMP', 'PYTHONUTF8', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')) {
        [Environment]::SetEnvironmentVariable($taskName, $taskPreviousEnvironment[$taskName], 'Process')
    }
}
exit $taskExitCode
