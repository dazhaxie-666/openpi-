param(
    [string]$Python
)

# Resolve the same B runtime as every other entry point; no old sibling venv.
$taskRunner = Join-Path (Split-Path -Parent $PSScriptRoot) 'run.ps1'
$taskArguments = @{ Mode = 'tabletop-ui' }
if ($Python) { $taskArguments.Python = $Python }
& $taskRunner @taskArguments
exit $LASTEXITCODE
