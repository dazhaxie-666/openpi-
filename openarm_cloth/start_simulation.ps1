param([Parameter(ValueFromRemainingArguments = $true)][string[]]$ScriptArgs = @())
$ErrorActionPreference = 'Stop'
& (Join-Path (Split-Path $PSScriptRoot -Parent) 'run.ps1') -Mode grasp -ScriptArgs $ScriptArgs
exit $LASTEXITCODE
