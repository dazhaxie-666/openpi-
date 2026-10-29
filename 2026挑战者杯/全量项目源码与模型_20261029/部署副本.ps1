param([string]$Destination='D:\openpi\challengecup_runtime_20261029')
$ErrorActionPreference='Stop'
$resolvedTarget=[System.IO.Path]::GetFullPath($Destination)
if($resolvedTarget -match '[^\x00-\x7F]'){throw 'Destination must contain ASCII characters only.'}
if(Test-Path -LiteralPath $resolvedTarget){throw 'Destination already exists. Choose a NEW directory; no existing data will be overwritten.'}
$sourceRuntime=Join-Path $PSScriptRoot 'runtime'
if(!(Test-Path -LiteralPath (Join-Path $sourceRuntime 'latest\openpi--main\run.ps1'))){throw 'Bundled project run.ps1 is missing.'}
New-Item -ItemType Directory -Path $resolvedTarget | Out-Null
Get-ChildItem -LiteralPath $sourceRuntime -Force | Copy-Item -Destination $resolvedTarget -Recurse
Write-Host "Deployed independent copy to: $resolvedTarget"
Write-Host 'Pass -Python to dataset/dataset.ps1; see docs for collection commands.'
