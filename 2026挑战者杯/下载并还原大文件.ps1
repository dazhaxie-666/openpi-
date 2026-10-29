$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskManifest = Get-Content -LiteralPath (Join-Path $taskRoot '大文件还原清单.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$taskArchiveInfo = $taskManifest.files | Where-Object { $_.url }
$taskArchive = Join-Path $taskRoot $taskArchiveInfo.path
if (-not (Test-Path -LiteralPath $taskArchive)) {
    $taskPartial = "$taskArchive.partial"
    Write-Host 'Downloading original project archive...'
    Invoke-WebRequest -Uri $taskArchiveInfo.url -OutFile $taskPartial -UseBasicParsing
    if ((Get-FileHash -LiteralPath $taskPartial -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskArchiveInfo.sha256) { throw 'Archive SHA-256 mismatch' }
    Move-Item -LiteralPath $taskPartial -Destination $taskArchive
}
if ((Get-FileHash -LiteralPath $taskArchive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskArchiveInfo.sha256) { throw 'Existing archive SHA-256 mismatch' }
Add-Type -AssemblyName System.IO.Compression.FileSystem
$taskZip = [System.IO.Compression.ZipFile]::OpenRead($taskArchive)
try {
    foreach ($taskItem in $taskManifest.files | Where-Object { $_.zip_entry }) {
        $taskDestination = Join-Path $taskRoot $taskItem.path
        if (Test-Path -LiteralPath $taskDestination) {
            if ((Get-FileHash -LiteralPath $taskDestination -Algorithm SHA256).Hash.ToLowerInvariant() -eq $taskItem.sha256) { continue }
            throw "Existing model hash mismatch: $($taskItem.path)"
        }
        $taskEntry = $taskZip.GetEntry($taskItem.zip_entry)
        if ($null -eq $taskEntry) { throw "Missing archive entry: $($taskItem.zip_entry)" }
        [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($taskDestination)) | Out-Null
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($taskEntry, $taskDestination, $false)
        if ((Get-FileHash -LiteralPath $taskDestination -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskItem.sha256) { throw "Extracted model hash mismatch: $($taskItem.path)" }
    }
} finally { $taskZip.Dispose() }
Write-Host 'PASS: all large files restored and SHA-256 verified.'
