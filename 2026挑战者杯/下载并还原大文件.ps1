$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskManifest = Get-Content -LiteralPath (Join-Path $taskRoot '大文件还原清单.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$taskArchiveInfo = $taskManifest.files | Where-Object { $_.url }
$taskArchive = Join-Path $taskRoot $taskArchiveInfo.path
if (-not (Test-Path -LiteralPath $taskArchive)) {
    foreach ($taskPart in $taskArchiveInfo.parts) {
        $taskPartPath = Join-Path $taskRoot $taskPart.path
        if (-not (Test-Path -LiteralPath $taskPartPath)) {
            Write-Host "Downloading $($taskPart.path)..."
            Invoke-WebRequest -Uri $taskPart.url -OutFile "$taskPartPath.partial" -UseBasicParsing
            if ((Get-FileHash -LiteralPath "$taskPartPath.partial" -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskPart.sha256) { throw 'Part SHA-256 mismatch' }
            Move-Item -LiteralPath "$taskPartPath.partial" -Destination $taskPartPath
        }
        if ((Get-FileHash -LiteralPath $taskPartPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskPart.sha256) { throw 'Existing part SHA-256 mismatch' }
    }
    Write-Host 'Combining verified parts...'
    $taskCombined = [System.IO.File]::Create("$taskArchive.partial")
    try {
        foreach ($taskPart in $taskArchiveInfo.parts) {
            $taskInput = [System.IO.File]::OpenRead((Join-Path $taskRoot $taskPart.path))
            try { $taskInput.CopyTo($taskCombined) } finally { $taskInput.Dispose() }
        }
    } finally { $taskCombined.Dispose() }
    if ((Get-FileHash -LiteralPath "$taskArchive.partial" -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskArchiveInfo.sha256) { throw 'Combined ZIP SHA-256 mismatch' }
    Move-Item -LiteralPath "$taskArchive.partial" -Destination $taskArchive
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
