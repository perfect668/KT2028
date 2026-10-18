param([string]$Root = $PSScriptRoot, [switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path -LiteralPath $Root).Path
$tool = Join-Path $Root 'tools/aria2-1.37.0-win-64bit-build1/aria2c.exe'
$inputList = Join-Path $Root 'metadata/surgen413_aria2.txt'
$manifest = Join-Path $Root 'metadata/surgen413_download_manifest.json'
foreach ($p in @($tool, $inputList, $manifest)) {
    if (!(Test-Path -LiteralPath $p -PathType Leaf)) { throw "Missing required file: $p" }
}
$items = Get-Content -LiteralPath $manifest -Raw -Encoding UTF8 | ConvertFrom-Json
if ($items.Count -ne 413 -or @($items.filename | Select-Object -Unique).Count -ne 413) {
    throw 'Expected exactly 413 unique source images.'
}
$wsi = Join-Path $Root 'wsi'
$logs = Join-Path $Root 'logs'
$tempDir = Join-Path $Root 'tmp'
foreach ($p in @($wsi, $logs, $tempDir)) { New-Item -ItemType Directory -Force -Path $p | Out-Null }
$lock = $null
try {
    $lock = [IO.File]::Open((Join-Path $logs 'download.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
} catch { throw 'Another download process is using this folder. Do not launch twice.' }
$oldTemp = $env:TEMP; $oldTmp = $env:TMP; $oldTmpDir = $env:TMPDIR
$exitCode = 1
try {
    $env:TEMP = $tempDir; $env:TMP = $tempDir; $env:TMPDIR = $tempDir
    $total = [long](($items | Measure-Object -Property size -Sum).Sum)
    $occupied = [long]0
    foreach ($item in $items) {
        $p = Join-Path $wsi $item.filename
        if (Test-Path -LiteralPath $p) {
            $n = (Get-Item -LiteralPath $p).Length
            if ($n -gt [long]$item.size) { throw "Oversize source file; preserve and investigate: $p" }
            $occupied += $n
        }
    }
    $free = (Get-PSDrive -Name ([IO.Path]::GetPathRoot($Root).Substring(0,1))).Free
    if ($free -lt ($total - $occupied + 20GB)) { throw 'Insufficient disk space including 20 GiB reserve.' }
    Write-Host "Images: 413; total bytes: $total; remaining disk bytes: $free"
    Write-Host "Raw images and .aria2 checkpoints: $wsi"
    Write-Host 'Press Ctrl+C once to stop gracefully. Run the SAME command to resume.'
    if ($CheckOnly) { $exitCode = 0 } else {
        $options = @('--no-conf=true', "--input-file=$inputList", "--dir=$wsi",
            '--continue=true', '--always-resume=true', '--auto-file-renaming=false',
            '--allow-overwrite=false', '--file-allocation=none', '--max-concurrent-downloads=3',
            '--split=4', '--max-connection-per-server=4', '--min-split-size=20M',
            '--max-tries=20', '--retry-wait=15', '--connect-timeout=20', '--timeout=60',
            '--auto-save-interval=10', '--save-session-interval=10',
            "--save-session=$(Join-Path $logs 'unfinished_session.txt')",
            "--log=$(Join-Path $logs 'aria2_download.log')", '--log-level=notice',
            '--summary-interval=10', '--download-result=full', '--enable-color=false')
        & $tool @options
        $exitCode = $LASTEXITCODE
        $ready = 0; $pending = @()
        foreach ($item in $items) {
            $p = Join-Path $wsi $item.filename
            if ((Test-Path -LiteralPath $p) -and !(Test-Path -LiteralPath ($p+'.aria2')) -and
                (Get-Item -LiteralPath $p).Length -eq [long]$item.size) { $ready++ }
            else { $pending += $item.filename }
        }
        [pscustomobject]@{ expected=413; size_checked_complete=$ready; pending=$pending;
            official_checksum_verified=$false; aria2_exit_code=$exitCode;
            at=(Get-Date -Format o) } | ConvertTo-Json -Depth 4 |
            Set-Content -LiteralPath (Join-Path $logs 'download_size_check.json') -Encoding UTF8
        Write-Host "Size-checked complete: $ready / 413. Official checksums were not supplied in the source list."
        if ($ready -ne 413 -and $exitCode -eq 0) { $exitCode = 2 }
    }
} finally {
    $env:TEMP = $oldTemp; $env:TMP = $oldTmp; $env:TMPDIR = $oldTmpDir
    $lock.Dispose()
}
exit $exitCode
