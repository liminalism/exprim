# G0 census driver: runs `probe census` over every PDF under -Roots, resumable.
# Usage: powershell -File g0-census.ps1 -Probe <probe.exe> -Roots a,b [-Out file.jsonl]
#        [-MaxPages 6] [-TimeoutSec 20] [-Limit 0]
# The output names every scanned file: keep it out of the repository.
param(
    [Parameter(Mandatory = $true)][string]$Probe,
    [Parameter(Mandatory = $true)][string[]]$Roots,
    [string]$Out = (Join-Path $PSScriptRoot '..\..\.agent\scratch\g0g1\census\census.jsonl'),
    [int]$MaxPages = 6,
    [int]$TimeoutSec = 20,
    [int]$Limit = 0,
    [int]$Stride = 1
)
$ErrorActionPreference = 'Continue'

function Esc([string]$s) { return $s.Replace('\', '\\').Replace('"', '\"') }

$outDir = Split-Path -Parent $Out
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Force $outDir | Out-Null }
$tmpOut = Join-Path $outDir 'tmp-stdout.txt'
$tmpErr = Join-Path $outDir 'tmp-stderr.txt'

# Files already done (resume): the "file" field of every line, JSON-unescaped.
$done = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
if (Test-Path $Out) {
    foreach ($line in [System.IO.File]::ReadLines($Out)) {
        if ($line -match '^\{"file":"((?:[^"\\]|\\.)*)"') {
            [void]$done.Add($Matches[1].Replace('\\', '\').Replace('\"', '"'))
        }
    }
}

$files = foreach ($r in $Roots) {
    if (Test-Path $r) {
        Get-ChildItem -LiteralPath $r -Recurse -File -Filter *.pdf -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName }
    }
}
$files = @($files)
if ($Stride -gt 1) { $files = @(for ($i = 0; $i -lt $files.Count; $i += $Stride) { $files[$i] }) }
if ($Limit -gt 0 -and $files.Count -gt $Limit) { $files = $files[0..($Limit - 1)] }
Write-Host ("census: {0} files found, {1} already done" -f $files.Count, $done.Count)

$n = 0; $timeouts = 0; $crashes = 0; $t0 = Get-Date
$utf8 = New-Object System.Text.UTF8Encoding($false)
foreach ($f in $files) {
    $n++
    if ($done.Contains($f)) { continue }
    $argLine = 'census "{0}" --max-pages {1}' -f $f, $MaxPages
    Remove-Item -ErrorAction SilentlyContinue $tmpOut, $tmpErr
    $p = Start-Process -FilePath $Probe -ArgumentList $argLine -PassThru -NoNewWindow `
        -RedirectStandardOutput $tmpOut -RedirectStandardError $tmpErr
    $lines = @()
    $null = $p.Handle  # cache the handle so ExitCode is readable after exit
    if (-not $p.WaitForExit($TimeoutSec * 1000)) {
        try { $p.Kill() } catch {}
        $p.WaitForExit(2000) | Out-Null
        $timeouts++
        $lines += ('{{"file":"{0}","error":"timeout {1}s"}}' -f (Esc $f), $TimeoutSec)
    } else {
        $got = @()
        if (Test-Path $tmpOut) { $got = @(Get-Content -LiteralPath $tmpOut -Encoding UTF8 | Where-Object { $_ -like '{*' }) }
        if ($p.ExitCode -ne 0) {
            $crashes++
            $lines += ('{{"file":"{0}","error":"exit code {1}"}}' -f (Esc $f), $p.ExitCode)
        }
        if ($got.Count -eq 0 -and $p.ExitCode -eq 0) {
            $lines += ('{{"file":"{0}","error":"no output"}}' -f (Esc $f))
        }
        $lines = @($got) + $lines
    }
    [System.IO.File]::AppendAllText($Out, (($lines -join "`n") + "`n"), $utf8)
    if ($n % 50 -eq 0) {
        Write-Host ("census: {0}/{1} files, {2} timeouts, {3} crashes, {4:N0}s" -f $n, $files.Count, $timeouts, $crashes, ((Get-Date) - $t0).TotalSeconds)
    }
}
Write-Host ("census done: {0} files, {1} timeouts, {2} crashes, {3:N0}s" -f $n, $timeouts, $crashes, ((Get-Date) - $t0).TotalSeconds)
