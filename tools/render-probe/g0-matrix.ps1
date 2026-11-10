param(
    # The git-ignored manifest with local paths (written by g0_manifest.py).
    [string]$Manifest = (Join-Path $PSScriptRoot '..\..\docs\benchmarks\2026-11-10-attribution\manifest.local.json'),
    [string]$Probe = (Join-Path $PSScriptRoot '..\..\.agent\scratch\g0g1\bin\probe-work.exe'),
    [string]$ProbeProf = (Join-Path $PSScriptRoot '..\..\.agent\scratch\g0g1\bin\probe-work-prof.exe'),
    [string]$Out = (Join-Path $PSScriptRoot '..\..\docs\benchmarks\2026-11-10-attribution\raw'),
    [string[]]$Only = @(),
    [string[]]$Sets = @(),
    [int]$BudgetMinutes = 75
)
$ErrorActionPreference = 'Continue'
$env:RAYON_NUM_THREADS = '1'
$env:PDF_RENDERER_IMAGE_ROW_PAR = '0'
$env:PDF_RENDERER_JPX_CONCURRENCY = 'serial'
New-Item -ItemType Directory -Force $Out | Out-Null
$log = Join-Path $Out 'matrix.log'
$root = ((Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path)
$entries = Get-Content -Raw $Manifest | ConvertFrom-Json
if ($Only.Count -gt 0) { $entries = @($entries | Where-Object { $Only -contains $_.id }) }
$start = Get-Date
$script:stop = $false

function Count-Meta($file, $id) {
    if (-not (Test-Path $file)) { return 0 }
    $pat = '"kind"\s*:\s*"meta".*"id"\s*:\s*"' + [regex]::Escape($id) + '"'
    $n = 0
    foreach ($l in [System.IO.File]::ReadLines($file)) { if ($l -match $pat) { $n++ } }
    return $n
}
function First-Page($spec) { return (($spec -split ',')[0] -split '-')[0] }
function EntryPath($e) {
    if ([System.IO.Path]::IsPathRooted($e.path)) { return $e.path }
    return (Join-Path $root $e.path)
}

function Invoke-One($set, $exe, $e, $argList, $file, $need) {
    if ($script:stop) { return }
    if (((Get-Date) - $start).TotalMinutes -gt $BudgetMinutes) {
        $script:stop = $true
        Add-Content $log "BUDGET EXCEEDED: stopping before set=$set id=$($e.id)"
        return
    }
    if ((Count-Meta $file $e.id) -ge $need) { return }
    $pdf = EntryPath $e
    $full = "bench `"$pdf`" --id $($e.id) --hash --jsonl `"$file`" " + $argList
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $so = Join-Path $env:TEMP 'g0m_stdout.txt'; $se = Join-Path $env:TEMP 'g0m_stderr.txt'
    $p = Start-Process -FilePath $exe -ArgumentList $full -PassThru -NoNewWindow -RedirectStandardOutput $so -RedirectStandardError $se
    $null = $p.Handle   # cache the handle so ExitCode is readable
    $done = $p.WaitForExit(15 * 60 * 1000)
    $code = ''
    if (-not $done) {
        & taskkill /F /T /PID $p.Id | Out-Null
        $code = 'TIMEOUT'
    } else { $p.WaitForExit(); $code = $p.ExitCode }
    $sw.Stop()
    $tail = ''
    if ($code -ne 0) {
        $tail = ' | ' + ((Get-Content $se -Tail 2 -ErrorAction SilentlyContinue) -join ' / ')
    }
    Add-Content $log ("{0} set={1} id={2} dur={3:N1}s exit={4}{5}" -f (Get-Date -Format s), $set, $e.id, $sw.Elapsed.TotalSeconds, $code, $tail)
}

$S = @(
    @{n='low-rerender-144'; p='n'; api='low'; sc='rerender'; dpi='144'; rep=5; roles=@('*'); pg='spec'},
    @{n='low-rerender-72'; p='n'; api='low'; sc='rerender'; dpi='72'; rep=5; roles=@('*'); pg='spec'},
    @{n='low-rerender-98.64'; p='n'; api='low'; sc='rerender'; dpi='98.64'; rep=5; roles=@('*'); pg='spec'},
    @{n='session-rerender-144'; p='n'; api='session'; sc='rerender'; dpi='144'; rep=5; roles=@('*'); pg='spec'},
    @{n='session-rerender-72'; p='n'; api='session'; sc='rerender'; dpi='72'; rep=5; roles=@('*'); pg='spec'},
    @{n='prof-stage-low-144'; p='p'; api='low'; sc='rerender'; dpi='144'; rep=3; det='stage'; roles=@('*'); pg='spec'},
    @{n='prof-stage-low-72'; p='p'; api='low'; sc='rerender'; dpi='72'; rep=3; det='stage'; roles=@('*'); pg='spec'},
    @{n='prof-stage-session-144'; p='p'; api='session'; sc='rerender'; dpi='144'; rep=3; det='stage'; roles=@('*'); pg='spec'},
    @{n='prof-fill-low-144'; p='p'; api='low'; sc='rerender'; dpi='144'; rep=3; det='fill'; roles=@('anchor','vector','synthetic-vector'); pg='spec'},
    @{n='cold-low-144'; p='n'; api='low'; sc='cold'; dpi='144'; roles=@('anchor'); pg='cold'; need=9},
    @{n='cold-session-144'; p='n'; api='session'; sc='cold'; dpi='144'; roles=@('anchor'); pg='cold'; need=9},
    @{n='zoom-low-144'; p='n'; api='low'; sc='zoom'; dpi='144'; rep=5; zoom=$true; roles=@('anchor','vector','synthetic-vector'); pg='first'},
    @{n='zoom-session-144'; p='n'; api='session'; sc='zoom'; dpi='144'; rep=5; zoom=$true; roles=@('anchor','vector','synthetic-vector'); pg='first'},
    @{n='prof-stage-zoom-low-144'; p='p'; api='low'; sc='zoom'; dpi='144'; rep=3; det='stage'; zoom=$true; roles=@('anchor','vector','synthetic-vector'); pg='first'},
    @{n='low-rerender-300'; p='n'; api='low'; sc='rerender'; dpi='300'; rep=3; maxpx=40000000; roles=@('anchor','synthetic-vector'); pg='first'},
    @{n='seq-low-144'; p='n'; api='low'; sc='seq'; dpi='144'; roles=@('*'); pg='seq'},
    @{n='seq-session-144'; p='n'; api='session'; sc='seq'; dpi='144'; roles=@('*'); pg='seq'}
)
$seqPages = @{ 'vector-tables' = '0-199'; 'text' = '100-499'; 'drawing-a' = '0-21' }

foreach ($s in $S) {
    if ($Sets.Count -gt 0 -and ($Sets -notcontains $s.n)) { continue }
    $file = Join-Path $Out ($s.n + '.jsonl')
    $exe = if ($s.p -eq 'p') { $ProbeProf } else { $Probe }
    foreach ($e in $entries) {
        if ($script:stop) { break }
        if ($s.roles -notcontains '*' -and $s.roles -notcontains $e.role) { continue }
        $a = "--api $($s.api) --scenario $($s.sc) --dpi $($s.dpi)"
        $need = 1
        if ($s.pg -eq 'seq' -and -not $seqPages.ContainsKey($e.id)) { continue }
        switch ($s.pg) {
            'spec' { $a += " --pages `"$($e.pages_selected)`"" }
            'first' { $a += " --pages $(First-Page $e.pages_selected)" }
            'cold' {
                $pg = if ($e.id -eq 'text') { '100' } else { '0' }
                $a += " --pages $pg"; $need = $s.need
            }
            'seq' {
                $a += " --pages $($seqPages[$e.id])"
            }
        }
        if ($s.rep) { $a += " --repeat $($s.rep) --warmup 1" }
        if ($s.det) { $a += " --detail $($s.det)" }
        if ($s.zoom) { $a += ' --zoom 4 --viewport 1280x800' }
        if ($s.maxpx) { $a += " --max-pixels $($s.maxpx)" }
        if ($s.pg -eq 'cold') {
            for ($i = 0; $i -lt $need; $i++) { Invoke-One $s.n $exe $e $a $file ($i + 1) }
        } else {
            Invoke-One $s.n $exe $e $a $file $need
        }
    }
}
Add-Content $log ("{0} DONE stop={1} elapsed={2:N1}min" -f (Get-Date -Format s), $script:stop, ((Get-Date) - $start).TotalMinutes)
