# Sampled CPU profiles (flamegraphs) of the attribution anchors.
#
# `flamegraph` records through ETW on Windows and must run from an ELEVATED
# PowerShell ("Run as administrator"); an ordinary shell fails with
# `could not profile using blondie: NotAnAdmin`.
#
#   .\g0-flamegraph.ps1 [-Probe <probe.exe>] [-Out <dir>]
#                       [-Map <pdf>] [-Tables <pdf>] [-Text <pdf>]
#
# The three inputs default to the `map`, `vector-tables` and `text` entries of
# the git-ignored `manifest.local.json` (see g0_manifest.py). The SVGs name
# functions only and can be published; the `runs\*.jsonl` timing rows record
# the input paths and are git-ignored.
#
# Without -Probe the script builds the probe itself (non-profiling,
# `debug-fast`: release codegen with symbols) under the probe's own `target`
# directory, so the samples are of the code as shipped and not of the stage
# timers. Sample shares stay valid on a busy machine: samples are taken only
# while the process runs.
param(
  [string]$Probe = '',
  [string]$Out = '',
  [string]$Map = '',
  [string]$Tables = '',
  [string]$Text = ''
)
$ErrorActionPreference = 'Stop'
$harness = $PSScriptRoot
$repo = (Resolve-Path (Join-Path $harness '..\..')).Path
if (-not $Out) { $Out = Join-Path $repo 'docs\benchmarks\2026-11-10-attribution\profiles' }
if (-not ($Map -and $Tables -and $Text)) {
  $local = Join-Path $repo 'docs\benchmarks\2026-11-10-attribution\manifest.local.json'
  if (-not (Test-Path $local)) { throw "pass -Map/-Tables/-Text, or create $local with g0_manifest.py" }
  $entries = Get-Content -Raw -Encoding UTF8 $local | ConvertFrom-Json
  function Local-Path($id) { ($entries | Where-Object { $_.id -eq $id } | Select-Object -First 1).path }
  if (-not $Map) { $Map = Local-Path 'map' }
  if (-not $Tables) { $Tables = Local-Path 'vector-tables' }
  if (-not $Text) { $Text = Local-Path 'text' }
}

if (-not (Get-Command flamegraph -ErrorAction SilentlyContinue)) {
  throw 'flamegraph is not on PATH (cargo install flamegraph)'
}
if (-not $Probe) {
  $manifest = Join-Path $harness 'probe\Cargo.toml'
  $Probe = Join-Path $harness 'probe\target\debug-fast\lege-render-probe.exe'
  Write-Host "== building probe ($manifest)"
  & cargo build --profile debug-fast --manifest-path $manifest
  if ($LASTEXITCODE -ne 0) { throw "probe build failed (exit $LASTEXITCODE)" }
}
if (-not (Test-Path $Probe)) { throw "probe not found: $Probe" }
foreach ($pdf in $Map, $Tables, $Text) {
  if (-not $pdf -or -not (Test-Path $pdf)) { throw "input not found: $pdf" }
}

New-Item -ItemType Directory -Force -Path (Join-Path $Out 'runs') | Out-Null
$env:RAYON_NUM_THREADS = '1'
$env:PDF_RENDERER_IMAGE_ROW_PAR = '0'
$env:PDF_RENDERER_JPX_CONCURRENCY = 'serial'
$runs = @(
  @{ name = 'map-144-rerender';  args = @('bench', $Map,    '--scenario', 'rerender', '--repeat', '40', '--warmup', '1', '--dpi', '144', '--id', 'map') },
  @{ name = 'tables-144-seq';    args = @('bench', $Tables, '--scenario', 'seq', '--pages', '0-299', '--dpi', '144', '--id', 'vector-tables') },
  @{ name = 'text-144-seq';      args = @('bench', $Text,   '--scenario', 'seq', '--pages', '100-599', '--dpi', '144', '--id', 'text') },
  @{ name = 'text-144-rerender'; args = @('bench', $Text,   '--scenario', 'rerender', '--pages', '100-139', '--repeat', '20', '--warmup', '1', '--dpi', '144', '--id', 'text') }
)
$failed = 0
foreach ($run in $runs) {
  $svg = Join-Path $Out ($run.name + '.svg')
  $jsonl = Join-Path $Out ('runs\' + $run.name + '.jsonl')
  Remove-Item -ErrorAction SilentlyContinue $jsonl
  Write-Host "== $($run.name)"
  & flamegraph -o $svg -- $Probe @($run.args) --jsonl $jsonl
  if ($LASTEXITCODE -ne 0) { $failed++; Write-Warning "flamegraph failed for $($run.name) (exit $LASTEXITCODE)" }
}
Get-ChildItem $Out -Filter *.svg | Select-Object Name, Length
if ($failed -gt 0) { exit 1 }
