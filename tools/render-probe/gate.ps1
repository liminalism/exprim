# A/B output-identity gate: render every configured document with two probe
# binaries and compare the per-page hashes.
#
#   gate.ps1 -A <baseline probe.exe> -B <new probe.exe>
#            [-Docs <list>] [-Scales <list>] [-ExtraArgsB <probe args for B only>]
#
# A document entry is a name (no extension) under the local test-document
# directory `$env:EXPRIM_GATE_DIR`, or a full path,
# optionally followed by `|<page spec>` (probe `--pages` syntax, e.g.
# `D:\x\book.pdf|0-9,100,455`). `$env:GATE_EXTRA` appends `;`-separated entries.
# `-EnvB` sets environment variables for the B runs only (e.g. a knob that must
# reproduce A's bytes): `-EnvB @{ PDF_RENDERER_PATH_SEGMENTS = '16' }`.
param(
  [string]$A,
  [string]$B,
  [string[]]$Docs = @(),
  [string[]]$Scales = @('1.0','1.37','2.0'),
  [hashtable]$EnvB = @{}
)
$docs = @($Docs)
if($env:GATE_EXTRA){ $docs += ($env:GATE_EXTRA -split ';') }
if($docs.Count -eq 0){ throw 'name the documents to compare: -Docs <names or paths>, or $env:GATE_EXTRA' }
$modes=@(@('--fresh-backend'),@('--repeat','2'))
$bad=0;$n=0
function HashRun($bin,$d,$s,$m,$envs){
  $parts = $d -split '\|',2
  $name = $parts[0]
  if($name -notlike '*:*' -and -not $env:EXPRIM_GATE_DIR){ throw 'set $env:EXPRIM_GATE_DIR to the local test-document directory' }
  $path = if($name -like '*:*'){$name}else{ Join-Path $env:EXPRIM_GATE_DIR "$name.pdf" }
  $extra = @()
  if($parts.Count -gt 1 -and $parts[1]){ $extra = @('--pages',$parts[1]) }
  $saved = @{}
  foreach($k in $envs.Keys){ $saved[$k] = [Environment]::GetEnvironmentVariable($k,'Process'); [Environment]::SetEnvironmentVariable($k,[string]$envs[$k],'Process') }
  try {
    & $bin run $path --scale $s --hash @extra @m 2>$null | Where-Object {$_ -notmatch '^#'} | ForEach-Object { $f=$_ -split "`t"; "$($f[0]):$($f[1]):$($f[-1])" }
  } finally {
    foreach($k in $envs.Keys){ [Environment]::SetEnvironmentVariable($k,$saved[$k],'Process') }
  }
}
foreach($d in $docs){foreach($s in $Scales){foreach($g in '0','1'){foreach($m in $modes){
  $env:PROBE_GRAY8=$g
  $ha=HashRun $A $d $s $m @{}
  $hb=HashRun $B $d $s $m $EnvB
  $n++
  if(($ha -join ',') -ne ($hb -join ',') -or -not $ha){$bad++; "DIFF $d scale=$s gray=$g $($m -join ' ') (pages a=$($ha.Count) b=$($hb.Count))"}
}}}}
$env:PROBE_GRAY8=$null
"gate: $n configs, $bad differ"
if($bad -gt 0){ exit 1 }
