<#
.SYNOPSIS
  Build the exprim agent, install it as ~/.local/bin/exprim.exe, and register
  it with Claude Code as the user-scope MCP server `exprim`, replacing the old
  `lege-pdf` server.

.DESCRIPTION
  Run it from a plain terminal while no Claude Code session is open: every open
  session holds its MCP server process (and the .exe file) and can rewrite
  ~/.claude.json behind this script. The script refuses to start while
  claude.exe, lege-pdf.exe or exprim.exe is running, unless -Force.

  Steps: preflight -> build (debug-fast) -> smoke test the new binary ->
  install -> re-register (remove lege-pdf, add exprim) -> retire
  lege-pdf.exe as lege-pdf.exe.retired. Nothing is deleted.

  New sessions see the tools as mcp__exprim__pdf_* (previously
  mcp__lege-pdf__pdf_*); the tool names themselves are unchanged.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File D:\projects\exprim\agent\install-mcp.ps1

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File D:\projects\exprim\agent\install-mcp.ps1 -DryRun
#>
param(
    [string]$BinDir = (Join-Path $HOME '.local\bin'),
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$agentDir = $PSScriptRoot
$exprimRoot = Split-Path $agentDir -Parent
$manifest = Join-Path $agentDir 'Cargo.toml'
$built = Join-Path $agentDir 'target\debug-fast\exprim.exe'
$installed = Join-Path $BinDir 'exprim.exe'
$oldExe = Join-Path $BinDir 'lege-pdf.exe'
$fixture = Join-Path $exprimRoot 'chaos-tests\tests\fixtures\hello_world.pdf'

function Step($text) { Write-Host "==> $text" -ForegroundColor Cyan }
function Fail($text) { Write-Host "error: $text" -ForegroundColor Red; exit 1 }
function Run($description, [scriptblock]$action) {
    if ($DryRun) { Write-Host "    (dry run) would $description" } else { & $action }
}

# --- Preflight ------------------------------------------------------------
Step 'Preflight'
$busy = @(Get-Process -Name 'claude', 'lege-pdf', 'exprim' -ErrorAction SilentlyContinue)
if ($busy.Count -gt 0) {
    $list = ($busy | ForEach-Object { "$($_.ProcessName) (pid $($_.Id))" }) -join ', '
    if ($Force) {
        Write-Host "    warning: running anyway (-Force): $list" -ForegroundColor Yellow
    } else {
        Fail "close these first, or pass -Force: $list"
    }
}
$claude = (Get-Command claude -ErrorAction SilentlyContinue).Source
if (-not $claude) { Fail 'the claude CLI is not on PATH' }
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) { Fail 'cargo is not on PATH' }
if (-not (Test-Path $manifest)) { Fail "no agent manifest at $manifest" }
$legeOcr = Join-Path (Split-Path $exprimRoot -Parent) 'Lege-ecosystem\lege-ocr'
if (-not (Test-Path $legeOcr)) { Fail "the agent's OCR needs $legeOcr (Lege-ecosystem beside exprim)" }
New-Item -ItemType Directory -Force $BinDir | Out-Null
Write-Host "    exprim:  $exprimRoot"
Write-Host "    install: $installed"

# --- Build ----------------------------------------------------------------
Step 'Build (debug-fast: release optimisation without fat LTO)'
Run 'cargo build --profile debug-fast' {
    & cargo build --profile debug-fast --manifest-path $manifest --bin exprim
    if ($LASTEXITCODE -ne 0) { Fail 'cargo build failed' }
}

# --- Smoke test -------------------------------------------------------------
Step 'Smoke test the new binary'
Run 'run an MCP initialize handshake and inspect a fixture' {
    $init = '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"install-mcp","version":"0"}}}'
    $reply = ($init | & $built mcp | Out-String)
    if ($reply -notmatch '"result"' -or $reply -notmatch 'exprim') {
        Fail "MCP handshake did not answer as exprim:`n$reply"
    }
    $inspect = (& $built inspect $fixture | Out-String)
    if ($inspect -notmatch 'open: ok') { Fail "inspect failed:`n$inspect" }
    Write-Host '    MCP initialize: ok; inspect: ok'
}

# --- Install ----------------------------------------------------------------
Step 'Install'
Run "copy $built to $installed" {
    if (Test-Path $installed) { Copy-Item $installed "$installed.bak" -Force }
    Copy-Item $built $installed -Force
    Write-Host "    installed $installed"
}

# --- Register -----------------------------------------------------------------
Step 'Re-register the MCP server (user scope)'
$registered = (& $claude mcp list 2>&1 | Out-String)
Run 'remove lege-pdf and (re)add exprim' {
    if ($registered -match '(?m)^lege-pdf:') {
        & $claude mcp remove lege-pdf --scope user
    }
    if ($registered -match '(?m)^exprim:') {
        & $claude mcp remove exprim --scope user
    }
    & $claude mcp add --scope user exprim -- $installed mcp
    if ($LASTEXITCODE -ne 0) { Fail 'claude mcp add failed' }
}

# --- Retire the old binary ------------------------------------------------------
Step 'Retire lege-pdf.exe'
if (Test-Path $oldExe) {
    Run "rename $oldExe to lege-pdf.exe.retired" {
        Move-Item $oldExe "$oldExe.retired" -Force
        Write-Host "    renamed to $oldExe.retired"
    }
} else {
    Write-Host '    not present; nothing to retire'
}

# --- Verify -------------------------------------------------------------------
Step 'Verify'
Run 'show the registration' { & $claude mcp get exprim }

Write-Host ''
Write-Host 'Done. Start a new Claude Code session to pick up mcp__exprim__pdf_*.' -ForegroundColor Green
Write-Host 'Rollback:'
Write-Host "  claude mcp remove exprim --scope user"
Write-Host "  Move-Item `"$oldExe.retired`" `"$oldExe`""
Write-Host "  claude mcp add --scope user lege-pdf -- `"$oldExe`" mcp"
