$ErrorActionPreference = 'Stop'
$benchRoot = $PSScriptRoot
$vendorRoot = Join-Path $benchRoot 'vendor'
$fontRoot = Join-Path $benchRoot 'hayro\fonts'
New-Item -ItemType Directory -Force -Path $vendorRoot,$fontRoot | Out-Null

function Fetch-Checked($url, $destination, $expected) {
    if (-not (Test-Path -LiteralPath $destination)) {
        Invoke-WebRequest -Uri $url -OutFile $destination
    }
    $actual = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected) { throw "SHA-256 mismatch for $destination ($actual)" }
}

Fetch-Checked 'https://github.com/bblanchon/pdfium-binaries/releases/download/chromium/8125/pdfium-win-x64.tgz' (Join-Path $vendorRoot 'pdfium.tgz') '9a964261760cff52570aabf038ab346d3a5034dd8b30d0d5fe8861235d356b8b'
& tar -xzf (Join-Path $vendorRoot 'pdfium.tgz') -C $vendorRoot
if ($LASTEXITCODE) { throw 'PDFium archive extraction failed' }
$dllPath = Join-Path $vendorRoot 'bin\pdfium.dll'
if ((Get-FileHash -LiteralPath $dllPath).Hash.ToLowerInvariant() -ne '4b0dec92555d69437bef0df37be630181ab3e83a6823ca264ede355a4f2ef994') { throw 'PDFium DLL hash mismatch' }

$fontBase = 'https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF'
Fetch-Checked "$fontBase/Japanese/NotoSansCJKjp-Regular.otf" (Join-Path $fontRoot 'NotoSansCJKjp-Regular.otf') '68a3fc98800b2a27b371f2fb79991daf3633bd89309d4ffaa6946fd587f375b5'
Fetch-Checked "$fontBase/Korean/NotoSansCJKkr-Regular.otf" (Join-Path $fontRoot 'NotoSansCJKkr-Regular.otf') '6bcb2a0703aa137e874fc2dffa85f6c21ba9a67fa329e81b8c801663af7e992a'
Fetch-Checked "$fontBase/SimplifiedChinese/NotoSansCJKsc-Regular.otf" (Join-Path $fontRoot 'NotoSansCJKsc-Regular.otf') '2c76254f6fc379fddfce0a7e84fb5385bb135d3e399294f6eeb6680d0365b74b'
Copy-Item -LiteralPath (Join-Path $benchRoot 'FONT-LICENSE') -Destination (Join-Path $fontRoot 'LICENSE')

& python -m venv (Join-Path $benchRoot '.venv')
& (Join-Path $benchRoot '.venv\Scripts\python.exe') -m pip install -r (Join-Path $benchRoot 'requirements.txt')
if ($LASTEXITCODE) { throw 'Python dependency installation failed' }
& cargo build --release --locked --manifest-path (Join-Path $benchRoot 'exprim\Cargo.toml')
if ($LASTEXITCODE) { throw 'exprim adapter build failed' }
& cargo build --release --locked --manifest-path (Join-Path $benchRoot 'hayro\Cargo.toml')
if ($LASTEXITCODE) { throw 'hayro adapter build failed' }
$env:BENCH_PDFIUM_DLL = $dllPath
Write-Output "Setup complete. BENCH_PDFIUM_DLL=$dllPath"
