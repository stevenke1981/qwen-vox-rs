param(
    [string]$OutDir = "dist",
    [switch]$SkipCuda
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$dist = Join-Path $repoRoot $OutDir
New-Item -ItemType Directory -Force -Path $dist | Out-Null

Push-Location $repoRoot
try {
    Write-Host "Building CPU release..."
    cargo build --release -p qwen-vox-cli --bin qwen-vox
    Copy-Item -LiteralPath "target\release\qwen-vox.exe" -Destination (Join-Path $dist "qwen-vox-cpu.exe") -Force

    if (-not $SkipCuda) {
        Write-Host "Building CUDA release..."
        cargo build --release -p qwen-vox-cli --bin qwen-vox --features cuda
        Copy-Item -LiteralPath "target\release\qwen-vox.exe" -Destination (Join-Path $dist "qwen-vox-cuda.exe") -Force
    }

    $commit = (git rev-parse --short HEAD 2>$null)
    $builtAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    $manifestPath = Join-Path $dist "BUILD_INFO.txt"
    @(
        "qwen-vox-rs release build"
        "commit=$commit"
        "built_at_utc=$builtAt"
        "cpu=dist\qwen-vox-cpu.exe"
        "cuda=dist\qwen-vox-cuda.exe"
        ""
        "Example:"
        'dist\qwen-vox-cuda.exe generate --device cuda --language chinese --speaker vivian --text "Hello from Qwen3 TTS." --output out\speech.wav'
    ) | Set-Content -Encoding UTF8 -Path $manifestPath

    Write-Host "Release artifacts:"
    Get-ChildItem -LiteralPath $dist | Select-Object Name, Length, LastWriteTime
}
finally {
    Pop-Location
}
