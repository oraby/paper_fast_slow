<#
Run the rlmodel test suite via uv — from the repo root, no parent-dir dance.

`uv run` uses the project's virtualenv (.venv, created by `uv sync`) whose PyPI
wheels self-contain their native DLLs, so there is no conda-activation / DLL
search-path problem (the 0xc06d007f native fault). It invokes the `pytest`
console script (not `python -m pytest`), which keeps cwd off sys.path, so the
local `code/` package never shadows the stdlib `code` module — that is why
running from the repo root now works.

Usage (from anywhere):
    powershell -File code\rlmodel\run_tests.ps1                    # full suite
    powershell -File code\rlmodel\run_tests.ps1 -k model_compare -q   # pass-through args
#>
$ErrorActionPreference = "Stop"

# repo = paper_fast_slow (script lives at <repo>/code/rlmodel/run_tests.ps1)
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

# Prefer uv on PATH; fall back to the default per-user install location.
$cmd = Get-Command uv -ErrorAction SilentlyContinue
$uv  = if ($cmd) { $cmd.Source } else { Join-Path $env:USERPROFILE ".local\bin\uv.exe" }
if (-not (Test-Path $uv)) {
    throw "uv not found on PATH or at $uv. Install uv (https://docs.astral.sh/uv/) or add it to PATH."
}

Write-Host "uv   : $uv"
Write-Host "repo : $repo"
Write-Host ""

Push-Location $repo
try {
    & $uv run pytest @args
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
