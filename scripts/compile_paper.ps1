# Compile paper/latex/bops_ieee_draft.tex to PDF using bundled Tectonic.
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$tectonic = Join-Path $repo "tools\tectonic\tectonic.exe"
$latexDir = Join-Path $repo "paper\latex"
$tex = Join-Path $latexDir "bops_ieee_draft.tex"

if (-not (Test-Path $tectonic)) {
    Write-Error "Tectonic not found at $tectonic. Download from https://github.com/tectonic-typesetting/tectonic/releases"
}

Push-Location $latexDir
try {
    & $tectonic $tex
    Write-Host "Wrote $(Join-Path $latexDir 'bops_ieee_draft.pdf')"
} finally {
    Pop-Location
}
