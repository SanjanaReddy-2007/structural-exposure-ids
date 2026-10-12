$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'The project environment is missing. Follow the setup instructions in README.md.'
}
if (-not (Test-Path -LiteralPath 'data\synthetic\synthetic.csv')) {
    & $projectPython -m exposure_ids.cli demo --out data/synthetic
    if ($LASTEXITCODE -ne 0) { throw 'Demo generation failed.' }
}
$demoOutput = 'runs/demo-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
& $projectPython -m exposure_ids.cli run --data data/synthetic --out $demoOutput --protocol random --seeds 1 2 --epochs 30 --rewire
if ($LASTEXITCODE -ne 0) { throw 'Demo experiment failed.' }
Write-Host "Synthetic software check completed. Results: $demoOutput"
Write-Host 'These are not CICIDS2017 research results.'
