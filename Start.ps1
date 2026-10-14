param([switch]$Demo)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & (Join-Path $PSScriptRoot 'Setup.ps1')
}
$visionArguments = @('-m', 'bibazu_vision')
if ($Demo) { $visionArguments += '--demo' }
& .\.venv\Scripts\python.exe @visionArguments
if ($LASTEXITCODE -ne 0) { throw 'Die GUI wurde mit einem Fehler beendet.' }
