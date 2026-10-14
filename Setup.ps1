param()
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (Get-Command uv -ErrorAction SilentlyContinue) {
    & uv sync --extra dev
    if ($LASTEXITCODE -ne 0) { throw 'Installation mit uv fehlgeschlagen.' }
} else {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        if (Get-Command py -ErrorAction SilentlyContinue) { & py -3 -m venv .venv }
        elseif (Get-Command python -ErrorAction SilentlyContinue) { & python -m venv .venv }
        else { throw 'Python 3.11+ oder uv installieren und Setup erneut starten.' }
        if ($LASTEXITCODE -ne 0) { throw 'Virtuelle Python-Umgebung konnte nicht erstellt werden.' }
    }
    & .\.venv\Scripts\python.exe -m pip install -e '.[dev]'
    if ($LASTEXITCODE -ne 0) { throw 'Paketinstallation fehlgeschlagen.' }
}
Write-Host 'Bereit. Mit Start.cmd oder ./Start.ps1 starten.'
