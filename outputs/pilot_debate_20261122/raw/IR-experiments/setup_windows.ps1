$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.11, then rerun this script.' }
}
$agentirPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $agentirPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed.' }
if (Test-Path -LiteralPath 'requirements-lock-windows-cpu.txt') {
    & $agentirPython -m pip install -r requirements-lock-windows-cpu.txt
    if ($LASTEXITCODE -ne 0) { throw 'Locked environment installation failed.' }
    & $agentirPython -m pip install --no-deps -e '.[test]'
} else {
    & $agentirPython -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed.' }
    & $agentirPython -m pip install -e '.[test]'
}
if ($LASTEXITCODE -ne 0) { throw 'Project dependency installation failed.' }
& $agentirPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
Write-Host 'Ready. Select the .venv interpreter in VS Code.'
