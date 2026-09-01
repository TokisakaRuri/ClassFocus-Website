$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Python = if (Test-Path $VenvPython) { $VenvPython } else { (Get-Command python -ErrorAction Stop).Source }
$FrontendRoot = Join-Path $ProjectRoot "frontend"
$FrontendEntry = Join-Path $FrontendRoot "dist\index.html"
$Npm = if (Test-Path "D:\Nodejs\npm.cmd") { "D:\Nodejs\npm.cmd" } else { (Get-Command npm.cmd -ErrorAction Stop).Source }

Write-Host "Starting ClassFocus local services..." -ForegroundColor Cyan
Write-Host "Workspace: $ProjectRoot"

if (-not (Test-Path $FrontendEntry)) {
    Write-Host "Building the React workbench..." -ForegroundColor Cyan
    if (-not (Test-Path (Join-Path $FrontendRoot "node_modules"))) {
        & $Npm ci --ignore-scripts --no-audit --no-fund --prefix $FrontendRoot
        if ($LASTEXITCODE -ne 0) { throw "Failed to install frontend dependencies." }
    }
    & $Npm run build --prefix $FrontendRoot
    if ($LASTEXITCODE -ne 0) { throw "Failed to build the React workbench." }
}

$Backend = Start-Process `
    -WindowStyle Hidden `
    -FilePath $Python `
    -ArgumentList @("-m", "uvicorn", "app_api.main:app", "--host", "127.0.0.1", "--port", "8000") `
    -WorkingDirectory $ProjectRoot `
    -PassThru

$Worker = Start-Process `
    -WindowStyle Hidden `
    -FilePath $Python `
    -ArgumentList @("scripts\run_worker.py") `
    -WorkingDirectory $ProjectRoot `
    -PassThru

$HealthUrl = "http://127.0.0.1:8000/api/health"
$WorkbenchUrl = "http://127.0.0.1:8000"
$BackendReady = $false
$WorkerReady = $false

for ($Attempt = 0; $Attempt -lt 60; $Attempt++) {
    if (-not $BackendReady) {
        try {
            $Response = Invoke-WebRequest -UseBasicParsing -Uri $HealthUrl -TimeoutSec 1
            $BackendReady = $Response.StatusCode -eq 200
        } catch { }
    }
    if ($BackendReady -and -not $WorkerReady) {
        try {
            $Health = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 1
            $WorkerReady = $Health.worker.online -eq $true
        } catch { }
    }
    if ($BackendReady -and $WorkerReady) { break }
    Start-Sleep -Milliseconds 500
}

Write-Host ""
Write-Host "Workbench: $WorkbenchUrl" -ForegroundColor Green
Write-Host "FastAPI:   $HealthUrl (PID $($Backend.Id))" -ForegroundColor Green
Write-Host "Worker:    PID $($Worker.Id)" -ForegroundColor Green

if (-not ($BackendReady -and $WorkerReady)) {
    Write-Warning "Services are still starting. Wait a few seconds, then open the workbench URL."
}
