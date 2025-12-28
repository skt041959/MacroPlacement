# run_training.ps1

# Define the virtual environment path
$VENV_PATH = "$PSScriptRoot\.venv"

# Check if uv is installed
if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
    Write-Host "Error: 'uv' is not found in PATH. Please install uv first." -ForegroundColor Red
    Exit 1
}

# Ensure the virtual environment exists
if (-not (Test-Path $VENV_PATH)) {
    Write-Host "Virtual environment not found at $VENV_PATH. Creating one..." -ForegroundColor Yellow
    uv venv $VENV_PATH
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create virtual environment." -ForegroundColor Red
        Exit 1
    }
}

# Set environment variable for uv to use the specific venv
$env:VIRTUAL_ENV = $VENV_PATH

Write-Host "Starting Training..." -ForegroundColor Green

# Run the training script using uv
uv run src/train.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "Training completed successfully." -ForegroundColor Green
    if (Test-Path "dashboard.html") {
        Write-Host "Opening Dashboard..." -ForegroundColor Cyan
        Start-Process "dashboard.html"
    } else {
        Write-Host "Dashboard file not found." -ForegroundColor Yellow
    }
} else {
    Write-Host "Training failed with exit code $LASTEXITCODE." -ForegroundColor Red
}

Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
