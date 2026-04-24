if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "Missing .venv. Create it first with: python -m venv .venv"
    exit 1
}

& ".\.venv\Scripts\python.exe" "-m" "streamlit" "run" "app/app.py"
