$ErrorActionPreference = "Stop"

Write-Host "Creating Python virtual environment..."
py -3.12 -m venv .venv

Write-Host "Activating environment..."
& .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing dependencies..."
pip install -r requirements.txt

Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Copy your train/ and test/ folders into data/"
Write-Host "2. Run: python -m banana_ai.ml.dataset_report"
Write-Host "3. Run: python -m banana_ai.ml.train"
Write-Host "4. Start DB: docker compose up -d db"
Write-Host "5. Initialize DB: python -m banana_ai.db.init_db"
Write-Host "6. Run UI: streamlit run src/banana_ai/app.py"
