@echo off
setlocal
cd /d "%~dp0"
where git >nul 2>nul
if errorlevel 1 (
  echo Git is required. Install Git for Windows, then reopen this terminal.
  exit /b 1
)
if not exist .venv\Scripts\python.exe (
  py -3.11 -m venv .venv
  if errorlevel 1 (
    echo Install Python 3.11 with the Python launcher, then run this script again.
    exit /b 1
  )
)
.venv\Scripts\python -m pip install --upgrade pip
if errorlevel 1 exit /b 1
.venv\Scripts\python -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 exit /b 1
.venv\Scripts\python -m pip install -e ".[test,paradee]"
if errorlevel 1 exit /b 1
.venv\Scripts\python -m spacy download en_core_web_sm
if errorlevel 1 exit /b 1
.venv\Scripts\python -m pytest -q
if errorlevel 1 exit /b 1
echo Setup complete. Run .venv\Scripts\python -m speechloop test-paradee to verify the actual model.
