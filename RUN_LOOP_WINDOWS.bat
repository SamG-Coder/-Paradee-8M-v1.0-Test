@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run SETUP_WINDOWS.bat first.
  exit /b 1
)
.venv\Scripts\python -m speechloop loop --engine paradee --generate 120 --rounds 3 --fresh 100 --steps 300 --output runs\loop %*
exit /b %errorlevel%
