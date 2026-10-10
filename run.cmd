@echo off
cd /d "%~dp0"
if exist "portable\HeritageStudio\runtime\pythonw.exe" (
    start "" "portable\HeritageStudio\runtime\pythonw.exe" "portable\HeritageStudio\app\main.py"
    exit /b
)
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" "main.py"
    exit /b
)
echo Please run setup.ps1 first, or use the portable/HeritageStudio package.
pause
