@echo off
cd /d "%~dp0"
if exist "portable\HeritageStudio\runtime\pythonw.exe" (
    start "" "portable\HeritageStudio\runtime\pythonw.exe" "portable\HeritageStudio\app\main.py" --results
) else (
    start "" ".venv\Scripts\pythonw.exe" "main.py" --results
)
