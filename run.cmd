@echo off
cd /d "%~dp0"
if exist "portable\HeritageStudio\_internal\runtime\pythonw.exe" (
 start "" "portable\HeritageStudio\_internal\runtime\pythonw.exe" "portable\HeritageStudio\_internal\app\main.py"
 exit /b
)
if exist ".venv\Scripts\pythonw.exe" (
 start "" ".venv\Scripts\pythonw.exe" "main.py"
 exit /b
)
echo Please run setup.ps1 first or download the Windows release.
pause
