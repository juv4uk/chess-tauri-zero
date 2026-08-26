@echo off
rem Runs chess-tauri-zero-current-windows-x86_64.exe with the correct
rem working directory. Real bug found and fixed 2026-08-26: the
rem binary's tauri.conf.json "frontendDist": "../web" resolves
rem relative to the CURRENT WORKING DIRECTORY at launch time
rem (bundle.active is false, nothing is embedded into the binary) --
rem NOT relative to wherever the .exe file itself sits. Double-clicking
rem the .exe directly from this release folder (Windows Explorer sets
rem cwd to the exe's own folder) launches a process with no visible
rem window and no error message, because the frontend assets silently
rem fail to resolve.
setlocal
set REPO_ROOT=%~dp0..

where python >nul 2>nul
if errorlevel 1 (
    echo Python не знайдено на PATH. Встанови Python 3.10+ ^(python.org або winget install Python.Python.3^) і повтори.
    exit /b 1
)
python "%REPO_ROOT%\scripts\bootstrap_venv.py"
if errorlevel 1 (
    echo Встановлення Python-залежностей не вдалося -- дивись помилку вище.
    exit /b 1
)

cd /d "%REPO_ROOT%\app\src-tauri"
"%REPO_ROOT%\release\chess-tauri-zero-current-windows-x86_64.exe"
