@echo off
rem For the laziest: one command from a fresh clone to an open window.
rem Just delegates to release\run-windows.bat (venv bootstrap +
rem correct working directory) -- see release\README.md for details.
set REPO_ROOT=%~dp0
call "%REPO_ROOT%release\run-windows.bat"
