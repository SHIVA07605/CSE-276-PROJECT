@echo off
REM Material Match: double-click to start the model website at http://localhost:8060
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
python web_app.py %*
pause
