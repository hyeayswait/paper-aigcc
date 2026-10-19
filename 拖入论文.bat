@echo off
chcp 65001 >nul
setlocal
set "HERE=%~dp0"
if "%~1"=="" goto :noarg
start "" pythonw "%HERE%app.py" "%~1"
goto :eof
:noarg
start "" pythonw "%HERE%app.py"
