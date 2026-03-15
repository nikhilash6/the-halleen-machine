@echo off
setlocal
cls
title The Halleen Machine

rem === Config ===
set "APP=app.py"
rem Fixed port so we can kill the right thing
if not defined GRADIO_SERVER_PORT set "GRADIO_SERVER_PORT=7867"
if not defined GRADIO_SERVER_NAME set "GRADIO_SERVER_NAME=0.0.0.0"

echo Stopping anything listening on port %GRADIO_SERVER_PORT%...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$port=[int]$env:GRADIO_SERVER_PORT; " ^
  "$procs = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue | " ^
  "         Select-Object -Expand OwningProcess -Unique; " ^
  "if($procs){ $procs | ForEach-Object { try { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue } catch {} }; " ^
  "            Write-Host ('Killed ' + ($procs | Measure-Object).Count + ' process(es).') } else { Write-Host 'None found.' }"

rem brief pause to release the port
timeout /t 1 /nobreak >nul

echo Starting: python "%APP%"  (port %GRADIO_SERVER_PORT%, host %GRADIO_SERVER_NAME%)
set "PYTHONUNBUFFERED=1"
pushd "%~dp0"
python "%APP%"
popd

endlocal
