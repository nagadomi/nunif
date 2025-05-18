@echo off
setlocal enabledelayedexpansion

call "%~dp0\..\setenv.bat"
if "%ROOT_DIR%"=="" goto :on_error
if "%PYTHON_DIR%"=="" goto :on_error

python -m pip --version > nul 2>&1
if %ERRORLEVEL% neq 0 (
   echo Python not installed
   goto :on_error
)

call :install_python_dev
if !ERRORLEVEL! neq 0 goto :on_error
echo Successfully installed python310 developer header/lib
pause
exit /b 0


:install_python_dev
  set TMP_DIR=%ROOT_DIR%\tmp
  set PYTHON_DEV_URL=https://github.com/nagadomi/nunif/releases/download/0.0.0/python310-dev.zip

  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
             "Start-BitsTransfer -Source $env:PYTHON_DEV_URL -Destination $env:TMP_DIR\python310-dev.zip"
  if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
             "Expand-Archive -Force $env:TMP_DIR\python310-dev.zip $env:PYTHON_DIR"
  if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

  exit /b 0


:on_error
  echo Error!
  pause
  exit /b 1
