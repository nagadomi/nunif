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
echo Successfully installed python developer files
pause
exit /b 0


:install_python_dev
  set TMP_DIR=%ROOT_DIR%\tmp
  set PYTHON_312_10_DEV_URL=https://github.com/nagadomi/nunif/releases/download/python_dev_release/python-3.12.10-dev.zip
  set PYTHON_310_11_DEV_URL=https://github.com/nagadomi/nunif/releases/download/python_dev_release/python-3.10.11-dev.zip

  for /f %%A in ('python -c "import sys; print('.'.join([str(v) for v in sys.version_info[0:3]]))"') do set PYVER=%%A
  if "%PYVER%" == "3.12.10" (
    set PYTHON_DEV_URL=%PYTHON_312_10_DEV_URL%
  ) else if "%PYVER%" == "3.10.11" (
    set PYTHON_DEV_URL=%PYTHON_310_11_DEV_URL%
  ) else (
    echo Unsupported python version: %PYVER%
    goto on_error
  )
  echo Installing %PYTHON_DEV_URL%

  set ZIP_FILE=%TMP_DIR%\python-%PYVER%-dev.zip
  if not exist "%TMP_DIR%" mkdir "%TMP_DIR%"

  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
             "Start-BitsTransfer -Source $env:PYTHON_DEV_URL -Destination $env:ZIP_FILE -ErrorAction Stop"
  if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
             "Expand-Archive -Force $env:ZIP_FILE $env:PYTHON_DIR -ErrorAction Stop"
  if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

  exit /b 0


:on_error
  echo Error!
  pause
  exit /b 1
