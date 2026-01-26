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

for /f %%A in ('python -c "import torch; print(torch.__version__.split('+')[0].split('-')[0])"') do set TORCH_VER=%%A
if "%TORCH_VER%" == "2.7.1" (
  set TRITON_WINDOWS="triton-windows<3.4"
) else if "%TORCH_VER%" == "2.9.1" (
  set TRITON_WINDOWS="triton-windows<3.6"
) else (
  echo Unsupported pytorch version: %TORCH_VER%
  goto on_error
)

echo Reinstalling !TRITON_WINDOWS!
python -m pip uninstall --yes triton-windows
python -m pip install --no-cache-dir --upgrade !TRITON_WINDOWS!
if %ERRORLEVEL% neq 0 goto :on_error
echo Successfully installed triton-windows
pause
exit /b 0


:on_error
  echo Error!
  pause
  exit /b 1
