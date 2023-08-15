@echo off

setlocal enabledelayedexpansion
call %~dp0\setenv.bat


@rem check to make sure the variables are available
if "%ROOT_DIR%"=="" goto :on_error
if "%PYTHON_DIR%"=="" goto :on_error
if "%MINGIT_DIR%"=="" goto :on_error
if "%NUNIF_DIR%"=="" goto :on_error

@rem try python
python -m pip --version > nul 2>&1
if %ERRORLEVEL% neq 0 (
  call :install_python
  if !ERRORLEVEL! neq 0 goto :on_error
)


@rem try git
git --version > nul 2>&1
if %ERRORLEVEL% neq 0 (
  call :install_git
  if !ERRORLEVEL! neq 0 goto :on_error
)


@rem try nunif
if not exist %NUNIF_DIR% (
  git clone https://github.com/nagadomi/nunif.git %NUNIF_DIR%
  if !ERRORLEVEL! neq 0 goto :on_error
) else (
  git -C %NUNIF_DIR% pull --ff
  if !ERRORLEVEL! neq 0 (
    git -C %NUNIF_DIR% reset --hard
    if !ERRORLEVEL! neq 0 goto :on_error
    git -C %NUNIF_DIR% pull --ff
    if !ERRORLEVEL! neq 0 goto :on_error
  )
)


echo Install Python Packages...
python -m pip install --no-cache-dir --upgrade pip
if %ERRORLEVEL% neq 0 goto :on_error
python -m pip install --no-cache-dir --upgrade -r %NUNIF_DIR%\requirements-torch.txt
if %ERRORLEVEL% neq 0 goto :on_error
python -m pip install --no-cache-dir --upgrade -r %NUNIF_DIR%\requirements.txt
if %ERRORLEVEL% neq 0 goto :on_error
python -m pip install --no-cache-dir --upgrade -r %NUNIF_DIR%\requirements-gui.txt
if %ERRORLEVEL% neq 0 goto :on_error


echo Download Models...
pushd %NUNIF_DIR% && python -m waifu2x.download_models && popd
if %ERRORLEVEL% neq 0 goto :on_error

pushd %NUNIF_DIR% && python -m iw3.download_models && popd
if %ERRORLEVEL% neq 0 goto :on_error


@rem warmup, create pyc
pushd %NUNIF_DIR% && python -m iw3.gui --help > nul && popd
if %ERRORLEVEL% neq 0 goto :on_error
pushd %NUNIF_DIR% && python -m waifu2x.gui --help > nul && popd
if %ERRORLEVEL% neq 0 goto :on_error


@rem all succeeded
echo Successfully installed nunif
pause
exit /b 0


:install_git
  echo Install MinGit...

  setlocal
  set MINGIT_URL=https://github.com/git-for-windows/git/releases/download/v2.41.0.windows.2/MinGit-2.41.0.2-64-bit.zip
  set TMP_DIR=%ROOT_DIR%\tmp

  if not exist %TMP_DIR% mkdir %TMP_DIR%

  @rem Install MinGit
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
             "Start-BitsTransfer -Source $env:MINGIT_URL -Destination $env:TMP_DIR\mingit.zip"
  if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
             "Expand-Archive -Force $env:TMP_DIR\mingit.zip $env:MINGIT_DIR"
  if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

  echo Successfully installed MinGit

  exit /b 0


:install_python
  echo Install Python...

  setlocal
  set PYTHON_URL=https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip
  set GETPIP_URL=https://bootstrap.pypa.io/get-pip.py
  set PTH_PATH=%PYTHON_DIR%\python310._pth
  set TMP_DIR=%ROOT_DIR%\tmp

  if not exist %TMP_DIR% mkdir %TMP_DIR%

  @rem Install Embeddable Python
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
             "Start-BitsTransfer -Source $env:PYTHON_URL -Destination $env:TMP_DIR\python.zip"
  if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
             "Expand-Archive -Force $env:TMP_DIR\python.zip $env:PYTHON_DIR"
  if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

  @rem setup pth file to work isolated mode
  echo ..\nunif>> %PTH_PATH%
  echo import site>> %PTH_PATH%

  echo Successfully installed Python

  echo Install pip...

  @rem ensure pip
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
             "Start-BitsTransfer -Source $env:GETPIP_URL -Destination $env:PYTHON_DIR\get-pip.py"
  if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

  python %PYTHON_DIR%\get-pip.py
  if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

  echo Successfully installed pip
  exit /b 0


:on_error
  echo Error!
  pause
  exit /b 1
