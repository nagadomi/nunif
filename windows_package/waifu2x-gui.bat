@echo off

call "%~dp0setenv.bat"

pushd "%NUNIF_DIR%" && python -m waifu2x.download_models && popd
if %ERRORLEVEL% neq 0 goto :on_error

pushd "%NUNIF_DIR%" && start "" pythonw -m waifu2x.gui && popd
exit /b 0

:on_error
  echo Error!
  pause
  exit /b 1
