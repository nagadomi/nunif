@echo off

call %~dp0setenv.bat
pushd %NUNIF_DIR% && start "" pythonw -m waifu2x.gui && popd
exit /b 0
