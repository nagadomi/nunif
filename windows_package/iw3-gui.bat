@echo off

call %~dp0setenv.bat
pushd %NUNIF_DIR% && start "" pythonw -m iw3.gui && popd
exit /b 0
