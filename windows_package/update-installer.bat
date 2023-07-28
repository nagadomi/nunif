@echo off

setlocal enabledelayedexpansion
call %~dp0\setenv.bat

copy %NUNIF_DIR%\windows_package\update.bat %~dp0\update.bat
copy %NUNIF_DIR%\windows_package\setenv.bat %~dp0\setenv.bat

pause
exit /b 0
