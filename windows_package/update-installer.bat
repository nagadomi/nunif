@echo off

setlocal enabledelayedexpansion
call %~dp0\setenv.bat

copy %NUNIF_DIR%\windows_package\update.bat %~dp0\update.bat

pause
exit /b 0
