@rem set environment variables

set ROOT_DIR=%~dp0
set NUNIF_DIR=%ROOT_DIR%nunif
set PYTHON_DIR=%ROOT_DIR%python
set MINGIT_DIR=%ROOT_DIR%git
set PATH=C:\Windows\system32;C:\Windows;C:\WINDOWS\System32\WindowsPowerShell\v1.0;%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%MINGIT_DIR%\cmd

@rem clear python related variables
set PYTHONHOME=
set PYTHONPATH=
set PYTHONSTARTUP=
set PYTHONUSERBASE=
set PYTHONEXECUTABLE=
set PIP_TARGET=

@rem git env
set GIT_CONFIG_NOSYSTEM=1
