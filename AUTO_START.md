# Auto Start
Script to start miner when system start.

## Windows
Set the variables `PATH_EXE` and `MINER_PARAMETERS`.
```bat
@echo off
set PATH_EXE="FULLPATH MINER EXE"
set MINER_PARAMETERS="YOUR PARAMETERS"

:loop
rem Wait until miner.exe is not running
:wait
tasklist /fi "imagename eq miner.exe" | find /i "miner.exe" >nul
if not errorlevel 1 (
    timeout /t 10 /nobreak >nul
    goto wait
)

rem Start miner.exe with HIGH priority
start "" /high "%PATH_EXE%\miner.exe" %MINER_PARAMETERS%

timeout /t 5 /nobreak >nul
goto loop
```
