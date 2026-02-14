@echo off
setlocal EnableExtensions

cd /d "%~dp0"

where git >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Git was not found on this machine.
    echo Install Git for Windows from:
    echo   https://git-scm.com/download/win
    echo During setup, choose the option that adds Git to PATH.
    pause
    endlocal & exit /b 1
)

git rev-parse --is-inside-work-tree >nul 2>nul
if errorlevel 1 (
    echo [ERROR] This folder is not a Git repository.
    echo Open the project folder that was cloned from GitHub/Git.
    pause
    endlocal & exit /b 1
)

echo Fetching latest refs...
git fetch --prune
if errorlevel 1 (
    echo [ERROR] Fetch failed.
    echo Check internet access and remote permissions, then retry.
    pause
    endlocal & exit /b 1
)

echo Pulling latest changes...
git pull --ff-only
if errorlevel 1 (
    echo [ERROR] Pull failed.
    echo This usually means you have local changes or branch divergence.
    echo Commit/stash your changes, then run this script again.
    pause
    endlocal & exit /b 1
)

echo [OK] Repository is up to date.
pause
endlocal & exit /b 0
