@echo off
echo Attempting to create symlink 'venv' pointing to '.venv'...
mklink /D venv .venv
if %errorlevel% equ 0 (
    echo Symlink created successfully.
) else (
    echo Failed to create symlink. Please run this script as an Administrator.
    echo You can right-click on Command Prompt or PowerShell and select "Run as administrator".
)
pause
