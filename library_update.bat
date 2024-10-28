@echo off
setlocal

REM Define variables
set "REPO_URL=https://github.com/ultralytics/ultralytics.git"
set "TEMP_DIR=%TEMP%\ultralytics_temp"
set "DEST_DIR=%CD%\ultralytics"

REM Remove any existing temporary directory
IF EXIST "%TEMP_DIR%" (
    rd /s /q "%TEMP_DIR%"
)

REM Clone the ultralytics repository into the temporary directory
git clone "%REPO_URL%" "%TEMP_DIR%"

REM Check if the clone was successful
IF ERRORLEVEL 1 (
    echo Failed to clone the ultralytics repository.
    exit /b 1
)

REM Remove existing ultralytics folder in models
IF EXIST "%DEST_DIR%" (
    rd /s /q "%DEST_DIR%"
)

REM Copy the ultralytics source code to models directory
xcopy /E /I /Y "%TEMP_DIR%\ultralytics" "%DEST_DIR%"

REM Remove the temporary directory
rd /s /q "%TEMP_DIR%"

echo Ultralytics source code has been updated in "%DEST_DIR%"
pause
endlocal