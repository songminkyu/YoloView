@echo off
setlocal

REM Define variables
set "REPO_URL=https://github.com/roboflow/supervision.git"
set "TEMP_DIR=%TEMP%\supervision_temp"
set "DEST_DIR=%CD%\supervision"

REM Remove any existing temporary directory
IF EXIST "%TEMP_DIR%" (
    rd /s /q "%TEMP_DIR%"
)

REM Clone the supervision repository into the temporary directory
git clone "%REPO_URL%" "%TEMP_DIR%"

REM Check if the clone was successful
IF ERRORLEVEL 1 (
    echo Failed to clone the supervision repository.
    exit /b 1
)

REM Remove existing supervision folder in models
IF EXIST "%DEST_DIR%" (
    rd /s /q "%DEST_DIR%"
)

REM Copy the supervision source code to models directory
xcopy /E /I /Y "%TEMP_DIR%\supervision" "%DEST_DIR%"

REM Remove the temporary directory
rd /s /q "%TEMP_DIR%"

echo Ultralytics source code has been updated in "%DEST_DIR%"
pause
endlocal