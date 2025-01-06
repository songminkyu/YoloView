@echo off
setlocal

REM Define variables
set "REPO_URL=https://github.com/ultralytics/ultralytics.git"
set "TEMP_DIR=%TEMP%\ultralytics_temp"
set "DEST_ULTRALYTICS=%CD%\ultralytics"
set "DEST_DOCS=%CD%\docs"

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

REM Remove existing ultralytics folder
IF EXIST "%DEST_ULTRALYTICS%" (
    rd /s /q "%DEST_ULTRALYTICS%"
)

REM Copy the ultralytics source code to its destination
xcopy /E /I /Y "%TEMP_DIR%\ultralytics" "%DEST_ULTRALYTICS%"

REM Check if docs/en exists, and copy its content to docs at the same level as ultralytics
IF EXIST "%TEMP_DIR%\docs\en" (
    xcopy /E /I /Y "%TEMP_DIR%\docs\en" "%DEST_DOCS%\Ultralytics"
)

REM Remove the temporary directory
rd /s /q "%TEMP_DIR%"

echo Ultralytics source code has been updated in "%DEST_DIR%"
endlocal