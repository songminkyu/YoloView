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

REM Check if docs/en exists, and copy it to docs at the same level as ultralytics
IF EXIST "%TEMP_DIR%\docs\en" (
    rd /s /q "%DEST_DOCS%" 2>nul
    mkdir "%DEST_DOCS%"
    xcopy /E /I /Y "%TEMP_DIR%\docs\en" "%DEST_DOCS%"
)

REM Remove the temporary directory
rd /s /q "%TEMP_DIR%"

echo Ultralytics source code has been updated in "%DEST_DIR%"
endlocal


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