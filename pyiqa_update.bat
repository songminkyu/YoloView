@echo off
setlocal

REM Define variables
set "REPO_URL=https://github.com/chaofengc/IQA-PyTorch.git"
set "TEMP_DIR=%TEMP%\iqa-pytorch-temp"
set "DEST_PYIQA=%CD%\pyiqa"
set "DEST_DOCS=%CD%\docs"

REM Remove any existing temporary directory
IF EXIST "%TEMP_DIR%" (
    rd /s /q "%TEMP_DIR%"
)

REM Clone the pyiqa repository into the temporary directory
git clone "%REPO_URL%" "%TEMP_DIR%"

REM Check if the clone was successful
IF ERRORLEVEL 1 (
    echo Failed to clone the ultralytics repository.
    exit /b 1
)

REM Remove existing ultralytics folder
IF EXIST "%DEST_PYIQA%" (
    rd /s /q "%DEST_PYIQA%"
)

REM Copy the pyiqa source code to its destination
xcopy /E /I /Y "%TEMP_DIR%\pyiqa" "%DEST_PYIQA%"

REM Remove the temporary directory
rd /s /q "%TEMP_DIR%"

echo Pyiqa source code has been updated in "%DEST_DIR%"
endlocal