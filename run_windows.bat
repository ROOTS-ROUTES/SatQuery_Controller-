@echo off
title SatQuery AI - Starting...
cd /d "%~dp0"

set VENV_DIR=venv
set MARKER=%VENV_DIR%\.setup_complete
set VENV_PY=%VENV_DIR%\Scripts\python.exe

REM --- Find a Python interpreter to CREATE the venv with. Once the venv
REM     exists, everything else below calls %VENV_PY% by its full path -
REM     never a bare "python"/"pip" command - so PATH resolution issues
REM     (e.g. a Windows Store Python stub or another install shadowing the
REM     real one) can't silently point us at the wrong interpreter. ---
where python >nul 2>&1
if %ERRORLEVEL%==0 (
    set PYTHON_CMD=python
) else (
    where py >nul 2>&1
    if %ERRORLEVEL%==0 (
        set PYTHON_CMD=py -3
    ) else (
        echo.
        echo ============================================================
        echo  Python was not found on this machine.
        echo  Install Python 3.10 or newer from https://python.org
        echo  ^(tick "Add python.exe to PATH" during install^), then
        echo  double-click this file again.
        echo ============================================================
        pause
        exit /b 1
    )
)

REM --- First run only: create a virtual environment and install dependencies.
REM     A marker file means every run after the first just launches straight
REM     away instead of re-running pip install. ---
if not exist "%MARKER%" (
    echo ============================================================
    echo  First-time setup - this can take several minutes.
    echo  This only happens once; every run after this is instant.
    echo ============================================================
    echo.

    if not exist "%VENV_DIR%" (
        %PYTHON_CMD% -m venv %VENV_DIR%
        if %ERRORLEVEL% NEQ 0 (
            echo Failed to create the virtual environment. See the error above.
            pause
            exit /b 1
        )
    )

    if not exist "%VENV_PY%" (
        echo.
        echo ============================================================
        echo  The virtual environment looks incomplete ^(%VENV_PY% is
        echo  missing^). Delete the "venv" folder and run this file again
        echo  for a clean setup.
        echo ============================================================
        pause
        exit /b 1
    )

    "%VENV_PY%" -m pip install --upgrade pip

    REM llama-cpp-python has no prebuilt wheel on PyPI - a plain install
    REM would build from source and fail on Windows due to a long-path
    REM limit while extracting llama.cpp's bundled web UI assets. Install
    REM it first from the maintainer's prebuilt CPU wheel index instead.
    REM For GPU offload later, see the CUDA wheel index note in README.
    echo Installing the brain LLM engine, prebuilt wheel...
    "%VENV_PY%" -m pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo ============================================================
        echo  Failed to install llama-cpp-python from the prebuilt wheel
        echo  index. Check your internet connection and try again, or see
        echo  README.md for alternative install options.
        echo ============================================================
        pause
        exit /b 1
    )

    "%VENV_PY%" -m pip install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo ============================================================
        echo  Setup failed - see the error above.
        echo  Common fix: if llama-cpp-python or bitsandbytes failed to
        echo  install, see the notes in README.md, fix it, then run this
        echo  file again ^(it will resume from here, venv is already made^).
        echo ============================================================
        pause
        exit /b 1
    )

    echo done> "%MARKER%"
    echo.
    echo Setup complete.
    echo.
)

REM --- Sanity check: confirm the venv's Python can actually import the key
REM     packages before launching, so a broken/partial install shows a clear
REM     message here instead of a confusing traceback from server.py. ---
"%VENV_PY%" -c "import fastapi, uvicorn" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ============================================================
    echo  The virtual environment exists but is missing required
    echo  packages ^(fastapi/uvicorn^). Re-installing now...
    echo ============================================================
    "%VENV_PY%" -m pip install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo Re-install failed - see the error above.
        pause
        exit /b 1
    )
)

"%VENV_PY%" -c "import llama_cpp" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ============================================================
    echo  llama-cpp-python is missing. Installing the prebuilt CPU
    echo  wheel now ^(this avoids the long-path build-from-source issue^)...
    echo ============================================================
    "%VENV_PY%" -m pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
    if %ERRORLEVEL% NEQ 0 (
        echo Install failed - see the error above.
        pause
        exit /b 1
    )
)

REM --- Launch. The browser UI appears within a second or two; the brain
REM     and vision models load in the background after that - see the
REM     "Loading models..." state in the UI while that finishes. ---
title SatQuery AI
"%VENV_PY%" server.py

echo.
echo Server stopped.
pause
