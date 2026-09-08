#!/usr/bin/env bash
# Run this to start SatQuery AI: ./run_mac_linux.sh
# First run does one-time setup (creates a virtual environment, installs
# dependencies); every run after that launches straight away.
set -e
cd "$(dirname "$0")"

VENV_DIR="venv"
MARKER="$VENV_DIR/.setup_complete"
VENV_PY="$VENV_DIR/bin/python"

if ! command -v python3 >/dev/null 2>&1; then
    echo "============================================================"
    echo " Python 3 was not found on this machine."
    echo " Install Python 3.10+ (e.g. from https://python.org or your"
    echo " package manager), then run this script again."
    echo "============================================================"
    exit 1
fi

if [ ! -f "$MARKER" ]; then
    echo "============================================================"
    echo " First-time setup - this can take several minutes."
    echo " This only happens once; every run after this is instant."
    echo "============================================================"
    echo

    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
    fi

    if [ ! -x "$VENV_PY" ]; then
        echo "The virtual environment looks incomplete ($VENV_PY is missing)."
        echo "Delete the venv folder and run this script again for a clean setup."
        exit 1
    fi

    "$VENV_PY" -m pip install --upgrade pip

    # Prebuilt wheel index for llama-cpp-python (the brain LLM engine) —
    # more reliable than a bare install, which can still fall back to
    # building from source on some platforms/Python versions.
    echo "Installing the brain LLM engine (llama-cpp-python, prebuilt wheel)..."
    "$VENV_PY" -m pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

    "$VENV_PY" -m pip install -r requirements.txt

    touch "$MARKER"
    echo
    echo "Setup complete."
    echo
fi

# Sanity check, same as the Windows launcher: confirm the venv's Python can
# actually import the key packages before launching, rather than a
# confusing traceback from server.py if the environment is somehow broken.
if ! "$VENV_PY" -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "============================================================"
    echo " The virtual environment exists but is missing required"
    echo " packages. Re-installing now..."
    echo "============================================================"
    "$VENV_PY" -m pip install -r requirements.txt
fi

if ! "$VENV_PY" -c "import llama_cpp" 2>/dev/null; then
    echo "============================================================"
    echo " llama-cpp-python is missing. Installing the prebuilt CPU"
    echo " wheel now..."
    echo "============================================================"
    "$VENV_PY" -m pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
fi

# The browser UI appears within a second or two; the brain and vision
# models load in the background after that. Always call the venv's own
# interpreter by full path — never a bare "python" — so this can't
# silently pick up a different Python installation.
"$VENV_PY" server.py
