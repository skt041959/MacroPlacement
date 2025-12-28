#!/bin/bash

# run_training.sh

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VENV_PATH="$SCRIPT_DIR/.venv"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not found in PATH. Please install uv first."
    exit 1
fi

# Ensure the virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Virtual environment not found at $VENV_PATH. Creating one..."
    uv venv "$VENV_PATH"
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment."
        exit 1
    fi
fi

# Set environment variable for uv
export VIRTUAL_ENV="$VENV_PATH"

echo "Starting Training..."

# Run the training script
uv run src/train.py

if [ $? -eq 0 ]; then
    echo "Training completed successfully."
    if [ -f "dashboard.html" ]; then
        echo "Opening Dashboard..."
        # Try to open based on OS
        if [[ "$OSTYPE" == "darwin"* ]]; then
            open dashboard.html
        elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
            start dashboard.html
        else
            xdg-open dashboard.html
        fi
    else
        echo "Dashboard file not found."
    fi
else
    echo "Training failed."
    exit 1
fi
