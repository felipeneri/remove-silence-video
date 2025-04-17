#!/bin/bash

# Wrapper script to run transcribe_video.py with the correct Python interpreter

# Path to the Python interpreter that has WhisperX installed
PYTHON_PATH="/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10"

# Path to the transcribe_video.py script (in the same directory as this script)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRANSCRIBE_SCRIPT="$SCRIPT_DIR/transcribe_video.py"

# Run the script with all arguments passed to this wrapper
"$PYTHON_PATH" "$TRANSCRIBE_SCRIPT" "$@" 