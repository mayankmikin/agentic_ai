import sys
import os
import io
import json
import contextlib
import urllib.request
# Security: confine list_directory_contents to this base directory and its descendants
# Set to the current working directory when the script starts
SAFE_BASE_DIR = os.path.abspath(os.getcwd())

# Security: cap the runtime of the Python interpreter tool (seconds)
PYTHON_EXEC_TIMEOUT = 5

def list_directory_contents(path: str = ".") -> str:
    """Lists files and directories within a path, constrained to the safe base directory."""
    try:
        # Resolve to an absolute path and verify it sits inside SAFE_BASE_DIR
        # This blocks traversal attempts like '../../etc' or absolute paths like '/'
        requested = os.path.abspath(os.path.join(SAFE_BASE_DIR, path))
        if not (requested == SAFE_BASE_DIR or requested.startswith(SAFE_BASE_DIR + os.sep)):
            return (
                f"Error: Access denied. The path '{path}' resolves outside the "
                f"permitted workspace ({SAFE_BASE_DIR})."
            )

        if not os.path.exists(requested):
            return f"Error: The path '{path}' does not exist."

        if not os.path.isdir(requested):
            return f"Error: The path '{path}' is not a directory."

        entries = sorted(os.listdir(requested))
        if not entries:
            return f"The directory '{path}' is empty."

        lines = [f"Contents of '{path}' ({len(entries)} item(s)):"]
        for name in entries:
            full = os.path.join(requested, name)
            if os.path.isdir(full):
                lines.append(f"  [DIR]  {name}/")
            else:
                try:
                    size = os.path.getsize(full)
                    lines.append(f"  [FILE] {name} ({size} bytes)")
                except OSError:
                    lines.append(f"  [FILE] {name}")
        return "\n".join(lines)

    except Exception as e:
        return f"Error listing directory '{path}': {e}"