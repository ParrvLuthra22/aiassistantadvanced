"""AppleScript helpers for macOS command execution."""

from __future__ import annotations

import subprocess


def run_applescript(script: str) -> str:
    """Run inline AppleScript via `osascript -e` and return stdout."""
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        error = result.stderr.strip() or "Unknown AppleScript error"
        raise RuntimeError(error)
    return result.stdout.strip()


def run_applescript_file(path: str) -> str:
    """Run a compiled AppleScript file (.scpt) and return stdout."""
    result = subprocess.run(
        ["osascript", path],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        error = result.stderr.strip() or "Unknown AppleScript file error"
        raise RuntimeError(error)
    return result.stdout.strip()
