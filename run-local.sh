#!/usr/bin/env bash
# Local launcher for Windows (Git Bash): exposes ffmpeg and espeak-ng, then starts the FastAPI server.
set -euo pipefail
cd "$(dirname "$0")"

FFMPEG_BIN="$LOCALAPPDATA/Microsoft/WinGet/Packages/Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.1-essentials_build/bin"
ESPEAK_DIR="/c/Program Files/eSpeak NG"

export PATH="$FFMPEG_BIN:$ESPEAK_DIR:$PATH"
export PHONEMIZER_ESPEAK_LIBRARY="C:\\Program Files\\eSpeak NG\\libespeak-ng.dll"
export PHONEMIZER_ESPEAK_PATH="C:\\Program Files\\eSpeak NG\\espeak-ng.exe"
export PYTHONUTF8=1
# hf-xet parallel downloads stall without an HF token; plain HTTP resumes reliably.
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

exec .venv/Scripts/python.exe -m uvicorn server:app --host 127.0.0.1 --port "${PORT:-8000}"
