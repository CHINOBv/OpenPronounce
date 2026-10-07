# Local launcher for Windows (PowerShell): exposes ffmpeg and espeak-ng, then starts the FastAPI server.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$ffmpegBin = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-essentials_build\bin"
$espeakDir = "C:\Program Files\eSpeak NG"

$env:PATH = "$ffmpegBin;$espeakDir;$env:PATH"
$env:PHONEMIZER_ESPEAK_LIBRARY = Join-Path $espeakDir "libespeak-ng.dll"
$env:PHONEMIZER_ESPEAK_PATH = Join-Path $espeakDir "espeak-ng.exe"
$env:PYTHONUTF8 = "1"
# hf-xet parallel downloads stall without an HF token; plain HTTP resumes reliably.
$env:HF_HUB_DISABLE_XET = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

$port = if ($env:PORT) { $env:PORT } else { "8000" }
& .\.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port $port
