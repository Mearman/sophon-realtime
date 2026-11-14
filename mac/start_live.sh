#!/bin/bash
# Live mode on a Mac: the GPT-Live and OpenAI Realtime gateway (live/sophon_live) in front of the Mac engine, in one process.
#   bash start_live.sh                    # English model, gateway on port 8780
#   LANG_=zh LIVE_PORT=8780 bash start_live.sh
# env: SOPHON_HOME (default ~/sophon-mac), LANG_ (zh|en, default en), LIVE_PORT, LIVE_HOST (default 127.0.0.1), SOPHON_LIVE_ASR (default mlx-community/Qwen3-ASR-0.6B-8bit),
#      SOPHON_LIVE_BACKEND, SOPHON_LIVE_BACKEND_URL, SOPHON_LIVE_BACKEND_KEY (a larger OpenAI-compatible model that answers delegated requests; see live/README.md)
# Needs `aiohttp` and `onnxruntime` in the Mac install's environment (install.sh adds them).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; H="${SOPHON_HOME:-$HOME/sophon-mac}"
PY="$H/.venv/bin/python"; [ -x "$PY" ] || { echo "no Python environment at $H/.venv: run bash install.sh first"; exit 1; }
export SOPHON_MAC_DIR="$HERE" SOPHON_LIVE_LANG="${LANG_:-en}" SOPHON_LIVE_ASR="${SOPHON_LIVE_ASR:-mlx-community/Qwen3-ASR-0.6B-8bit}"
cd "$HERE/../live"
exec "$PY" -m sophon_live.gateway --engine mlx --mlx-root "$H/models" --mlx-lang "${LANG_:-en}" --host "${LIVE_HOST:-127.0.0.1}" --port "${LIVE_PORT:-8780}"
