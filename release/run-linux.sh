#!/bin/bash
# Runs chess-tauri-zero-current-linux-x86_64 with the correct working
# directory. Real bug found and fixed 2026-08-26: the binary's
# tauri.conf.json "frontendDist": "../web" resolves relative to the
# CURRENT WORKING DIRECTORY at launch time (bundle.active is false, so
# nothing is embedded into the binary) -- NOT relative to wherever the
# executable file itself happens to sit. Double-clicking the binary
# directly from this release/ folder (or any folder other than
# app/src-tauri/) launches a process with no visible window and no
# error message, because the frontend assets silently fail to resolve.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if command -v git >/dev/null 2>&1 && [ -d "$REPO_ROOT/.git" ]; then
    echo "Оновлюю репозиторій (git pull)..."
    git -C "$REPO_ROOT" pull --ff-only \
        || echo "git pull не вдався (локальні зміни або немає інтернету?) -- продовжую з тим, що вже є."
fi

PY=python3
command -v python3 >/dev/null 2>&1 || PY=python
"$PY" "$REPO_ROOT/scripts/bootstrap_venv.py"

cd "$REPO_ROOT/app/src-tauri"
exec "$REPO_ROOT/release/chess-tauri-zero-current-linux-x86_64"
