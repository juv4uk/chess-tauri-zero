#!/bin/bash
# For the laziest: one command from a fresh clone to an open window.
# Just delegates to release/run-linux.sh (venv bootstrap + correct
# working directory) -- see release/README.md for what that actually
# does and why a plain double-click on the binary itself doesn't work.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$REPO_ROOT/release/run-linux.sh"
