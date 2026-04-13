#!/usr/bin/env bash
# Launch the OmniTeleop frontend (React/Vite dev server).
#
# Usage:
#   ./launch.sh               # default port 5173
#   ./launch.sh --port 3000   # custom port
#
# NOTE: --record-mode is a backend flag, not a frontend flag.
#   Start the backend separately:
#     python -m internal.backend_app.app_backend                # no recording
#     python -m internal.backend_app.app_backend --record-mode  # with recording
#   The frontend detects the mode automatically via the WebSocket state stream.

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure node/npm/npx are on PATH when installed in a non-standard location.
for _node_dir in "$HOME/nodejs/bin" "$HOME/.nvm/versions/node"/*/bin; do
  [ -x "$_node_dir/node" ] && export PATH="$_node_dir:$PATH" && break
done
PORT=5173

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1"
      echo "Usage: $0 [--port N]"
      echo ""
      echo "To enable recording, start the backend with:"
      echo "  python -m internal.backend_app.app_backend --record-mode"
      exit 1
      ;;
  esac
done

cd "$DIR"

if [ ! -d node_modules ]; then
  echo "→ Installing dependencies..."
  if command -v pnpm &>/dev/null; then pnpm install
  elif command -v npm &>/dev/null; then npm install
  else echo "Error: npm or pnpm not found"; exit 1
  fi
fi

echo "→ Frontend: http://localhost:${PORT}"
PORT="$PORT" npx vite --host 0.0.0.0 --port "$PORT"
