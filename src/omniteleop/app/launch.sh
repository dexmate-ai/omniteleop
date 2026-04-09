#!/usr/bin/env bash
# Launch OmniTeleop backend + frontend together.
#
# Usage:
#   ./launch.sh                # normal mode
#   ./launch.sh --rpi-mode     # RPi leader mode (bypasses local hw checks)
#   ./launch.sh --port 3000    # custom frontend port (default 5173)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

# Ensure node/npx are on PATH when installed in a non-standard location.
for _node_dir in "$HOME/nodejs/bin" "$HOME/.nvm/versions/node"/*/bin; do
  [ -x "$_node_dir/node" ] && export PATH="$_node_dir:$PATH" && break
done

PORT=5173
RPI_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rpi-mode) RPI_FLAG="--rpi-mode"; shift ;;
    --port)     PORT="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--rpi-mode] [--port N]"
      exit 1
      ;;
  esac
done

BACKEND_PID=""
FRONTEND_PID=""

kill_pid() {
  local pid="$1" name="$2"
  [[ -z "$pid" ]] && return
  if kill -0 "$pid" 2>/dev/null; then
    echo "→ Stopping $name (pid $pid)..."
    kill -TERM "$pid" 2>/dev/null || true
    # Wait up to 5 s then force-kill
    for _ in $(seq 1 10); do
      kill -0 "$pid" 2>/dev/null || return
      sleep 0.5
    done
    echo "→ Force-killing $name (pid $pid)..."
    kill -KILL "$pid" 2>/dev/null || true
  fi
}

cleanup() {
  echo ""
  kill_pid "$BACKEND_PID"  "backend"
  kill_pid "$FRONTEND_PID" "frontend"
  echo "→ Done."
}
trap cleanup EXIT INT TERM

# ── Backend ──────────────────────────────────────────────────────────────────
echo "→ Starting backend${RPI_FLAG:+ (rpi-mode)}..."
cd "$REPO_ROOT"
python -m omniteleop.app.backend.app_backend $RPI_FLAG &
BACKEND_PID=$!

# ── Frontend ─────────────────────────────────────────────────────────────────
echo "→ Starting frontend on port $PORT..."
cd "$FRONTEND_DIR"
if [ ! -d node_modules ]; then
  echo "→ Installing frontend dependencies..."
  if command -v npm &>/dev/null; then
    npm install
  else
    echo "Error: npm not found"; exit 1
  fi
fi
npx vite --host 0.0.0.0 --port "$PORT" &
FRONTEND_PID=$!

echo ""
echo "  Backend  PID : $BACKEND_PID"
echo "  Frontend PID : $FRONTEND_PID"
echo "  Frontend URL : http://localhost:${PORT}"
echo ""
echo "  Press Ctrl+C to stop both."
echo ""

wait "$BACKEND_PID" "$FRONTEND_PID"
