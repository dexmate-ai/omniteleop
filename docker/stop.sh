#!/bin/bash
# Stop and remove the omniteleop Docker container.
#
# Directory-independent: run from anywhere, e.g.:#   /path/to/omniteleop/docker/stop.sh
#   ./stop.sh   (when already in docker/)
#   docker/stop.sh   (when in repo root)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load Jetson credentials (same as run.sh)
if [ -f "$SCRIPT_DIR/.env.local" ]; then
  source "$SCRIPT_DIR/.env.local"
fi

JETSON_IP="${JETSON_IP:-192.168.50.20}"
JETSON_USER="${JETSON_USER:-dexmate}"
JETSON_PASSWORD="${JETSON_PASSWORD:-hello-dex}"

# Build SSH command
if [ -n "$JETSON_PASSWORD" ] && command -v sshpass &> /dev/null; then
  SSH_CMD="sshpass -p '$JETSON_PASSWORD' ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no"
else
  SSH_CMD="ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no"
fi

# Stop dexsensor on Jetson
echo "Stopping dexsensor on Jetson..."
if eval "$SSH_CMD $JETSON_USER@$JETSON_IP 'pkill -f \"dexsensor launch\"'" 2>/dev/null; then
  echo "✓ dexsensor stopped on Jetson"
else
  echo "⚠ dexsensor was not running on Jetson (or failed to stop)"
fi

echo ""
echo "Stopping omniteleop container..."
echo "  Compose file: $SCRIPT_DIR/docker-compose.yml"
echo "  Project root: $PROJECT_ROOT"

# Change to project root directory so context paths work correctly
cd "$PROJECT_ROOT"

# Check which docker compose command is available
if docker compose version &>/dev/null 2>&1; then
  COMPOSE_CMD="docker compose"
elif command -v docker-compose &>/dev/null; then
  COMPOSE_CMD="docker-compose"
else
  echo "Error: Neither 'docker compose' nor 'docker-compose' is available"
  echo "Please install docker compose: sudo apt-get install docker-compose-plugin"
  exit 1
fi

# Stop and remove the container
$COMPOSE_CMD -f docker/docker-compose.yml down

echo ""
echo "Container stopped and removed!"
echo ""
echo "To start again, run:"
echo "  $SCRIPT_DIR/run.sh"
