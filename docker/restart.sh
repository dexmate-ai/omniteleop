#!/bin/bash
# Restart the omniteleop Docker container.
#
# Directory-independent: run from anywhere, e.g.:
#   /path/to/omniteleop/docker/restart.sh
#   ./restart.sh   (when already in docker/)
#   docker/restart.sh   (when in repo root)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Restarting omniteleop container..."
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

# Restart the container
echo ""
echo "Stopping container..."
$COMPOSE_CMD -f docker/docker-compose.yml down

echo ""
echo "Starting container..."
$COMPOSE_CMD -f docker/docker-compose.yml up -d

echo ""
echo "Container restarted!"
echo ""
echo "Useful commands:"
echo "  View logs:     cd $PROJECT_ROOT && $COMPOSE_CMD -f docker/docker-compose.yml logs -f"
echo "  Stop:          cd $PROJECT_ROOT && $COMPOSE_CMD -f docker/docker-compose.yml down"
echo "  Shell access:  docker exec -it omniteleop bash"
echo "  Process logs:  docker exec -it omniteleop tail -f /app/logs/<process>.log"
