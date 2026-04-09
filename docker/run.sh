#!/bin/bash
# Run the omniteleop Docker container with docker-compose.
#
# Directory-independent: run from anywhere, e.g.:
#   /path/to/omniteleop/docker/run.sh
#   ./run.sh   (when already in docker/)
#
# Environment variables (optional):
#   ROBOT_CONFIG     - Robot configuration (default: vega_1u_f5d6)
#                      Example: ROBOT_CONFIG=vega_1u_gripper ./run.sh
#   RECORDER_SAVE_DIR - Directory to save recordings (default: /exchange/data_sink)
#   JETSON_IP        - Jetson IP address (default: 192.168.50.20)
#   JETSON_USER      - Jetson username (default: dexmate)
#   JETSON_PASSWORD  - Jetson password (default: hello-dex)
#
# Or create docker/.env.local to set these permanently (not committed to git)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load local config if it exists (not committed to git)
if [ -f "$SCRIPT_DIR/.env.local" ]; then
  source "$SCRIPT_DIR/.env.local"
  echo "✓ Loaded settings from .env.local"
fi

# Always fetch ROBOT_NAME from Jetson (ignoring any local environment variable)
# Unset ROBOT_NAME to ensure we use the Jetson value, not terminal value
unset ROBOT_NAME

JETSON_IP="${JETSON_IP:-192.168.50.20}"
JETSON_USER="${JETSON_USER:-dexmate}"
JETSON_PASSWORD="${JETSON_PASSWORD:-hello-dex}"  # Set your Jetson password here

echo "Fetching ROBOT_NAME from Jetson at $JETSON_IP..."

# Build SSH command based on whether password is provided
if [ -n "$JETSON_PASSWORD" ]; then
  # Use sshpass if password is provided
  if command -v sshpass &> /dev/null; then
    SSH_CMD="sshpass -p '$JETSON_PASSWORD' ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no"
  else
    echo "⚠ Warning: JETSON_PASSWORD is set but sshpass is not installed"
    echo "  Install with: sudo apt-get install sshpass"
    echo "  Or set up SSH keys instead for better security"
    SSH_CMD="ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no"
  fi
else
  # Try key-based auth or interactive password prompt
  SSH_CMD="ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no"
fi

# Try to get ROBOT_NAME via SSH (from systemd user environment)
if FETCHED_ROBOT_NAME=$(eval "$SSH_CMD $JETSON_USER@$JETSON_IP 'systemctl --user show-environment 2>/dev/null | grep \"^ROBOT_NAME=\" | cut -d= -f2-'" 2>/dev/null); then
  if [ -n "$FETCHED_ROBOT_NAME" ]; then
    export ROBOT_NAME="$FETCHED_ROBOT_NAME"
    echo "✓ ROBOT_NAME fetched from Jetson: $ROBOT_NAME"
  else
    echo "⚠ Warning: ROBOT_NAME not set on Jetson, using default from docker-compose.yml"
  fi
else
  echo "⚠ Warning: Could not connect to Jetson to fetch ROBOT_NAME"
  echo "  Using default from docker-compose.yml"
fi

# Launch dexsensor on Jetson
echo "Launching dexsensor on Jetson..."
# Use timeout to prevent hanging, and run with full environment via bash -l (login shell)
if timeout 3 bash -c "eval \"$SSH_CMD $JETSON_USER@$JETSON_IP 'bash -l -c \\\"cd ~ && (nohup dexsensor launch --config data_collect.toml > /tmp/dexsensor.log 2>&1 &) && sleep 0.1 && exit 0\\\"'\"" 2>/dev/null; then
  echo "✓ dexsensor launch command sent to Jetson"
  sleep 1  # Give it a moment to start
  # Verify it's running
  if eval "$SSH_CMD $JETSON_USER@$JETSON_IP 'pgrep -f \"dexsensor launch\"'" >/dev/null 2>&1; then
    echo "✓ dexsensor is running on Jetson (logs: /tmp/dexsensor.log)"
  else
    echo "⚠ Warning: dexsensor may not have started (check logs on Jetson)"
  fi
else
  echo "⚠ Warning: Failed to launch dexsensor on Jetson"
fi

# Create logs directory on host (next to docker/)
mkdir -p "$SCRIPT_DIR/logs"

# Create recording directory on host if RECORDER_SAVE_DIR is set
if [ -n "$RECORDER_SAVE_DIR" ]; then
  echo "Creating recording directory: $RECORDER_SAVE_DIR"
  mkdir -p "$RECORDER_SAVE_DIR"
fi

# Set USB device permissions on host (requires sudo)
if [ -e /dev/ttyUSB0 ]; then
  echo "Setting USB device permissions (requires sudo)..."
  sudo chmod 666 /dev/ttyUSB0
  echo "✓ /dev/ttyUSB0 permissions set"
fi

echo "Starting omniteleop container..."
echo "  Compose file: $SCRIPT_DIR/docker-compose.yml"
echo "  Project root: $PROJECT_ROOT"
echo "  Robot config: ${ROBOT_CONFIG:-vega_1u_f5d6}"
echo "  Recordings will be saved to: ${RECORDER_SAVE_DIR:-/exchange/data_sink}"

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

# Run docker compose from project root with relative path to docker-compose.yml
$COMPOSE_CMD -f docker/docker-compose.yml up -d

echo ""
echo "Container started!"
echo ""
echo "Useful commands:"
echo "  View logs:     cd $PROJECT_ROOT && $COMPOSE_CMD -f docker/docker-compose.yml logs -f"
echo "  Stop:          cd $PROJECT_ROOT && $COMPOSE_CMD -f docker/docker-compose.yml down"
echo "  Shell access:  docker exec -it omniteleop bash"
echo "  Process logs:  docker exec -it omniteleop tail -f /app/logs/<process>.log"
