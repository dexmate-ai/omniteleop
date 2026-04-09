#!/bin/bash
# Entrypoint script for running multiple processes in background
# Each process runs in the background with output redirected to log files

set -e

# Create logs directory
LOG_DIR="/app/logs"
mkdir -p "$LOG_DIR"

# Source any environment variables if provided
if [ -f "/app/env_vars.sh" ]; then
    source /app/env_vars.sh
fi

# Export common environment variables
# Most values are hardcoded, only ROBOT_CONFIG and RECORDER_SAVE_DIR should be passed via CLI
export ROBOT_CONFIG="${ROBOT_CONFIG:-vega_1u_f5d6}"  # Default config, override with -e if needed
export ROBOT_NAME="${ROBOT_NAME:-}"  # Auto-fetched from Jetson (do not set manually)
export JETSON_IP="192.168.50.20"  # Hardcoded Jetson IP
export JETSON_USER="dexmate"  # Hardcoded Jetson user
export JETSON_PASSWORD="hello-dex"  # Hardcoded Jetson password
export DEXCONTROL_DISABLE_HEARTBEAT=1  # Always disabled for data collection
export HOST_UID="${HOST_UID:-1000}"  # Default to 1000, override with -e if needed
export HOST_GID="${HOST_GID:-1000}"  # Default to 1000, override with -e if needed
export RECORDER_SAVE_DIR="${RECORDER_SAVE_DIR:-/exchange/data_sink}"  # Default recording path

# Auto-set ZENOH_CONFIG if not provided (uses mounted $HOME/.dexmate)
if [ -z "$ZENOH_CONFIG" ]; then
    export ZENOH_CONFIG="${HOME}/.dexmate/comm/zenoh/world_engine/zenoh_peer_config.json5"
fi

export PYTHONPATH=.

echo "=================================================="
echo "Starting Omniteleop Docker Container"
echo "ROBOT_CONFIG: $ROBOT_CONFIG"
echo "ROBOT_NAME: $ROBOT_NAME (will be auto-fetched if empty)"
echo "RECORDER_SAVE_DIR: $RECORDER_SAVE_DIR"
echo "JETSON_IP: $JETSON_IP"
echo "JETSON_USER: $JETSON_USER"
echo "ZENOH_CONFIG: $ZENOH_CONFIG"
echo "PYTHONPATH: $PYTHONPATH"
echo "=================================================="

# ============================================================
# FETCH ROBOT_NAME FROM JETSON (if not already set)
# ============================================================
if [ -z "$ROBOT_NAME" ]; then
    echo ""
    echo "ROBOT_NAME not set, fetching from Jetson at $JETSON_IP..."

    # Build SSH command
    if [ -n "$JETSON_PASSWORD" ] && command -v sshpass &> /dev/null; then
        SSH_CMD="sshpass -p '$JETSON_PASSWORD' ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no"
    else
        SSH_CMD="ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no"
    fi

    # Fetch ROBOT_NAME from Jetson systemd user environment
    if FETCHED_ROBOT_NAME=$(eval "$SSH_CMD $JETSON_USER@$JETSON_IP 'systemctl --user show-environment 2>/dev/null | grep \"^ROBOT_NAME=\" | cut -d= -f2-'" 2>/dev/null); then
        if [ -n "$FETCHED_ROBOT_NAME" ]; then
            export ROBOT_NAME="$FETCHED_ROBOT_NAME"
            echo "✓ ROBOT_NAME fetched from Jetson: $ROBOT_NAME"
        else
            echo "⚠ Warning: ROBOT_NAME not set on Jetson, using default: dm/abcdefgh-1"
            export ROBOT_NAME="dm/abcdefgh-1"
        fi
    else
        echo "⚠ Warning: Could not connect to Jetson to fetch ROBOT_NAME"
        echo "  Using default: dm/abcdefgh-1"
        export ROBOT_NAME="dm/abcdefgh-1"
    fi
else
    echo ""
    echo "✓ ROBOT_NAME already set: $ROBOT_NAME"
fi

# ============================================================
# LAUNCH DEXSENSOR ON JETSON
# ============================================================
echo ""
echo "Launching dexsensor on Jetson $JETSON_IP..."

# Build SSH command (if not already built above)
if [ -z "$SSH_CMD" ]; then
    if [ -n "$JETSON_PASSWORD" ] && command -v sshpass &> /dev/null; then
        SSH_CMD="sshpass -p '$JETSON_PASSWORD' ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no"
    else
        SSH_CMD="ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no"
    fi
fi

# Launch dexsensor on Jetson using timeout to prevent hanging
if timeout 3 bash -c "eval \"$SSH_CMD $JETSON_USER@$JETSON_IP 'bash -l -c \\\"cd ~ && (nohup dexsensor launch --config data_collect.toml > /tmp/dexsensor.log 2>&1 &) && sleep 0.1 && exit 0\\\"'\"" 2>/dev/null; then
    echo "✓ dexsensor launch command sent to Jetson"
    sleep 1  # Give it a moment to start

    # Verify it's running
    if eval "$SSH_CMD $JETSON_USER@$JETSON_IP 'pgrep -f \"dexsensor launch\"'" >/dev/null 2>&1; then
        echo "✓ dexsensor is running on Jetson (logs: /tmp/dexsensor.log)"
    else
        echo "⚠ Warning: dexsensor may not have started (check logs on Jetson: /tmp/dexsensor.log)"
    fi
else
    echo "⚠ Warning: Failed to launch dexsensor on Jetson"
    echo "  Check Jetson connectivity and dexsensor installation"
fi

echo "=================================================="

# ============================================================
# CLEANUP HANDLER - Graceful Shutdown
# ============================================================
CLEANUP_DONE=0
cleanup() {
    # Guard against double cleanup (SIGINT triggers cleanup, then exit 0 triggers EXIT trap)
    if [ "$CLEANUP_DONE" -eq 1 ]; then
        return
    fi
    CLEANUP_DONE=1

    echo ""
    echo "=================================================="
    echo "Received shutdown signal, cleaning up..."
    echo "=================================================="

    # 1. Stop dexsensor on Jetson (CRITICAL!)
    if [ -n "$JETSON_IP" ] && [ -n "$JETSON_USER" ]; then
        echo "Stopping dexsensor on Jetson $JETSON_IP..."

        # Build SSH command based on whether password is provided
        if [ -n "$JETSON_PASSWORD" ] && command -v sshpass &> /dev/null; then
            SSH_CMD="sshpass -p '$JETSON_PASSWORD' ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no"
        else
            SSH_CMD="ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no"
        fi

        # Stop dexsensor process on Jetson (with timeout to prevent hanging during cleanup)
        if timeout 5 bash -c "eval \"$SSH_CMD $JETSON_USER@$JETSON_IP 'pkill -f \\\"dexsensor launch\\\"'\"" 2>/dev/null; then
            echo "✓ dexsensor stopped on Jetson"
        else
            echo "⚠ dexsensor was not running on Jetson (or failed to stop)"
        fi
    else
        echo "⚠ Skipping dexsensor cleanup (JETSON_IP or JETSON_USER not set)"
    fi

    # 2. Stop all background processes using PID files
    echo "Stopping all background processes..."
    local stopped_count=0
    local pids_to_kill=()

    # First pass: send SIGTERM to all tracked processes
    for pid_file in "$LOG_DIR"/*.pid; do
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            local process_name=$(basename "$pid_file" .pid)

            if kill -0 "$pid" 2>/dev/null; then
                echo "  Sending SIGTERM to $process_name (PID: $pid)..."
                kill -TERM "$pid" 2>/dev/null
                pids_to_kill+=("$pid")
                stopped_count=$((stopped_count + 1))
            fi

            rm -f "$pid_file"
        fi
    done

    # Give processes 2 seconds to exit gracefully
    if [ ${#pids_to_kill[@]} -gt 0 ]; then
        echo "  Waiting 2s for graceful shutdown..."
        sleep 2

        # Second pass: SIGKILL anything still alive
        for pid in "${pids_to_kill[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Force killing PID $pid..."
                kill -KILL "$pid" 2>/dev/null
            fi
        done
    fi

    echo "✓ Stopped $stopped_count background processes"
    echo "Cleanup complete!"
    echo "=================================================="
    exit 0
}

# Register trap handlers for graceful shutdown
trap cleanup SIGTERM SIGINT EXIT

# ============================================================

# ============================================================
# SET USB DEVICE PERMISSIONS (leveraging --privileged mode)
# ============================================================
echo ""
echo "Setting USB device permissions..."

# With --privileged, we can set permissions for all USB devices
USB_DEVICES_FOUND=0

# Check for common USB serial devices (Dynamixel uses USB serial)
for device in /dev/ttyUSB* /dev/ttyACM* /dev/ttyUSB0; do
    if [ -e "$device" ] && [ ! -d "$device" ]; then
        chmod 666 "$device" 2>/dev/null && echo "✓ Set permissions for $device" && USB_DEVICES_FOUND=$((USB_DEVICES_FOUND + 1))
    fi
done

if [ $USB_DEVICES_FOUND -eq 0 ]; then
    echo "⚠ Warning: No USB serial devices found in /dev/ttyUSB* or /dev/ttyACM*"
    echo "  Devices will be checked again when arm_reader starts"
else
    echo "✓ Set permissions for $USB_DEVICES_FOUND USB device(s)"
fi

# Set permissions for input devices (JoyCon)
if [ -e /dev/input ]; then
    chmod -R 666 /dev/input/* 2>/dev/null || true
    echo "✓ Set permissions for /dev/input devices"
fi

echo "=================================================="

# ============================================================
# FIX RECORDINGS DIRECTORY OWNERSHIP
# ============================================================
echo ""
echo "Fixing recordings directory ownership..."

# If RECORDER_SAVE_DIR is set and HOST_UID/HOST_GID are available, fix ownership
if [ -n "$RECORDER_SAVE_DIR" ] && [ -n "$HOST_UID" ] && [ -n "$HOST_GID" ]; then
    # If directory exists on mounted filesystem, fix its ownership
    if [ -d "$RECORDER_SAVE_DIR" ]; then
        echo "  Changing ownership of $RECORDER_SAVE_DIR to $HOST_UID:$HOST_GID"
        chown "$HOST_UID:$HOST_GID" "$RECORDER_SAVE_DIR" 2>/dev/null || true
        echo "✓ Recordings directory ownership fixed"
    else
        echo "  Recordings directory doesn't exist yet (will be created by recorder)"
    fi
else
    echo "⚠ Skipping ownership fix (RECORDER_SAVE_DIR, HOST_UID, or HOST_GID not set)"
fi

echo "=================================================="

# Function to start a process in background
start_process() {
    local name=$1
    local command=$2
    local log_file="$LOG_DIR/${name}.log"
    
    echo "Starting $name..."
    echo "  Command: $command"
    echo "  Log: $log_file"
    
    # Run command in background, redirect stdout and stderr to log file
    # Use 'exec' so the python process replaces bash (single PID, no orphaned children)
    bash -c "exec $command" > "$log_file" 2>&1 &
    local pid=$!
    echo "  PID: $pid"
    echo "$pid" > "$LOG_DIR/${name}.pid"
}

# ============================================================
# START YOUR PROCESSES HERE
# Uncomment and modify these lines based on your requirements
# ============================================================

# ============================================================
# RUN CLEAR_ERROR.PY FIRST (synchronously)
# ============================================================
echo ""
echo "Running clear_error.py to clear any previous errors..."

if python3 clear_error.py; then
    echo "✓ clear_error.py completed successfully"
else
    echo "⚠ Warning: clear_error.py failed with exit code $?"
    echo "  Continuing with process startup anyway..."
fi

echo "=================================================="
echo ""

# ============================================================
# SPAWN BACKGROUND PROCESSES
# ============================================================

# Example process 1: JoyCon reader
start_process "joycon_reader" "python3 src/omniteleop/leader/joycon_reader.py --debug"

# Example process 2: Command processor
start_process "command_processor" "python3 src/omniteleop/follower/command_processor.py --debug"

# Example process 3: Robot controller
start_process "robot_controller" "python3 src/omniteleop/follower/robot_controller.py --debug"

# Example process 4: MDP Recorder
start_process "mdp_recorder" "python3 internal/we_record/recorder_backend.py --record-mode webapp"

# Example process 5: Arm reader
start_process "arm_reader" "python3 src/omniteleop/leader/arm_reader.py --debug"

# ============================================================

echo ""
echo "All processes started. Logs available in $LOG_DIR"
echo "To view logs: tail -f $LOG_DIR/<process_name>.log"
echo ""

# Keep the container running by waiting for all background processes
# This allows trap handlers (cleanup function) to work properly when SIGTERM/SIGINT is received
echo "Container is running. Press Ctrl+C to stop or send SIGTERM to trigger cleanup."
wait