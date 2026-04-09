#!/bin/bash
# Build the omniteleop Docker image.
#
# Directory-independent: run from anywhere, e.g.:
#   /path/to/omniteleop/docker/build.sh
#   ./build.sh   (when already in docker/)
#   docker/build.sh   (when in repo root)
#
# The script finds the repo root as the parent of the directory containing
# this script, then builds from there so COPY . and pip install -e . work correctly.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Building omniteleop Docker image..."
echo "  Script dir:  $SCRIPT_DIR"
echo "  Project root: $PROJECT_ROOT"
echo "  Home dir: $HOME"

cd "$PROJECT_ROOT"

docker build \
    --no-cache \
    --build-arg HOME_DIR="$HOME" \
    -f docker/Dockerfile \
    -t omniteleop:latest \
    .

echo ""
echo "Build complete! Image: omniteleop:latest"
echo ""
echo "Run the container from anywhere with:"
echo "  $SCRIPT_DIR/run.sh"
echo ""
echo "Or: docker run -d --name omniteleop --network host omniteleop:latest"
