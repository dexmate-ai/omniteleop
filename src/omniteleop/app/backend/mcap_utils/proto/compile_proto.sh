#!/bin/bash
# Compile protobuf schemas to generate Python modules
# Requires: protoc (protobuf compiler)
# Install with: sudo apt install protobuf-compiler

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "Compiling protobuf schemas in: $SCRIPT_DIR"

protoc --python_out=. robot.proto camera.proto policy.proto

if [ $? -eq 0 ]; then
    echo "Successfully compiled protobuf schemas:"
    echo "  - robot_pb2.py"
    echo "  - camera_pb2.py"
    echo "  - policy_pb2.py"
else
    echo "Error: Failed to compile protobuf schemas"
    echo "Make sure protoc is installed: sudo apt install protobuf-compiler"
    exit 1
fi

