# Base image: Ubuntu 24.04
FROM ubuntu:24.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set timezone (adjust as needed)
ENV TZ=UTC

# Build argument for home directory (passed from docker-compose or build command)
ARG HOME_DIR=/root
ENV HOME=${HOME_DIR}

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     python3 \
#     python3-pip \
#     python3-venv \
#     git \
#     curl \
#     wget \
#     sudo \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    sudo \
    iputils-ping \
    bluez \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    cmake \
    sshpass \
    tini \
    && rm -rf /var/lib/apt/lists/*

    # Create home directory to match host paths (for .dexmate config)
RUN mkdir -p ${HOME}

# Set working directory
WORKDIR /app/omniteleop

# Copy the omniteleop project into the container
COPY . /app/omniteleop/

# Install Python packages
# Note: Ubuntu 24.04 requires --break-system-packages for pip install outside venv
RUN pip3 install --break-system-packages \
    dextop \
    dexmotion \
    dexmate-urdf \
    dexcomm \
    dexcontrol==0.4.1

# Install dynamixelAPI (editable install)
RUN pip3 install --break-system-packages -e /app/omniteleop/dynamixelAPI/

# Install omniteleop itself (editable install)
RUN pip3 install --break-system-packages -e /app/omniteleop/

# Install MCAP packages
RUN pip3 install --break-system-packages \
    mcap \
    mcap-protobuf-support
RUN pip3 install --break-system-packages \
    fastapi \ 
    uvicorn
# Install OMPL
RUN pip3 install --break-system-packages ompl

# Use tini as PID 1 init process - properly forwards signals (SIGINT/SIGTERM) to entrypoint.sh
ENTRYPOINT ["/usr/bin/tini", "--", "/app/omniteleop/docker/entrypoint.sh"]

# Keep container running (wait will be called in entrypoint.sh)
CMD []
