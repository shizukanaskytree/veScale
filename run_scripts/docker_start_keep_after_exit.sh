#!/bin/bash

# This script runs the veScale Docker container with the following configuration:
# 1. Maps the local folder /data/xiaofengwu/vescale_prj to /root/vescale_prj inside the container.
# 2. Ensures the container persists after you exit, allowing you to reattach later.

# Set the container name
CONTAINER_NAME="vescale_container"

# Set the local path and the Docker runtime path
LOCAL_PATH="/data/xiaofengwu/vescale_prj"
DOCKER_RUNTIME_PATH="/root/vescale_prj"

# Check if the container already exists
if [ "$(docker ps -a -q -f name=${CONTAINER_NAME})" ]; then
    echo "Container with name ${CONTAINER_NAME} already exists."
    echo "Starting and attaching to the existing container..."

    # Start the existing container and attach to it
    docker start -ai ${CONTAINER_NAME}
else
    echo "Running a new container named ${CONTAINER_NAME}..."

    # Run a new Docker container with the specified volume mapping and name
    docker run -it \
        -v ${LOCAL_PATH}:${DOCKER_RUNTIME_PATH} \
        --name ${CONTAINER_NAME} \
        shizukanaskytree/vescale:latest
fi
