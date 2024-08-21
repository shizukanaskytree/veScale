#!/bin/bash

# This script runs the veScale Docker container with the following configuration:
# 1. Maps the local folder /data/xiaofengwu/vescale_prj to /root/vescale_prj inside the container.
# 2. Deletes the Docker runtime (container) after exiting.

# Set the container name
CONTAINER_NAME="vescale_container"

# Set the local path and the Docker runtime path
LOCAL_PATH="/data/xiaofengwu/vescale_prj"
DOCKER_RUNTIME_PATH="/root/vescale_prj"

# Run the Docker container with the specified volume mapping and name
docker run -it --rm \
    -v ${LOCAL_PATH}:${DOCKER_RUNTIME_PATH} \
    --name ${CONTAINER_NAME} \
    shizukanaskytree/vescale:latest

# After exiting the container, it will be automatically removed due to the --rm flag
echo "Container ${CONTAINER_NAME} has been removed."