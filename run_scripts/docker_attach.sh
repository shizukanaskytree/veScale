#!/bin/bash

# This script attaches to an existing Docker container that has exited but not been deleted.
# 1. Checks if the container with the specified name exists and has exited.
# 2. If it exists, it will start and attach to the container.

# Set the container name
CONTAINER_NAME="vescale_container"

# Check if the container exists and has exited
if [ "$(docker ps -a -q -f name=${CONTAINER_NAME} -f status=exited)" ]; then
    echo "Container ${CONTAINER_NAME} has exited. Starting and attaching to it..."

    # Start the exited container and attach to it
    docker start -ai ${CONTAINER_NAME}
else
    echo "No exited container with the name ${CONTAINER_NAME} found."
    echo "Ensure the container exists and has exited, then try again."
fi
