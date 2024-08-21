#!/bin/bash

# This script runs the veScale Docker container with the following configuration:
# 1. Maps the local folder /data/xiaofengwu/vescale_prj to /root/vescale_prj inside the container.
# 2. Ensures the container persists after you exit, allowing you to reattach later.
# 3. Sets NCCL environment variables to mitigate potential distributed training issues.

# Set the container name
CONTAINER_NAME="vescale_container"

# Set the local path and the Docker runtime path
LOCAL_PATH="/data/xiaofengwu/vescale_prj"
DOCKER_RUNTIME_PATH="/root/vescale_prj"

# Set NCCL environment variables
NCCL_DEBUG="INFO"
NCCL_SHM_DISABLE="1"
NCCL_P2P_DISABLE="1"
NCCL_IB_DISABLE="1"

# Check if the container already exists
if [ "$(docker ps -a -q -f name=${CONTAINER_NAME})" ]; then
    echo "Container with name ${CONTAINER_NAME} already exists."
    echo "Starting and attaching to the existing container..."

    # Start the existing container and attach to it, passing NCCL environment variables
    docker start ${CONTAINER_NAME}
    docker exec -it ${CONTAINER_NAME} \
        bash -c "export NCCL_DEBUG=${NCCL_DEBUG} NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE} NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} NCCL_IB_DISABLE=${NCCL_IB_DISABLE} && bash"
else
    echo "Running a new container named ${CONTAINER_NAME}..."

    # Run a new Docker container with the specified volume mapping and name, and set shm-size to 16GB

    docker run -it --gpus all --shm-size=16g --ulimit memlock=-1:-1 \
        -v ${LOCAL_PATH}:${DOCKER_RUNTIME_PATH} \
        -u $(id -u):$(id -g) \
        --name ${CONTAINER_NAME} \
        --env NCCL_DEBUG=${NCCL_DEBUG} \
        --env NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE} \
        --env NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} \
        --env NCCL_IB_DISABLE=${NCCL_IB_DISABLE} \
        shizukanaskytree/vescale:latest
fi