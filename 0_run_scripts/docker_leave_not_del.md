To leave or detach from a Docker container without stopping or deleting it, you can use the following key combination:

### Detach from the Container
While inside the Docker container, press:

```bash
Ctrl + P, then Ctrl + Q
```

### Explanation:
- **`Ctrl + P` followed by `Ctrl + Q`**: This key sequence detaches your terminal from the Docker container, leaving the container running in the background. The container will continue to run, and you can reattach to it later.

### Reattaching to the Container
If you want to reattach to the container later, you can use:

```bash
docker attach <container_name_or_id>
```

or

```bash
docker start -ai <container_name_or_id>
```

This allows you to resume work inside the container without losing the state of the container's runtime.