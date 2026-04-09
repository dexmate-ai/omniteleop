# Omniteleop Docker

Build and run the omniteleop stack in a container. **Directory-independent**: you can run the scripts from anywhere.

## Build

From any directory:

```bash
# Using the full path to the script
/path/to/omniteleop/docker/build.sh

# Or from repo root
cd /path/to/omniteleop && docker/build.sh

# Or from inside docker/
cd /path/to/omniteleop/docker && ./build.sh
```

The script finds the repo root (parent of `docker/`) and runs `docker build` from there. The Dockerfile copies the repo into the image and runs `pip install -e dynamixelAPI/` and `pip install -e .` inside the container, so no dependency on your current working directory.

## Run

From any directory:

```bash
/path/to/omniteleop/docker/run.sh
```

Or `./run.sh` when already in `docker/`.

## Stop

From any directory:

```bash
# Using the full path to the script
/path/to/omniteleop/docker/stop.sh

# Or from repo root
cd /path/to/omniteleop && docker/stop.sh

# Or from inside docker/
cd /path/to/omniteleop/docker && ./stop.sh
```

This stops and removes the container cleanly using `docker compose down`.

## Restart

From any directory:

```bash
# Using the full path to the script
/path/to/omniteleop/docker/restart.sh

# Or from repo root
cd /path/to/omniteleop && docker/restart.sh

# Or from inside docker/
cd /path/to/omniteleop/docker && ./restart.sh
```

This stops the container, then starts it again. Useful when you've modified docker-compose.yml or want to reload the container configuration.

## View Logs

View logs from all processes:

```bash
docker compose -f /path/to/omniteleop/docker/docker-compose.yml logs -f
```

View logs from a specific process inside the container:

```bash
docker exec -it omniteleop tail -f /app/logs/joycon_reader.log
docker exec -it omniteleop tail -f /app/logs/command_processor.log
docker exec -it omniteleop tail -f /app/logs/robot_controller.log
docker exec -it omniteleop tail -f /app/logs/mdp_recorder.log
```

## Quick Reference

All scripts work from any directory:

- `docker/build.sh` - Build the Docker image
- `docker/run.sh` - Start the container
- `docker/stop.sh` - Stop and remove the container
- `docker/restart.sh` - Restart the container (stop + start)
- `docker exec -it omniteleop bash` - Get a shell inside the container
