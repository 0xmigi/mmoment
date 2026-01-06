# Native Camera Server → Container Migration Plan

## Goal
Move the native C++ camera server INTO the Docker container to eliminate:
- Socket file mounting race conditions
- Duplicate systemd services
- Complex startup ordering
- The `/tmp/native_inference.sock` garbage

## Current State (BACKUP REFERENCE)

### Files to preserve for rollback:
- `/etc/systemd/system/native-camera-server.service` (system-level)
- `~/.config/systemd/user/native-camera-server.service` (user-level, already disabled)
- Current `docker-compose.yml` socket mount

### Current architecture:
```
HOST: native_camera_server binary
  ↓ Unix socket
DOCKER: camera-service (Python)
  ↓
WebRTC stream
```

## New Architecture

```
DOCKER: camera-service container
  ├── native_camera_server (C++ binary, starts first)
  ├── Python camera service (connects via local socket INSIDE container)
  └── All TensorRT engines
```

Single container. No external dependencies. Start/stop cleanly.

---

## Implementation Steps

### Step 1: Modify Dockerfile
Add native binary, shared library, and TensorRT engines to the camera-service image.

**Files to add:**
- `native/build/native_camera_server` (1.1MB)
- `native/build/libpreprocess_cuda.so` (1.0MB)
- `native/*.engine` (3 files, ~102MB total)

**Dependencies needed:**
- GStreamer (libgstapp-1.0) - may need to install
- libnvbufsurface (tegra libs) - should be in jetpack base image

### Step 2: Create entrypoint script
New entrypoint that:
1. Starts native_camera_server in background
2. Waits for socket to be ready
3. Starts Python main.py

### Step 3: Update docker-compose.yml
- Remove socket mount: `- /tmp/native_inference.sock:/tmp/native_inference.sock`
- Add camera devices back:
  ```yaml
  devices:
    - /dev/video0:/dev/video0
    - /dev/video1:/dev/video1
  ```

### Step 4: Update native_client.py socket path
Change from `/tmp/native_inference.sock` to local path inside container.
(Actually, path can stay the same since it's now inside the container)

### Step 5: Disable host systemd service
```bash
sudo systemctl stop native-camera-server.service
sudo systemctl disable native-camera-server.service
```

### Step 6: Rebuild and test
```bash
docker compose build camera-service
docker compose up -d camera-service
docker compose logs -f camera-service
```

---

## Rollback Procedure

If this doesn't work:

1. Stop container: `docker compose down`
2. Re-enable host service:
   ```bash
   sudo systemctl enable native-camera-server.service
   sudo systemctl start native-camera-server.service
   ```
3. Revert docker-compose.yml to use socket mount
4. Revert Dockerfile changes
5. `docker compose up -d`

---

## Files Modified

1. `services/camera-service/Dockerfile` - add native binary + engines
2. `services/camera-service/entrypoint.sh` - new file, starts both processes
3. `docker-compose.yml` - remove socket mount, add camera devices
4. Systemd service - disable on host

## Success Criteria

- [ ] Single `docker compose up` starts everything
- [ ] WebRTC stream shows real camera feed (not test pattern)
- [ ] No processes running on host (except Docker)
- [ ] `docker compose down` stops everything cleanly
- [ ] Container restart works without manual intervention
