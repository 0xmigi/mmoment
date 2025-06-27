# Jetson Camera System Setup

This document provides a comprehensive overview of the Jetson Camera system setup and architecture.

## System Architecture

The system consists of four main components:

1. **Camera API Service** (port 5002): Handles camera operation, gesture detection, face recognition
2. **Solana Middleware** (port 5004): Handles wallet connections and NFT minting
3. **Frontend Bridge** (port 5003): Acts as a proxy between frontend clients and backend services
4. **Cloudflare Tunnel**: Securely exposes the Frontend Bridge to the internet

All services are configured as systemd services for automatic startup and management.

```
+---------------------+     +--------------------+
| Frontend (Web App)  | --> | Cloudflare Tunnel  |
+---------------------+     +--------------------+
                                      |
                                      v
                            +--------------------+
                            | Frontend Bridge    |
                            | (Port 5003)        |
                            +--------------------+
                                 /         \
                                /           \
                               v             v
+---------------------+     +--------------+
| Solana Middleware   | <-- | Camera API   |
| (Port 5004)         |     | (Port 5002)  |
+---------------------+     +--------------+
```

## Directory Structure

The Jetson Camera system is organized in the following directory structure:

```
/home/azuolas/mmoment/app/orin_nano/
├── camera_service_new/     # Buffer-based camera service
│   ├── config/             # Camera configuration
│   ├── faces/              # Facial recognition data
│   ├── services/           # Service modules
│   ├── logs/               # Log files
│   ├── photos/             # Captured photos
│   ├── videos/             # Recorded videos
│   ├── models/             # ML models
│   └── main.py             # Entry point
├── frontend_bridge/        # Frontend bridge files
│   ├── frontend_bridge.py  # Main bridge service
│   └── templates/          # HTML templates
├── solana_middleware/      # Solana middleware files
│   └── solana_middleware.py # Solana service
├── systemd/                # Systemd service files
├── docs/                   # Documentation
│   ├── API_ENDPOINTS.md    # API reference
│   ├── SYSTEM_SETUP.md     # This file 
│   ├── FRONTEND_INTEGRATION.md # Frontend integration guide
│   └── HARDWARE.md         # Hardware setup guide
└── data/                   # Shared data storage
```

## Service Files

All services are configured using systemd service files located in `/home/azuolas/mmoment/app/orin_nano/systemd/`:

### 1. Camera API Service

**File**: `/home/azuolas/mmoment/app/orin_nano/systemd/camera-service.service`

```ini
[Unit]
Description=Lightweight Buffer-Based Jetson Camera API Service
After=network.target
Wants=solana-middleware.service

[Service]
Type=simple
User=azuolas
WorkingDirectory=/home/azuolas/mmoment/app/orin_nano/camera_service_new
Environment="SOLANA_MIDDLEWARE_URL=http://localhost:5004"
Environment="CAMERA_PDA=WT9oJrL7sbNip8Rc2w5LoWFpwsUcZZJnnjE2zZjMuvD"
Environment="PROGRAM_ID=Hx5JaUCZXQqvcYzTcdgm9ZE3sqhMWqwAhNXZBrzWm45S"
Environment="CAMERA_DEVICE=/dev/video1"
# Run port cleanup script before starting
ExecStartPre=/bin/bash -c "lsof -ti:5002 | xargs -r kill -9"
ExecStart=/usr/bin/python3 /home/azuolas/mmoment/app/orin_nano/camera_service_new/main.py --port 5002
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### 2. Solana Middleware Service

**File**: `/home/azuolas/mmoment/app/orin_nano/systemd/solana-middleware.service`

```ini
[Unit]
Description=Solana Middleware Service for NFT Minting
After=network.target

[Service]
Type=simple
User=azuolas
WorkingDirectory=/home/azuolas/mmoment/app/orin_nano/solana_middleware
ExecStartPre=/bin/bash -c "lsof -ti:5004 | xargs -r kill -9"
ExecStart=/usr/bin/python3 /home/azuolas/mmoment/app/orin_nano/solana_middleware/solana_middleware.py --port 5004
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### 3. Frontend Bridge Service

**File**: `/home/azuolas/mmoment/app/orin_nano/systemd/frontend-bridge.service`

```ini
[Unit]
Description=Jetson Camera Frontend Bridge
After=network.target
Wants=camera-service.service

[Service]
Type=simple
User=azuolas
WorkingDirectory=/home/azuolas/mmoment/app/orin_nano/frontend_bridge
ExecStart=/usr/bin/python3 /home/azuolas/mmoment/app/orin_nano/frontend_bridge/frontend_bridge.py --camera-port 5002 --frontend-port 5003
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### 4. Cloudflare Tunnel Service

**File**: `/home/azuolas/mmoment/app/orin_nano/systemd/cloudflared-compat.service`

```ini
[Unit]
Description=Cloudflare Tunnel (compatible version)
After=network.target
Wants=frontend-bridge.service

[Service]
Type=simple
User=azuolas
WorkingDirectory=/home/azuolas
ExecStart=/usr/local/bin/cloudflared tunnel run 6257e873-7943-4b85-b8a3-72b5b9d0a500
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

## Configuration Files

### Camera Configuration

**File**: `/home/azuolas/mmoment/app/orin_nano/camera_service_new/config/camera_config.json`

```json
{
    "camera": {
        "preferred_device": "/dev/video1",
        "width": 1280,
        "height": 720,
        "fps": 30,
        "buffersize": 3,
        "reconnect_attempts": 5,
        "reconnect_delay": 0.5
    },
    "detection": {
        "detection_interval": 0.03,
        "recognition_interval": 0.5,
        "min_face_size": [
            60,
            60
        ],
        "scale_factor": 1.1,
        "min_neighbors": 5
    }
}
```

### Cloudflare Tunnel Configuration

**File**: `/home/azuolas/.cloudflared/config.yml`

```yaml
# Cloudflare Tunnel configuration for Jetson Camera
tunnel: 6257e873-7943-4b85-b8a3-72b5b9d0a500
credentials-file: /home/azuolas/.cloudflared/6257e873-7943-4b85-b8a3-72b5b9d0a500.json

# Basic ingress rules - no fancy options, just simple routing
ingress:
  # Route all jetson.mmoment.xyz requests to the frontend bridge
  - hostname: jetson.mmoment.xyz
    service: http://127.0.0.1:5013
    originRequest:
      noTLSVerify: true
      disableChunkedEncoding: true
  
  # Catch-all for everything else
  - service: http_status:404
```

## Installing and Starting Services

### Installing Service Files

To install the service files into systemd, run:

```bash
sudo cp /home/azuolas/mmoment/app/orin_nano/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
```

### Starting All Services

Use the provided script:

```bash
cd /home/azuolas/mmoment/app/orin_nano
sudo ./start_services.sh
```

Or manually start each service:

```bash
sudo systemctl start camera-service.service
sudo systemctl start solana-middleware.service
sudo systemctl start frontend-bridge.service
sudo systemctl start cloudflared-compat.service
```

### Checking Service Status

To check if services are running correctly:

```bash
# Check status of all services
systemctl status camera-service.service solana-middleware.service frontend-bridge.service cloudflared-compat.service

# Check specific service
systemctl status camera-service.service
```

### Viewing Service Logs

To view service logs:

```bash
# Camera service logs
journalctl -u camera-service.service -f

# Cloudflare tunnel logs
journalctl -u cloudflared-compat.service -f
```

## Common Issues and Troubleshooting

### Camera Not Detected

If the camera is not detected:

1. Check the physical connection of the camera
2. Verify the camera device path:
   ```bash
   v4l2-ctl --list-devices
   ```

3. Update the camera device path in the `camera-service.service` file:
   ```bash
   sudo nano /etc/systemd/system/camera-service.service
   ```
   Change the `CAMERA_DEVICE` environment variable to the correct path

4. Reload and restart the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart camera-service.service
   ```

### Port Conflicts

If services fail to start due to port conflicts:

1. Check which process is using the port:
   ```bash
   sudo lsof -i :5002  # Replace with the port number
   ```

2. Kill the process:
   ```bash
   sudo kill -9 <process_id>
   ```

3. Restart the service:
   ```bash
   sudo systemctl restart camera-service.service
   ```

### Cloudflare Tunnel Issues

If the Cloudflare tunnel is not working:

1. Check the tunnel status:
   ```bash
   cloudflared tunnel info 6257e873-7943-4b85-b8a3-72b5b9d0a500
   ```

2. Verify the tunnel configuration:
   ```bash
   cat ~/.cloudflared/config.yml
   ```

3. Restart the tunnel:
   ```bash
   sudo systemctl restart cloudflared-compat.service
   ```

## Performance Optimization

For best performance:

1. Set Jetson power mode to maximum:
   ```bash
   sudo nvpmodel -m 0
   sudo jetson_clocks
   ```

2. Monitor system performance:
   ```bash
   tegrastats
   ```

3. Check camera frame rate in logs:
   ```bash
   tail -f /home/azuolas/mmoment/app/orin_nano/camera_service_new/logs/camera_service.log | grep "fps"
   ```

## Endpoints Documentation

For a detailed list of available API endpoints and their usage, see:
[API Endpoints Documentation](API_ENDPOINTS.md) and 
[Frontend Integration Guide](FRONTEND_INTEGRATION.md) 