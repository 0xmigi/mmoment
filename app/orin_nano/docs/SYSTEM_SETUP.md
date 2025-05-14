# Jetson Camera System Setup

This document provides a comprehensive overview of the Jetson Camera system setup and architecture.

## System Architecture

The system consists of four main components:

1. **Camera API Service** (port 5002): Handles camera operation, gesture detection, face recognition
2. **Solana Middleware** (port 5001): Handles wallet connections and NFT minting
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
| (Port 5001)         |     | (Port 5002)  |
+---------------------+     +--------------+
```

## Directory Structure

The Jetson Camera system is organized in the following directory structure:

```
/home/azuolas/jetson_system/
├── camera_service/     # Camera API service files
├── frontend_bridge/    # Frontend bridge files
│   └── api_test.html   # Test page for API functionality
├── solana_middleware/  # Solana middleware files
├── systemd/            # Systemd service files
├── data/               # Data files
│   ├── faces/          # Facial recognition data
│   ├── face_embeddings/# Face embeddings for recognition
│   ├── videos/         # Recorded videos
│   ├── images/         # Captured images
│   └── recordings/     # Additional video recordings
└── docs/               # Documentation
    ├── SYSTEM_SETUP.md        # This file
    └── FRONTEND_INTEGRATION.md# Frontend integration guide
```

## Service Files

All services are configured using systemd service files located in `/home/azuolas/jetson_system/systemd/`:

### 1. Camera API Service

**File**: `/home/azuolas/jetson_system/systemd/camera-service.service`

```ini
[Unit]
Description=Jetson Camera API Service
After=network.target

[Service]
Type=simple
User=azuolas
WorkingDirectory=/home/azuolas/jetson_system/camera_service
ExecStartPre=/bin/bash -c "lsof -ti:5002 | xargs -r kill -9"
ExecStart=/usr/bin/python3 /home/azuolas/jetson_system/camera_service/api.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### 2. Solana Middleware Service

**File**: `/home/azuolas/jetson_system/systemd/solana-middleware.service`

```ini
[Unit]
Description=Solana Middleware Service
After=network.target

[Service]
Type=simple
User=azuolas
WorkingDirectory=/home/azuolas/jetson_system/solana_middleware
ExecStart=/usr/bin/python3 /home/azuolas/jetson_system/solana_middleware/solana_middleware.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### 3. Frontend Bridge Service

**File**: `/home/azuolas/jetson_system/systemd/frontend-bridge.service`

```ini
[Unit]
Description=Jetson Camera Frontend Bridge
After=network.target
Wants=camera-service.service

[Service]
Type=simple
User=azuolas
WorkingDirectory=/home/azuolas/jetson_system/frontend_bridge
ExecStart=/usr/bin/python3 /home/azuolas/jetson_system/frontend_bridge/frontend_bridge.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### 4. Cloudflare Tunnel Service

**File**: `/home/azuolas/jetson_system/systemd/cloudflared-compat.service`

```ini
[Unit]
Description=Cloudflare Tunnel (compatible version)
After=network.target
Wants=frontend-bridge.service

[Service]
Type=simple
User=azuolas
WorkingDirectory=/home/azuolas
ExecStart=/usr/local/bin/cloudflared-compat tunnel run 6257e873-7943-4b85-b8a3-72b5b9d0a500
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

## Configuration Files

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
    service: http://127.0.0.1:5003
    originRequest:
      noTLSVerify: true
      disableChunkedEncoding: true
  
  # Catch-all for everything else
  - service: http_status:404
```

## Starting and Managing Services

### Installing Service Files

To install the service files into systemd, run:

```bash
sudo cp /home/azuolas/jetson_system/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
```

### Starting All Services

Use the provided script:

```bash
sudo /home/azuolas/jetson_system/start_services.sh
```

This script checks if each service is running and starts it if necessary.

### Restarting Services Individually

```bash
# Restart Camera API
sudo systemctl restart camera-service.service

# Restart Solana Middleware
sudo systemctl restart solana-middleware.service

# Restart Frontend Bridge
sudo systemctl restart frontend-bridge.service

# Restart Cloudflare Tunnel
sudo systemctl restart cloudflared-compat.service
```

### Viewing Service Logs

```bash
# Camera API logs
sudo journalctl -u camera-service.service -f

# Solana Middleware logs
sudo journalctl -u solana-middleware.service -f

# Frontend Bridge logs
sudo journalctl -u frontend-bridge.service -f
# or
tail -f /home/azuolas/jetson_system/frontend_bridge/frontend_bridge.log

# Cloudflare Tunnel logs
sudo journalctl -u cloudflared-compat.service -f
```

## Troubleshooting

### Common Issues

1. **Service Won't Start**
   - Check logs: `journalctl -u service-name.service -n 50`
   - Verify permissions: `ls -la /home/azuolas/jetson_system/`

2. **Tunnel Not Working**
   - Check tunnel status: `systemctl status cloudflared-compat.service`
   - Verify tunnel can connect to Cloudflare: `cloudflared-compat tunnel info 6257e873-7943-4b85-b8a3-72b5b9d0a500`
   - Try accessing locally first: `curl http://localhost:5003/health`

3. **Stream Not Working**
   - Check if camera is detected: `v4l2-ctl --list-devices`
   - Verify camera API is running: `curl http://localhost:5002/health`
   - Try direct stream access: `curl -I http://localhost:5002/stream`

### Recovery Procedures

If the system becomes unresponsive or services fail repeatedly:

1. Stop all services:
   ```bash
   sudo systemctl stop cloudflared-compat.service frontend-bridge.service solana-middleware.service camera-service.service
   ```

2. Check for any orphan processes:
   ```bash
   ps aux | grep -E "cloudflared|frontend_bridge|solana_middleware|api.py" | grep -v grep
   ```

3. Start services in order:
   ```bash
   sudo systemctl start camera-service.service
   sudo systemctl start solana-middleware.service
   sudo systemctl start frontend-bridge.service
   sudo systemctl start cloudflared-compat.service
   ```

## Maintenance

### Updating Configuration

To update any service configuration:

1. Edit the relevant service file in `/home/azuolas/jetson_system/systemd/`
2. Copy to systemd directory: `sudo cp /home/azuolas/jetson_system/systemd/service-name.service /etc/systemd/system/`
3. Reload systemd: `sudo systemctl daemon-reload`
4. Restart the service: `sudo systemctl restart service-name.service`

### Backup

Regular backups of the following directories are recommended:

- `/home/azuolas/jetson_system/` - All application code
- `/home/azuolas/.cloudflared/` - Tunnel configuration and credentials
- `/home/azuolas/jetson_system/data/` - All data files including faces, recordings, etc.

### Security

- The Cloudflare tunnel provides secure encrypted access without opening ports
- All services run as the non-root azuolas user
- Access to the system should be restricted to authorized personnel

## Endpoints Documentation

For a detailed list of available API endpoints and their usage, see:
[Frontend Integration Guide](/home/azuolas/jetson_system/docs/FRONTEND_INTEGRATION.md) 