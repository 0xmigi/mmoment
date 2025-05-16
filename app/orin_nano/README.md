# Jetson Camera System

This repository contains the Jetson Camera System which provides facial recognition, gesture detection, and camera control capabilities for the mmoment project.

## Overview

The Jetson Camera System consists of the following components:

1. **Camera Service** - Provides low-level camera controls and vision processing
2. **Frontend Bridge** - Provides an API for frontend applications to interact with the camera
3. **Solana Middleware** - Handles blockchain integration for NFT moments

## Quick Start

To get the system up and running:

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/mmoment.git
   cd mmoment/app/orin_nano
   ```

2. Run the installation script:
   ```bash
   ./install_services.sh
   ```

3. Start all services:
   ```bash
   ./start_services.sh
   ```

4. Access the camera system through:
   - Web Interface: http://localhost:5003
   - Camera Stream: http://localhost:5003/stream
   - API Test Page: http://localhost:5003/api-test

## File Structure

```
mmoment/app/orin_nano/
├── camera_service_new/     # Buffer-based camera service
├── frontend_bridge/        # Frontend API bridge
├── solana_middleware/      # Blockchain integration
├── systemd/                # Systemd service definitions
├── docs/                   # Documentation
│   ├── API_ENDPOINTS.md    # API reference
│   ├── SYSTEM_SETUP.md     # System setup guide 
│   ├── FRONTEND_INTEGRATION.md # Frontend integration guide
│   └── HARDWARE.md         # Hardware setup guide
├── data/                   # Data storage
└── [other configuration files]
```

## Documentation

All documentation for the system can be found in the `docs/` directory:

- [API Endpoints](docs/API_ENDPOINTS.md) - Complete reference of all available API endpoints
- [System Setup](docs/SYSTEM_SETUP.md) - How to set up the system from scratch
- [Frontend Integration](docs/FRONTEND_INTEGRATION.md) - Guide for frontend developers
- [Hardware Setup](docs/HARDWARE.md) - Information about hardware requirements and setup

## Camera Service

The optimized Camera Service provides:

- High-performance frame buffer (~30fps)
- Face recognition and gesture detection
- Media capture (photos and videos)
- Status and health monitoring

For more details, see the [Camera Service documentation](camera_service_new/README.md).

## Frontend Bridge

The Frontend Bridge provides:

- RESTful API for frontend applications
- MJPEG streaming
- Session management
- CORS handling for web applications

## Solana Middleware

The Solana Middleware provides:

- Wallet connection management
- NFT minting functionality
- Blockchain transaction handling
- PDA verification

## Managing Services

The system uses systemd for service management. To control services:

```bash
# Start all services
sudo systemctl start camera-service.service solana-middleware.service frontend-bridge.service cloudflared-compat.service

# Check status
sudo systemctl status camera-service.service

# Restart a specific service
sudo systemctl restart frontend-bridge.service

# Stop all services
sudo systemctl stop camera-service.service solana-middleware.service frontend-bridge.service cloudflared-compat.service
```

## Troubleshooting

See the [System Setup](docs/SYSTEM_SETUP.md) document for troubleshooting steps and common issues.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is proprietary software owned by mmoment. 