# Jetson Camera System

This is a cleaned-up and reorganized version of the Jetson Camera System with a proper directory structure.

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
    ├── SYSTEM_SETUP.md        # System setup documentation
    └── FRONTEND_INTEGRATION.md# Frontend integration guide
```

## Setup Instructions

To complete the setup of the cleaned-up system:

1. **Install the services**:
   ```bash
   sudo /home/azuolas/jetson_system/install_services.sh
   ```

2. **Start the services**:
   ```bash
   sudo /home/azuolas/jetson_system/start_services.sh
   ```

3. **Test the system**:
   Open a browser and navigate to `https://jetson.mmoment.xyz/api-test`
   
4. **Clean up duplicates**:
   After verifying that everything works correctly:
   ```bash
   /home/azuolas/jetson_system/cleanup_duplicates.sh
   ```

## Available Scripts

- `install_services.sh`: Installs systemd service files
- `start_services.sh`: Starts all services
- `cleanup_duplicates.sh`: Removes duplicate data directories

## Documentation

For more details on the system:

- [System Setup](docs/SYSTEM_SETUP.md)
- [Frontend Integration](docs/FRONTEND_INTEGRATION.md)

## Example URLs

- API Test Page: `https://jetson.mmoment.xyz/api-test`
- Health Check: `https://jetson.mmoment.xyz/health`
- Stream Test: `https://jetson.mmoment.xyz/test-stream`

## Important Note

The original system files in `/home/azuolas/jetson_camera_service` are preserved until you manually delete them after confirming the new system works correctly. 