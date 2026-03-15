#!/bin/bash
# Install native-camera-server as a systemd service
# Run this script with: sudo ./install-service.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_FILE="$SCRIPT_DIR/native-camera-server.service"

echo "Installing native-camera-server systemd service..."

# Stop any running instance
pkill -f native_camera_server 2>/dev/null || true
sleep 1

# Copy service file
cp "$SERVICE_FILE" /etc/systemd/system/

# Reload systemd
systemctl daemon-reload

# Enable service to start on boot
systemctl enable native-camera-server

# Start the service now
systemctl start native-camera-server

# Check status
sleep 2
systemctl status native-camera-server --no-pager

echo ""
echo "=========================================="
echo "Native camera server installed!"
echo ""
echo "Commands:"
echo "  sudo systemctl status native-camera-server   # Check status"
echo "  sudo systemctl restart native-camera-server  # Restart"
echo "  sudo systemctl stop native-camera-server     # Stop"
echo "  sudo journalctl -u native-camera-server -f   # View logs"
echo "=========================================="
