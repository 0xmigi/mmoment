#!/bin/bash
# Service installation script for Jetson Camera System

echo "Installing Jetson Camera System services..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (with sudo)"
  exit 1
fi

# Copy service files
echo "Copying service files to /etc/systemd/system/"
cp /home/azuolas/jetson_system/systemd/*.service /etc/systemd/system/

# Reload systemd
echo "Reloading systemd daemon"
systemctl daemon-reload

# Enable services to start on boot
echo "Enabling services to start on boot"
systemctl enable camera-service.service
systemctl enable solana-middleware.service
systemctl enable frontend-bridge.service
systemctl enable cloudflared-compat.service

echo "Installation complete. You can now start services with:"
echo "sudo /home/azuolas/jetson_system/start_services.sh" 