#!/bin/bash
# Setup script for Pi Zero 2W — run once after flashing
set -e

echo "=== MMOMENT Pi Zero 2W Setup ==="

# System dependencies
sudo apt-get update
sudo apt-get install -y \
    python3-picamera2 \
    python3-pip \
    python3-venv \
    ffmpeg \
    cloudflared \
    network-manager \
    libcamera-apps \
    python3-opencv \
    libatlas-base-dev \
    libjpeg-dev

# Python venv (use system picamera2 since it needs system libs)
python3 -m venv --system-site-packages ~/mmoment/venv

# Install Python dependencies
~/mmoment/venv/bin/pip install --upgrade pip
~/mmoment/venv/bin/pip install \
    flask \
    flask-cors \
    aiortc \
    aiohttp \
    av \
    numpy \
    cryptography \
    PyNaCl \
    base58 \
    solders \
    requests \
    pyyaml

# Directories
mkdir -p ~/.mmoment
mkdir -p ~/camera_files/photos
mkdir -p ~/camera_files/videos

# Install systemd service
sudo cp ~/mmoment/app/pi_zero_2w/camera_service.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable camera_service
sudo systemctl start camera_service

echo ""
echo "Setup complete. Camera service started."
echo "Device pubkey will be shown in: sudo journalctl -u camera_service -f"
echo ""
echo "Next: open the MMOMENT web app and scan the setup QR with this device."
