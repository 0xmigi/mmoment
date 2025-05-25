#!/bin/bash

# Camera Service Installation Script
echo "Installing Lightweight Buffer-Based Camera Service..."

# Create directories if they don't exist
echo "Creating required directories..."
mkdir -p ~/mmoment/app/orin_nano/camera_service_new/logs
mkdir -p ~/mmoment/app/orin_nano/camera_service_new/photos
mkdir -p ~/mmoment/app/orin_nano/camera_service_new/videos
mkdir -p ~/mmoment/app/orin_nano/camera_service_new/faces

# Install required dependencies
echo "Installing required Python packages..."
pip3 install flask opencv-python numpy

# Ask to install optional dependencies
read -p "Install optional dependencies for face recognition and gesture detection? (y/n): " install_optional

if [[ $install_optional == "y" || $install_optional == "Y" ]]; then
    echo "Installing face_recognition and mediapipe..."
    pip3 install face_recognition mediapipe
fi

# Make scripts executable
echo "Setting permissions..."
chmod +x main.py

# Install systemd service
echo "Installing systemd service..."
sudo cp camera-service.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable camera-service.service

# Success message
echo "Installation completed successfully."
echo "You can start the service with: sudo systemctl start camera-service.service"
echo "Access the API at: http://localhost:5003/"
echo "Access the test page at: http://localhost:5003/test-page" 