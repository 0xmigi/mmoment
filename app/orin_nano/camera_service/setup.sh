#!/bin/bash

# Camera Service Setup Script
echo "Setting up the new Buffer-Based Camera Service..."

# Create required directories
echo "Creating required directories..."
mkdir -p ~/mmoment/app/orin_nano/camera_service_new/logs
mkdir -p ~/mmoment/app/orin_nano/camera_service_new/photos
mkdir -p ~/mmoment/app/orin_nano/camera_service_new/videos
mkdir -p ~/mmoment/app/orin_nano/camera_service_new/faces
mkdir -p ~/mmoment/app/orin_nano/camera_service_new/models/facenet_model

# Install required dependencies
echo "Installing required Python packages..."
pip3 install flask opencv-python numpy requests

# Install FaceNet dependencies
echo "Installing TensorFlow and FaceNet dependencies..."
pip3 install tensorflow mtcnn pillow

# Ask if user wants to download the FaceNet model now
read -p "Download and prepare FaceNet model now? (y/n): " download_model
if [[ $download_model == "y" || $download_model == "Y" ]]; then
    echo "Downloading FaceNet model (this may take a while)..."
    chmod +x download_facenet_model.py
    python3 download_facenet_model.py --model mobilefacenet
else
    echo "You can download the model later with: python3 download_facenet_model.py"
fi

# Ask to install gesture detection dependencies
read -p "Install optional dependencies for gesture detection? (y/n): " install_gesture

if [[ $install_gesture == "y" || $install_gesture == "Y" ]]; then
    echo "Installing mediapipe for gesture detection..."
    pip3 install mediapipe
fi

# Make scripts executable
echo "Setting permissions..."
chmod +x ~/mmoment/app/orin_nano/camera_service_new/main.py
chmod +x download_facenet_model.py

# Create systemd service file
echo "Creating systemd service file..."
cat > /tmp/camera-service.service << EOF
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
# Run port cleanup script before starting to avoid conflicts
ExecStartPre=/bin/bash -c "lsof -ti:5003 | xargs -r kill -9"
ExecStart=/usr/bin/python3 /home/azuolas/mmoment/app/orin_nano/camera_service_new/main.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Install the service
echo "Installing the service..."
sudo mv /tmp/camera-service.service /etc/systemd/system/camera-service.service
sudo systemctl daemon-reload

# Stop the old service and start the new one
echo "Stopping old service and starting new service..."
sudo systemctl stop camera-service.service
sudo systemctl start camera-service.service
sudo systemctl enable camera-service.service

# Restart related services
echo "Restarting related services..."
sudo systemctl restart solana-middleware.service frontend-bridge.service cloudflared-compat.service

# Check status
echo "Camera service status:"
sudo systemctl status camera-service.service

echo ""
echo "Setup completed successfully!"
echo "You can access the test page at: http://localhost:5003/test-page"
echo ""
echo "To restart all services, run:"
echo "sudo systemctl restart camera-service.service solana-middleware.service frontend-bridge.service cloudflared-compat.service" 