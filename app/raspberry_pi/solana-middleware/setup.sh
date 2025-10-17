#!/bin/bash
# Setup script for Solana Transaction Verification Middleware

set -e

echo "Installing required packages..."
pip install --break-system-packages -r requirements.txt

echo "Setting up service..."
sudo cp solana-middleware.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable solana-middleware.service

echo "Making middleware script executable..."
chmod +x middleware.py

echo "Starting service..."
sudo systemctl start solana-middleware.service
sleep 2
sudo systemctl status solana-middleware.service

echo "Setup complete! Middleware is running on port 5002."
echo "Test with: curl http://localhost:5002/api/health" 