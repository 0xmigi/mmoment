#!/bin/bash
# Service startup script for Jetson Camera System

echo "Starting Jetson Camera System Services..."

# Check and start camera service
echo "Checking Camera Service..."
if ! systemctl is-active --quiet camera-service.service; then
    echo "Starting Camera Service..."
    sudo systemctl restart camera-service.service
    sleep 3
else
    echo "Camera Service is already running."
fi

# Check and start Solana middleware
echo "Checking Solana Middleware..."
if ! systemctl is-active --quiet solana-middleware.service; then
    echo "Starting Solana Middleware..."
    sudo systemctl restart solana-middleware.service
    sleep 2
else
    echo "Solana Middleware is already running."
fi

# Check and start Frontend Bridge
echo "Checking Frontend Bridge..."
if ! systemctl is-active --quiet frontend-bridge.service; then
    echo "Starting Frontend Bridge..."
    sudo systemctl restart frontend-bridge.service
    sleep 2
else
    echo "Frontend Bridge is already running."
fi

# Check and start Cloudflare Tunnel
echo "Checking Cloudflare Tunnel..."
if ! systemctl is-active --quiet cloudflared-compat.service; then
    echo "Starting Cloudflare Tunnel..."
    sudo systemctl restart cloudflared-compat.service
else
    echo "Cloudflare Tunnel is already running."
fi

echo "All services have been checked and started if needed."
echo "You can access the API at https://jetson.mmoment.xyz/" 