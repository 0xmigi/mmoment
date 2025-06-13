#!/bin/bash

# Development helper script for Camera Service
# This ensures clean development environment with auto-reloading

echo "🚀 Starting Camera Service Development Environment"
echo "=================================================="

# Make sure we're in the right directory
cd "$(dirname "$0")"

# Stop any existing containers
echo "📦 Stopping existing containers..."
sudo docker compose down 2>/dev/null || true

# Clean up any orphaned containers
echo "🧹 Cleaning up..."
sudo docker ps -a --filter "name=unified-camera-service" --format "table {{.Names}}" | grep -v NAMES | xargs -r sudo docker rm -f

# Start with development overrides
echo "🔄 Starting development environment..."
echo "   • Flask debug mode: ON"
echo "   • Auto-reload: ON" 
echo "   • Live code editing: ON"
echo ""

# Use both compose files for development
sudo docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build

echo ""
echo "🎯 Development environment ready!"
echo "   • Camera Service: http://localhost:5003"
echo "   • Local Test Page: http://localhost:5003/local-test"
echo "   • Make changes to code and they'll auto-reload!" 