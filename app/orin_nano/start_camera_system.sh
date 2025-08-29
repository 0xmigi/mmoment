#!/bin/bash

echo "🎬 Starting MMoment Camera System..."

# Navigate to the project directory
cd /mnt/nvme/mmoment/app/orin_nano

# Load environment variables
if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    export $(grep -v '^#' .env | xargs)
fi

# Stop any existing containers
echo "Stopping existing containers..."
docker compose down 2>/dev/null

# Set DISPLAY variable for GUI applications
export DISPLAY=:0

# Try starting with full docker-compose first
echo "Attempting full container startup..."
if docker compose up -d 2>/dev/null; then
    echo "✅ Full container stack started successfully!"
    echo "Services running:"
    echo "  📹 Camera Service: http://192.168.1.232:5002"
    echo "  🔐 Biometric Security: http://192.168.1.232:5003"
    echo "  ⛓️  Solana Middleware: http://192.168.1.232:5001"
else
    echo "❌ Full stack failed, trying camera-only mode..."
    
    # Create test compose if it doesn't exist
    if [ ! -f docker-compose.test.yml ]; then
        echo "Creating test compose file..."
        cat > docker-compose.test.yml << 'EOF'
version: '3.8'

services:
  camera-service:
    build:
      context: ./services/camera-service
      dockerfile: Dockerfile
    environment:
      - DISPLAY=${DISPLAY:-}
      - CUDA_VISIBLE_DEVICES=0
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - PYTHONUNBUFFERED=1
      - CAMERA_DEVICE=/dev/video1
      # Livepeer streaming configuration
      - LIVEPEER_API_KEY=${LIVEPEER_API_KEY}
      - LIVEPEER_STREAM_KEY=${LIVEPEER_STREAM_KEY}
      - LIVEPEER_INGEST_URL=rtmp://rtmp.livepeer.com/live
      - LIVEPEER_PLAYBACK_ID=${LIVEPEER_PLAYBACK_ID}
    privileged: true
    runtime: nvidia
    devices:
      - /dev/video0:/dev/video0
      - /dev/video1:/dev/video1
      - /dev/video2:/dev/video2
    volumes:
      - /dev:/dev
      - /tmp/.X11-unix:/tmp/.X11-unix
      # GPU model cache
      - /mnt/nvme/jetson_cache/insightface:/root/.insightface
      - /mnt/nvme/jetson_cache/ultralytics:/root/.config/Ultralytics
      # Temporary local storage
      - ./temp_photos:/workspace/photos
      - ./temp_videos:/workspace/videos
      - ./temp_logs:/workspace/logs
    restart: unless-stopped
    network_mode: host
EOF
    fi
    
    # Start camera service only
    mkdir -p temp_photos temp_videos temp_logs
    docker compose -f docker-compose.test.yml up camera-service -d
    
    if [ $? -eq 0 ]; then
        echo "✅ Camera service started in standalone mode!"
        echo "  📹 Camera Service: http://192.168.1.232:5002"
        echo "  ⚠️  Biometric & Solana services not available"
    else
        echo "❌ Failed to start camera service"
        exit 1
    fi
fi

# Wait for service to initialize
echo "Waiting for service to initialize..."
sleep 15

# Check service health
echo "Checking service health..."
if curl -s http://192.168.1.232:5002/api/health > /dev/null; then
    echo "✅ Camera service is healthy!"
    echo ""
    echo "🎬 MMoment Camera System Ready!"
    echo "Dashboard: http://192.168.1.232:5002"
    echo "Livepeer Playback: https://lvpr.tv/?v=\${LIVEPEER_PLAYBACK_ID}"
else
    echo "⚠️  Service starting... (may take another minute)"
fi

echo ""
echo "To start Livepeer streaming:"
echo "  curl -X POST http://192.168.1.232:5002/api/stream/livepeer/start" 