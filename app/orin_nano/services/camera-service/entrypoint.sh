#!/bin/bash
# Container entrypoint - starts native camera server then Python service

set -e

SOCKET_PATH="/tmp/native_inference.sock"
NATIVE_DIR="/app/native"
NATIVE_SERVER="$NATIVE_DIR/build/native_camera_server"

echo "=== MMOMENT Camera Service Container ==="

# Set library path for native binary (TensorRT + nvidia runtime libs + tegra EGL)
# Note: build directory must come BEFORE native dir to prioritize mounted libs over any copied placeholders
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/tegra-egl:/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/nvidia:$NATIVE_DIR/build:$NATIVE_DIR:$LD_LIBRARY_PATH"

# Headless EGL - unset DISPLAY to use device EGL instead of X11
unset DISPLAY

echo "Starting native camera server..."

# Clean up any stale socket
rm -f "$SOCKET_PATH"

# Start native camera server in background (from native directory where engines are)
cd "$NATIVE_DIR"
"$NATIVE_SERVER" &
NATIVE_PID=$!

echo "Native server started (PID: $NATIVE_PID)"

# Wait for socket to be ready (max 30 seconds)
echo "Waiting for socket..."
for i in {1..30}; do
    if [ -S "$SOCKET_PATH" ]; then
        echo "Socket ready after ${i}s"
        break
    fi
    if ! kill -0 $NATIVE_PID 2>/dev/null; then
        echo "ERROR: Native server died during startup"
        exit 1
    fi
    sleep 1
done

if [ ! -S "$SOCKET_PATH" ]; then
    echo "ERROR: Socket not created after 30s"
    kill $NATIVE_PID 2>/dev/null || true
    exit 1
fi

# Trap to clean up on exit
cleanup() {
    echo "Shutting down..."
    kill $NATIVE_PID 2>/dev/null || true
    wait $NATIVE_PID 2>/dev/null || true
    echo "Cleanup complete"
}
trap cleanup EXIT SIGTERM SIGINT

# Start Python service
echo "Starting Python camera service..."
cd /app

# Force unbuffered Python output so logs appear immediately
# This is critical for debugging freezes - otherwise logs are stuck in buffer
export PYTHONUNBUFFERED=1

exec python3 main.py
