# Camera Service - Native TensorRT Mode
# Uses dustynv base for Tegra libraries (GStreamer, NVMM), but minimal Python deps
FROM dustynv/l4t-pytorch:r36.4.0

# Cache busting argument
ARG REBUILD_TIMESTAMP
ENV REBUILD_TIMESTAMP=${REBUILD_TIMESTAMP}

# Set working directory
WORKDIR /app

# Install only the minimal additional system dependencies
# gstreamer1.0-plugins-good provides v4l2src for camera capture
# ffmpeg needed for video recording
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    socat \
    libzbar0 \
    ffmpeg \
    gstreamer1.0-plugins-good \
    && rm -rf /var/lib/apt/lists/*

# Create required directories
RUN mkdir -p /app/logs /app/photos /app/videos /app/faces /app/native

# Copy rebuild timestamp to bust cache
COPY .env.rebuild /tmp/rebuild_marker

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies - LIGHTWEIGHT!
# Native server does all GPU inference, Python only needs networking/web libs
# We use dustynv's Python but DON'T install torch/insightface/onnx
# --ignore-installed needed for blinker (distutils package)
RUN pip3 install --no-cache-dir --ignore-installed -i https://pypi.org/simple \
    flask==3.0.3 flask-cors==4.0.1 \
    opencv-python-headless==4.10.0.84 \
    numpy==1.24.4 \
    requests==2.32.3 \
    cryptography==44.0.0 \
    solders==0.21.0 base58==2.1.1 pynacl==1.5.0 \
    pyzbar==0.1.9 \
    PyYAML==6.0.2 \
    aiortc==1.14.0 \
    "python-socketio[asyncio]==5.9.0" \
    "aiohttp>=3.10.0" \
    av==14.0.0 \
    Pillow \
    psutil

# Copy application code
COPY . .

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Expose port
EXPOSE 5002

# Health check (longer start period for native server init)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5002/api/health || exit 1

# Run via entrypoint that starts native server first
ENTRYPOINT ["/app/entrypoint.sh"]
