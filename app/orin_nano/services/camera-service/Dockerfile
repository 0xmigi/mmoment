# Unified Camera Service with GPU Face Recognition
# Based on the working ultralytics container from the GPU demo

FROM ultralytics/ultralytics:latest-jetson-jetpack6

# Cache busting argument
ARG REBUILD_TIMESTAMP
ENV REBUILD_TIMESTAMP=${REBUILD_TIMESTAMP}

# Set working directory
WORKDIR /app

# Install additional system dependencies including FFmpeg for streaming and NetworkManager for WiFi
RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=America/New_York apt-get install -y \
    curl \
    ffmpeg \
    network-manager \
    socat \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*

# Create required directories
RUN mkdir -p /root/mmoment/app/orin_nano/camera_service/logs \
    && mkdir -p /app/logs \
    && mkdir -p /app/photos \
    && mkdir -p /app/videos \
    && mkdir -p /app/faces

# Copy rebuild timestamp to bust cache
COPY .env.rebuild /tmp/rebuild_marker

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with exact versions to avoid conflicts
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5002/api/health || exit 1

# Run the application
CMD ["python", "main.py"] 