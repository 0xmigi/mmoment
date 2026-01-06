# Build container for native inference pipeline
# Uses L4T base with TensorRT and CUDA

FROM nvcr.io/nvidia/l4t-tensorrt:r8.6.2-runtime

# Install build tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Check if this L4T image has OpenCV with CUDA
# If not, we'll use NPP directly

WORKDIR /build

# Copy source
COPY . /build/

# Build
RUN mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

CMD ["/bin/bash"]
