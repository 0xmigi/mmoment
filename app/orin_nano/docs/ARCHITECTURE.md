# Mmoment Jetson Camera System Architecture

## Overview

Three-container Docker application on NVIDIA Jetson providing AI-powered camera functionality with Solana blockchain integration. Streaming via WebRTC/WHIP. External access via Cloudflare tunnel at `https://[camera-pda].mmoment.xyz`.

## Container Architecture

```
+---------------------------------------------------------------------+
|                        JETSON HOST SYSTEM                            |
|                                                                      |
|  +-----------------+  +------------------+  +---------------------+  |
|  |   CAMERA        |  |  BIOMETRIC       |  |   SOLANA            |  |
|  |   SERVICE       |  |  SECURITY        |  |   MIDDLEWARE        |  |
|  |   (Port 5002)   |  |  (Port 5003)     |  |   (Port 5001)      |  |
|  |                 |  |                  |  |                     |  |
|  | - Main API      |  | - AES-256       |  | - Blockchain ops    |  |
|  | - CV/AI (native)|  | - NFT packaging |  | - Wallet sessions   |  |
|  | - WebRTC/WHIP   |  | - Secure purge  |  | - Tx building       |  |
|  | - App Manager   |  | - No raw storage|  | - Device reg        |  |
|  +-----------------+  +------------------+  +---------------------+  |
|                                                                      |
|  External: Cloudflare Tunnel -> [camera-pda].mmoment.xyz -> :5002    |
+---------------------------------------------------------------------+
```

## Camera Service (Port 5002)

Primary container and API gateway. Runs with NVIDIA runtime for GPU access.

### Core Services (initialized in main.py)
- **Buffer Service** — camera I/O, frame ring buffer, GPU memory
- **Capture Service** — photo/video capture, file storage, metadata
- **Session Service** — user auth, access control, timeouts
- **WebRTC Service** — P2P streaming via Socket.IO signaling
- **Dual WHIP Publishers** — clean stream + annotated stream
- **Blockchain Sync** — auto-enables face visualization on check-in
- **Device Registration** — on-chain camera identity, dynamic PDA/DNS
- **App Manager** — loads CV apps (pushups, basketball) as plugins from `/opt/mmoment/apps`

### AI Pipeline
- **Native Mode** (default): C++ TensorRT inference server inside container
  - RetinaFace/SCRFD for face detection
  - ArcFace/AdaFace for face recognition
  - YOLOv8n-pose for body tracking
  - OSNet for person re-identification
- **Python fallback**: YOLOv8 + InsightFace via Python services

### Hardware Access
- GPU via NVIDIA runtime (`runtime: nvidia`)
- Camera devices: `/dev/video0`, `/dev/video1` (Logitech StreamCam)
- DRI devices for headless EGL
- TensorRT engines mounted from `./native/`

### Network
- `network_mode: host` (required for Jetson kernel compatibility)
- Biometric/Solana containers reached via Docker network aliases

## Biometric Security (Port 5003)

CPU-only container on private Docker network (`172.20.0.3`). Not externally accessible.

- AES-256 encryption of facial embeddings
- NFT-compatible metadata package generation
- Cryptographic deletion (secure purge)
- No persistent storage of raw biometric data
- Session-scoped encrypted temporary storage only

## Solana Middleware (Port 5001)

CPU-only container on private Docker network (`172.20.0.2`). Not externally accessible.

- **Program ID**: `E67WTa1NpFVoapXwYYQmXzru3pyhaN9Kj3wPdZEyyZsL`
- **Camera PDA**: set dynamically after device registration
- **Network**: Solana Devnet
- Wallet session management
- Transaction building for frontend consumption
- Device keypair management (encrypted, hardware-bound)

## Data Flow

### Frame Processing
```
Camera -> Buffer Service -> Native TensorRT Pipeline -> Visual Overlays
                                                            |
                                              +-------------+-------------+
                                              |                           |
                                    Clean WHIP Stream          Annotated WHIP Stream
                                              |                           |
                                        WebRTC/WHEP                 WebRTC/WHEP
```

### Authentication
```
Frontend -> Cloudflare Tunnel -> Camera Service (:5002)
                                       |
                                 Session Service
                                       |
                                 Solana Middleware (:5001)
                                       |
                                 Solana Blockchain
```

### Inter-Container Communication
All containers communicate via localhost HTTP APIs:
- Camera Service -> Biometric Security (encryption requests)
- Camera Service -> Solana Middleware (session validation, blockchain ops)
- Biometric Security -> Solana Middleware (secure data transfer)

**Camera service never handles blockchain operations directly.** It only makes HTTP calls to other containers and syncs blockchain state for visual effects automation.

## File System

```
/mnt/nvme/mmoment/app/orin_nano/
+-- data/                      # Persistent data
|   +-- face_embeddings/       # Face recognition data
|   +-- faces/                 # Face photos
|   +-- recordings/            # Recording sessions
|   +-- config/                # Configuration files
|   +-- device/                # Device keypair (encrypted)
|
+-- services/                  # Docker service source
|   +-- camera-service/        # Port 5002
|   +-- biometric-security/    # Port 5003
|   +-- solana-middleware/     # Port 5001
|
+-- apps/                      # CV app plugins
|   +-- base_competition_app.py
|   +-- pushup/
|   +-- basketball/
|
+-- native/                    # TensorRT engines + C++ binary
|   +-- build/
|   +-- retinaface.engine
|   +-- arcface_r50.engine
|   +-- yolov8n-pose-native.engine
|   +-- osnet_x0_25.engine
|
+-- cv_dev/                    # CV dev environment + test videos
+-- docs/                      # Documentation
+-- docker-compose.yml         # Service orchestration
```

### Key Volume Mounts
```yaml
camera-service:
  # Media on NVMe SSD (not in repo)
  - /mnt/nvme/mmoment-photos:/app/photos
  - /mnt/nvme/mmoment-videos:/app/videos
  # Persistent data
  - ./data/face_embeddings:/app/face_embeddings
  - ./data/config:/app/config
  - ./data/device:/opt/mmoment/device
  # CV apps loaded as plugins
  - ./apps:/opt/mmoment/apps
  # Native TensorRT engines
  - ./native/build:/app/native/build:ro
```

## Security

- Biometric and Solana containers on isolated Docker network, no external access
- Camera service is sole public entry point via Cloudflare tunnel
- Biometric data encrypted at rest, never stored in plain text
- Device keypair encrypted with hardware-bound key (`/etc/machine-id`)
- CORS configured for frontend integration
- Health checks on all containers with auto-restart

## Docker Compose Key Config

- **Network**: `host` mode for camera-service, bridge network (`172.20.0.0/16`) for internal services
- **GPU**: NVIDIA runtime, privileged mode, NET_ADMIN/NET_RAW capabilities
- **Health Checks**: curl to `/api/health` every 30s, 3 retries
- **Dependencies**: camera-service depends on biometric-security and solana-middleware
- **Restart**: `unless-stopped` on all containers
