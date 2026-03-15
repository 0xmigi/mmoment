# Mmoment Jetson Camera System Architecture

## Overview
The Mmoment camera system is a **four-container Docker application** running on NVIDIA Jetson that provides AI-powered camera functionality with blockchain integration. The system is designed for real-time computer vision processing with Livepeer streaming and Solana blockchain integration.

## Container Architecture

The system consists of four Docker containers that communicate via localhost:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        JETSON HOST SYSTEM                          │
│                                                                     │
│  ┌──────────────┬──────────────┬─────────────┬──────────────────┐  │
│  │   CAMERA     │  BIOMETRIC   │   SOLANA    │    CV APPS       │  │
│  │   SERVICE    │  SECURITY    │ MIDDLEWARE  │    SERVICE       │  │
│  │ (Port 5002)  │ (Port 5003)  │ (Port 5001) │  (Port 5004)     │  │
│  │              │              │             │                  │  │
│  │ • Main API   │ • Encryption │ • Blockchain│ • Push-up App    │  │
│  │ • CV Core    │ • NFT Package│ • Sessions  │ • Basketball App │  │
│  │ • Streaming  │ • Purging    │ • Tx Build  │ • Pose Est.      │  │
│  │ • GPU/AI     │ • Storage    │ • NFT Mint  │ • Competitions   │  │
│  └──────────────┴──────────────┴─────────────┴──────────────────┘  │
│                                                                     │
│  External Access: Cloudflare Tunnel → jetson.mmoment.xyz → :5002   │
└─────────────────────────────────────────────────────────────────────┘
```

## Camera Service Container (Port 5002)
**Primary container** - Handles camera operations and serves as the main API gateway.

### Core Services Architecture
The camera service uses a **service-oriented architecture** with the following components:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      CAMERA SERVICE                                 │
├─────────────────┬─────────────────┬─────────────────┬───────────────┤
│ Buffer Service  │ GPU Face Service│ Gesture Service │ Capture Service│
│                 │                 │                 │               │
│ • Camera I/O    │ • YOLOv8 Detect │ • MediaPipe     │ • Photo/Video │
│ • Frame Buffer  │ • InsightFace   │ • Hand Tracking │ • File Storage│
│ • Ring Buffer   │ • Face Recogn.  │ • Gesture Recog │ • Metadata    │
│ • GPU Memory    │ • Visual Overlay│ • Visual Overlay│ • Timestamps  │
├─────────────────┼─────────────────┼─────────────────┼───────────────┤
│ Session Service │ Livepeer Service│ Routes/API      │ Blockchain    │
│                 │                 │                 │ Session Sync  │
│ • User Sessions │ • RTMP Streaming│ • /api/* routes │ • Auto Check-in│
│ • Access Control│ • Hardware Accel│ • Legacy routes │ • Visual Enable│
│ • Validation    │ • Stream Status │ • CORS handling │ • HTTP API Only│
│ • Timeouts      │ • Auto Recovery │ • Error handling│ • No Direct BC│
└─────────────────┴─────────────────┴─────────────────┴───────────────┘
```

### Key Features
- **GPU-Accelerated AI**: YOLOv8 for face detection, InsightFace for recognition
- **Real-time Streaming**: Hardware-accelerated RTMP to Livepeer network
- **Visual Effects**: Face boxes and gesture overlays in both stream and local MJPEG
- **Session Management**: User authentication and access control
- **Blockchain Session Sync**: Automatically enables face visualization when users check in on-chain
- **Media Capture**: Photos and videos with metadata

### Hardware Access
- Direct GPU access via NVIDIA runtime
- Camera device access (`/dev/video0`, `/dev/video1`, `/dev/video2`)
- Hardware video encoding for streaming
- Model caching on fast storage (`/mnt/nvme/jetson_cache/`)

## Biometric Security Container (Port 5003)
**Security-focused container** - Handles encryption and secure biometric data operations.

### Responsibilities
- **Facial Embedding Encryption**: AES-256 encryption of facial embeddings
- **NFT Package Generation**: Solana-compatible metadata creation
- **Secure Data Purging**: Cryptographic deletion of sensitive data
- **Session-based Security**: Temporary encrypted storage only

### Security Features
- No persistent storage of raw biometric data
- Encrypted temporary storage with automatic cleanup
- Audit logging for all operations
- Secure inter-container communication

## Solana Middleware Container (Port 5001)
**Blockchain integration container** - Handles all Solana network interactions.

### Configuration
- **Program ID**: `Hx5JaUCZXQqvcYzTcdgm9ZE3sqhMWqwAhNXZBrzWm45S` (hardcoded)
- **Camera PDA**: `WT9oJrL7sbNip8Rc2w5LoWFpwsUcZZJnnjE2zZjMuvD` (hardcoded)
- **Network**: Solana Devnet (`https://api.devnet.solana.com`)

### Features
- **Wallet Session Management**: User authentication via wallet connection
- **Face NFT Minting**: Encrypted facial embeddings as blockchain NFTs
- **Transaction Building**: Frontend-ready transaction serialization
- **PDA Management**: Program Derived Address calculations

## CV Apps Container (Port 5004)
**Computer vision applications container** - Runs competition and tracking apps independently from core camera service.

### Purpose
Isolates CV applications from core camera functionality to provide:
- **Fault Isolation**: App crashes don't affect camera/face recognition
- **Independent Scaling**: Apps can be updated without touching core services
- **Resource Management**: GPU/CPU allocation per app
- **Hot-swapping**: Load/unload apps dynamically

### Architecture
```
CV Apps Service (5004)
├── App Loader (main.py)
│   └── Dynamic app loading
├── Process API (/api/process)
│   └── Frame + detections → stats
└── Loaded Apps
    ├── Push-up Competition
    ├── Basketball Tracking
    └── Future apps...
```

### Communication Flow
```
Camera Service → HTTP POST /api/process → CV Apps Service
   (frame + detections)              (MediaPipe, pose estimation)
                                               ↓
                                     Competition state/stats
                                               ↓
Camera Service ← HTTP Response ← overlay data, rep counts, etc.
```

### Features
- **MediaPipe Pose Estimation**: 17-point body tracking
- **Competition Management**: Start/end, scoring, winners
- **Form Validation**: Biomechanical analysis
- **Real-time Overlays**: Stats, timers, leaderboards
- **Betting Integration**: Crypto wagering (future)

## Data Flow Architecture

### 1. Camera Frame Processing
```
Physical Camera → Buffer Service → [GPU Face, Gesture, Capture] Services
                                                    ↓
                                          Visual Overlay Processing
                                                    ↓
                                    ┌─────────────────────────────┐
                                    │                             │
                                    ▼                             ▼
                            Livepeer RTMP Stream          Local MJPEG Stream
                                    │                      (/stream endpoint)
                                    ▼
                            Livepeer Network CDN
```

### 2. User Authentication & Session Flow
```
Frontend → Cloudflare Tunnel → Camera Service (:5002)
                                      ↓
                               Session Service
                                      ↓
                               Solana Middleware (:5001)
                                      ↓
                               Solana Blockchain
```

### 3. Face Recognition & NFT Flow
```
Camera Frame → Buffer Service → GPU Face Service → Face Detection/Recognition
                                                           ↓
                                               Biometric Security (:5003)
                                                           ↓
                                                  Encryption & NFT Package
                                                           ↓
                                               Solana Middleware (:5001)
                                                           ↓
                                                 Face NFT on Blockchain
```

### 4. Blockchain Session Sync Flow
```
Solana Blockchain → Solana Middleware (:5001) → Camera Service (:5002)
                                                        ↓
                                                Blockchain Session Sync
                                                        ↓
                                           Auto-enable Face Visualization
                                                        ↓
                                              User sees face boxes/identity
```

### 5. Inter-Container Communication
All containers communicate via localhost HTTP APIs:
- Camera Service ↔ Biometric Security (encryption requests)
- Camera Service ↔ Solana Middleware (session validation, NFT operations)
- Biometric Security ↔ Solana Middleware (secure data transfer)

**Important**: Camera service does NOT handle blockchain operations directly. It only:
- Makes HTTP API calls to other containers
- Syncs blockchain state for visual effects automation
- Never duplicates blockchain functionality

## API Architecture

### Public API (Port 5002)
**Accessible via `jetson.mmoment.xyz/api/`**

The camera service exposes a comprehensive REST API with:
- **System Status**: `/api/health`, `/api/status`
- **Camera Control**: `/api/capture`, `/api/record`
- **Computer Vision**: `/api/face/*`, `/api/gesture/current`
- **Session Management**: `/api/session/*`
- **Streaming**: `/api/stream/*`, `/stream` (MJPEG)
- **Media Access**: `/api/photos`, `/api/videos`

### Internal APIs
- **Biometric Security (5003)**: `/api/biometric/*`
- **Solana Middleware (5001)**: `/api/blockchain/*`, `/api/session/*`

## Service Initialization Process

Based on `main.py`, the system starts up as follows:

1. **Directory Creation**: Ensures required directories exist
2. **Service Initialization**: 
   - Buffer Service (camera I/O)
   - GPU Face Service (YOLOv8 + InsightFace)
   - Gesture Service (MediaPipe)
   - Capture Service (media storage)
   - Session Service (user management)
   - Livepeer Service (streaming)
   - Blockchain Session Sync (auto check-in)

3. **Service Injection**: Services are injected into each other for communication
4. **Flask App Creation**: Routes are registered and CORS is configured
5. **Health Monitoring**: Each container has health checks

## Key Configuration

### Docker Compose
- **Network Mode**: `host` (required for Jetson kernel limitations)
- **GPU Access**: NVIDIA runtime for camera service
- **Volume Mounts**: Persistent storage for photos, videos, logs, model cache
- **Health Checks**: Automatic restart on failure
- **Dependencies**: Camera service depends on biometric and Solana containers

### Environment Variables
- **Camera Device**: `/dev/video1` (configurable)
- **Livepeer Config**: API keys, stream keys, playback IDs
- **Service URLs**: Inter-container communication endpoints
- **Solana Config**: RPC URL, camera name, owner wallet

## Hardware Requirements

### NVIDIA Jetson Specifications
- **GPU**: CUDA-capable NVIDIA GPU
- **Memory**: 8GB+ RAM recommended
- **Storage**: 32GB+ with fast model cache storage
- **Camera**: USB/CSI camera support
- **Network**: Stable internet for blockchain and streaming

### Performance Optimizations
- **Model Caching**: Persistent AI model storage
- **Hardware Encoding**: GPU-accelerated video encoding
- **Efficient Buffering**: Ring buffer for frame management
- **Service Singletons**: Shared service instances across the application

## Security Architecture

### Container Isolation
- Each service runs in isolated Docker container
- Inter-container communication via localhost only
- No direct external access to biometric or Solana containers

### Data Protection
- Biometric data never stored in plain text
- Encrypted temporary storage with automatic cleanup
- Secure key derivation for encryption
- Audit logging for all sensitive operations

### Network Security
- Cloudflare tunnel for secure external access
- HTTPS/TLS for all external communication
- CORS configuration for frontend integration
- Health check endpoints for monitoring

## External Integrations

### Livepeer Network
- **RTMP Streaming**: Hardware-accelerated stream to Livepeer
- **Global CDN**: Worldwide stream distribution
- **Playback URLs**: Frontend-accessible stream URLs

### Solana Blockchain
- **Devnet Integration**: Real blockchain transactions
- **Face NFTs**: Encrypted biometric data as NFTs
- **Camera Registry**: On-chain camera and user management
- **Session Validation**: Blockchain-based authentication

### Cloudflare Tunnel
- **Public Access**: `jetson.mmoment.xyz` domain
- **SSL Termination**: Secure HTTPS access
- **Load Balancing**: Traffic distribution
- **DDoS Protection**: Network security

This architecture provides a robust, scalable foundation for AI-powered camera applications with blockchain integration while maintaining security and performance on edge devices.

## File System Structure

### Top-Level Organization
```
/mnt/nvme/mmoment/app/orin_nano/
├── data/                      # ALL persistent data (single source of truth)
│   ├── images/               # Camera captures
│   ├── videos/               # Recorded videos
│   ├── recordings/           # Recording sessions
│   ├── face_embeddings/      # Face recognition data
│   ├── faces/                # Face photos
│   ├── config/               # Configuration files
│   └── device/               # Device keypair
│
├── services/                 # Docker service implementations
│   ├── camera-service/       # Core camera + face recognition (Port 5002)
│   ├── biometric-security/   # Encryption service (Port 5003)
│   ├── solana-middleware/    # Blockchain ops (Port 5001)
│   └── cv-apps/              # CV competition apps runner (Port 5004)
│
├── apps/                     # CV app definitions (plugins)
│   ├── base_competition_app.py  # Reusable base class
│   ├── pushup/               # Push-up competition
│   └── basketball/           # Basketball tracking
│
├── docs/                     # Documentation
├── docker-compose.yml        # Service orchestration
└── test_pushup_app.py        # Integration tests
```

### Service Structure (Example: camera-service)
```
services/camera-service/
├── main.py                   # Service entry point
├── routes.py                 # API routes
├── Dockerfile                # Container build
├── requirements.txt          # Python dependencies
│
├── services/                 # Internal modules (organized!)
│   ├── gpu_face_service.py
│   ├── identity_tracker.py
│   ├── cv_apps_client.py
│   ├── routes_cv_apps.py
│   ├── pipe_integration.py
│   └── ...
│
└── templates/                # HTML templates
```

### Key Principles

1. **Single Data Directory**: All persistent data lives in `/data`
   - Docker volumes mount from `/data` → container paths
   - No data stored in service directories
   - Prevents duplicate/empty folders

2. **Service Isolation**: Each service is self-contained
   - Only `main.py`, `routes.py`, `Dockerfile`, `requirements.txt` at root
   - Everything else in subdirectories (`services/`, `templates/`)
   - Clean, navigable structure

3. **Apps as Plugins**: CV apps are separate from core services
   - `/apps` = app definitions (the logic)
   - `/services/cv-apps` = app runner (the loader)
   - Hot-swappable without touching camera service

4. **No Empty Folders**: Docker volume mounts can create empty directories
   - Fixed by proper volume paths in docker-compose
   - `.gitignore` patterns prevent commits
   - Delete any empty `config/`, `faces/`, etc. in service dirs

### Docker Volume Mapping
```yaml
camera-service:
  volumes:
    - ./data/images:/app/photos              # Host → Container
    - ./data/face_embeddings:/app/face_embeddings
    - ./services/camera-service:/app         # Dev: live code reload

cv-apps:
  volumes:
    - ./apps:/app/apps                       # Apps loaded dynamically
    - ./services/cv-apps:/app                # Dev: live code reload
```

This structure ensures clean organization, prevents clutter, and makes the system easy to navigate and maintain. 