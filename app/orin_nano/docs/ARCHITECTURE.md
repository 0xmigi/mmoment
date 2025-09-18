# Mmoment Jetson Camera System Architecture

## Overview
The Mmoment camera system is a **three-container Docker application** running on NVIDIA Jetson that provides AI-powered camera functionality with blockchain integration. The system is designed for real-time computer vision processing with Livepeer streaming and Solana blockchain integration.

## Container Architecture

The system consists of three Docker containers that communicate via localhost:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        JETSON HOST SYSTEM                          │
│                                                                     │
│  ┌─────────────────┬─────────────────┬─────────────────────────────┐ │
│  │   CAMERA        │   BIOMETRIC     │      SOLANA                 │ │
│  │   SERVICE       │   SECURITY      │   MIDDLEWARE                │ │
│  │   (Port 5002)   │   (Port 5003)   │   (Port 5001)               │ │
│  │                 │                 │                             │ │
│  │ • Main API      │ • Encryption    │ • Blockchain API            │ │
│  │ • Computer Vision│ • NFT Packaging │ • Wallet Sessions           │ │
│  │ • Livepeer RTMP │ • Secure Storage│ • Transaction Building      │ │
│  │ • GPU Processing│ • Data Purging  │ • Face NFT Minting          │ │
│  └─────────────────┴─────────────────┴─────────────────────────────┘ │
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