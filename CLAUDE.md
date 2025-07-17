# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MMOMENT is a blockchain-integrated camera network system that creates identity-aware cameras for capturing social content. The system combines AI-powered computer vision, Solana blockchain integration, and real-time streaming to enable seamless content capture at predictable interaction points.

## Development Commands

### Web Frontend (React/TypeScript)
```bash
cd app/web
yarn install
yarn dev                    # Development server
yarn dev:devnet            # Development with devnet cluster
yarn build                 # Production build
yarn lint                  # ESLint
yarn preview               # Preview production build
```

### Backend (Node.js/Express)
```bash
cd app/backend
yarn install
yarn dev                   # Development server with ts-node
yarn build                 # Compile TypeScript
yarn start                 # Run compiled JavaScript
yarn test                  # Run Jest tests
```

### Solana Smart Contracts (Anchor/Rust)
```bash
anchor build               # Build Solana programs
anchor test                # Run tests
anchor deploy              # Deploy to configured cluster
anchor run camera-network  # Run camera network client
anchor run face-test       # Run face recognition tests
anchor run face-checkin    # Run face check-in tests
```

### Camera Services (Python/Docker)
```bash
# Jetson Orin Nano
cd app/orin_nano
docker-compose up -d       # Start all services
docker-compose down        # Stop all services
docker-compose logs -f     # View logs

# Raspberry Pi
cd app/raspberry_pi/new-camera-service
python -m camera_service.main  # Start camera service
```

## Architecture Overview

The system uses a **three-container Docker architecture** on NVIDIA Jetson devices:

### 1. Camera Service Container (Port 5002)
- **Primary API gateway** accessible via `jetson.mmoment.xyz`
- Computer vision processing (YOLOv8 face detection, InsightFace recognition)
- Real-time streaming via Livepeer RTMP
- Session management and access control
- Photo/video capture with metadata

### 2. Biometric Security Container (Port 5003)
- AES-256 encryption of facial embeddings
- NFT package generation for Solana
- Secure data purging with cryptographic deletion
- No persistent storage of raw biometric data

### 3. Solana Middleware Container (Port 5001)
- Blockchain integration for Solana devnet
- Wallet session management
- Face NFT minting with encrypted embeddings
- Transaction building for frontend consumption

### Web Application Architecture
```
src/
├── auth/           # Solana wallet authentication, Dynamic.xyz integration
├── blockchain/     # Anchor client, Solana provider setup
├── camera/         # Camera connection, face enrollment, recording
├── media/          # Video streaming, gallery, media viewer
├── storage/        # IPFS (Pinata), Walrus storage integration
├── timeline/       # Activity timeline and events
└── ui/             # Layout components, controls, settings
```

## Key Configuration

### Solana Program
- **Program ID**: `Hx5JaUCZXQqvcYzTcdgm9ZE3sqhMWqwAhNXZBrzWm45S`
- **Network**: Devnet (`https://api.devnet.solana.com`)
- **Test Wallet**: `./test-wallet.json`

### Camera Hardware
- **Primary Device**: NVIDIA Jetson Orin Nano
- **Fallback Device**: Raspberry Pi 5
- **Camera Access**: `/dev/video0`, `/dev/video1`, `/dev/video2`
- **Public Access**: `jetson.mmoment.xyz` via Cloudflare tunnel

### Streaming Integration
- **Livepeer Network**: Hardware-accelerated RTMP streaming
- **WebRTC Support**: Low-latency frontend streaming
- **Global CDN**: Worldwide stream distribution

## Development Guidelines

### Package Management
- **Always use `yarn`** for all JavaScript/TypeScript projects
- Install dependencies with `yarn install`, not `npm install`
- Add packages with `yarn add <package>`, not `npm install <package>`

### Code Structure
- **Frontend**: React with TypeScript, Tailwind CSS, Vite build system
- **Backend**: Node.js with Express, Socket.IO for real-time features
- **Smart Contracts**: Anchor framework for Solana programs
- **Camera Services**: Python Flask with OpenCV, MediaPipe, YOLOv8

### Inter-Service Communication
- All Docker containers communicate via localhost HTTP APIs
- Camera Service never handles blockchain operations directly
- Biometric Security provides encryption services to other containers
- Solana Middleware handles all blockchain interactions

### Important Patterns
- **Service Injection**: Services are injected into each other for communication
- **Session Management**: User authentication via Solana wallet connection
- **Visual Automation**: Face visualization auto-enables when users check in on-chain
- **Hardware Acceleration**: GPU-accelerated AI processing and video encoding

### Security Considerations
- Biometric data never stored in plain text
- Encrypted temporary storage with automatic cleanup
- Inter-container communication via localhost only
- Cloudflare tunnel for secure external access

## Testing

### Anchor Tests
Run comprehensive blockchain tests:
```bash
anchor test                # Full test suite
yarn run ts-mocha -p ./tsconfig.json -t 1000000 tests/**/*.ts
```

### Camera System Tests
- Face recognition tests via `scripts/face-recognition-test.js`
- Check-in functionality via `scripts/face-recognition-checkin.js`
- Camera network client via `scripts/camera-network-client.js`

### Frontend Testing
The web application includes manual testing workflows for:
- Wallet connection and authentication
- Camera discovery and connection
- Face enrollment and recognition
- Video streaming and recording
- Media gallery and sharing

## Deployment

### Docker Deployment (Jetson)
```bash
cd app/orin_nano
./deploy.sh               # Deploy production containers
./start_camera_system.sh  # Start camera services
```

### Service Management
Services are managed via Docker Compose with:
- Health checks for automatic restart
- Persistent storage for photos, videos, models
- GPU access for AI processing
- Network host mode for Jetson kernel compatibility

## External Integrations

- **Livepeer**: Decentralized streaming network
- **Solana**: Blockchain for identity and content ownership
- **Cloudflare**: Tunnel for secure public access
- **IPFS/Pinata**: Decentralized storage for media
- **Walrus**: Alternative storage solution
- **Dynamic.xyz**: Wallet connection and authentication