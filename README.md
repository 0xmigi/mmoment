# MOMENT: ambient photo booth for IRL presence

Mmoment is an ambient photo booth system that captures content automatically when users check in at physical locations. Identity-aware cameras handle face recognition, streaming, and media capture on-device — users just tap in and the camera does the rest. All user data is encrypted and stored on-chain (Solana), owned by the user.

https://github.com/user-attachments/assets/7d560159-c1f3-4f4b-b32c-5c44a421203d

## Architecture

```
[User taps in via NFC/QR] → [Camera device] → [On-device CV processing]
                                                       ↓
                                             [Encrypted on-chain PDAs]
                                                       ↓
                                             [Backend decrypts if authorized]
                                                       ↓
                                             [Frontend displays content]
```

User data lives on-chain in encrypted Solana PDAs (CameraTimeline, UserSessionChain). The backend is a decryption and coordination layer, not a data store.

## Repository Structure

```
mmoment/
├── app/
│   ├── orin_nano/          # NVIDIA Jetson Orin Nano camera
│   ├── pi_zero_2w/         # Raspberry Pi Zero 2W camera
│   ├── pi_5/               # Raspberry Pi 5 camera (legacy)
│   ├── web/                # React frontend
│   └── backend/            # Node.js backend (Express, Socket.IO)
├── programs/
│   ├── camera-network/     # Core Solana program (cameras, identity, sessions)
│   └── competition-escrow/ # Competition escrow program
└── scripts/
```

## Camera Devices

Two camera implementations are actively used.

### NVIDIA Jetson Orin Nano (`app/orin_nano/`)

GPU-accelerated camera running containerized microservices:

- Real-time face detection/recognition (YOLOv8 + InsightFace via TensorRT)
- Pose estimation and gesture detection
- Native C++ inference pipeline (RetinaFace, ArcFace, OSNet)
- Encrypted facial embedding storage
- H.264 live streaming
- On-device Solana wallet integration

### Raspberry Pi Zero 2W (`app/pi_zero_2w/`)

Lightweight Python Flask camera service:

- Dual-mode: QR scanning for onboarding, H.264 streaming via `rpicam-vid` + ffmpeg
- Auto-provisioned Cloudflare tunnels for remote access
- Device-level Solana signing for check-in/checkout

### Raspberry Pi 5 (`app/pi_5/`) — Legacy

Earlier iteration kept as reference. Superseded by the above.

## Web Frontend (`app/web/`)

React + TypeScript + Tailwind CSS + Vite

- Solana wallet auth (Dynamic Labs, Privy, Phantom)
- Camera discovery, live stream viewing, and session management
- Media timeline with photo/video browsing
- Decentralized storage integration (Walrus on Sui, Pinata)
- NFC-based check-in flow
- Livepeer streaming integration

## Backend (`app/backend/`)

Node.js + Express + Socket.IO

- Decryption service for on-chain content (authorized access only)
- File storage relay (Walrus blob storage with AES-256-GCM encryption)
- Gas sponsorship for user transactions (Kora)
- Competition API for escrow settlement
- Real-time session coordination via WebSockets
- Camera device config and relay status
- Session cleanup cron jobs and timeline write operations

## Solana Programs (`programs/`)

Built with the Anchor framework, deployed on devnet.

### camera-network (`E67WTa1NpFVoapXwYYQmXzru3pyhaN9Kj3wPdZEyyZsL`)

- Camera registration and management
- User identity enrollment with encrypted face embeddings
- On-chain session tracking with access keys
- Compressed timeline entries via Light Protocol
- Moment metadata and content attribution

### competition-escrow (`EpczQBF7WmPcyzTtYJfzrPNXSVxM3YJsND7Vx8zpTLAj`)

- Competition creation with invited participants
- Stake deposits and withdrawals
- Settlement with configurable payout rules (winner-take-all, split)

## Stack

| Layer | Tech |
|-------|------|
| Frontend | React, TypeScript, Tailwind CSS, Vite |
| Backend | Node.js, Express, Socket.IO, SQLite |
| Smart Contracts | Anchor (Rust), Light Protocol |
| Primary Camera | Python, OpenCV, YOLOv8, InsightFace, TensorRT, Docker |
| Lightweight Camera | Python, Flask, rpicam-vid, ffmpeg |
| Storage | Walrus (Sui), Pinata, on-chain PDAs |
| Auth | Dynamic Labs, Privy, Phantom, Solana wallets |
| Streaming | Livepeer, H.264/WebRTC |

## License

This project is proprietary software owned by mmoment. All rights reserved.
