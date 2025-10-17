# mmoment Computer Vision Platform - 3-Container Architecture

## 🚀 Quick Start
```bash
./deploy.sh build    # Build all containers
./deploy.sh start    # Start the platform  
./deploy.sh status   # Check status
```

## 📦 Services

### 🎥 Camera Service (Port 5002)
**Location:** `services/camera-service/`
- GPU-accelerated YOLOv8 + InsightFace
- Real-time detection & recognition
- Video streaming & capture
- Your existing working face recognition system

### 🔐 Biometric Security (Port 5003)  
**Location:** `services/biometric-security/`
- Facial embedding encryption (AES-256)
- NFT package creation
- Session management & secure purging
- CPU-only container

### ⛓️ Solana Middleware (Port 5001)
**Location:** `services/solana-middleware/`
- Blockchain operations
- NFT minting & management
- Wallet authentication
- Cross-service coordination

## 🔄 Data Flow
1. **Check-in:** Wallet auth → Face recognition → Session created
2. **Session:** User is "recognizable" for gestures/apps (2 hours)
3. **NFT Mint:** Encrypted facial embedding → Blockchain storage
4. **Check-out:** Secure purge across all services

## 🛠 Management
```bash
./deploy.sh help     # See all commands
./deploy.sh logs camera-service  # View specific service logs
./deploy.sh health   # Check all service health
```

## 🎯 Vision
Physical-digital bridge platform for gesture-based apps with privacy-preserving identity. 