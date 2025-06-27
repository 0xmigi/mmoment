# Standardized API Endpoints

This document outlines the API endpoints for the mmoment camera system, clearly separating **public frontend APIs** from **internal service APIs**.

## 🌐 PUBLIC FRONTEND ENDPOINTS 
**(Accessible via jetson.mmoment.xyz/api/...)**

All these endpoints are accessible to your frontend application and are the **ONLY ones** you should use in your frontend code.

### Health & Status
- `GET /api/health` - Health check with service status
- `GET /api/status` - **[MAIN ENDPOINT]** Comprehensive system status including streaming and recording state

### Camera Actions  
- `POST /api/capture` - Take a photo (requires session)
- `POST /api/record` - Start/stop video recording (requires session)

### Media Access
- `GET /api/photos` - List available photos
- `GET /api/videos` - List available videos
- `GET /api/photos/{filename}` - Get specific photo
- `GET /api/videos/{filename}` - Get specific video

### Session Management
- `POST /api/session/connect` - Connect wallet and create session
- `POST /api/session/disconnect` - Disconnect wallet and end session

### Computer Vision (Jetson-specific)
- `POST /api/face/enroll` - Enroll face for recognition (requires session)
- `POST /api/face/recognize` - Recognize faces in current frame
- `POST /api/face/detect` - Detect faces in current frame (no recognition)
- `GET /api/gesture/current` - Get current detected gesture
- `POST /api/visualization/face` - Toggle face visualization
- `POST /api/visualization/gesture` - Toggle gesture visualization

### Facial NFT Endpoints (Jetson-specific)
- `POST /api/face/enroll/prepare-transaction` - Prepare facial NFT transaction (requires session)
- `POST /api/face/enroll/confirm` - Confirm facial NFT transaction (requires session)

### Streaming (Jetson-specific - Livepeer)
- `POST /api/stream/livepeer/start` - Start Livepeer streaming
- `POST /api/stream/livepeer/stop` - Stop Livepeer streaming
- `GET /api/stream/livepeer/status` - Get Livepeer streaming status (detailed)

---

## 🔒 INTERNAL SERVICE ENDPOINTS
**(NOT accessible from frontend - Internal container communication only)**

These endpoints are used for communication between Docker containers and services. **DO NOT** call these from your frontend application.

### Solana Middleware (Internal Port 5001)
- `GET /api/health` - Internal health check
- `POST /api/session/connect` - Internal blockchain session creation
- `POST /api/session/disconnect` - Internal blockchain session cleanup
- `GET /api/session/status` - Internal session status
- `GET /api/wallet/status` - Internal wallet status
- `POST /api/blockchain/encrypt-face` - Internal encryption
- `POST /api/blockchain/decrypt-face` - Internal decryption
- `POST /api/blockchain/verify-nft` - Internal NFT verification
- `POST /api/blockchain/mint-moment` - Internal NFT minting
- `POST /api/blockchain/mint-facial-nft` - Internal facial NFT minting
- `POST /api/blockchain/purge-session` - Internal session cleanup

### Biometric Security Service (Internal Port 5003)
- `GET /api/health` - Internal health check
- `GET /api/biometric/status` - Internal biometric service status
- `POST /api/biometric/encrypt-embedding` - Internal embedding encryption
- `POST /api/biometric/decrypt-for-session` - Internal embedding decryption
- `POST /api/biometric/get-nft-package` - Internal NFT package creation
- `POST /api/biometric/create-session` - Internal biometric session
- `POST /api/biometric/purge-session` - Internal session cleanup

## Request/Response Format

### Session Connect Request
```json
{
  "wallet_address": "string"
}
```

### Session Connect Response
```json
{
  "success": true,
  "session_id": "string",
  "wallet_address": "string",
  "created_at": 1234567890,
  "expires_at": 1234567890
}
```

### Status Response (Key Endpoint)
```json
{
  "success": true,
  "timestamp": 1234567890,
  "data": {
    "isOnline": true,
    "isStreaming": true,
    "isRecording": false,
    "lastSeen": 1234567890,
    "streamInfo": {
      "playbackId": "24583deg6syfcql",
      "isActive": true,
      "format": "livepeer"
    },
    "recordingInfo": {
      "isActive": false,
      "currentFilename": null
    }
  }
}
```

### Health Response
```json
{
  "status": "ok",
  "buffer_service": "Healthy", 
  "buffer_fps": 9.96,
  "active_sessions": 0,
  "timestamp": 1234567890,
  "camera": {
    "index": 1,
    "preferred_device": "/dev/video1",
    "resolution": "1280x720",
    "target_fps": 15
  }
}
```

### Stream Info Response
```json
{
  "success": true,
  "playbackId": "jetson-camera-stream",
  "isActive": true,
  "streamType": "mjpeg",
  "resolution": "1280x720", 
  "fps": 9.94,
  "streamUrl": "/stream"
}
```

## Frontend Integration

**IMPORTANT:** Your frontend should ONLY call the public endpoints listed above. Here are the key patterns:

### 🎯 Essential Endpoints for Frontend

```javascript
// 1. CHECK STREAMING STATUS (Most Important!)
const status = await fetch('/api/status').then(r => r.json());
const isStreaming = status.data.isStreaming;
const isRecording = status.data.isRecording;

// 2. HEALTH CHECK
const health = await fetch('/api/health').then(r => r.json());

// 3. SESSION MANAGEMENT
const session = await fetch('/api/session/connect', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ wallet_address: 'your_wallet' })
}).then(r => r.json());

// 4. STREAMING CONTROL (Jetson Only)
const streamStart = await fetch('/api/stream/livepeer/start', {
  method: 'POST'
}).then(r => r.json());

const streamStop = await fetch('/api/stream/livepeer/stop', {
  method: 'POST'
}).then(r => r.json());

// 5. COMPUTER VISION (Jetson Only)
const gesture = await fetch('/api/gesture/current').then(r => r.json());

const faceEnroll = await fetch('/api/face/enroll', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ wallet_address: 'your_wallet', session_id: 'session_id' })
}).then(r => r.json());

// 6. MEDIA ACCESS
const photos = await fetch('/api/photos').then(r => r.json());
const videos = await fetch('/api/videos').then(r => r.json());
```

### ❌ DO NOT Call These From Frontend
- Any `/api/blockchain/*` endpoints - These are internal only
- Any `/api/biometric/*` endpoints - These are internal only  
- Port 5001 or 5003 endpoints - These are not exposed publicly

### ✅ Frontend Architecture
```
Frontend → jetson.mmoment.xyz/api/* → Camera Service (Port 5002)
                                   ↓
                              Internal Services:
                              - Solana Middleware (Port 5001)
                              - Biometric Security (Port 5003)
```

The Camera Service handles all the blockchain and biometric operations internally. Your frontend only needs to talk to the main API.

## Legacy Endpoints (Still Supported)

Some older endpoints are still supported for backward compatibility, but use the `/api/` versions above:
- `/health` → `/api/health`
- `/connect` → `/api/session/connect`
- `/disconnect` → `/api/session/disconnect`
- `/capture_moment` → `/api/capture`
- `/start_recording` → `/api/record`
- `/enroll_face` → `/api/face/enroll`
- `/recognize_face` → `/api/face/recognize`

## Summary

**Frontend developers:** Use only the 🌐 PUBLIC FRONTEND ENDPOINTS section above. All blockchain and biometric operations happen automatically behind the scenes when you use the main camera API endpoints.