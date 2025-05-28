# Standardized API Endpoints

This document outlines the standardized API endpoints that are consistent across all camera devices in the mmoment network (Pi5, Jetson Orin Nano, etc.).

## Overview

All camera services now support standardized endpoints under the `/api/` prefix, while maintaining backward compatibility with legacy endpoints. This ensures consistent frontend integration across the entire camera network.

## Camera Service Endpoints (Port 5002)

### Health & Status
- `GET /api/health` - Health check with service status
- `GET /api/stream/info` - Stream metadata (playbackId, isActive, streamType, etc.)

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
- `GET /api/gesture/current` - Get current detected gesture
- `POST /api/visualization/face` - Toggle face visualization
- `POST /api/visualization/gesture` - Toggle gesture visualization

## Solana Middleware Endpoints (Port 5001)

### Health & Status
- `GET /api/health` - Health check

### Session Management
- `POST /api/session/connect` - Connect wallet to blockchain
- `POST /api/session/disconnect` - Disconnect wallet from blockchain
- `GET /api/session/status` - Get session status

### Blockchain Operations
- `POST /api/blockchain/encrypt-face` - Encrypt face embedding
- `POST /api/blockchain/decrypt-face` - Decrypt face embedding
- `POST /api/blockchain/verify-nft` - Verify NFT ownership
- `POST /api/blockchain/mint-moment` - Mint moment as NFT

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

## Legacy Endpoints

All legacy endpoints are still supported for backward compatibility:
- `/health` → `/api/health`
- `/connect` → `/api/session/connect`
- `/disconnect` → `/api/session/disconnect`
- `/capture_moment` → `/api/capture`
- `/start_recording` → `/api/record`
- `/list_photos` → `/api/photos`
- `/list_videos` → `/api/videos`
- `/enroll_face` → `/api/face/enroll`
- `/recognize_face` → `/api/face/recognize`
- `/current_gesture` → `/api/gesture/current`

## Frontend Integration

Your frontend can now use the same API calls for basic camera functions across all devices:

```javascript
// Health check (works on all devices)
const health = await fetch('/api/health').then(r => r.json());

// Connect session (works on all devices)
const session = await fetch('/api/session/connect', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ wallet_address: 'your_wallet' })
}).then(r => r.json());

// Get photos (works on all devices)
const photos = await fetch('/api/photos').then(r => r.json());

// Jetson-specific features
const gesture = await fetch('/api/gesture/current').then(r => r.json());
```

## Device Detection

You can detect device capabilities by checking the root endpoint:

```javascript
const info = await fetch('/').then(r => r.json());
console.log(info.standardized_endpoints); // Available endpoints
console.log(info.version); // API version
```

Jetson devices will include computer vision endpoints, while Pi5 devices will have simpler camera functionality. 