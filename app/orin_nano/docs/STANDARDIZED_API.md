# Standardized API Endpoints

This document outlines the API endpoints for the mmoment camera system, clearly separating **public frontend APIs** from **internal service APIs**.

## 🔐 Authentication Model

Most endpoints require a valid **session** established via Ed25519 signature verification:

1. User signs message: `{wallet_address}|{timestamp}|{nonce}` with their wallet
2. Call `/api/checkin` with the signature
3. Subsequent requests include `wallet_address`, `request_timestamp`, `request_nonce`, `request_signature`

**Session-protected endpoints** will return `403` with `"Invalid session - please check in first"` if not authenticated.

---

## 🌐 PUBLIC FRONTEND ENDPOINTS
**(Accessible via jetson.mmoment.xyz/api/...)**

All these endpoints are accessible to your frontend application and are the **ONLY ones** you should use in your frontend code.

### Health & Status (Public - No Session Required)
- `GET /api/health` - Health check with service status
- `GET /api/status` - **[MAIN ENDPOINT]** Comprehensive system status including streaming and recording state
- `GET /api/stream/info` - Get all available stream URLs and connection info

### Session Management
- `POST /api/checkin` - **Check in to camera** (requires Ed25519 signature)
- `POST /api/checkout` - **Check out from camera** (requires session)
- `GET /api/session/status/{wallet_address}` - Check if a wallet is checked in (public)

> **Note:** `/api/session/disconnect` and `/disconnect` have been **removed**. Use `/api/checkout` instead.

### Camera Actions (Session Required)
- `POST /api/capture` - Take a photo. Supports `annotated: true` for CV overlay capture.
- `POST /api/record` - Start/stop video recording

### Media Access (Session Required)
- `GET /list_videos` - List available videos
- `GET /list_photos` - List available photos
- `GET /photos/{filename}` - Get specific photo (currently public - signed URLs planned)
- `GET /videos/{filename}` - Get specific video (currently public - signed URLs planned)

### Computer Vision (Session Required)
- `POST /api/face/enroll` - Enroll face for recognition
- `POST /recognize_face` - Recognize faces in current frame
- `POST /toggle_face_detection` - Toggle face detection
- `POST /toggle_face_visualization` - Toggle face visualization
- `POST /toggle_face_boxes` - Toggle face boxes
- `POST /toggle_gesture_visualization` - Toggle gesture visualization
- `POST /toggle_pose_visualization` - Toggle pose skeleton visualization

### CV Apps (Session Required)
- `POST /api/apps/load` - Load a CV app (e.g., pushup counter)
- `POST /api/apps/activate` - Activate a loaded CV app
- `POST /api/apps/deactivate` - Deactivate current CV app
- `GET /api/apps/status` - Get current app status and state
- `POST /api/apps/competition/start` - Start a competition with recognized users
- `POST /api/apps/competition/end` - End the current competition

### Admin Actions (Session Required - Owner-only planned)
- `POST /camera/reset` - Reset camera connection
- `POST /clear_enrolled_faces` - Clear all enrolled faces

> **Note:** These admin endpoints currently require any valid session. Future update will restrict to camera owner wallet only.

### Streaming (Public)

**Note:** Video streaming uses WebRTC via Socket.IO signaling, NOT REST API endpoints.

- `GET /api/stream/webrtc/status` - Get P2P WebRTC service status
- `GET /api/stream/whip/status` - Get WHIP publisher status (for WHEP fallback)

#### Stream Types (Dual Stream)
- **Clean stream**: Raw video without CV annotations (default) - public
- **Annotated stream**: Video with face boxes, skeletons, app overlays - should be session-gated (planned)

---

## 🛠️ CV DEVELOPMENT ENDPOINTS
**(Only available when `CV_DEV_MODE=true`)**

These endpoints allow CV app development using pre-recorded video files instead of live camera.

### Dev Status & Video Management
- `GET /api/dev/status` - Get CV dev environment status
- `GET /api/dev/videos` - List available test videos
- `POST /api/dev/load` - Load a video file: `{"path": "video.mp4"}`
- `GET /api/dev/help` - Get help on all dev endpoints

### Playback Controls
- `GET /api/dev/playback/state` - Get current playback state
- `POST /api/dev/playback/play` - Resume playback
- `POST /api/dev/playback/pause` - Pause playback
- `POST /api/dev/playback/seek` - Seek: `{"frame": N}` or `{"time": S}` or `{"progress": 0.5}`
- `POST /api/dev/playback/speed` - Set speed: `{"speed": 0.5}`
- `POST /api/dev/playback/loop` - Toggle loop: `{"enabled": true}`
- `POST /api/dev/playback/step` - Step frame: `{"direction": "forward"}`
- `POST /api/dev/playback/rotation` - Toggle rotation: `{"enabled": false}`
- `POST /api/dev/restart` - Restart video from beginning

### Example Usage
```bash
# List available videos
curl localhost:5002/api/dev/videos

# Load a video
curl -X POST localhost:5002/api/dev/load \
  -H "Content-Type: application/json" \
  -d '{"path": "pushup_sample.mp4"}'

# Control playback
curl -X POST localhost:5002/api/dev/playback/pause
curl -X POST localhost:5002/api/dev/playback/seek -d '{"frame": 100}'
curl -X POST localhost:5002/api/dev/playback/speed -d '{"speed": 0.5}'
```

### Adding Test Videos
Place videos in `cv_dev/videos/` directory:
```bash
cd /mnt/nvme/mmoment/app/orin_nano/cv_dev/videos/
yt-dlp -f "best[height<=720]" "https://www.youtube.com/watch?v=VIDEO_ID" -o my_video.mp4
# Or download from Pexels:
curl -L "https://videos.pexels.com/video-files/VIDEO_ID/..." -o video.mp4
```

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

---

## Request/Response Format

### Check-in Request (Ed25519 Signature Required)
```json
{
  "wallet_address": "ABC123...",
  "request_signature": "base58-encoded-signature",
  "request_timestamp": 1234567890123,
  "request_nonce": "random-uuid",
  "display_name": "User Name",
  "username": "username"
}
```

### Session-Protected Request Format
All session-protected endpoints require:
```json
{
  "wallet_address": "ABC123...",
  "request_timestamp": 1234567890123,
  "request_nonce": "random-uuid",
  "request_signature": "base58-encoded-signature",
  // ... other endpoint-specific fields
}
```

### Capture Request
```json
{
  "wallet_address": "string",
  "request_timestamp": 1234567890123,
  "request_nonce": "random-uuid",
  "request_signature": "base58-signature",
  "annotated": false,
  "event_metadata": {
    "app_id": "pushup",
    "event_type": "pushup_peak",
    "rep": 5
  }
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
      "p2p_connected": true,
      "whip_publishing": true,
      "active_viewers": 2,
      "format": "webrtc"
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

---

## Frontend Integration

### 🎯 Essential Flow

```javascript
// 1. CHECK-IN (Required for most actions)
const checkinResponse = await fetch('/api/checkin', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    wallet_address: walletPublicKey,
    request_signature: signedMessage,  // Ed25519 signature
    request_timestamp: Date.now(),
    request_nonce: crypto.randomUUID(),
    display_name: 'User Name'
  })
});

// 2. USE SESSION-PROTECTED ENDPOINTS
// All subsequent requests need signature fields
const captureResponse = await fetch('/api/capture', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    wallet_address: walletPublicKey,
    request_signature: newSignature,
    request_timestamp: Date.now(),
    request_nonce: crypto.randomUUID(),
    annotated: true
  })
});

// 3. CHECK-OUT when done
await fetch('/api/checkout', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    wallet_address: walletPublicKey,
    request_signature: signature,
    request_timestamp: Date.now(),
    request_nonce: crypto.randomUUID()
  })
});
```

### ❌ DO NOT Call These From Frontend
- Any `/api/blockchain/*` endpoints - Internal only
- Any `/api/biometric/*` endpoints - Internal only
- Port 5001 or 5003 endpoints - Not exposed publicly

### ✅ Frontend Architecture
```
Frontend → jetson.mmoment.xyz/api/* → Camera Service (Port 5002)
                                   ↓
                              Internal Services:
                              - Solana Middleware (Port 5001)
                              - Biometric Security (Port 5003)
```

---

## Removed/Deprecated Endpoints

These endpoints have been **removed**:
- `/disconnect` - Use `/api/checkout` instead
- `/api/session/disconnect` - Use `/api/checkout` instead

The `/api/checkout` endpoint properly handles:
- Session termination
- Timeline activity buffering
- Backend notification

---

## Summary

**Frontend developers:**
1. Use `/api/checkin` with Ed25519 signature to establish session
2. Include signature fields in all session-protected requests
3. Use `/api/checkout` to end session (NOT `/disconnect`)
4. Only use 🌐 PUBLIC FRONTEND ENDPOINTS listed above
