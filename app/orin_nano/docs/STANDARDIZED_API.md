# Standardized API Endpoints

This document outlines the API endpoints for the mmoment camera system, clearly separating **public frontend APIs** from **internal service APIs**.

## 🌐 PUBLIC FRONTEND ENDPOINTS 
**(Accessible via jetson.mmoment.xyz/api/...)**

All these endpoints are accessible to your frontend application and are the **ONLY ones** you should use in your frontend code.

### Health & Status
- `GET /api/health` - Health check with service status
- `GET /api/status` - **[MAIN ENDPOINT]** Comprehensive system status including streaming and recording state

### Camera Actions
- `POST /api/capture` - Take a photo (requires session). Supports `annotated: true` parameter for CV overlay capture.
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
- `POST /api/visualization/pose` - Toggle pose skeleton visualization

### CV Apps (Jetson-specific)
- `POST /api/apps/load` - Load a CV app (e.g., pushup counter)
- `POST /api/apps/activate` - Activate a loaded CV app
- `POST /api/apps/deactivate` - Deactivate current CV app
- `GET /api/apps/status` - Get current app status and state
- `POST /api/apps/competition/start` - Start a competition with recognized users
- `POST /api/apps/competition/end` - End the current competition

### Facial NFT Endpoints (Jetson-specific)
- `POST /api/face/enroll/prepare-transaction` - Prepare facial  transaction (requires session)
- `POST /api/face/enroll/confirm` - Confirm facial NFT transaction (requires session)

### Streaming (Jetson-specific - WebRTC)

**Note:** Video streaming uses WebRTC via Socket.IO signaling, NOT REST API endpoints.

- `GET /api/stream/webrtc/status` - Get P2P WebRTC service status
- `GET /api/stream/whip/status` - Get WHIP publisher status (for WHEP fallback)
- `GET /api/stream/info` - Get all available stream URLs and connection info

#### WebRTC Architecture
```
Frontend ──Socket.IO──> Backend Signaling Server ──Socket.IO──> Jetson Camera
              │
              └── Events: register-viewer, webrtc-offer, webrtc-answer, webrtc-ice-candidate
```

#### Stream Types (Dual Stream)
- **Clean stream**: Raw video without CV annotations (default)
- **Annotated stream**: Video with face boxes, skeletons, app overlays

#### Connection Methods
1. **P2P WebRTC** (lowest latency, local network) - via Socket.IO signaling
2. **WHEP Fallback** (remote viewing) - direct connection to MediaMTX server

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

### Capture Request
```json
{
  "wallet_address": "string",
  "annotated": false,
  "event_metadata": {
    "app_id": "pushup",
    "event_type": "pushup_peak",
    "rep": 5
  }
}
```

**Parameters:**
- `wallet_address` (required): Wallet address for the capture
- `annotated` (optional, default: false): If true, capture with CV annotations (face boxes, skeletons, app overlays)
- `event_metadata` (optional): App-provided context stored with the capture

### Capture Response
```json
{
  "success": true,
  "path": "/data/images/capture_123.jpg",
  "timestamp": 1234567890,
  "annotated": false,
  "event_metadata": null
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

### Stream Info Response
```json
{
  "success": true,
  "camera_pda": "your-camera-pda",
  "streams": {
    "clean": {
      "description": "Raw video stream without CV annotations",
      "p2p": {
        "available": true,
        "signaling": "Socket.IO via backend"
      },
      "whep": {
        "url": "http://mediamtx-server:8889/{camera_pda}/",
        "available": true
      }
    },
    "annotated": {
      "description": "Video stream with CV annotations (face boxes, skeletons, app overlays)",
      "p2p": {
        "available": true,
        "signaling": "Socket.IO via backend"
      },
      "whep": {
        "url": "http://mediamtx-server:8889/{camera_pda}-annotated/",
        "available": true
      }
    }
  },
  "default_stream": "clean"
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

// 4. STREAMING (WebRTC via Socket.IO - NOT REST API)
// P2P WebRTC streaming uses Socket.IO signaling, not REST endpoints.
// The frontend connects to the backend signaling server which relays to the camera.
//
// Socket.IO Events (sent to backend signaling server):
//   - 'register-viewer': { viewerId, streamType: 'clean' | 'annotated' }
//   - 'viewer-wants-connection': { viewerId, streamType, cellularMode }
//   - 'webrtc-answer': { viewerId, answer }
//   - 'webrtc-ice-candidate': { viewerId, candidate }
//
// Socket.IO Events (received from backend):
//   - 'webrtc-offer': { viewerId, offer }
//   - 'webrtc-ice-candidate': { viewerId, candidate }
//
// Stream types:
//   - 'clean': Raw video without CV annotations (default)
//   - 'annotated': Video with face boxes, skeletons, app overlays
//
// For WHEP fallback (remote viewing), use the URLs from /api/stream/info

// Get available stream URLs
const streamInfo = await fetch('/api/stream/info').then(r => r.json());

// 5. COMPUTER VISION (Jetson Only)
const gesture = await fetch('/api/gesture/current').then(r => r.json());

const faceEnroll = await fetch('/api/face/enroll', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ wallet_address: 'your_wallet', session_id: 'session_id' })
}).then(r => r.json());

// Toggle pose visualization
await fetch('/api/visualization/pose', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ enabled: true })
});

// 6. CV APPS (Jetson Only)
// Load pushup counter app
await fetch('/api/apps/load', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ app_name: 'pushup' })
});

// Activate pushup counter app (starts processing frames, shows skeleton)
await fetch('/api/apps/activate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ app_name: 'pushup' })
});

// Start competition (automatically uses currently recognized users)
await fetch('/api/apps/competition/start', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    competitors: [
      { wallet_address: 'user_wallet', display_name: 'User Name' }
    ],
    duration_limit: 300  // Optional: time limit in seconds
  })
});

// Check app status (poll this to get live rep counts)
const appStatus = await fetch('/api/apps/status').then(r => r.json());
/* Returns:
{
  success: true,
  active_app: 'pushup',
  loaded_apps: ['pushup'],
  state: {
    active: true,  // Competition running
    competitors: [
      {
        wallet_address: '...',
        display_name: '...',
        stats: {
          reps: 7,
          in_down_position: false,
          current_angle: 165,
          view: 'left',  // 'front', 'left', or 'right'
          last_rep_time: 1234567890
        },
        track_id: 123
      }
    ],
    elapsed: 45.2,  // Seconds elapsed
    time_remaining: 254.8  // Seconds remaining (if duration_limit set)
  }
}
*/

// End competition
await fetch('/api/apps/competition/end', { method: 'POST' });

// Deactivate app (stops processing entirely)
await fetch('/api/apps/deactivate', { method: 'POST' });

// 7. MEDIA ACCESS
const photos = await fetch('/api/photos').then(r => r.json());
const videos = await fetch('/api/videos').then(r => r.json());

// 8. PHOTO CAPTURE (with optional CV annotations)
// Clean capture (default - raw frame without CV overlays)
const cleanCapture = await fetch('/api/capture', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    wallet_address: 'your_wallet'
  })
}).then(r => r.json());

// Annotated capture (with face boxes, skeletons, app overlays)
const annotatedCapture = await fetch('/api/capture', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    wallet_address: 'your_wallet',
    annotated: true,
    event_metadata: {
      app_id: 'pushup',
      event_type: 'rep_complete',
      rep: 10
    }
  })
}).then(r => r.json());
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