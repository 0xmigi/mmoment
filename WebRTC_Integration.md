# WebRTC Integration

## Overview
Added WebRTC streaming capability to reduce latency from 6-15 seconds (RTMP→HLS) to sub-second streaming.

## Components Added

### Backend (app/backend/src/index.ts)
- **WebRTC signaling server** using existing Socket.IO infrastructure
- Handles peer-to-peer connection setup between Jetson cameras and web viewers
- Events: `register-camera`, `register-viewer`, `webrtc-offer`, `webrtc-answer`, `webrtc-ice-candidate`

### Frontend (app/web/src/media/)
- **WebRTCStreamPlayer.tsx** - New component for WebRTC video streaming
- **Modified StreamPlayer.tsx** - Now tries WebRTC first, falls back to Livepeer
- Uses browser WebRTC APIs for direct peer-to-peer video

## Architecture
```
User Browser ←→ Your Backend (signaling) ←→ Jetson Camera
     ↓                                        ↑
     └────── Direct WebRTC Video ──────────────┘
```

## How It Works
1. **User visits stream**: Frontend tries WebRTC connection
2. **Signaling**: Backend coordinates initial handshake 
3. **Direct connection**: Video flows peer-to-peer (bypasses backend)
4. **Fallback**: If WebRTC fails, automatically switches to Livepeer HLS

## Next Steps (for Jetson)
On your Jetson, you'll need to add a WebRTC server that:
1. Connects to your backend signaling server
2. Registers as a camera
3. Streams video directly to browsers via WebRTC

## Benefits
- **Sub-second latency** vs 6-15 seconds with RTMP→HLS
- **Same user experience** - automatic fallback to existing Livepeer
- **No infrastructure changes** - uses your existing backend
- **Battery/bandwidth efficient** - direct peer-to-peer after initial setup

## Testing
1. Start your backend: `cd app/backend && yarn dev`
2. Start your frontend: `cd app/web && yarn dev`
3. WebRTC will attempt to connect, fall back to Livepeer if no Jetson WebRTC server

The UI shows connection status and allows switching between WebRTC/HLS modes.