# NVIDIA Jetson Orin Nano Camera API

A high-performance, gesture-controlled camera system with real-time face recognition, automatic content capture, and blockchain integration.

## 🎯 Overview

This system provides a complete camera API running on NVIDIA Jetson Orin Nano hardware with:

- **Real-time video streaming** at ~10-15 FPS (1280x720)
- **Face recognition** with enrollment and identification
- **Gesture detection** with automatic content capture
- **Photo/video capture** with user attribution
- **Blockchain integration** via Solana middleware
- **RESTful API** for frontend integration

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │  Camera Service │    │ Solana Middle-  │
│   (React/TS)    │◄──►│   (Port 5002)   │◄──►│  ware (5004)    │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │  Hardware       │
                       │  - Camera       │
                       │  - MediaPipe    │
                       │  - OpenCV       │
                       └─────────────────┘
```

### Core Services

1. **Buffer Service**: High-performance frame buffering and processing
2. **Face Service**: MTCNN-based face detection and simple recognition
3. **Gesture Service**: MediaPipe hand gesture recognition
4. **Capture Service**: Photo/video recording with MOV/MP4 output
5. **Session Service**: User session management
6. **Solana Integration**: Blockchain wallet and NFT support

## 🚀 Quick Start for Frontend Integration

### Base URL
```
http://localhost:5002  # When port-forwarded from Jetson
```

### Authentication Flow
```typescript
// 1. Connect wallet to camera
const connectResponse = await fetch('/connect', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ wallet_address: 'your_wallet_address' })
});

const session = await connectResponse.json();
// Returns: { success: true, session_id: "...", wallet_address: "...", camera_pda: "..." }
```

## 📡 API Endpoints

### 🔗 Session Management

#### Connect Wallet
```http
POST /connect
Content-Type: application/json

{
  "wallet_address": "string"
}
```

**Response:**
```json
{
  "success": true,
  "session_id": "uuid",
  "wallet_address": "string",
  "camera_pda": "string",
  "timestamp": 1234567890
}
```

#### Disconnect
```http
POST /disconnect
Content-Type: application/json

{
  "wallet_address": "string",
  "session_id": "string"
}
```

### 📹 Video Streaming

#### Live Stream (MJPEG)
```http
GET /stream
```
Returns continuous MJPEG stream with face boxes and gesture labels when enabled.

### 👤 Face Recognition

#### Enroll Face
```http
POST /enroll_face
Content-Type: application/json

{
  "wallet_address": "string",
  "session_id": "string"
}
```

**Response:**
```json
{
  "success": true,
  "wallet_address": "string",
  "message": "Face enrolled successfully",
  "include_image": true,
  "image": "base64_encoded_image"
}
```

#### Recognize Faces
```http
POST /recognize_face
Content-Type: application/json

{
  "wallet_address": "string",
  "session_id": "string"
}
```

**Response:**
```json
{
  "success": true,
  "detected_faces": 1,
  "recognized_faces": 1,
  "recognized_data": {
    "wallet_address": {
      "confidence": 0.85
    }
  },
  "wallet_recognized": true,
  "wallet_confidence": 0.85,
  "include_image": true,
  "image": "base64_encoded_image"
}
```

#### List Enrolled Faces
```http
GET /get_enrolled_faces
```

#### Clear All Faces
```http
POST /clear_enrolled_faces
```

### ✋ Gesture Recognition

#### Get Current Gesture
```http
GET /current_gesture
```

**Response:**
```json
{
  "success": true,
  "gesture": "peace|thumbs_up|palm|none",
  "confidence": 0.85,
  "timestamp": 1234567890
}
```

#### Toggle Gesture Visualization
```http
POST /toggle_gesture_visualization
Content-Type: application/json

{
  "enabled": true
}
```

### 📸 Content Capture

#### Capture Photo
```http
POST /capture_moment
Content-Type: application/json

{
  "wallet_address": "string",
  "session_id": "string"
}
```

**Response:**
```json
{
  "success": true,
  "path": "/path/to/photo.jpg",
  "filename": "wallet_photo_timestamp_id.jpg",
  "timestamp": 1234567890,
  "size": 123456,
  "width": 1280,
  "height": 720,
  "image_data": "data:image/jpeg;base64,..."
}
```

#### Start Recording
```http
POST /start_recording
Content-Type: application/json

{
  "wallet_address": "string",
  "session_id": "string",
  "duration": 10  // Optional: seconds, 0 = until stopped
}
```

#### Stop Recording
```http
POST /stop_recording
Content-Type: application/json

{
  "wallet_address": "string",
  "session_id": "string"
}
```

**Response:**
```json
{
  "success": true,
  "path": "/path/to/video.mov",
  "filename": "wallet_video_timestamp_id.mov",
  "size": 985271,
  "recording": false
}
```

### 📁 Media Access

#### List Photos
```http
GET /list_photos?limit=10
```

#### List Videos
```http
GET /list_videos?limit=10
```

#### Access Media Files
```http
GET /photos/{filename}     # Returns JPEG image
GET /videos/{filename}     # Returns MOV/MP4 video
```

**Note:** Video files are automatically served as MP4 when available for better browser compatibility.

### 🎛️ Visualization Controls

#### Toggle Face Boxes
```http
POST /toggle_face_visualization
Content-Type: application/json

{
  "enabled": true
}
```

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "buffer_service": "healthy",
  "buffer_fps": 10.5,
  "active_sessions": 1,
  "timestamp": 1234567890,
  "camera": {
    "index": 1,
    "preferred_device": "/dev/video1",
    "resolution": "1280x720",
    "target_fps": 15
  }
}
```

## 🎮 Gesture-to-Action System

The system supports automatic content capture based on gestures:

| Gesture | Action | Confidence Threshold |
|---------|--------|---------------------|
| ✌️ Peace Sign | 📸 Capture Photo | 70% |
| 👍 Thumbs Up | 🎥 Start Recording | 70% |
| 🖐️ Open Palm | ⏹️ Stop Recording | 70% |

### Gesture Polling Example
```typescript
// Enable gesture recognition
await fetch('/toggle_gesture_visualization', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ enabled: true })
});

// Poll for gestures
const pollGestures = async () => {
  const response = await fetch('/current_gesture');
  const data = await response.json();
  
  if (data.gesture !== 'none' && data.confidence > 0.7) {
    console.log(`Detected: ${data.gesture} (${data.confidence})`);
    // Handle gesture-based actions
  }
};

setInterval(pollGestures, 500); // Poll every 500ms
```

## 🔧 Frontend Integration Patterns

### React Hook Example
```typescript
import { useState, useEffect } from 'react';

interface CameraSession {
  sessionId: string;
  walletAddress: string;
  cameraPda: string;
}

export const useCameraSession = (walletAddress: string) => {
  const [session, setSession] = useState<CameraSession | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  const connect = async () => {
    try {
      const response = await fetch('http://localhost:5002/connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ wallet_address: walletAddress })
      });
      
      const data = await response.json();
      if (data.success) {
        setSession({
          sessionId: data.session_id,
          walletAddress: data.wallet_address,
          cameraPda: data.camera_pda
        });
        setIsConnected(true);
      }
    } catch (error) {
      console.error('Camera connection failed:', error);
    }
  };

  const disconnect = async () => {
    if (!session) return;
    
    try {
      await fetch('http://localhost:5002/disconnect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          wallet_address: session.walletAddress,
          session_id: session.sessionId
        })
      });
      
      setSession(null);
      setIsConnected(false);
    } catch (error) {
      console.error('Camera disconnection failed:', error);
    }
  };

  return { session, isConnected, connect, disconnect };
};
```

### Video Stream Component
```typescript
import React, { useRef, useEffect } from 'react';

interface CameraStreamProps {
  isConnected: boolean;
}

export const CameraStream: React.FC<CameraStreamProps> = ({ isConnected }) => {
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    if (isConnected && imgRef.current) {
      imgRef.current.src = `http://localhost:5002/stream?t=${Date.now()}`;
    }
  }, [isConnected]);

  return (
    <div className="camera-stream">
      {isConnected ? (
        <img
          ref={imgRef}
          alt="Camera stream"
          style={{ maxWidth: '100%', borderRadius: '8px' }}
          onError={() => console.error('Stream failed to load')}
        />
      ) : (
        <div className="stream-placeholder">
          Connect to camera to view stream
        </div>
      )}
    </div>
  );
};
```

### Gesture Detection Service
```typescript
export class GestureDetectionService {
  private polling: NodeJS.Timeout | null = null;
  private onGestureCallback?: (gesture: string, confidence: number) => void;

  start(callback: (gesture: string, confidence: number) => void) {
    this.onGestureCallback = callback;
    
    // Enable gesture visualization
    fetch('http://localhost:5002/toggle_gesture_visualization', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: true })
    });

    // Start polling
    this.polling = setInterval(async () => {
      try {
        const response = await fetch('http://localhost:5002/current_gesture');
        const data = await response.json();
        
        if (data.success && data.gesture !== 'none' && data.confidence > 0.7) {
          this.onGestureCallback?.(data.gesture, data.confidence);
        }
      } catch (error) {
        console.error('Gesture polling error:', error);
      }
    }, 500);
  }

  stop() {
    if (this.polling) {
      clearInterval(this.polling);
      this.polling = null;
    }
    
    // Disable gesture visualization
    fetch('http://localhost:5002/toggle_gesture_visualization', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: false })
    });
  }
}
```

## 🔒 Security & Session Management

- **Session Validation**: All capture endpoints require valid session
- **3-Second Cooldown**: Gesture actions have built-in cooldown to prevent spam
- **Wallet Attribution**: All media is tagged with wallet address
- **Automatic Cleanup**: Old media files are automatically cleaned up

## 📊 Performance Characteristics

- **Video Stream**: ~10-15 FPS at 1280x720
- **Face Detection**: MTCNN with OpenCV fallback
- **Gesture Recognition**: MediaPipe hands with 70% confidence threshold
- **Recording Format**: MOV (primary) with MP4 conversion for web compatibility
- **Photo Format**: JPEG with 95% quality
- **Session Timeout**: Configurable (default: no timeout)

## 🛠️ Development & Testing

### Test Interface
Access the built-in test interface at:
```
http://localhost:5002/direct-test
```

This provides a complete testing environment for all API endpoints.

### Port Forwarding (Development)
```bash
ssh -L 5002:localhost:5002 azuolas@192.168.1.232
```

### Service Management
```bash
# Check status
sudo systemctl status camera-service

# Restart service
sudo systemctl restart camera-service

# View logs
journalctl -u camera-service -f
```

## 🔄 Integration with Existing Frontend

Based on the existing `/app/web` structure, integrate with:

1. **Camera Provider** (`src/camera/CameraProvider.tsx`)
2. **Camera Service** (`src/camera/camera-service.ts`)
3. **Camera Client** (`src/camera/camera-client.ts`)

The API endpoints are designed to work seamlessly with the existing camera service architecture.

## 📝 Notes

- **Video Playback**: MOV files are automatically served as MP4 when available for better browser compatibility
- **Gesture Sensitivity**: Adjust confidence thresholds in gesture service if needed
- **Face Recognition**: Uses simple pixel-based matching - works best with consistent lighting
- **Media Storage**: Files are stored locally and cleaned up automatically based on limits

---

**Ready for frontend integration!** 🚀

The gesture-controlled capture system is fully functional and ready to be integrated with your React frontend. 