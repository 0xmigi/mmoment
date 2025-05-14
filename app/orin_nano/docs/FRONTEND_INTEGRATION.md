# Jetson Camera API - Frontend Integration Guide

This document provides information on how to integrate with the Jetson Camera API via the Cloudflare tunnel.

## Connection Information

- **Production URL**: `https://jetson.mmoment.xyz`
- **Camera PDA**: `WT9oJrL7sbNip8Rc2w5LoWFpwsUcZZJnnjE2zZjMuvD`
- **Program ID**: `Hx5JaUCZXQqvcYzTcdgm9ZE3sqhMWqwAhNXZBrzWm45S`

## Endpoint Status

Below is the current status of all endpoints:

| Endpoint | Status | Notes |
|----------|--------|-------|
| `/health` | ✅ Working | Returns API status information |
| `/connect` | ✅ Working | Successfully creates sessions |
| `/stream` | ✅ Working | Returns MJPEG stream from camera |
| `/current_gesture` | ✅ Working | Returns current detected gesture |
| `/test-stream` | ✅ Working | HTML test page for stream |
| `/toggle_face_detection` | ✅ Working | Toggles face detection visualization |
| `/toggle_face_visualization` | ✅ Working | Toggles face box visualization |
| `/toggle_gesture_visualization` | ✅ Working | Toggles gesture visualization |
| `/recognize_face` | ✅ Working | Recognizes faces with reduced requirements |
| `/detect_gesture` | ✅ Working | Detects gestures with reduced requirements |
| `/enroll_face` | ✅ Working | Enrolls faces for recognition |
| `/capture_moment` | ✅ Working | Captures camera image, optional NFT minting |

## Endpoint Format Notice

**IMPORTANT**: All endpoints are now available in both **kebab-case** (with hyphens) and **snake_case** (with underscores). For example:
- `/toggle-face-detection` and `/toggle_face_detection` both work
- `/recognize-face` and `/recognize_face` both work

For frontend development, we recommend using the **snake_case** versions (with underscores) as they tend to have fewer issues with CORS in browser environments.

## API Test Page

A comprehensive API test page is available at:
```
https://jetson.mmoment.xyz/api-test
```

This page allows you to test all available endpoints directly in your browser, including:
- Visualization controls (face detection, face boxes, gesture tags)
- Wallet connection
- Face enrollment and recognition
- Gesture detection
- Capturing moments

Use this page to verify that all endpoints are working correctly through the Cloudflare tunnel.

## Available Endpoints

The following endpoints are available through the Cloudflare tunnel:

### General Endpoints

#### `GET /`
Returns basic information about the API.

**Example Response:**
```json
{
  "name": "Jetson Camera API Bridge",
  "description": "API bridge for Jetson camera service",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "connect": "/connect",
    "disconnect": "/disconnect",
    "enroll-face": "/enroll-face",
    "recognize-face": "/recognize-face",
    "detect-gesture": "/detect-gesture",
    "capture-moment": "/capture-moment",
    "stream": "/stream"
  },
  "camera_pda": "WT9oJrL7sbNip8Rc2w5LoWFpwsUcZZJnnjE2zZjMuvD"
}
```

#### `GET /health`
Checks the health of the camera API and solana middleware.

**Example Response:**
```json
{
  "status": "ok",
  "camera_service": "ok",
  "solana_middleware": "skipped",
  "camera_pda": "WT9oJrL7sbNip8Rc2w5LoWFpwsUcZZJnnjE2zZjMuvD",
  "program_id": "Hx5JaUCZXQqvcYzTcdgm9ZE3sqhMWqwAhNXZBrzWm45S",
  "active_sessions": 0
}
```

#### `GET /test-stream`
A test page to verify camera streaming functionality.

### Streaming Endpoints

#### `GET /stream`
Returns a live MJPEG stream from the camera. This endpoint should be embedded directly in an `<img>` tag.

**HTML Example:**
```html
<img src="https://jetson.mmoment.xyz/stream" alt="Camera Stream">
```

### Session Management Endpoints

#### `POST /connect`
Connects a wallet to the camera.

**Request Body:**
```json
{
  "wallet_address": "YourWalletAddress123"
}
```

**Example Response:**
```json
{
  "success": true,
  "session_id": "c1d3c7db33d543f8",
  "camera_pda": "WT9oJrL7sbNip8Rc2w5LoWFpwsUcZZJnnjE2zZjMuvD",
  "message": "Connected to camera successfully"
}
```

#### `POST /disconnect`
Disconnects a wallet from the camera.

**Request Body:**
```json
{
  "wallet_address": "YourWalletAddress123",
  "session_id": "1234567890abcdef"
}
```

**Example Response:**
```json
{
  "success": true,
  "message": "Disconnected from camera successfully"
}
```

### Facial Recognition Endpoints

#### `POST /enroll-face`
Enrolls a face for facial recognition.

**Request Body:**
```json
{
  "wallet_address": "YourWalletAddress123",
  "session_id": "1234567890abcdef"
}
```

**Example Response:**
```json
{
  "success": true,
  "face_image_url": "base64_encoded_image_data",
  "message": "Face enrolled successfully"
}
```

#### `POST /recognize-face`
Recognizes a face against enrolled users.

**Request Body:**
```json
{
  "wallet_address": "YourWalletAddress123"
}
```

**Example Response:**
```json
{
  "success": true,
  "recognized": true,
  "confidence": 0.95,
  "face_image_url": "base64_encoded_image_data"
}
```

### Gesture Detection Endpoints

#### `POST /detect-gesture`
Detects a gesture from the camera.

**Request Body:**
```json
{
  "wallet_address": "YourWalletAddress123",
  "session_id": "1234567890abcdef"
}
```

**Example Response:**
```json
{
  "success": true,
  "gesture": "thumbs_up",
  "confidence": 0.89,
  "image_url": "base64_encoded_image_data"
}
```

#### `GET /current_gesture`
Returns the current gesture detected by the camera (polling endpoint).

**Example Response:**
```json
{
  "success": true,
  "gesture": "none",
  "confidence": 0,
  "timestamp": 1746819043780
}
```

### Capture Endpoints

#### `POST /capture-moment`
Captures and mints a moment as NFT.

**Request Body:**
```json
{
  "wallet_address": "YourWalletAddress123",
  "session_id": "1234567890abcdef"
}
```

**Example Response:**
```json
{
  "success": true,
  "image_url": "base64_encoded_image_data",
  "nft_data": {
    "mint": "NFTMintAddress123",
    "metadata": "MetadataAddress123"
  },
  "message": "Moment captured and minted successfully"
}
```

### Visualization Control Endpoints

#### `POST /toggle-face-detection`
Toggles face detection on/off.

**Request Body:**
```json
{
  "enabled": true
}
```

#### `POST /toggle-face-visualization`
Toggles face visualization on/off.

**Request Body:**
```json
{
  "enabled": true
}
```

#### `POST /toggle-gesture-visualization`
Toggles gesture visualization on/off.

**Request Body:**
```json
{
  "enabled": true
}
```

## Error Handling

All API endpoints return appropriate HTTP status codes:
- `200 OK`: Request was successful
- `400 Bad Request`: Invalid request parameters
- `403 Forbidden`: Invalid session or unauthorized
- `500 Internal Server Error`: Server-side error

Error responses contain a JSON object with an `error` field containing a description of the error.

Example error response:
```json
{
  "error": "Invalid session"
}
```

## Troubleshooting

### Common Issues

1. **Stream not loading**: 
   - Make sure the camera service is running
   - Try accessing the test stream page at `/test-stream`
   - Check the browser console for CORS errors

2. **Connection errors**: 
   - Verify the Solana middleware is running
   - Make sure the wallet address is valid

3. **Face recognition issues**:
   - Make sure a face is properly enrolled
   - Check lighting conditions and camera positioning

### Testing the Connection

You can verify the camera API is accessible by visiting:
- `https://jetson.mmoment.xyz/health`
- `https://jetson.mmoment.xyz/test-stream`

If both pages load successfully, the API bridge is functioning correctly.

## Services Status

The following services are now configured as systemd services and will automatically start on boot:

1. **Camera API Service**: `camera-service.service`
2. **Solana Middleware**: `solana-middleware.service`
3. **Frontend Bridge**: `frontend-bridge.service`
4. **Cloudflare Tunnel**: `cloudflared-compat.service`

To restart all services, you can use:
```bash
sudo systemctl restart camera-service.service solana-middleware.service frontend-bridge.service cloudflared-compat.service
```

Or run the provided startup script:
```bash
sudo ./start_services.sh
```

## Additional Notes

- All services are configured to restart automatically if they crash
- The streaming endpoint uses MJPEG format which is compatible with most browsers
- Face enrollment requires a clear view of the face for best results
- Gestures are detected best when performed clearly in the camera's field of view 