# Camera Service API Endpoints

This document describes the available API endpoints for the Camera Service.

## User Session Management

### User Connect
- **Endpoint**: `/connect`
- **Method**: POST
- **Description**: Connect a wallet to the camera, creating a new session
- **Parameters**:
  - `wallet_address` (string, required): The user's wallet address
- **Response**:
  ```json
  {
    "success": true,
    "session_id": "session-uuid-string", 
    "wallet_address": "user-wallet-address",
    "created_at": 1234567890,
    "expires_at": 1234657890
  }
  ```

### User Disconnect
- **Endpoint**: `/disconnect`
- **Method**: POST
- **Description**: Disconnect a wallet from the camera, ending the session
- **Parameters**:
  - `wallet_address` (string, required): The user's wallet address
  - `session_id` (string, required): The session ID from the connect request
- **Response**:
  ```json
  {
    "success": true,
    "message": "Disconnected from camera successfully"
  }
  ```

## Face Recognition

### Enroll Face
- **Endpoint**: `/enroll_face`
- **Method**: POST
- **Description**: Enroll the user's face for future recognition
- **Parameters**:
  - `wallet_address` (string, required): The user's wallet address
  - `session_id` (string, required): The session ID from the connect request
- **Response**:
  ```json
  {
    "success": true,
    "wallet_address": "user-wallet-address",
    "include_image": true,
    "image": "base64-encoded-image",
    "encrypted": false,
    "nft_verified": false,
    "message": "Face enrolled successfully"
  }
  ```

### Recognize Face
- **Endpoint**: `/recognize_face`
- **Method**: POST
- **Description**: Recognize faces in the current camera view
- **Parameters**: None required
- **Response**:
  ```json
  {
    "success": true,
    "detected_count": 1,
    "recognized_count": 1,
    "recognized_users": [
      {
        "wallet_address": "user-wallet-address",
        "confidence": 0.95,
        "box": [100, 100, 200, 200]
      }
    ],
    "include_image": false
  }
  ```

### List Enrolled Faces
- **Endpoint**: `/get_enrolled_faces`
- **Method**: GET
- **Description**: Get a list of all enrolled faces
- **Parameters**: None
- **Response**:
  ```json
  {
    "success": true,
    "faces": ["wallet-address-1", "wallet-address-2"],
    "count": 2
  }
  ```

### Clear All Faces
- **Endpoint**: `/clear_enrolled_faces`
- **Method**: POST
- **Description**: Clear all enrolled faces from the system
- **Parameters**: None
- **Response**:
  ```json
  {
    "success": true,
    "message": "All faces cleared successfully"
  }
  ```

## Gesture Detection

### Current Gesture
- **Endpoint**: `/current_gesture`
- **Method**: GET
- **Description**: Get the current detected gesture
- **Parameters**: None
- **Response**:
  ```json
  {
    "success": true,
    "gesture": "thumbs_up",
    "confidence": 0.85,
    "timestamp": 1234567890
  }
  ```

### Toggle Gesture Visualization
- **Endpoint**: `/toggle_gesture_visualization`
- **Method**: POST
- **Description**: Enable or disable gesture visualization
- **Parameters**:
  - `enabled` (boolean, required): Whether to enable gesture visualization
- **Response**:
  ```json
  {
    "success": true,
    "enabled": true
  }
  ```

## Media Capture

### Manual Photo Capture
- **Endpoint**: `/capture_moment`
- **Method**: POST
- **Description**: Manually capture a photo
- **Parameters**:
  - `wallet_address` (string, required): The user's wallet address
  - `session_id` (string, required): The session ID from the connect request
- **Response**:
  ```json
  {
    "success": true,
    "filename": "photo_1234567890.jpg",
    "path": "/path/to/photos/photo_1234567890.jpg",
    "timestamp": 1234567890,
    "wallet_address": "user-wallet-address",
    "image_data": "data:image/jpeg;base64,..."
  }
  ```

### Manual Start Video Recording
- **Endpoint**: `/start_recording`
- **Method**: POST
- **Description**: Start recording a video
- **Parameters**:
  - `wallet_address` (string, required): The user's wallet address
  - `session_id` (string, required): The session ID from the connect request
  - `duration` (integer, optional): Recording duration in seconds (0 for unlimited)
- **Response**:
  ```json
  {
    "success": true,
    "recording_id": "recording-uuid",
    "start_time": 1234567890,
    "wallet_address": "user-wallet-address",
    "duration": 10
  }
  ```

### Manual Stop Video Recording
- **Endpoint**: `/stop_recording`
- **Method**: POST
- **Description**: Stop the current video recording
- **Parameters**:
  - `wallet_address` (string, required): The user's wallet address
  - `session_id` (string, required): The session ID from the connect request
- **Response**:
  ```json
  {
    "success": true,
    "filename": "video_1234567890.mp4",
    "path": "/path/to/videos/video_1234567890.mp4",
    "duration": 5.2,
    "size": 1024000,
    "wallet_address": "user-wallet-address"
  }
  ```

## Media Access

### List Photos
- **Endpoint**: `/list_photos`
- **Method**: GET
- **Description**: List available photos
- **Parameters**:
  - `limit` (integer, optional): Maximum number of photos to return (default: 10)
- **Response**:
  ```json
  {
    "success": true,
    "photos": [
      {
        "filename": "photo_1234567890.jpg",
        "path": "/path/to/photos/photo_1234567890.jpg",
        "timestamp": 1234567890,
        "size": 102400,
        "wallet_address": "user-wallet-address"
      }
    ],
    "count": 1
  }
  ```

### List Videos
- **Endpoint**: `/list_videos`
- **Method**: GET
- **Description**: List available videos
- **Parameters**:
  - `limit` (integer, optional): Maximum number of videos to return (default: 10)
- **Response**:
  ```json
  {
    "success": true,
    "videos": [
      {
        "filename": "video_1234567890.mp4",
        "path": "/path/to/videos/video_1234567890.mp4",
        "timestamp": 1234567890,
        "size": 1024000,
        "duration": 5.2,
        "wallet_address": "user-wallet-address"
      }
    ],
    "count": 1
  }
  ```

### Get Photo
- **Endpoint**: `/photos/<filename>`
- **Method**: GET
- **Description**: Get a specific photo by filename
- **Response**: Photo file (image/jpeg)

### Get Video
- **Endpoint**: `/videos/<filename>`
- **Method**: GET
- **Description**: Get a specific video by filename
- **Response**: Video file (video/mp4)

## Visualization Settings

### Toggle Face Detection
- **Endpoint**: `/toggle_face_detection`
- **Method**: POST
- **Description**: Enable or disable face detection
- **Parameters**:
  - `enabled` (boolean, required): Whether to enable face detection
- **Response**:
  ```json
  {
    "success": true,
    "enabled": true,
    "message": "Face detection is always enabled"
  }
  ```

### Toggle Face Visualization
- **Endpoint**: `/toggle_face_visualization`
- **Method**: POST
- **Description**: Enable or disable face visualization
- **Parameters**:
  - `enabled` (boolean, required): Whether to enable face visualization
- **Response**:
  ```json
  {
    "success": true,
    "enabled": true
  }
  ```

### Toggle Face Boxes
- **Endpoint**: `/toggle_face_boxes`
- **Method**: POST
- **Description**: Enable or disable face bounding boxes
- **Parameters**:
  - `enabled` (boolean, required): Whether to enable face boxes
- **Response**:
  ```json
  {
    "success": true,
    "enabled": true
  }
  ```

## Camera Management

### Camera Health Check
- **Endpoint**: `/health`
- **Method**: GET
- **Description**: Get camera service health status
- **Response**:
  ```json
  {
    "status": "ok",
    "buffer_service": "running",
    "buffer_fps": 30,
    "active_sessions": 1,
    "timestamp": 1234567890,
    "camera": {
      "index": 0,
      "preferred_device": "/dev/video0",
      "resolution": "1280x720",
      "target_fps": 30
    }
  }
  ```

### Camera Stream
- **Endpoint**: `/stream`
- **Method**: GET
- **Description**: Stream the camera feed as MJPEG
- **Response**: MJPEG stream

### Camera Diagnostics
- **Endpoint**: `/camera/diagnostics`
- **Method**: GET
- **Description**: Get detailed camera diagnostic information
- **Response**: JSON with camera status

### Camera Reset
- **Endpoint**: `/camera/reset`
- **Method**: POST
- **Description**: Reset the camera connection
- **Response**:
  ```json
  {
    "success": true,
    "message": "Camera reset successful",
    "was_running": true,
    "now_running": true,
    "camera_index": 0,
    "preferred_device": "/dev/video0"
  }
  ``` 