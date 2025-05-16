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
- **Description**: List all captured photos
- **Parameters**:
  - `wallet_address` (string, optional): Filter by wallet address
  - `page` (integer, optional): Page number for pagination
  - `limit` (integer, optional): Number of items per page
- **Response**:
  ```json
  {
    "success": true,
    "photos": [
      {
        "filename": "photo_1234567890.jpg",
        "path": "/path/to/photos/photo_1234567890.jpg",
        "timestamp": 1234567890,
        "wallet_address": "user-wallet-address",
        "url": "/photos/photo_1234567890.jpg"
      }
    ],
    "count": 1,
    "page": 1,
    "total_pages": 1
  }
  ```

### List Videos
- **Endpoint**: `/list_videos`
- **Method**: GET
- **Description**: List all recorded videos
- **Parameters**:
  - `wallet_address` (string, optional): Filter by wallet address
  - `page` (integer, optional): Page number for pagination
  - `limit` (integer, optional): Number of items per page
- **Response**:
  ```json
  {
    "success": true,
    "videos": [
      {
        "filename": "video_1234567890.mp4",
        "path": "/path/to/videos/video_1234567890.mp4",
        "timestamp": 1234567890,
        "duration": 10.5,
        "size": 1024000,
        "wallet_address": "user-wallet-address",
        "url": "/videos/video_1234567890.mp4"
      }
    ],
    "count": 1,
    "page": 1,
    "total_pages": 1
  }
  ```

## Camera Controls

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
    "enabled": true
  }
  ```

### Toggle Face Visualization
- **Endpoint**: `/toggle_face_visualization`
- **Method**: POST
- **Description**: Enable or disable face box visualization
- **Parameters**:
  - `enabled` (boolean, required): Whether to enable face box visualization
- **Response**:
  ```json
  {
    "success": true,
    "enabled": true
  }
  ```

### Reset Camera
- **Endpoint**: `/camera/reset`
- **Method**: POST
- **Description**: Reset the camera if it's not responding
- **Parameters**: None
- **Response**:
  ```json
  {
    "success": true,
    "message": "Camera reset successfully"
  }
  ```

### Camera Diagnostics
- **Endpoint**: `/camera/diagnostics`
- **Method**: GET
- **Description**: Get camera diagnostic information
- **Parameters**: None
- **Response**:
  ```json
  {
    "success": true,
    "device": "/dev/video1",
    "resolution": "1280x720",
    "fps": 30,
    "buffer_size": 3,
    "active": true,
    "reconnect_attempts": 0,
    "frame_counter": 1234
  }
  ```

## API Testing

### Health Check
- **Endpoint**: `/health`
- **Method**: GET
- **Description**: Check if the camera service is healthy
- **Parameters**: None
- **Response**:
  ```json
  {
    "status": "ok",
    "buffer": {
      "running": true,
      "fps": 29.8,
      "buffer_size": 30,
      "frame_count": 1234
    },
    "face_detection": {
      "enabled": true,
      "model": "facenet",
      "visualize": true
    },
    "gesture_detection": {
      "enabled": true,
      "visualize": true
    },
    "camera_device": "/dev/video1"
  }
  ```

### Version Information
- **Endpoint**: `/version`
- **Method**: GET
- **Description**: Get API version information
- **Parameters**: None
- **Response**:
  ```json
  {
    "api": "Camera Service API",
    "version": "1.0.0",
    "build_date": "2023-01-01",
    "git_hash": "abc123"
  }
  ``` 