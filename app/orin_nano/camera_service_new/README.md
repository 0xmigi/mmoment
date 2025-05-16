# Camera Service API

This is a lightweight, optimized camera service API for the Jetson Orin Nano. It provides functionality for face recognition, gesture detection, and media capture.

## Features

- User session management (connect/disconnect)
- Face recognition and enrollment
- Gesture detection
- Media capture (photos and videos)
- Camera streaming

## Installation

### Prerequisites

- Python 3.8+
- OpenCV 4.5+
- Flask

### Setup

1. Clone the repository and navigate to the camera service directory:

```bash
cd mmoment/app/orin_nano/camera_service_new
```

2. Run the setup script to install dependencies:

```bash
./setup.sh
```

3. Start the camera service:

```bash
python main.py
```

By default, the server runs on port 5003. You can specify a different port with the `--port` argument:

```bash
python main.py --port 8080
```

## Usage

Once the server is running, you can access the following pages:

- `/` - API information
- `/stream` - Live camera stream
- `/test-page` - General test page
- `/simple-test` - Simple API test page
- `/api-test` - Interactive API test page

## API Endpoints

See the [API_ENDPOINTS.md](API_ENDPOINTS.md) file for a complete list of available API endpoints.

### Quick Reference

- `/connect` - Connect a user session
- `/disconnect` - Disconnect a user session
- `/enroll_face` - Enroll a face for recognition
- `/recognize_face` - Recognize faces in the current frame
- `/get_enrolled_faces` - List all enrolled faces
- `/clear_enrolled_faces` - Clear all enrolled faces
- `/current_gesture` - Get the current detected gesture
- `/toggle_gesture_visualization` - Enable/disable gesture visualization
- `/capture_moment` - Capture a photo
- `/start_recording` - Start video recording
- `/stop_recording` - Stop video recording
- `/list_photos` - List available photos
- `/list_videos` - List available videos

## Testing

To test the API endpoints, you can run the test script:

```bash
python test_api_endpoints.py
```

This will test all the available endpoints and provide a summary of the results.

## Directory Structure

- `main.py` - Main entry point
- `routes.py` - API routes
- `services/` - Service implementations
  - `buffer_service.py` - Camera buffer service
  - `face_service.py` - Face recognition service
  - `gesture_service.py` - Gesture detection service
  - `capture_service.py` - Media capture service
  - `session_service.py` - Session management service
- `models/` - ML models
- `templates/` - HTML templates
- `static/` - Static assets
- `photos/` - Captured photos
- `videos/` - Recorded videos
- `faces/` - Enrolled face data

## Troubleshooting

If you encounter issues with the camera:

1. Check camera connections
2. Try resetting the camera with the `/camera/reset` endpoint
3. Check camera diagnostics with the `/camera/diagnostics` endpoint

For face recognition issues:

1. Ensure proper lighting
2. Position face directly in front of the camera
3. Check face model status with the `/facenet/check` endpoint

## License

This project is proprietary and confidential. 