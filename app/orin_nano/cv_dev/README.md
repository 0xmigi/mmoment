# CV App Development Environment

Development environment for building and testing CV apps using pre-recorded video instead of live camera feeds.

## Quick Start

### 1. Get a Test Video

```bash
# Install yt-dlp if not already installed
pip install yt-dlp

# Download a sample video (e.g., pushup tutorial)
cd /mnt/nvme/mmoment/app/orin_nano/cv_dev/videos/
yt-dlp -f "best[height<=720]" "https://www.youtube.com/watch?v=VIDEO_ID" -o pushups.mp4
```

### 2. Start in Dev Mode

```bash
cd /mnt/nvme/mmoment/app/orin_nano

# Option A: With video pre-loaded
CV_DEV_MODE=true CV_DEV_VIDEO=/app/cv_dev/videos/pushups.mp4 docker-compose up camera-service

# Option B: Load video via API after startup
CV_DEV_MODE=true docker-compose up camera-service
```

### 3. Control Playback

```bash
# List available videos
curl localhost:5002/api/dev/videos

# Load a video
curl -X POST localhost:5002/api/dev/load \
  -H "Content-Type: application/json" \
  -d '{"path": "pushups.mp4"}'

# Play/Pause
curl -X POST localhost:5002/api/dev/playback/play
curl -X POST localhost:5002/api/dev/playback/pause

# Seek to frame
curl -X POST localhost:5002/api/dev/playback/seek \
  -H "Content-Type: application/json" \
  -d '{"frame": 100}'

# Seek to time (seconds)
curl -X POST localhost:5002/api/dev/playback/seek \
  -H "Content-Type: application/json" \
  -d '{"time": 5.5}'

# Set playback speed (0.5 = half, 2.0 = double)
curl -X POST localhost:5002/api/dev/playback/speed \
  -H "Content-Type: application/json" \
  -d '{"speed": 0.5}'

# Step forward/backward one frame
curl -X POST localhost:5002/api/dev/playback/step \
  -H "Content-Type: application/json" \
  -d '{"direction": "forward"}'

# Get current state
curl localhost:5002/api/dev/playback/state

# Get help
curl localhost:5002/api/dev/help
```

## API Reference

### Video Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/dev/status` | GET | Get dev environment status |
| `/api/dev/videos` | GET | List videos in videos/ directory |
| `/api/dev/load` | POST | Load a video file |

### Playback Controls

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/api/dev/playback/state` | GET | - | Get playback state |
| `/api/dev/playback/play` | POST | - | Resume playback |
| `/api/dev/playback/pause` | POST | - | Pause playback |
| `/api/dev/playback/toggle` | POST | - | Toggle play/pause |
| `/api/dev/playback/seek` | POST | `{frame: N}` or `{time: S}` | Seek to position |
| `/api/dev/playback/speed` | POST | `{speed: 1.0}` | Set playback speed |
| `/api/dev/playback/loop` | POST | `{enabled: true}` | Enable/disable looping |
| `/api/dev/playback/step` | POST | `{direction: "forward"}` | Step one frame |
| `/api/dev/restart` | POST | - | Restart from beginning |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CV_DEV_MODE` | `false` | Enable dev mode (video instead of camera) |
| `CV_DEV_VIDEO` | - | Path to initial video file to load |

## Directory Structure

```
cv_dev/
├── __init__.py              # Module exports
├── video_buffer_service.py  # Drop-in replacement for BufferService
├── routes.py                # Flask blueprint with /api/dev/* endpoints
├── README.md                # This file
└── videos/                  # Store your test videos here
    └── .gitkeep
```

## How It Works

1. **VideoBufferService** replaces the camera's BufferService
2. It reads frames from a video file instead of `/dev/video0`
3. All other services (pose detection, face recognition, CV apps) work unchanged
4. You get playback controls to pause, seek, and step through frames

## Tips

### Finding Good Test Videos

- **YouTube workout videos** - pushups, pullups, squats, etc.
- **Sports highlights** - basketball, soccer, etc.
- **Dance videos** - for pose/gesture detection

### Efficient Iteration

```bash
# Max speed processing (no frame delay)
curl -X POST localhost:5002/api/dev/playback/speed \
  -H "Content-Type: application/json" \
  -d '{"speed": 0}'

# Slow motion for debugging
curl -X POST localhost:5002/api/dev/playback/speed \
  -H "Content-Type: application/json" \
  -d '{"speed": 0.25}'
```

### Viewing Output

The annotated stream is available at the same endpoints as live camera:
- WebRTC: Use existing frontend connection
- WHIP/WHEP: If enabled, stream includes CV annotations

## Roadmap

- **V1 (Current)**: Video playback with API controls
- **V2**: Mock user registry for identity testing
- **V3**: Developer SDK for external contributors
