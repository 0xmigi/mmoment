# Media Management & Cleanup System

This document explains how the Jetson camera system manages media files to prevent storage buildup and avoid committing large files to Git.

## 🚫 Git Ignore Rules

### Comprehensive Media Exclusion
All media files are automatically excluded from Git commits:

```gitignore
# Camera Service Media Files
app/orin_nano/camera_service/photos/
app/orin_nano/camera_service/videos/
app/orin_nano/camera_service/faces/
app/orin_nano/camera_service/recordings/

# All media file extensions
*.jpg *.jpeg *.png *.gif *.bmp *.tiff *.webp
*.mov *.mp4 *.avi *.mkv *.webm *.flv *.wmv *.m4v *.3gp
```

### Allowed Exceptions
Documentation images are still allowed:
- `README*.jpg/png`
- `docs/**/*.jpg/png`
- `*.example.jpg/png`

## 🧹 Automatic Cleanup System

### Built-in Service Cleanup
The `CaptureService` automatically manages storage:

- **Photos**: Max 100 files, cleanup at 90 files
- **Videos**: Max 20 files, cleanup at 18 files
- **Threshold**: 90% capacity triggers cleanup
- **Method**: Deletes oldest files first

### External Cleanup Script
Location: `app/orin_nano/scripts/cleanup_media.sh`

**Configuration:**
```bash
KEEP_PHOTOS=20      # Keep newest 20 photos
KEEP_VIDEOS=10      # Keep newest 10 videos  
KEEP_FACES=50       # Keep newest 50 face embeddings
```

**Cleanup Rules:**
- **By Count**: Keeps only the newest N files
- **By Age**: Removes files older than 7 days (photos/videos) or 30 days (faces)
- **Logs**: Cleanup activity logged to `camera_service/logs/cleanup.log`

### Automatic Scheduling
**Systemd Timer**: Runs cleanup every hour
```bash
# Check timer status
sudo systemctl status media-cleanup.timer

# View cleanup logs
sudo journalctl -u media-cleanup.service
```

## 📁 Directory Structure

```
camera_service/
├── photos/          # JPEG photos (auto-cleanup)
├── videos/          # MOV/MP4 videos (auto-cleanup)  
├── faces/           # Face embeddings (auto-cleanup)
├── logs/            # Service logs
└── scripts/         # Cleanup scripts
```

## 🔧 Manual Operations

### Run Cleanup Manually
```bash
cd /home/azuolas/mmoment
./app/orin_nano/scripts/cleanup_media.sh
```

### Check Current Storage
```bash
# Count files
ls -la app/orin_nano/camera_service/photos/ | wc -l
ls -la app/orin_nano/camera_service/videos/ | wc -l

# Check sizes
du -sh app/orin_nano/camera_service/photos/
du -sh app/orin_nano/camera_service/videos/
```

### Disable Auto-Cleanup
```bash
# Stop the timer
sudo systemctl stop media-cleanup.timer
sudo systemctl disable media-cleanup.timer
```

## 📊 Storage Limits

### Current Configuration
- **Photos**: 100 max (cleanup at 90)
- **Videos**: 20 max (cleanup at 18)
- **Face Data**: 50 max (cleanup at 45)

### Estimated Storage Usage
- **Photo**: ~500KB each → ~50MB for 100 photos
- **Video**: ~500KB each (3-10 seconds) → ~10MB for 20 videos
- **Total**: ~60MB maximum local storage

## 🚀 Benefits

1. **No Git Bloat**: Media files never committed to repository
2. **Controlled Storage**: Automatic limits prevent disk filling
3. **Performance**: Faster Git operations without large files
4. **Cleanup Logs**: Track what's being removed and when
5. **Configurable**: Easy to adjust limits and schedules

## ⚙️ Configuration Files

- **Git Ignore**: `.gitignore`
- **Cleanup Script**: `app/orin_nano/scripts/cleanup_media.sh`
- **Systemd Service**: `/etc/systemd/system/media-cleanup.service`
- **Systemd Timer**: `/etc/systemd/system/media-cleanup.timer`
- **Service Config**: `camera_service/services/capture_service.py`

This system ensures your repository stays clean while maintaining a reasonable amount of local media for testing and development. 