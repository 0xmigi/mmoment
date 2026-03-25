# Media Management & Cleanup System

How the Jetson camera system manages media files to prevent storage buildup and keep large files out of Git.

## Git Ignore Rules

All media files are excluded from Git commits:

```gitignore
# Media stored on NVMe SSD, outside the repo
# These patterns catch any accidental local copies
*.jpg *.jpeg *.png *.gif *.bmp *.tiff *.webp
*.mov *.mp4 *.avi *.mkv *.webm *.flv *.wmv *.m4v *.3gp
```

**Allowed exceptions**: documentation images (`docs/**/*.png`, `README*.png`, `*.example.jpg`).

## Storage Layout

Media is stored on the NVMe SSD, mounted into the camera-service container:

```
# Host paths (NVMe SSD)
/mnt/nvme/mmoment-photos/     -> container /app/photos
/mnt/nvme/mmoment-videos/     -> container /app/videos

# Repo-local persistent data (in docker volumes)
./data/face_embeddings/        -> container /app/face_embeddings
./data/faces/                  -> container /app/faces
./data/recordings/             -> container /app/recordings
```

## Automatic Cleanup

The `CaptureService` manages storage limits automatically:

- **Photos**: max 100 files, cleanup triggers at 90
- **Videos**: max 20 files, cleanup triggers at 18
- **Method**: deletes oldest files first when threshold is reached

### Estimated Storage Usage
- Photo: ~500KB each -> ~50MB for 100 photos
- Video: ~500KB each (3-10 seconds) -> ~10MB for 20 videos
- Total: ~60MB maximum local storage

## Manual Operations

### Check Current Storage
```bash
# From the Jetson host
ls -la /mnt/nvme/mmoment-photos/ | wc -l
ls -la /mnt/nvme/mmoment-videos/ | wc -l
du -sh /mnt/nvme/mmoment-photos/
du -sh /mnt/nvme/mmoment-videos/
```

### Run Cleanup Script
```bash
cd /mnt/nvme/mmoment/app/orin_nano
./scripts/cleanup_media.sh
```

**Cleanup script config:**
```bash
KEEP_PHOTOS=20      # Keep newest 20 photos
KEEP_VIDEOS=10      # Keep newest 10 videos
KEEP_FACES=50       # Keep newest 50 face embeddings
```

Cleanup also removes files older than 7 days (photos/videos) or 30 days (faces).
