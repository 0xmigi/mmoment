# Jetson Camera Hardware Setup

## Hardware Components

### Main Components

1. **NVIDIA Jetson Orin Nano Developer Kit**
   - Processor: NVIDIA Jetson Orin Nano (8GB)
   - Memory: 8GB LPDDR5
   - Storage: 64GB eMMC + NVMe SSD for media (`/mnt/nvme/`)

2. **Logitech StreamCam**
   - Resolution: 1080p/60fps
   - Connection: USB-C
   - Field of View: 78 diagonal
   - Mounted as: `/dev/video0` and `/dev/video1`

3. **Jetson IMX477 Camera** (Optional)
   - Resolution: 4032x3040
   - Connection: MIPI CSI-2 (requires nvargus-daemon socket)

### Environment Requirements

- **Power**: 19V barrel jack or USB-C PD (the Developer Kit requires ~25W)
- **Connectivity**: Ethernet or WiFi
- **Cooling**: Active cooling recommended for extended operation

## Camera Configuration

### Primary Camera

The system uses the Logitech StreamCam as the primary camera. Configured in `docker-compose.yml`:

```yaml
devices:
  # Logitech StreamCam on video0/video1
  - /dev/video0:/dev/video0
  - /dev/video1:/dev/video1
```

Operating at 1280x720, 15fps target for CV processing.

### Changing Camera Device

Set the camera device in `docker-compose.yml` environment variables or override at runtime. The buffer service auto-detects available cameras on startup.

## Camera Position

For optimal face recognition:
1. Place at eye level
2. Ensure consistent lighting on faces
3. Mount on a stable surface to reduce motion blur

## Network Setup

1. Ethernet preferred for stability
2. WiFi via NetworkManager:
   ```bash
   nmcli device wifi connect YOUR_SSID password YOUR_PASSWORD
   ```

## Performance

### Thermal Management
Monitor temperatures:
```bash
tegrastats
```

### Power Modes
```bash
# Maximum performance (25W)
sudo nvpmodel -m 0
sudo jetson_clocks

# Power saving (15W)
sudo nvpmodel -m 1
```

## Troubleshooting

### Camera Not Detected
1. Check physical USB connections
2. List camera devices:
   ```bash
   v4l2-ctl --list-devices
   ```
3. Test camera capture:
   ```bash
   v4l2-ctl --device=/dev/video0 --set-fmt-video=width=1280,height=720,pixelformat=MJPG --stream-mmap
   ```

### Performance Issues
1. Check utilization: `htop`
2. Check for thermal throttling: `tegrastats | grep CPU`
3. Ensure NVMe is mounted: `df -h /mnt/nvme`
