# Jetson Camera Hardware Setup

This document provides information about the hardware setup for the Jetson Camera system.

## Hardware Components

### Main Components

1. **NVIDIA Jetson Orin Nano Developer Kit**
   - Processor: NVIDIA Jetson Orin Nano (8GB)
   - Memory: 8GB LPDDR5
   - Storage: 64GB eMMC

2. **Logitech StreamCam**
   - Resolution: 1080p/60fps
   - Connection: USB-C
   - Field of View: 78° diagonal
   - Connected to: `/dev/video1` and `/dev/video2`

3. **Jetson IMX477 Camera** (Optional)
   - Resolution: 4032x3040
   - Connection: MIPI CSI-2
   - Connected to: `/dev/video0`

### Environment Requirements

- **Power**: 5V DC, 4A power supply
- **Connectivity**: Ethernet or WiFi
- **Cooling**: Active cooling recommended for extended operation

## Camera Setup

### Primary Camera Configuration

The system uses the Logitech StreamCam as the primary camera for face recognition and gesture detection. The camera is configured in the following way:

- **Device Path**: `/dev/video1`
- **Resolution**: 1280x720
- **Frame Rate**: 30fps

### Secondary/Backup Camera

If the Logitech StreamCam is not available, the system can fall back to using the Jetson IMX477 camera:

- **Device Path**: `/dev/video0`
- **Resolution**: 4032x3040 (will be scaled down for processing)

### Camera Environment Variables

The camera device can be changed by setting the `CAMERA_DEVICE` environment variable in the systemd service file:

```ini
Environment="CAMERA_DEVICE=/dev/video1"  # Logitech StreamCam
```

## Hardware Installation

### Camera Position

For optimal face recognition and gesture detection:

1. Place the camera at eye level
2. Ensure good, consistent lighting on the user's face
3. Mount camera on a stable surface to reduce motion blur

### Network Setup

1. Connect the Jetson Orin Nano to your network via Ethernet for the most stable connection
2. Alternatively, configure WiFi using the NetworkManager:
   ```bash
   nmcli device wifi connect YOUR_SSID password YOUR_PASSWORD
   ```

## Performance Considerations

### Thermal Management

The Jetson Orin Nano can generate significant heat when running intensive tasks like face recognition and gesture detection. To manage this:

1. Ensure proper ventilation around the device
2. Use the included fan or consider a larger cooling solution for extended operation
3. Monitor temperatures with:
   ```bash
   tegrastats
   ```

### Power Management

To optimize performance vs. power consumption:

1. Default power mode: 15W (balanced performance)
2. For maximum performance:
   ```bash
   sudo nvpmodel -m 0
   sudo jetson_clocks
   ```
3. For power saving:
   ```bash
   sudo nvpmodel -m 1
   ```

## Troubleshooting Hardware Issues

### Camera Not Detected

1. Check physical connections
2. Verify camera device paths:
   ```bash
   v4l2-ctl --list-devices
   ```
3. Test camera with:
   ```bash
   v4l2-ctl --device=/dev/video1 --set-fmt-video=width=1280,height=720,pixelformat=MJPG --stream-mmap
   ```

### Performance Issues

1. Check CPU/GPU utilization:
   ```bash
   htop
   ```
2. Verify thermal throttling is not occurring:
   ```bash
   tegrastats | grep CPU
   ```

## Upgrading Hardware

The system can be upgraded with:

1. **External SSD** - Connect via USB 3.0 for faster storage and less wear on the eMMC
2. **Additional Memory** - Not possible on the Orin Nano Developer Kit
3. **Higher Quality Camera** - Any USB camera supported by V4L2 should work 