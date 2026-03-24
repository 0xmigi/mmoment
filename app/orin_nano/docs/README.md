# Jetson Camera System Documentation

Documentation for the MMOMENT camera system running on NVIDIA Jetson Orin Nano.

## Documents

| Document | Description |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Three-container Docker architecture, data flow, file system layout |
| [STANDARDIZED_API.md](STANDARDIZED_API.md) | API reference — public endpoints, auth model, internal services, CV dev mode |
| [IDENTITY_REDESIGN.md](IDENTITY_REDESIGN.md) | Face recognition identity system — tracking, recovery, thresholds |
| [HARDWARE.md](HARDWARE.md) | Hardware components, camera setup, thermal/power management |
| [MEDIA_MANAGEMENT.md](MEDIA_MANAGEMENT.md) | Storage layout, cleanup system, git ignore rules |

## Quick Reference

- **Public access**: `https://[camera-pda].mmoment.xyz` (Cloudflare tunnel -> port 5002)
- **Program ID**: `E67WTa1NpFVoapXwYYQmXzru3pyhaN9Kj3wPdZEyyZsL`
- **Containers**: camera-service (5002), biometric-security (5003), solana-middleware (5001)
- **Streaming**: WebRTC via WHIP/WHEP (dual stream: clean + annotated)
- **AI pipeline**: C++ TensorRT native mode (RetinaFace, ArcFace, YOLOv8-pose, OSNet)
- **Media storage**: NVMe SSD at `/mnt/nvme/mmoment-photos/` and `/mnt/nvme/mmoment-videos/`
- **Deployment**: `docker-compose up -d` from `/mnt/nvme/mmoment/app/orin_nano/`
