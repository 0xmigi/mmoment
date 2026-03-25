---
paths:
  - "app/orin_nano/**"
  - "app/raspberry_pi/**"
---
# Camera Service Rules

## Architecture
Three-container Docker setup on NVIDIA Jetson:
1. **Camera Service** (port 5002) — primary API gateway, CV processing, streaming
2. **Biometric Security** (port 5003) — AES-256 encryption of facial embeddings, no persistent raw storage
3. **Solana Middleware** (port 5001) — blockchain integration, wallet sessions, transaction building

## Inter-Service Communication
- All containers communicate via localhost HTTP APIs
- Camera Service never handles blockchain operations directly
- Biometric Security provides encryption services to other containers
- Solana Middleware handles all blockchain interactions

## Security
- Biometric data never stored in plain text
- Encrypted temporary storage with automatic cleanup
- Inter-container communication via localhost only
- Cloudflare tunnel for secure external access
