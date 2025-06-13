# Containerized Unified Camera Service

## 🎯 **Phase 2 Complete: Containerized Privacy-First Architecture**

This update containerizes your unified GPU face recognition service, building on your existing Docker expertise and working `ultralytics/ultralytics:latest-jetson-jetpack6` setup.

## 🔄 **From Working Demo to Production Service**

### **Your Working Setup (Before)**
```bash
sudo docker run -it --rm --runtime nvidia --network host --privileged \
  -v /dev:/dev -v $(pwd):/workspace -w /workspace \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /mnt/nvme/jetson_cache/insightface:/root/.insightface \
  -v /mnt/nvme/jetson_cache/ultralytics:/root/.config/Ultralytics \
  ultralytics/ultralytics:latest-jetson-jetpack6 \
  bash -c "pip install flask insightface >/dev/null 2>&1 && python gpu_face_web_demo.py"
```

### **New Containerized Service (After)**
```bash
./start_containerized_camera_service.sh
```

## 🏗️ **Architecture Benefits**

### **Privacy & Security**
- 🔐 **Container Isolation**: Face data contained within container boundaries
- 💾 **Ephemeral Processing**: Face recognition data can be easily purged
- 🛡️ **Controlled Access**: Only specified directories mounted from host
- 🗑️ **Easy Cleanup**: `docker-compose down` removes all processing data

### **GPU Performance**
- 🚀 **Same Base Container**: Uses your working `ultralytics/ultralytics:latest-jetson-jetpack6`
- 🎯 **NVIDIA Runtime**: Direct GPU passthrough with `--runtime nvidia`
- 📊 **Model Caching**: Preserves your existing `/mnt/nvme/jetson_cache` setup
- ⚡ **Optimized Stack**: YOLOv8 + InsightFace pre-configured in container

### **Development & Deployment**
- 🔧 **Reproducible**: Same container works across environments
- 📝 **Version Control**: Dockerfile tracks all dependencies
- 🔄 **Easy Updates**: `docker-compose up --build` rebuilds service
- 📊 **Health Monitoring**: Built-in health checks and logging

## 🚀 **Quick Start**

### **1. Start the Containerized Service**
```bash
cd /home/azuolas/mmoment/app/orin_nano/camera_service
./start_containerized_camera_service.sh
```

### **2. Test the Service**
```bash
python3 test_containerized_service.py
```

### **3. Access the Service**
- **Main API**: http://192.168.1.232:5003
- **Health Check**: http://192.168.1.232:5003/api/health  
- **GPU Face Status**: http://192.168.1.232:5003/api/face/unified/status

## 📊 **Container Features**

### **GPU Acceleration**
- **Base Image**: `ultralytics/ultralytics:latest-jetson-jetpack6`
- **Runtime**: `nvidia` for GPU passthrough
- **Models**: YOLOv8 (person detection) + InsightFace (embeddings)
- **Cache**: Preserves model downloads in `/mnt/nvme/jetson_cache`

### **Camera Integration**
- **Device Access**: `/dev/video*` mounted for camera access
- **Buffer Service**: Real-time frame processing pipeline
- **Network**: Host networking for direct access (port 5003)

### **Data Management**
```bash
camera_service/
├── faces/          # Face embeddings (wallet-based)
├── photos/         # Captured photos
├── videos/         # Recorded videos  
├── logs/           # Service logs
└── config/         # Configuration files
```

## 🔧 **Container Management**

### **Start Service**
```bash
docker-compose up -d --build
```

### **View Logs**
```bash
docker-compose logs -f
```

### **Stop Service**
```bash
docker-compose down
```

### **Rebuild Service**
```bash
docker-compose down
docker-compose up --build -d
```

### **Shell Access** (for debugging)
```bash
docker-compose exec camera-service bash
```

## 🛡️ **Privacy Controls**

### **Data Isolation**
- Face embeddings stored in container-specific volumes
- Easy purging: `docker-compose down -v` removes all data
- No persistent face data on host (unless explicitly mounted)

### **Session-Based Access**
- Face enrollment requires valid blockchain session
- Automatic cleanup when session ends
- User-owned embeddings stored by wallet address

### **Blockchain Integration**
- Solana middleware runs separately (native, port 5004)
- Camera service connects via API calls
- Face data tied to wallet addresses, not personal identifiers

## 📡 **API Endpoints**

All endpoints maintain compatibility with your existing frontend:

### **Unified GPU Face Recognition**
- `GET /api/face/unified/status` - Service and GPU status
- `POST /api/face/unified/enroll` - Enroll face (requires session)
- `POST /api/face/unified/recognize` - Real-time recognition
- `GET /api/face/unified/enrolled` - List enrolled faces
- `POST /api/face/unified/threshold` - Set similarity threshold
- `POST /api/face/unified/clear` - Clear all faces

### **Camera & Streaming**
- `GET /api/health` - Service health check
- `POST /api/stream/livepeer/start` - Start streaming  
- `GET /mjpeg-stream` - MJPEG video feed
- `GET /mjpeg-stream-with-faces` - Video with face overlays

## 🐛 **Troubleshooting**

### **Container Won't Start**
```bash
# Check Docker status
docker info

# Check NVIDIA runtime
docker info | grep nvidia

# View build logs
docker-compose up --build
```

### **GPU Not Available**
```bash
# Verify NVIDIA runtime
nvidia-smi
docker run --rm --runtime nvidia nvidia/cuda:11.0-base nvidia-smi

# Check container GPU access
docker-compose exec camera-service nvidia-smi
```

### **Models Not Loading**
```bash
# Check cache directory
ls -la /mnt/nvme/jetson_cache/

# View container logs
docker-compose logs camera-service

# Shell into container to debug
docker-compose exec camera-service bash
```

### **Camera Access Issues**
```bash
# Check camera devices
ls -la /dev/video*

# Verify container mounts
docker-compose exec camera-service ls -la /dev/video*
```

## 🔮 **Blockchain Privacy Features**

### **User-Owned Biometrics**
- Face embeddings stored as NFTs in user wallets
- Camera gets temporary decryption rights only during sessions
- Automatic purging when users disconnect

### **Zero-Trust Architecture**
- Camera service validates every request via Solana middleware
- No persistent user data without blockchain authorization
- Session-based access control for all face operations

### **Compliance Ready**
- GDPR compliant: users own their biometric data
- Audit trail via blockchain transactions  
- Data minimization: only temporary face processing

## 📈 **Performance Metrics**

The containerized service provides real-time monitoring:

```bash
# Service health
curl http://192.168.1.232:5003/api/health

# GPU face service status  
curl http://192.168.1.232:5003/api/face/unified/status

# Container metrics
docker stats unified-camera-service
```

## 🎉 **What You've Achieved**

1. ✅ **Containerized your working GPU setup** into a production service
2. ✅ **Privacy-first architecture** with blockchain authentication
3. ✅ **GPU-accelerated face recognition** with YOLOv8 + InsightFace
4. ✅ **Easy deployment** via Docker Compose
5. ✅ **Data isolation** for enhanced privacy
6. ✅ **Scalable foundation** ready for multi-camera deployments

---

**🚀 Your camera service is now containerized and production-ready!** It combines the performance of your working GPU demo with the privacy-first architecture of your blockchain-authenticated system. 