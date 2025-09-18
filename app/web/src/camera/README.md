# Standardized Camera System

This directory contains the new standardized camera system that makes the frontend completely camera-agnostic. The system automatically detects camera types and routes operations to the appropriate implementation.

## Architecture Overview

### Core Components

1. **Camera Interface** (`camera-interface.ts`)
   - Defines the standard contract that all camera implementations must follow
   - Ensures consistent API across all camera types
   - Includes capabilities, status, media operations, and streaming

2. **Camera Registry** (`camera-registry.ts`)
   - Automatically detects and manages all cameras in the network
   - Provides health checking and status monitoring
   - Creates camera instances on demand

3. **Unified Camera Service** (`unified-camera-service.ts`)
   - Single point of access for all camera operations
   - Automatically routes operations to the correct camera implementation
   - Provides consistent error handling and logging

4. **Camera Implementations** (`implementations/`)
   - `jetson-camera.ts` - NVIDIA Jetson Orin Nano implementation
   - `pi5-camera.ts` - Raspberry Pi 5 implementation
   - Each implements the standard `ICamera` interface

## Adding a New Camera Type

To add a new camera type, you only need to:

1. **Create a new implementation** in `implementations/` that implements `ICamera`
2. **Register it in the registry** by adding it to the `createCameraInstance` method
3. **Add configuration** to the known cameras list in `initializeKnownCameras`

Example for a new "ESP32" camera:

```typescript
// implementations/esp32-camera.ts
export class ESP32Camera implements ICamera {
  // ... implement all ICamera methods
}

// camera-registry.ts - add to createCameraInstance
case 'esp32':
  return new ESP32Camera(entry.cameraId, entry.apiUrl);

// camera-registry.ts - add to initializeKnownCameras
{
  cameraId: 'your-esp32-camera-id',
  cameraType: 'esp32',
  apiUrl: 'https://esp32.example.com',
  name: 'ESP32 Camera',
  capabilities: {
    canTakePhotos: true,
    canRecordVideos: false,
    canStream: true,
    // ... other capabilities
  }
}
```

## Frontend Usage

The frontend now uses the unified camera service for all operations:

```typescript
import { unifiedCameraService } from './camera/unified-camera-service';

// Take a photo (works with any camera type)
const response = await unifiedCameraService.takePhoto(cameraId);

// Start streaming (automatically uses correct format)
const streamResponse = await unifiedCameraService.startStream(cameraId);

// Check capabilities
const supportsGestures = unifiedCameraService.cameraSupports(cameraId, 'canDetectGestures');
```

## Key Benefits

1. **Camera Agnostic**: Frontend code works with any camera type
2. **Automatic Detection**: System automatically detects and configures cameras
3. **Consistent API**: All cameras use the same interface
4. **Easy Expansion**: Adding new camera types requires minimal changes
5. **Health Monitoring**: Built-in health checking for all cameras
6. **Error Handling**: Consistent error handling across all camera types

## Migration from Old System

The old system had hardcoded camera type detection:

```typescript
// OLD - Hardcoded detection
const isJetsonCamera = (cameraId: string) => {
  return cameraId === CONFIG.JETSON_CAMERA_PDA;
};

if (isJetsonCamera(cameraId)) {
  // Use Jetson service
  await jetsonCameraService.takePhoto();
} else {
  // Use Pi5 service
  await cameraActionService.takePhoto();
}
```

The new system is completely unified:

```typescript
// NEW - Unified approach
await unifiedCameraService.takePhoto(cameraId);
```

## Configuration

Cameras are automatically registered from the configuration in `core/config.ts`. The registry reads the camera PDAs and API URLs to create the appropriate instances.

## Future Enhancements

1. **Auto-discovery**: Scan network for cameras automatically
2. **Dynamic registration**: Allow cameras to register themselves
3. **Load balancing**: Distribute operations across multiple cameras
4. **Failover**: Automatic failover to backup cameras
5. **Analytics**: Track camera usage and performance 