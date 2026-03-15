import { useState, useEffect } from 'react';
import { unifiedCameraPolling, CameraStatusData } from './unified-camera-polling';

/**
 * React hook for subscribing to camera status updates.
 * Uses the unified polling service to prevent duplicate requests.
 */
export function useCameraStatus(cameraId: string): CameraStatusData {
  const [status, setStatus] = useState<CameraStatusData>({
    owner: '',
    isLive: false,
    isStreaming: false,
    status: 'offline'
  });

  useEffect(() => {
    if (!cameraId) {
      setStatus({
        owner: '',
        isLive: false,
        isStreaming: false,
        status: 'offline'
      });
      return;
    }

    console.log(`[useCameraStatus] Subscribing to camera ${cameraId}`);

    // Subscribe to unified polling service
    const unsubscribe = unifiedCameraPolling.subscribe(cameraId, (newStatus) => {
      setStatus(newStatus);
    });

    // Cleanup: unsubscribe when component unmounts or cameraId changes
    return () => {
      console.log(`[useCameraStatus] Unsubscribing from camera ${cameraId}`);
      unsubscribe();
    };
  }, [cameraId]);

  return status;
} 