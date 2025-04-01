import { useState, useEffect } from 'react';
import { cameraStatus } from '../services/camera-status';

interface CameraStatusData {
  owner: string;
  isLive: boolean;
  isStreaming: boolean;
  status: 'ok' | 'error' | 'offline';
  lastSeen?: number;
}

export function useCameraStatus(_cameraId: string): CameraStatusData {
  const [status, setStatus] = useState<CameraStatusData>({
    owner: '',
    isLive: false,
    isStreaming: false,
    status: 'offline'
  });

  useEffect(() => {
    const unsubscribe = cameraStatus.subscribe((newStatus) => {
      setStatus(prev => ({
        ...prev,
        ...newStatus,
        status: newStatus.isLive ? 'ok' : 'offline'
      }));
    });

    return () => {
      unsubscribe();
    };
  }, []);

  return status;
} 