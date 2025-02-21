import { useState, useEffect } from 'react';
import { cameraStatus } from '../services/camera-status';

export function useCameraStatus() {
  const [status, setStatus] = useState<{ isLive: boolean; isStreaming: boolean }>(() => 
    cameraStatus.getCurrentStatus()
  );

  useEffect(() => {
    const unsubscribe = cameraStatus.subscribe(setStatus);
    return () => unsubscribe();
  }, []);

  return status;
} 