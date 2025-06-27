import { useState, useEffect, useRef } from 'react';
import { unifiedCameraService } from './unified-camera-service';

interface CameraStatusData {
  owner: string;
  isLive: boolean;
  isStreaming: boolean;
  status: 'ok' | 'error' | 'offline';
  lastSeen?: number;
}

// Cache for individual camera statuses to avoid excessive API calls
const cameraStatusCache = new Map<string, { data: CameraStatusData; timestamp: number }>();
const CACHE_TTL = 5000; // 5 seconds cache

export function useCameraStatus(cameraId: string): CameraStatusData {
  const [status, setStatus] = useState<CameraStatusData>({
    owner: '',
    isLive: false,
    isStreaming: false,
    status: 'offline'
  });

  const pollIntervalRef = useRef<NodeJS.Timeout>();
  const isMountedRef = useRef(true);

  const fetchCameraStatus = async (currentCameraId: string) => {
    if (!currentCameraId || !isMountedRef.current) {
      return;
    }

    // Check cache first
    const now = Date.now();
    const cached = cameraStatusCache.get(currentCameraId);
    if (cached && now - cached.timestamp < CACHE_TTL) {
      if (isMountedRef.current) {
        setStatus(cached.data);
      }
      return;
    }

    try {
      console.log(`[useCameraStatus] Fetching status for camera: ${currentCameraId}`);
      
      // Get comprehensive state from unified service
      const comprehensiveState = await unifiedCameraService.getComprehensiveState(currentCameraId);
      
      let newStatus: CameraStatusData;
      
      if (comprehensiveState.status.success && comprehensiveState.status.data) {
        const statusData = comprehensiveState.status.data;
        const streamData = comprehensiveState.streamInfo.data;
        
        newStatus = {
          owner: statusData.owner || '',
          isLive: statusData.isOnline,
          isStreaming: streamData?.isActive || false,
          status: statusData.isOnline ? 'ok' : 'offline',
          lastSeen: statusData.lastSeen
        };
        
        console.log(`[useCameraStatus] Camera ${currentCameraId} status:`, {
          isLive: newStatus.isLive,
          isStreaming: newStatus.isStreaming,
          status: newStatus.status
        });
      } else {
        // Camera is not accessible
        newStatus = {
          owner: '',
          isLive: false,
          isStreaming: false,
          status: 'offline',
          lastSeen: now
        };
        
        console.log(`[useCameraStatus] Camera ${currentCameraId} is offline:`, comprehensiveState.status.error);
      }

      // Cache the result
      cameraStatusCache.set(currentCameraId, { data: newStatus, timestamp: now });
      
      // Update state only if component is still mounted
      if (isMountedRef.current) {
        setStatus(newStatus);
      }
    } catch (error) {
      console.error(`[useCameraStatus] Error fetching status for ${currentCameraId}:`, error);
      
      const errorStatus: CameraStatusData = {
        owner: '',
        isLive: false,
        isStreaming: false,
        status: 'error',
        lastSeen: now
      };
      
      // Cache the error state briefly
      cameraStatusCache.set(currentCameraId, { data: errorStatus, timestamp: now });
      
      if (isMountedRef.current) {
        setStatus(errorStatus);
      }
    }
  };

  useEffect(() => {
    isMountedRef.current = true;
    
    if (!cameraId) {
      setStatus({
        owner: '',
        isLive: false,
        isStreaming: false,
        status: 'offline'
      });
      return;
    }

    // Initial fetch
    fetchCameraStatus(cameraId);

    // Set up polling interval - camera-specific (reduced frequency to prevent blinking)
    pollIntervalRef.current = setInterval(() => {
      fetchCameraStatus(cameraId);
    }, 12000); // Poll every 12 seconds

    return () => {
      isMountedRef.current = false;
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, [cameraId]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      isMountedRef.current = false;
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  return status;
} 