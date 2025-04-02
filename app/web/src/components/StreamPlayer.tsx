import { Player } from "@livepeer/react";
import { useState, useEffect, useCallback, useRef, memo } from "react";
import { CONFIG } from "../config";
import { useProgram, CAMERA_ACTIVATION_PROGRAM_ID } from '../anchor/setup';
import { useCamera } from './CameraProvider';

// Simple cache for stream info to reduce API calls
const streamInfoCache = { 
  data: null as any, 
  timestamp: 0 
};
const STREAM_CACHE_TTL = 10000; // 10 seconds

interface StreamInfo {
  playbackId: string;
  isActive: boolean;
}

const StreamPlayer = memo(() => {
  const [streamInfo, setStreamInfo] = useState<StreamInfo | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isMobile] = useState(() => /iPhone|iPad|iPod|Android/i.test(navigator.userAgent));
  const pollInterval = useRef<NodeJS.Timeout>();
  const lastFetchTime = useRef(0);
  const isCameraOperation = useRef(false);
  const { program } = useProgram();
  const { selectedCamera } = useCamera();

  const fetchStreamInfo = useCallback(async () => {
    // Skip fetching during camera operations
    if (isCameraOperation.current) return;

    // Prevent multiple rapid fetches
    const now = Date.now();
    if (now - lastFetchTime.current < 3000) return; // Increased minimum time between fetches
    
    // Check cache first
    if (streamInfoCache.data && now - streamInfoCache.timestamp < STREAM_CACHE_TTL) {
      console.log('[StreamPlayer] Using cached stream info');
      setStreamInfo(streamInfoCache.data);
      setError(null);
      setIsLoading(false);
      return;
    }
    
    lastFetchTime.current = now;

    try {
      console.log('[StreamPlayer] Fetching fresh stream info');
      const response = await fetch(`${CONFIG.CAMERA_API_URL}/api/stream/info`, {
        headers: {
          'Cache-Control': 'no-cache',
          'Pragma': 'no-cache'
        }
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch stream info');
      }
      
      const data = await response.json();
      const newStreamInfo = {
        playbackId: data.playbackId,
        isActive: data.isActive
      };
      
      // Update cache
      streamInfoCache.data = newStreamInfo;
      streamInfoCache.timestamp = now;
      
      // Update state only if different
      setStreamInfo(prev => {
        if (prev?.playbackId === data.playbackId && prev?.isActive === data.isActive) {
          return prev;
        }
        return newStreamInfo;
      });
      setError(null);
    } catch (err) {
      console.error('Failed to get stream info:', err);
      if (!isCameraOperation.current) {
        setError('Failed to get stream info');
      }
    } finally {
      if (!isCameraOperation.current) {
        setIsLoading(false);
      }
    }
  }, []);

  // Listen for camera operations
  useEffect(() => {
    const handleCameraOperation = (e: CustomEvent) => {
      isCameraOperation.current = e.detail.inProgress;
    };

    window.addEventListener('cameraOperation', handleCameraOperation as EventListener);
    return () => {
      window.removeEventListener('cameraOperation', handleCameraOperation as EventListener);
    };
  }, []);

  useEffect(() => {
    fetchStreamInfo();
    pollInterval.current = setInterval(fetchStreamInfo, 15000); // Increased polling interval to 15 seconds
    
    return () => {
      if (pollInterval.current) {
        clearInterval(pollInterval.current);
      }
    };
  }, [fetchStreamInfo]);

  // Add debug logging for program and camera availability
  useEffect(() => {
    console.log(`[StreamPlayer] Program ID: ${CAMERA_ACTIVATION_PROGRAM_ID.toString()}`);
    console.log(`[StreamPlayer] Program available: ${!!program}`);
    
    if (selectedCamera) {
      console.log(`[StreamPlayer] Camera loaded: ${selectedCamera.publicKey}`);
    } else {
      console.log(`[StreamPlayer] No camera loaded`);
    }
  }, [program, selectedCamera]);

  if (isLoading) {
    return (
      <div className="aspect-video bg-gray-900 rounded-lg flex items-center justify-center">
        <p className="text-gray-400">Loading stream...</p>
      </div>
    );
  }

  if (error || !streamInfo?.playbackId || !streamInfo.isActive) {
    return (
      <div className="aspect-video w-full bg-gray-800 rounded-lg overflow-hidden flex items-center justify-center">
        <p className="text-center text-gray-400">Stream is offline</p>
      </div>
    );
  }

  return (
    <div className="aspect-video bg-black rounded-lg overflow-hidden">
      <Player 
        title="Camera Stream"
        playbackId={streamInfo.playbackId}
        autoPlay
        muted
        controls={{
          autohide: 3000,
          hotkeys: false,
          defaultVolume: 0
        }}
        aspectRatio="16to9"
        showPipButton={!isMobile}
        objectFit="contain"
        priority
        showLoadingSpinner={true}
        lowLatency
      />
    </div>
  );
});

StreamPlayer.displayName = 'StreamPlayer';

export { StreamPlayer };