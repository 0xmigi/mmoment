import { Player } from "@livepeer/react";
import { useState, useEffect, useCallback, useRef, memo } from "react";
import { CONFIG } from "../config";

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

  const fetchStreamInfo = useCallback(async () => {
    // Skip fetching during camera operations
    if (isCameraOperation.current) return;

    // Prevent multiple rapid fetches
    const now = Date.now();
    if (now - lastFetchTime.current < 2000) return;
    lastFetchTime.current = now;

    try {
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
      setStreamInfo(prev => {
        if (prev?.playbackId === data.playbackId && prev?.isActive === data.isActive) {
          return prev;
        }
        return {
          playbackId: data.playbackId,
          isActive: data.isActive
        };
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
    pollInterval.current = setInterval(fetchStreamInfo, 5000);
    
    return () => {
      if (pollInterval.current) {
        clearInterval(pollInterval.current);
      }
    };
  }, [fetchStreamInfo]);

  if (isLoading) {
    return (
      <div className="aspect-video bg-gray-900 rounded-lg flex items-center justify-center">
        <p className="text-gray-400">Loading stream...</p>
      </div>
    );
  }

  if (error || !streamInfo?.playbackId || !streamInfo.isActive) {
    return (
      <div className="aspect-video bg-gray-900 rounded-lg flex items-center justify-center">
        <p className="text-gray-400">Stream is offline</p>
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