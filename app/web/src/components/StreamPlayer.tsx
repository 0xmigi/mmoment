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

// Longer cache for mobile to reduce API calls
const STREAM_CACHE_TTL = 10000; // 10 seconds
const MOBILE_CACHE_TTL = 30000; // 30 seconds for mobile

// Fetch timeout to avoid hanging
const FETCH_TIMEOUT = 8000; // 8 seconds
const MOBILE_FETCH_TIMEOUT = 10000; // 10 seconds for mobile

interface StreamInfo {
  playbackId: string;
  isActive: boolean;
}

const StreamPlayer = memo(() => {
  const [streamInfo, setStreamInfo] = useState<StreamInfo | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isMobile] = useState(() => /iPhone|iPad|iPod|Android/i.test(navigator.userAgent));
  const [loadingRetry, setLoadingRetry] = useState(0);
  const pollInterval = useRef<NodeJS.Timeout>();
  const lastFetchTime = useRef(0);
  const isCameraOperation = useRef(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const { program } = useProgram();
  const { selectedCamera } = useCamera();

  // Helper function for fetch with timeout
  const fetchWithTimeout = async (url: string, options: RequestInit = {}, timeout: number) => {
    // Cancel any ongoing fetch
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    
    const controller = new AbortController();
    abortControllerRef.current = controller;
    const { signal } = controller;
    
    const timeoutId = setTimeout(() => {
      controller.abort();
    }, timeout);
    
    try {
      const response = await fetch(url, { ...options, signal });
      clearTimeout(timeoutId);
      return response;
    } catch (error) {
      clearTimeout(timeoutId);
      throw error;
    } finally {
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
    }
  };

  const fetchStreamInfo = useCallback(async (forceRefresh = false) => {
    // Skip fetching during camera operations
    if (isCameraOperation.current) return;

    // Prevent multiple rapid fetches
    const now = Date.now();
    const minTimeBetweenFetches = isMobile ? 5000 : 3000;
    if (!forceRefresh && now - lastFetchTime.current < minTimeBetweenFetches) return;
    
    // Check cache first if not forced refresh
    const cacheTTL = isMobile ? MOBILE_CACHE_TTL : STREAM_CACHE_TTL;
    if (!forceRefresh && streamInfoCache.data && now - streamInfoCache.timestamp < cacheTTL) {
      console.log('[StreamPlayer] Using cached stream info');
      setStreamInfo(streamInfoCache.data);
      setError(null);
      setIsLoading(false);
      return;
    }
    
    lastFetchTime.current = now;

    try {
      console.log('[StreamPlayer] Fetching fresh stream info');
      const timeout = isMobile ? MOBILE_FETCH_TIMEOUT : FETCH_TIMEOUT;
      
      const response = await fetchWithTimeout(
        `${CONFIG.CAMERA_API_URL}/api/stream/info`, 
        {
          headers: {
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
          }
        },
        timeout
      );
      
      if (!response.ok) {
        throw new Error(`Failed to fetch stream info (${response.status})`);
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
      setLoadingRetry(0); // Reset retry counter on success
    } catch (err) {
      console.error('Failed to get stream info:', err);
      if (!isCameraOperation.current) {
        setError(`Failed to get stream info: ${err instanceof Error ? err.message : 'Network error'}`);
        
        // For mobile, retry a few times
        if (isMobile && loadingRetry < 3) {
          console.log(`[StreamPlayer] Retrying fetch (${loadingRetry + 1}/3)`);
          setLoadingRetry(prev => prev + 1);
          
          // Schedule retry after a delay
          setTimeout(() => {
            fetchStreamInfo(true);
          }, 5000);
        }
      }
    } finally {
      if (!isCameraOperation.current) {
        setIsLoading(false);
      }
    }
  }, [isMobile, loadingRetry]);

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
    // Use different polling intervals for mobile vs desktop
    const pollingInterval = isMobile ? 30000 : 15000;
    pollInterval.current = setInterval(() => fetchStreamInfo(), pollingInterval);
    
    return () => {
      if (pollInterval.current) {
        clearInterval(pollInterval.current);
      }
      // Cleanup any ongoing fetch
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
    };
  }, [fetchStreamInfo, isMobile]);

  // Add debug logging for program and camera availability
  useEffect(() => {
    console.log(`[StreamPlayer] Program ID: ${CAMERA_ACTIVATION_PROGRAM_ID.toString()}`);
    console.log(`[StreamPlayer] Program available: ${!!program}`);
    console.log(`[StreamPlayer] Mobile browser: ${isMobile}`);
    
    if (selectedCamera) {
      console.log(`[StreamPlayer] Camera loaded: ${selectedCamera.publicKey}`);
    } else {
      console.log(`[StreamPlayer] No camera loaded`);
    }
  }, [program, selectedCamera, isMobile]);

  if (isLoading) {
    return (
      <div className="aspect-video bg-gray-900 rounded-lg flex items-center justify-center">
        <p className="text-gray-400">
          Loading stream...
          {isMobile && loadingRetry > 0 && (
            <span className="block text-xs mt-1">Attempt {loadingRetry}/3</span>
          )}
        </p>
      </div>
    );
  }

  if (error || !streamInfo?.playbackId || !streamInfo.isActive) {
    return (
      <div className="aspect-video w-full bg-gray-800 rounded-lg overflow-hidden flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-400">Stream is offline</p>
          {isMobile && error && (
            <button 
              className="mt-2 px-3 py-1 text-xs bg-blue-500 text-white rounded"
              onClick={() => {
                setIsLoading(true);
                setError(null);
                fetchStreamInfo(true);
              }}
            >
              Retry
            </button>
          )}
        </div>
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