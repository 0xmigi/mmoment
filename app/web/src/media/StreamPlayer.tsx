import { useProgram, CAMERA_ACTIVATION_PROGRAM_ID } from "../anchor/setup";
import { useCamera } from "../camera/CameraProvider";
import { unifiedCameraService } from "../camera/unified-camera-service";
import { unifiedCameraPolling } from "../camera/unified-camera-polling";
import { WebRTCStreamPlayer } from "./WebRTCStreamPlayer";
import { Player } from "@livepeer/react";
import { useState, useEffect, useCallback, useRef, memo } from "react";
import { useParams } from "react-router-dom";

// Simple cache for stream info to reduce API calls
const streamInfoCache = {
  data: null as any,
  timestamp: 0,
};

// Longer cache for mobile to reduce API calls
const STREAM_CACHE_TTL = 10000; // 10 seconds
const MOBILE_CACHE_TTL = 30000; // 30 seconds for mobile

interface StreamInfo {
  playbackId: string;
  isActive: boolean;
  streamType?: string;
  streamUrl?: string;
}

// Helper to determine stream type from localStorage CV overlay state
const getStreamTypeFromToggles = (cameraId: string | null): 'clean' | 'annotated' => {
  if (!cameraId) return 'clean';
  // Use consolidated CV overlay key (with fallback to old keys for backwards compat)
  const cvOverlay = localStorage.getItem(`jetson_cv_overlay_${cameraId}`) === 'true';
  if (cvOverlay) return 'annotated';
  // Fallback: check old keys in case migration hasn't happened yet
  const faceViz = localStorage.getItem(`jetson_face_viz_${cameraId}`) === 'true';
  const poseViz = localStorage.getItem(`jetson_pose_viz_${cameraId}`) === 'true';
  return (faceViz || poseViz) ? 'annotated' : 'clean';
};

const StreamPlayer = memo(() => {
  const [streamInfo, setStreamInfo] = useState<StreamInfo | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isMobile] = useState(() =>
    /iPhone|iPad|iPod|Android/i.test(navigator.userAgent)
  );
  const [loadingRetry, setLoadingRetry] = useState(0);
  const [useWebRTC, setUseWebRTC] = useState(true); // WebRTC only for testing
  const pollInterval = useRef<NodeJS.Timeout>();
  const lastFetchTime = useRef(0);
  const isCameraOperation = useRef(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const { program } = useProgram();
  const { selectedCamera } = useCamera();
  const { cameraId } = useParams<{ cameraId: string }>();

  // Get current camera ID for stream type calculation
  const currentCameraIdForStream =
    cameraId ||
    selectedCamera?.publicKey ||
    localStorage.getItem("directCameraId");

  // Stream type based on visualization toggle states
  const [streamType, setStreamType] = useState<'clean' | 'annotated'>(() =>
    getStreamTypeFromToggles(currentCameraIdForStream)
  );

  // Listen for visualization toggle changes via storage events and custom events
  useEffect(() => {
    const updateStreamType = () => {
      const newStreamType = getStreamTypeFromToggles(currentCameraIdForStream);
      setStreamType(prev => {
        if (prev !== newStreamType) {
          console.log(`[StreamPlayer] 🔄 Stream type changing: ${prev} → ${newStreamType}`);
          return newStreamType;
        }
        return prev;
      });
    };

    // Listen for localStorage changes (from other tabs/windows)
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key?.startsWith('jetson_cv_overlay_') ||
          e.key?.startsWith('jetson_face_viz_') ||
          e.key?.startsWith('jetson_pose_viz_')) {
        console.log('[StreamPlayer] 📦 Storage event detected:', e.key);
        updateStreamType();
      }
    };

    // Listen for custom visualization toggle events (from same tab)
    const handleVisualizationChange = () => {
      console.log('[StreamPlayer] 🎛️ Visualization toggle event detected');
      updateStreamType();
    };

    window.addEventListener('storage', handleStorageChange);
    window.addEventListener('visualizationToggle', handleVisualizationChange);

    // Also check periodically in case events are missed
    const checkInterval = setInterval(updateStreamType, 2000);

    return () => {
      window.removeEventListener('storage', handleStorageChange);
      window.removeEventListener('visualizationToggle', handleVisualizationChange);
      clearInterval(checkInterval);
    };
  }, [currentCameraIdForStream]);

  const fetchStreamInfo = useCallback(
    async (forceRefresh = false) => {
      // Skip fetching during camera operations
      if (isCameraOperation.current) {
        return;
      }

      const currentCameraId =
        cameraId ||
        selectedCamera?.publicKey ||
        localStorage.getItem("directCameraId");
      if (!currentCameraId) {
        console.log("[StreamPlayer] No camera ID available");
        setStreamInfo(null);
        setIsLoading(false);
        return;
      }

      const now = Date.now();
      const cacheAge = now - streamInfoCache.timestamp;
      const cacheTTL = isMobile ? MOBILE_CACHE_TTL : STREAM_CACHE_TTL;

      // Use cache if available and not forcing refresh
      if (!forceRefresh && streamInfoCache.data && cacheAge < cacheTTL) {
        console.log("[StreamPlayer] Using cached stream info");
        setStreamInfo(streamInfoCache.data);
        setIsLoading(false);
        return;
      }

      // Throttle API calls
      if (now - lastFetchTime.current < 2000) {
        console.log("[StreamPlayer] Throttling API call");
        return;
      }
      lastFetchTime.current = now;

      try {
        console.log(
          "[StreamPlayer] Fetching stream info for camera:",
          currentCameraId
        );

        // Use unified polling to get cached status (prevents duplicate API calls)
        const cameraStatus = unifiedCameraPolling.getCachedStatus(currentCameraId);

        // Also fetch stream-specific info (playbackId, streamUrl) directly
        const streamResponse = await unifiedCameraService.getStreamInfo(currentCameraId);

        if (streamResponse.success && streamResponse.data) {
          const data = streamResponse.data;

          const newStreamInfo = {
            playbackId: data.playbackId || "",
            isActive: cameraStatus?.isStreaming ?? data.isActive, // Prefer cached status from unified polling
            streamType: data.format,
            streamUrl:
              data.streamUrl ||
              (await unifiedCameraService.getStreamUrl(currentCameraId)) ||
              "",
          };

          // Update cache
          streamInfoCache.data = newStreamInfo;
          streamInfoCache.timestamp = now;

          // Update state only if different (prevent unnecessary re-renders)
          setStreamInfo((prev) => {
            if (
              prev?.playbackId === newStreamInfo.playbackId &&
              prev?.isActive === newStreamInfo.isActive &&
              prev?.streamType === newStreamInfo.streamType &&
              prev?.streamUrl === newStreamInfo.streamUrl
            ) {
              // No change - return previous state to prevent re-render
              return prev;
            }

            // Only log significant changes to reduce console noise
            if (prev?.isActive !== newStreamInfo.isActive) {
              console.log("[StreamPlayer] 🔄 Stream status changed:", {
                wasActive: prev?.isActive,
                nowActive: newStreamInfo.isActive,
                playbackId: newStreamInfo.playbackId,
              });
            }
            return newStreamInfo;
          });
          setError(null);
          setLoadingRetry(0); // Reset retry counter on success
        } else {
          throw new Error(
            streamResponse.error ||
              "Failed to get stream info"
          );
        }
      } catch (err) {
        console.error("Failed to get stream info:", err);
        if (!isCameraOperation.current) {
          setError(
            `Failed to get stream info: ${
              err instanceof Error ? err.message : "Network error"
            }`
          );

          // For mobile, retry a few times
          if (isMobile && loadingRetry < 3) {
            console.log(
              `[StreamPlayer] Retrying fetch (${loadingRetry + 1}/3)`
            );
            setLoadingRetry((prev) => prev + 1);

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
    },
    [isMobile, loadingRetry, cameraId, selectedCamera?.publicKey]
  );

  // Listen for camera operations
  useEffect(() => {
    const handleCameraOperation = (e: CustomEvent) => {
      isCameraOperation.current = e.detail.inProgress;
    };

    window.addEventListener(
      "cameraOperation",
      handleCameraOperation as EventListener
    );
    return () => {
      window.removeEventListener(
        "cameraOperation",
        handleCameraOperation as EventListener
      );
    };
  }, []);

  useEffect(() => {
    // Fetch stream info for all cameras using standardized API
    fetchStreamInfo();
    // Reduce polling frequency to prevent blinking - now that status works correctly, we don't need aggressive polling
    const pollingInterval = isMobile ? 15000 : 10000; // 10s desktop, 15s mobile
    pollInterval.current = setInterval(
      () => fetchStreamInfo(),
      pollingInterval
    );

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
    console.log(
      `[StreamPlayer] Program ID: ${CAMERA_ACTIVATION_PROGRAM_ID.toString()}`
    );
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
      <div className="px-2">
        <div className="aspect-[9/16] md:aspect-video bg-gray-900 rounded-lg flex items-center justify-center">
          <p className="text-gray-400">
            Loading stream...
            {isMobile && loadingRetry > 0 && (
              <span className="block text-xs mt-1">
                Attempt {loadingRetry}/3
              </span>
            )}
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="px-2">
        <div className="aspect-[9/16] md:aspect-video w-full bg-gray-800 rounded-lg overflow-hidden flex items-center justify-center">
          <div className="text-center">
            <p className="text-gray-400">Failed to load stream</p>
            {isMobile && error && (
              <button
                className="mt-2 px-3 py-1 text-xs bg-primary text-white rounded"
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
      </div>
    );
  }

  // WebRTC only for testing (no fallback)
  if (useWebRTC) {
    return (
      <WebRTCStreamPlayer
        // Key changes when streamType changes to force reconnection with new stream
        key={`webrtc-${streamType}`}
        streamType={streamType}
        onError={(error) => {
          console.log("[StreamPlayer] WebRTC failed:", error);
          // Don't fallback during testing
        }}
      />
    );
  }

  // Handle Livepeer streams (Pi5 cameras and Jetson cameras)
  // Show Livepeer streams if playbackId exists, regardless of isActive status
  // because Jetson streams are always available
  if (streamInfo?.playbackId) {
    return (
      <div className="px-2">
        <div className="aspect-[9/16] md:aspect-video bg-black rounded-lg overflow-hidden relative">
          <Player
            title="Camera Stream"
            playbackId={streamInfo.playbackId}
            autoPlay
            muted
            controls={{
              autohide: 3000,
              hotkeys: false,
              defaultVolume: 0,
            }}
            aspectRatio="16to9"
            showPipButton={!isMobile}
            objectFit="contain"
            priority
            showLoadingSpinner={true}
            lowLatency
          />
          {/* Show helpful message for Jetson cameras - memoized to prevent re-renders */}
          {(() => {
            const currentCameraId =
              cameraId ||
              selectedCamera?.publicKey ||
              localStorage.getItem("directCameraId");
            const supportsLivepeer =
              currentCameraId &&
              unifiedCameraService.cameraSupports(
                currentCameraId,
                "hasLivepeerStreaming"
              );

            if (!supportsLivepeer) return null;

            return (
              <div className="absolute top-2 left-2 bg-black bg-opacity-70 text-white text-xs px-2 py-1 rounded">
                {streamInfo.isActive
                  ? "Live Stream Active"
                  : "Stream Available (Click Start to go Live)"}
              </div>
            );
          })()}
          <div className="absolute top-2 right-2">
            <button
              onClick={() => setUseWebRTC(true)}
              className="bg-primary bg-opacity-80 hover:bg-opacity-100 text-white text-xs px-2 py-1 rounded transition-all"
            >
              Try WebRTC
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Handle MJPEG streams (legacy - keeping for fallback)
  if (streamInfo?.streamType === "mjpeg" && streamInfo.streamUrl) {
    return (
      <div className="px-2">
        <div className="aspect-[9/16] md:aspect-video bg-black rounded-lg overflow-hidden">
          <img
            src={streamInfo.streamUrl}
            alt="Camera Stream"
            className="w-full h-full object-contain"
            style={{ imageRendering: "auto" }}
          />
        </div>
      </div>
    );
  }

  // Only show "Stream is offline" if we have no playback ID and no stream URL
  if (
    !streamInfo?.isActive &&
    !streamInfo?.playbackId &&
    !streamInfo?.streamUrl
  ) {
    return (
      <div className="px-2">
        <div className="aspect-[9/16] md:aspect-video w-full bg-gray-800 rounded-lg overflow-hidden flex items-center justify-center">
          <div className="text-center">
            <p className="text-gray-400">Stream is offline</p>
          </div>
        </div>
      </div>
    );
  }

  // Fallback for unknown stream types
  return (
    <div className="px-2">
      <div className="aspect-[9/16] md:aspect-video w-full bg-gray-800 rounded-lg overflow-hidden flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-400">Unsupported stream format</p>
        </div>
      </div>
    </div>
  );
});

StreamPlayer.displayName = "StreamPlayer";

export { StreamPlayer };
