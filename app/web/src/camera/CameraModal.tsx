/* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unused-vars */
import { useState, useEffect } from 'react';
import { Dialog } from '@headlessui/react';
import { X, ExternalLink, Camera, User } from 'lucide-react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { unifiedCameraService } from './unified-camera-service';
import { useCamera } from './CameraProvider';
import { CONFIG } from '../core/config';

interface CameraModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCheckStatusChange?: (isCheckedIn: boolean) => void;
  camera: {
    id: string;
    owner: string;
    ownerDisplayName?: string;
    ownerPfpUrl?: string;
    isLive: boolean;
    isStreaming: boolean;
    status: 'ok' | 'error' | 'offline';
    lastSeen?: number;
    // activityCounter?: number; // Replaced with live active users analytics
    model?: string;
    // New properties for development info
    showDevInfo?: boolean;
    defaultDevCamera?: string;
  };
}

export function CameraModal({ isOpen, onClose, onCheckStatusChange, camera }: CameraModalProps) {
  const { primaryWallet } = useDynamicContext();
  // Use unified check-in state from CameraProvider (Phase 3 Privacy Architecture)
  const { isCheckedIn, isCheckingIn, checkInError, checkIn, checkOut, refreshCheckInStatus } = useCamera();
  const [error, setError] = useState<string | null>(null);

  // Configuration states for Jetson camera features
  const [faceVisualization, setFaceVisualization] = useState(false);
  const [poseVisualization, setPoseVisualization] = useState(false);
  const [configLoading, setConfigLoading] = useState(false);

  // State for active users analytics
  const [activeUsersCount, setActiveUsersCount] = useState<number>(0);
  const [loadingActiveUsers, setLoadingActiveUsers] = useState(false);

  // Check if current user is the owner
  const isOwner = primaryWallet?.address === camera.owner;

  // Check if this is a Jetson camera (has advanced features)
  const isJetsonCamera = camera.id === CONFIG.JETSON_CAMERA_PDA || camera.model === 'jetson' || camera.model === 'jetson_orin_nano';

  // Notify parent component when check-in status changes (backwards compatibility)
  useEffect(() => {
    if (onCheckStatusChange) {
      console.log("[CameraModal] Notifying parent of check-in status:", isCheckedIn);
      onCheckStatusChange(isCheckedIn);
    }
  }, [isCheckedIn, onCheckStatusChange]);

  // Refresh check-in status and active users when modal opens
  useEffect(() => {
    if (!isOpen || !camera.id) return;

    console.log("[CameraModal] Modal opened - refreshing check-in status and active users");

    // Refresh check-in status from context (queries Jetson)
    if (primaryWallet?.address) {
      refreshCheckInStatus();
    }

    // Always fetch active users (doesn't need wallet)
    fetchActiveUsersForCamera();

    // Poll active users only (much less frequently - every 15 seconds)
    const intervalId = setInterval(() => {
      console.log("[CameraModal] Periodic active users check");
      fetchActiveUsersForCamera();
    }, 15000);

    return () => {
      console.log("[CameraModal] Cleaning up active users check");
      clearInterval(intervalId);
    };
  }, [isOpen, camera.id, primaryWallet?.address, refreshCheckInStatus]);

  // Clear errors and load configuration when modal opens
  useEffect(() => {
    const loadConfiguration = async () => {
      if (isOpen) {
        setError(null);
        
        // Load current configuration for Jetson cameras
        if (isJetsonCamera && camera.id) {
          try {
            console.log('[CameraModal] Loading current computer vision state...');

            // Load visualization states from localStorage (persist across modal opens)
            const storedFaceViz = localStorage.getItem(`jetson_face_viz_${camera.id}`) === 'true';
            const storedPoseViz = localStorage.getItem(`jetson_pose_viz_${camera.id}`) === 'true';

            setFaceVisualization(storedFaceViz);
            setPoseVisualization(storedPoseViz);

            console.log('[CameraModal] Loaded visualization states - Face:', storedFaceViz, 'Pose:', storedPoseViz);

            console.log('[CameraModal] Computer vision configuration loaded successfully');
          } catch (error) {
            console.error('Error loading computer vision configuration:', error);
            // Set defaults on error
            setFaceVisualization(false);
            setPoseVisualization(false);
          }
        } else {
          // Reset states for non-Jetson cameras
          setFaceVisualization(false);
          setPoseVisualization(false);
        }
      }
    };

    loadConfiguration();
  }, [isOpen, isJetsonCamera, camera.id]);

  // Expose test function to window for debugging
  useEffect(() => {
    if (isJetsonCamera && camera.id) {
      (window as any).testVisualizationEndpoints = testVisualizationEndpoints;
    }
    
    return () => {
      delete (window as any).testVisualizationEndpoints;
    };
  }, [isJetsonCamera, camera.id]);


  // NOTE: Session status checking is now handled by CameraProvider.refreshCheckInStatus()
  // which queries Jetson (the source of truth) and updates localStorage as needed

  // NEW PRIVACY ARCHITECTURE: Sessions are managed off-chain by Jetson
  // Active users count comes from Jetson's session management
  const fetchActiveUsersForCamera = async () => {
    if (!camera.id) return;

    try {
      setLoadingActiveUsers(true);
      console.log('[CameraModal] Fetching active users from Jetson for camera:', camera.id);

      // Try to get active session count from Jetson
      // This will be implemented in Phase 3 - for now, use camera status
      const statusResult = await unifiedCameraService.getStatus(camera.id);

      if (statusResult.success && statusResult.data) {
        // Jetson returns active session count in status (to be added in Phase 3)
        const count = (statusResult.data as any).activeSessionCount || 0;
        setActiveUsersCount(count);
        console.log('[CameraModal] Active users count from Jetson:', count);
      } else {
        console.log('[CameraModal] Could not fetch active users from Jetson');
        setActiveUsersCount(0);
      }
    } catch (error) {
      console.error('[CameraModal] Error fetching active users:', error);
      // Don't reset count on error - keep showing last known state
    } finally {
      setLoadingActiveUsers(false);
    }
  };

  const handleCameraExplorerClick = () => {
    window.open(`https://solscan.io/account/${camera.id}?cluster=devnet`, '_blank');
  };

  const handleOwnerExplorerClick = () => {
    window.open(`https://solscan.io/account/${camera.owner}?cluster=devnet`, '_blank');
  };

  // Format address for display
  const formatAddress = (address: string, start = 6, end = 6) => {
    if (!address) return '';
    return `${address.slice(0, start)}...${address.slice(-end)}`;
  };

  const handleDevCameraClick = () => {
    if (camera.defaultDevCamera) {
      // Redirect to the correct camera page URL
      const baseUrl = window.location.origin;
      window.location.href = `${baseUrl}/app/camera/${camera.defaultDevCamera}`;
    }
  };

  // NOTE: Check-in/check-out timeline events are now created by the Jetson camera
  // via buffer_checkin_activity() and buffer_checkout_activity() for proper encryption
  // and privacy-preserving timeline architecture. Frontend no longer emits these directly.

  // Test visualization endpoints directly
  const testVisualizationEndpoints = async () => {
    if (!isJetsonCamera || !camera.id) return;
    
    console.log('[CameraModal] Testing visualization endpoints...');
    
    try {
      // Test both visualization endpoints through unified camera service
      const currentCameraId = camera.id;
      
      console.log('Testing face visualization endpoint...');
      const faceResult = await unifiedCameraService.toggleFaceVisualization?.(currentCameraId, true);
      console.log('Face viz result:', faceResult);
      
      console.log('Testing gesture visualization endpoint...');
      const gestureResult = await unifiedCameraService.toggleGestureVisualization?.(currentCameraId, true);
      console.log('Gesture viz result:', gestureResult);
      

      
    } catch (error) {
      console.error('Error testing visualization endpoints:', error);
    }
  };


  // Handle face visualization toggle
  const handleFaceVisualizationToggle = async () => {
    setConfigLoading(true);
    try {
      const newState = !faceVisualization;
      console.log('[CameraModal] Toggling face visualization to:', newState);
      
      const result = await unifiedCameraService.toggleFaceVisualization(camera.id, newState);
      
      if (result.success) {
        setFaceVisualization(newState);
        // Persist state to localStorage
        localStorage.setItem(`jetson_face_viz_${camera.id}`, newState.toString());
        console.log('[CameraModal] Face visualization toggled successfully to:', newState);
        
        // Force refresh the stream to show changes immediately
        const streamElements = document.querySelectorAll('img[src*="/stream"], video');
        streamElements.forEach(element => {
          if (element instanceof HTMLImageElement && element.src.includes('/stream')) {
            const currentSrc = element.src;
            element.src = '';
            setTimeout(() => {
              element.src = currentSrc + (currentSrc.includes('?') ? '&' : '?') + 't=' + Date.now();
            }, 100);
          }
        });
      } else {
        console.error('[CameraModal] Failed to toggle face visualization:', result.error);
        setError(result.error || 'Failed to toggle face visualization');
      }
    } catch (error) {
      console.error('[CameraModal] Error toggling face visualization:', error);
      setError(error instanceof Error ? error.message : 'Failed to toggle face visualization');
    } finally {
      setConfigLoading(false);
    }
  };

  // Handle pose visualization toggle
  const handlePoseVisualizationToggle = async () => {
    setConfigLoading(true);
    try {
      const newState = !poseVisualization;
      console.log('[CameraModal] Toggling pose visualization to:', newState);

      const result = await unifiedCameraService.togglePoseVisualization(camera.id, newState);

      if (result.success) {
        setPoseVisualization(newState);
        // Persist state to localStorage
        localStorage.setItem(`jetson_pose_viz_${camera.id}`, newState.toString());
        console.log('[CameraModal] Pose visualization toggled successfully to:', newState);

        // Force refresh the stream to show changes immediately
        const streamElements = document.querySelectorAll('img[src*="/stream"], video');
        streamElements.forEach(element => {
          if (element instanceof HTMLImageElement && element.src.includes('/stream')) {
            const currentSrc = element.src;
            element.src = '';
            setTimeout(() => {
              element.src = currentSrc + (currentSrc.includes('?') ? '&' : '?') + 't=' + Date.now();
            }, 100);
          }
        });
      } else {
        console.error('[CameraModal] Failed to toggle pose visualization:', result.error);
        setError(result.error || 'Failed to toggle pose visualization');
      }
    } catch (error) {
      console.error('[CameraModal] Error toggling pose visualization:', error);
      setError(error instanceof Error ? error.message : 'Failed to toggle pose visualization');
    } finally {
      setConfigLoading(false);
    }
  };

  // Check-in handler using unified context (Phase 3 Privacy Architecture)
  const handleCheckIn = async () => {
    setError(null);

    console.log('[CameraModal] Starting check-in via CameraProvider context...');
    const success = await checkIn();

    if (success) {
      console.log('[CameraModal] Check-in successful');
      // Refresh active users count
      await fetchActiveUsersForCamera();
    } else {
      // Show error from context
      setError(checkInError || 'Failed to check in to camera');
    }
  };

  // Check-out handler using unified context (Phase 3 Privacy Architecture)
  const handleCheckOut = async () => {
    setError(null);

    console.log('[CameraModal] Starting check-out via CameraProvider context...');
    const success = await checkOut();

    if (success) {
      console.log('[CameraModal] Check-out successful');
      // Refresh active users count
      await fetchActiveUsersForCamera();
    } else {
      // Show error from context
      setError(checkInError || 'Failed to check out from camera');
    }
  };

  if (!isOpen) return null;

  return (
    <Dialog
      open={isOpen}
      onClose={onClose}
      className="relative z-[100]"
    >
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/30" aria-hidden="true" />

      {/* Full-screen container */}
      <div className="fixed inset-0 flex items-end sm:items-center justify-center p-2 sm:p-0">
        <Dialog.Panel className="mx-auto w-full sm:w-[360px] rounded-xl bg-white shadow-xl">
          {/* Header with close button */}
          <div className="flex items-center justify-between p-3 border-b border-gray-100">
            <Dialog.Title className="text-base font-medium">
              Camera Details
            </Dialog.Title>
            <button
              onClick={onClose}
              className="p-1 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <X className="w-4 h-4 text-gray-500" />
            </button>
          </div>

          {/* Camera Content */}
          <div className="p-4">
            {camera.showDevInfo ? (
              // Development section when no camera is connected
              <div className="space-y-4">
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 mb-3">
                  <h3 className="text-sm font-medium text-yellow-800 mb-1">Development Mode</h3>
                  <p className="text-xs text-yellow-700 mb-3">
                    No camera is currently connected. Connect to a physical camera below:
                  </p>
                  <div className="space-y-2">
                    <button
                      onClick={handleDevCameraClick}
                      className="text-xs bg-yellow-100 hover:bg-yellow-200 text-yellow-800 px-3 py-1.5 rounded flex items-center justify-center w-full transition-colors"
                    >
                      Connect to Pi5 <ExternalLink className="h-3 w-3 ml-1" />
                    </button>
                    <button
                      onClick={() => {
                        const baseUrl = window.location.origin;
                        window.location.href = `${baseUrl}/app/camera/${CONFIG.JETSON_CAMERA_PDA}`;
                      }}
                      className="text-xs bg-blue-100 hover:bg-blue-200 text-blue-800 px-3 py-1.5 rounded flex items-center justify-center w-full transition-colors"
                    >
                      Connect to Orin Nano <ExternalLink className="h-3 w-3 ml-1" />
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              // Original camera details layout
              <>
                {/* Camera PDA */}
                <div className="flex items-center mb-4">
                  <div className="w-10 h-10 rounded-full bg-blue-50 flex-shrink-0 flex items-center justify-center overflow-hidden">
                    <Camera className="w-5 h-5 text-blue-500" />
                  </div>
                  <div className="ml-3 flex-1">
                    <div className="text-sm text-gray-700">Camera</div>
                    <div className="text-sm font-medium">{formatAddress(camera.id)}</div>
                  </div>
                  <button
                    onClick={handleCameraExplorerClick}
                    className="text-xs text-blue-600 hover:text-blue-700 transition-colors flex items-center"
                  >
                    View <ExternalLink className="w-3 h-3 ml-1" />
                  </button>
                </div>

                {/* Owner */}
                <div className="flex items-center mb-4">
                  <div className="w-10 h-10 rounded-full bg-gray-100 flex-shrink-0 flex items-center justify-center overflow-hidden ml-0">
                    <User className="w-5 h-5 text-gray-500" />
                  </div>
                  <div className="ml-3 flex-1">
                    <div className="text-sm text-gray-700 flex items-center">
                      Owner
                      {isOwner && <span className="ml-2 text-xs bg-green-100 text-green-800 px-1.5 py-0.5 rounded">you</span>}
                    </div>
                    <div className="text-sm font-medium">
                      {formatAddress(camera.owner, 9, 5)}
                    </div>
                  </div>
                  <button
                    onClick={handleOwnerExplorerClick}
                    className="text-xs text-blue-600 hover:text-blue-700 transition-colors flex items-center"
                  >
                    View <ExternalLink className="w-3 h-3 ml-1" />
                  </button>
                </div>

                {/* Camera Name */}
                {camera.ownerDisplayName && (
                  <div className="mb-4">
                    <div className="text-sm text-gray-700">Camera Name</div>
                    <div className="text-sm">{camera.ownerDisplayName}</div>
                  </div>
                )}

                {/* Type */}
                <div className="mb-4">
                  <div className="text-sm text-gray-700">Type</div>
                  <div className="text-sm">{camera.model || "pi5"}</div>
                </div>

                {/* Active Users Analytics */}
                <div className="mb-4">
                  <div className="text-sm text-gray-700 flex items-center">
                    Users checked in
                    {loadingActiveUsers && (
                      <div className="ml-2 w-3 h-3 border border-gray-400 border-t-transparent rounded-full animate-spin"></div>
                    )}
                  </div>
                  <div className="flex items-center mt-1">
                    <span className={`w-2 h-2 rounded-full mr-2 ${
                      activeUsersCount > 0 ? 'bg-green-500' : 'bg-gray-400'
                    }`}></span>
                    <span className={`text-sm font-medium ${
                      activeUsersCount > 0 ? 'text-green-600' : 'text-gray-500'
                    }`}>
                      {activeUsersCount} {activeUsersCount === 1 ? 'user' : 'users'} currently active
                    </span>
                  </div>
                </div>

                {/* Computer Vision Controls - Only for Jetson cameras */}
                {isJetsonCamera && (
                  <div className="mb-4 pt-4 border-t border-gray-200">
                    <div className="space-y-2">
                      {/* Face Visualization Toggle */}
                      <div className="flex items-center justify-between py-1">
                        <div className="flex-1">
                          <div className="text-sm font-medium">User Recognition Overlay</div>
                          <div className="text-xs text-gray-500">Shows body detection and recognizes enrolled users</div>
                        </div>
                        <button
                          onClick={handleFaceVisualizationToggle}
                          disabled={configLoading}
                          className={`ml-3 relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                            faceVisualization
                              ? 'bg-blue-600 hover:bg-blue-700'
                              : 'bg-gray-200 hover:bg-gray-300'
                          } ${configLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                        >
                          <span
                            className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                              faceVisualization ? 'translate-x-5' : 'translate-x-0.5'
                            }`}
                          />
                        </button>
                      </div>

                      {/* Pose Visualization Toggle */}
                      <div className="flex items-center justify-between py-1">
                        <div className="flex-1">
                          <div className="text-sm font-medium">Pose Skeleton Overlay</div>
                          <div className="text-xs text-gray-500">Shows body pose skeleton tracking</div>
                        </div>
                        <button
                          onClick={handlePoseVisualizationToggle}
                          disabled={configLoading}
                          className={`ml-3 relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                            poseVisualization
                              ? 'bg-blue-600 hover:bg-blue-700'
                              : 'bg-gray-200 hover:bg-gray-300'
                          } ${configLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                        >
                          <span
                            className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                              poseVisualization ? 'translate-x-5' : 'translate-x-0.5'
                            }`}
                          />
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                {/* Error Message */}
                {error && (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4">
                    <h3 className="text-sm font-medium text-red-800">Error</h3>
                    <p className="text-xs text-red-700 mt-1">{error}</p>
                  </div>
                )}

                {/* Check In/Out Button */}
                {isCheckedIn ? (
                  <button
                    onClick={handleCheckOut}
                    disabled={isCheckingIn}
                    className="w-full bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg transition-colors"
                  >
                    {isCheckingIn ? 'Processing...' : 'Check Out'}
                  </button>
                ) : (
                  <button
                    onClick={handleCheckIn}
                    disabled={isCheckingIn}
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
                  >
                    {isCheckingIn ? 'Processing...' : 'Check In'}
                  </button>
                )}
              </>
            )}
          </div>
        </Dialog.Panel>
      </div>
    </Dialog>
  );
}