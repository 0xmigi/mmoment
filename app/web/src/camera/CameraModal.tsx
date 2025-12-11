/* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unused-vars */
import { useState, useEffect } from 'react';
import { Dialog } from '@headlessui/react';
import { X, ExternalLink, Camera, User, KeyRound, Loader2 } from 'lucide-react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { isSolanaWallet } from '@dynamic-labs/solana';
import { unifiedCameraService } from './unified-camera-service';
import { useCamera } from './CameraProvider';
import { CONFIG } from '../core/config';
import { useUserSessionChain, fetchAuthorityPublicKey } from '../hooks/useUserSessionChain';
import { useProgram, findUserSessionChainPDA } from '../anchor/setup';
import { useConnection } from '@solana/wallet-adapter-react';
import { PublicKey, SystemProgram } from '@solana/web3.js';

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

  // Session keychain state (required for privacy architecture)
  const { hasSessionChain, isLoading: sessionChainLoading, refetch: refetchSessionChain } = useUserSessionChain();
  const { program } = useProgram();
  const { connection } = useConnection();
  const [isCreatingSessionChain, setIsCreatingSessionChain] = useState(false);
  const [sessionChainError, setSessionChainError] = useState<string | null>(null);

  // Configuration state for Jetson camera CV overlay (consolidated from face+pose)
  const [cvOverlayEnabled, setCvOverlayEnabled] = useState(false);
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

            // Load CV overlay state from localStorage (persist across modal opens)
            // Check new key first, then migrate from old keys for backwards compatibility
            let storedCvOverlay = localStorage.getItem(`jetson_cv_overlay_${camera.id}`) === 'true';

            // Migrate from old separate keys if new key doesn't exist
            if (!localStorage.getItem(`jetson_cv_overlay_${camera.id}`)) {
              const oldFaceViz = localStorage.getItem(`jetson_face_viz_${camera.id}`) === 'true';
              const oldPoseViz = localStorage.getItem(`jetson_pose_viz_${camera.id}`) === 'true';
              storedCvOverlay = oldFaceViz || oldPoseViz;
              // Clean up old keys
              localStorage.removeItem(`jetson_face_viz_${camera.id}`);
              localStorage.removeItem(`jetson_pose_viz_${camera.id}`);
              if (storedCvOverlay) {
                localStorage.setItem(`jetson_cv_overlay_${camera.id}`, 'true');
              }
            }

            setCvOverlayEnabled(storedCvOverlay);
            console.log('[CameraModal] Loaded CV overlay state:', storedCvOverlay);

            console.log('[CameraModal] Computer vision configuration loaded successfully');
          } catch (error) {
            console.error('Error loading computer vision configuration:', error);
            // Set default on error
            setCvOverlayEnabled(false);
          }
        } else {
          // Reset state for non-Jetson cameras
          setCvOverlayEnabled(false);
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


  // Handle CV overlay toggle (consolidated face + pose into one toggle)
  // This switches between 'clean' and 'annotated' WebRTC streams
  const handleCvOverlayToggle = async () => {
    setConfigLoading(true);
    try {
      const newState = !cvOverlayEnabled;
      console.log('[CameraModal] Toggling CV overlay to:', newState);

      // Toggle both face and pose visualization on the Jetson
      // The annotated stream includes all CV overlays when enabled
      const faceResult = await unifiedCameraService.toggleFaceVisualization(camera.id, newState);
      const poseResult = await unifiedCameraService.togglePoseVisualization(camera.id, newState);

      if (faceResult.success && poseResult.success) {
        setCvOverlayEnabled(newState);
        // Persist state to localStorage
        localStorage.setItem(`jetson_cv_overlay_${camera.id}`, newState.toString());
        console.log('[CameraModal] CV overlay toggled successfully to:', newState);

        // Dispatch custom event to notify StreamPlayer to switch stream type
        window.dispatchEvent(new CustomEvent('visualizationToggle', {
          detail: { cameraId: camera.id, type: 'cv_overlay', enabled: newState }
        }));
      } else {
        const errorMsg = faceResult.error || poseResult.error || 'Failed to toggle CV overlay';
        console.error('[CameraModal] Failed to toggle CV overlay:', errorMsg);
        setError(errorMsg);
      }
    } catch (error) {
      console.error('[CameraModal] Error toggling CV overlay:', error);
      setError(error instanceof Error ? error.message : 'Failed to toggle CV overlay');
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

  // Handler for creating session keychain (one-time setup before first check-in)
  const handleCreateSessionChain = async () => {
    if (!primaryWallet?.address) {
      setSessionChainError('Wallet not connected');
      return;
    }

    if (!isSolanaWallet(primaryWallet)) {
      setSessionChainError('Not a Solana wallet');
      return;
    }

    if (!program || !connection) {
      setSessionChainError('Please wait for wallet to fully load and try again');
      return;
    }

    setIsCreatingSessionChain(true);
    setSessionChainError(null);

    try {
      const userPublicKey = new PublicKey(primaryWallet.address);
      const [sessionChainPda] = findUserSessionChainPDA(userPublicKey);

      // Fetch authority public key from backend
      console.log('[CameraModal] Fetching authority for session chain...');
      const authority = await fetchAuthorityPublicKey();
      if (!authority) {
        throw new Error('Could not connect to backend. Please try again.');
      }

      console.log('[CameraModal] Creating session chain...');
      console.log('[CameraModal] User:', userPublicKey.toString());
      console.log('[CameraModal] Authority:', authority.toString());

      // Build the transaction
      const tx = await program.methods
        .createUserSessionChain()
        .accounts({
          user: userPublicKey,
          authority: authority,
          userSessionChain: sessionChainPda,
          systemProgram: SystemProgram.programId,
        })
        .transaction();

      // Get blockhash
      const { blockhash, lastValidBlockHeight } = await connection.getLatestBlockhash();
      tx.recentBlockhash = blockhash;
      tx.feePayer = userPublicKey;

      // Sign with Dynamic wallet
      console.log('[CameraModal] Requesting signature...');
      const signer = await (primaryWallet as any).getSigner();
      const signedTx = await signer.signTransaction(tx);

      // Send and confirm
      console.log('[CameraModal] Sending transaction...');
      const signature = await connection.sendRawTransaction(signedTx.serialize());
      await connection.confirmTransaction({ signature, blockhash, lastValidBlockHeight }, 'confirmed');

      console.log('[CameraModal] Session keychain created successfully!');

      // Refresh the session chain status
      await refetchSessionChain();

    } catch (error: any) {
      console.error('[CameraModal] Session chain creation error:', error);

      let errorMessage = 'Failed to create session keychain';
      if (error.message?.includes('User rejected')) {
        errorMessage = 'Transaction cancelled. Please try again.';
      } else if (error.message?.includes('insufficient funds')) {
        errorMessage = 'Insufficient SOL for transaction (~0.003 SOL needed)';
      } else if (error.message?.includes('already in use')) {
        errorMessage = 'Session keychain already exists!';
        // Refresh to update UI
        await refetchSessionChain();
      } else if (error.message) {
        errorMessage = error.message;
      }

      setSessionChainError(errorMessage);
    } finally {
      setIsCreatingSessionChain(false);
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
                      className="text-xs bg-primary-light hover:bg-primary-muted text-primary px-3 py-1.5 rounded flex items-center justify-center w-full transition-colors"
                    >
                      Connect to Pi5 <ExternalLink className="h-3 w-3 ml-1" />
                    </button>
                    <button
                      onClick={() => {
                        const baseUrl = window.location.origin;
                        window.location.href = `${baseUrl}/app/camera/${CONFIG.JETSON_CAMERA_PDA}`;
                      }}
                      className="text-xs bg-primary-light hover:bg-primary-muted text-primary px-3 py-1.5 rounded flex items-center justify-center w-full transition-colors"
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
                  <div className="w-10 h-10 rounded-full bg-primary-light flex-shrink-0 flex items-center justify-center overflow-hidden">
                    <Camera className="w-5 h-5 text-primary" />
                  </div>
                  <div className="ml-3 flex-1">
                    <div className="text-sm text-gray-700">Camera</div>
                    <div className="text-sm font-medium">{formatAddress(camera.id)}</div>
                  </div>
                  <button
                    onClick={handleCameraExplorerClick}
                    className="text-xs text-primary hover:text-primary-hover transition-colors flex items-center"
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
                    className="text-xs text-primary hover:text-primary-hover transition-colors flex items-center"
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
                    {/* CV Overlay Toggle - consolidated face + pose */}
                    <div className="flex items-center justify-between py-1">
                      <div className="flex-1">
                        <div className="text-sm font-medium">Computer Vision Overlay</div>
                        <div className="text-xs text-gray-500">Shows user recognition, pose tracking, and activity overlays</div>
                      </div>
                      <button
                        onClick={handleCvOverlayToggle}
                        disabled={configLoading}
                        className={`ml-3 relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 ${
                          cvOverlayEnabled
                            ? 'bg-primary hover:bg-primary-hover'
                            : 'bg-gray-200 hover:bg-gray-300'
                        } ${configLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                      >
                        <span
                          className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                            cvOverlayEnabled ? 'translate-x-5' : 'translate-x-0.5'
                          }`}
                        />
                      </button>
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

                {/* Check In/Out Button or Session Keychain Setup */}
                {isCheckedIn ? (
                  // User is checked in - show check out button
                  <button
                    onClick={handleCheckOut}
                    disabled={isCheckingIn}
                    className="w-full bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg transition-colors"
                  >
                    {isCheckingIn ? 'Processing...' : 'Check Out'}
                  </button>
                ) : !hasSessionChain && !sessionChainLoading ? (
                  // User needs to create session keychain first (one-time setup)
                  <div className="space-y-3">
                    <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <KeyRound className="w-5 h-5 text-amber-600" />
                        <h3 className="font-medium text-amber-900">One-Time Setup Required</h3>
                      </div>
                      <p className="text-sm text-amber-800 mb-1">
                        Create your Session Keychain to ensure your camera history is permanently stored on-chain.
                      </p>
                      <p className="text-xs text-amber-700">
                        This is a one-time action (~0.003 SOL) that enables decentralized access to your session history.
                      </p>
                    </div>

                    {sessionChainError && (
                      <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                        <p className="text-sm text-red-700">{sessionChainError}</p>
                      </div>
                    )}

                    <button
                      onClick={handleCreateSessionChain}
                      disabled={isCreatingSessionChain}
                      className="w-full flex items-center justify-center gap-2 bg-amber-600 hover:bg-amber-700 text-white px-4 py-3 rounded-lg transition-colors font-medium disabled:opacity-50"
                    >
                      {isCreatingSessionChain ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin" />
                          Creating Keychain...
                        </>
                      ) : (
                        <>
                          <KeyRound className="w-4 h-4" />
                          Create Session Keychain
                        </>
                      )}
                    </button>
                  </div>
                ) : sessionChainLoading ? (
                  // Loading session chain status
                  <button
                    disabled
                    className="w-full flex items-center justify-center gap-2 bg-gray-300 text-gray-600 px-4 py-2 rounded-lg"
                  >
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Loading...
                  </button>
                ) : (
                  // User has session keychain - show normal check in button
                  <button
                    onClick={handleCheckIn}
                    disabled={isCheckingIn}
                    className="w-full bg-primary hover:bg-primary-hover text-white px-4 py-2 rounded-lg transition-colors"
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