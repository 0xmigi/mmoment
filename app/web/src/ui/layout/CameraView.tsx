/* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unused-vars, @typescript-eslint/no-non-null-assertion */
import { useDynamicContext, useEmbeddedWallet } from '@dynamic-labs/sdk-react-core';
import { useRef, useState, useEffect, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import { Timeline } from '../../timeline/Timeline';
import MediaGallery from '../../media/Gallery';
import { CameraControls } from './MobileControls';
import { ToastMessage } from '../../core/types/toast';
import { ToastContainer } from '../feedback/ToastContainer';
import { StreamPlayer } from '../../media/StreamPlayer';
import { useCamera, CameraData, fetchCameraByPublicKey } from '../../camera/CameraProvider';
import { PublicKey, Connection, Transaction, TransactionInstruction } from '@solana/web3.js';
import { useProgram, CAMERA_ACTIVATION_PROGRAM_ID } from '../../anchor/setup';
import { unifiedCameraService } from '../../camera/unified-camera-service';
import { unifiedIpfsService } from '../../storage/ipfs/unified-ipfs-service';
import { useConnection } from '@solana/wallet-adapter-react';
import { TransactionModal } from '../../auth/components/TransactionModal';
import { StopCircle, Play, Camera, Video, Loader, Link2, CheckCircle } from 'lucide-react';
import { useCameraStatus } from '../../camera/useCameraStatus';
import { CameraModal } from '../../camera/CameraModal';
import { cameraStatus } from '../../camera/camera-status';
import { CONFIG } from '../../core/config';
import { Program, AnchorProvider } from '@coral-xyz/anchor';
import { IDL } from '../../anchor/idl';

type TimelineEventType =
  | 'photo_captured'
  | 'video_recorded'
  | 'stream_started'
  | 'stream_ended'
  | 'initialization';

interface TimelineEvent {
  type: TimelineEventType;
  user: {
    address: string;
    username?: string;
    displayName?: string;
    pfpUrl?: string;
  };
  timestamp: number;
  transactionId?: string;
  mediaUrl?: string;
  cameraId?: string;
}


// Update the CameraIdDisplay component to add a forced refresh when the modal is closed
const CameraIdDisplay = ({ cameraId, selectedCamera, cameraAccount, timelineRef }: {
  cameraId: string | undefined;
  selectedCamera: CameraData | null;
  cameraAccount: string | null;
  timelineRef?: React.RefObject<{ refreshTimeline?: () => void; refreshEvents?: () => void }>;
}) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const cameraStatus = useCameraStatus(selectedCamera?.publicKey || cameraAccount || cameraId || '');
  const { primaryWallet } = useDynamicContext();
  const { connection } = useConnection();
  const [isCheckedIn, setIsCheckedIn] = useState(false);

  // Get the direct ID from localStorage if available (most reliable source)
  const directId = localStorage.getItem('directCameraId');

  // Determine which ID to display (in order of preference)
  const displayId = selectedCamera?.publicKey || cameraAccount || directId || cameraId;

  // Simple blockchain check
  const checkBlockchainStatus = async () => {
    if (!displayId || !primaryWallet?.address || !connection) return;

    try {
      console.log(`[BLOCKCHAIN CHECK] Checking status for ${displayId.slice(0, 8)}...`);
      
      const provider = new AnchorProvider(
        connection,
        {
          publicKey: new PublicKey(primaryWallet.address),
          signTransaction: async (tx: Transaction) => tx,
          signAllTransactions: async (txs: Transaction[]) => txs,
        } as any,
        { commitment: 'confirmed' }
      );

      const program = new Program(IDL as any, CAMERA_ACTIVATION_PROGRAM_ID, provider);

      const [sessionPda] = PublicKey.findProgramAddressSync(
        [
          Buffer.from('session'),
          new PublicKey(primaryWallet.address).toBuffer(),
          new PublicKey(displayId).toBuffer()
        ],
        CAMERA_ACTIVATION_PROGRAM_ID
      );

      try {
        const session = await program.account.userSession.fetch(sessionPda);
        console.log(`[BLOCKCHAIN CHECK] ✅ SESSION FOUND:`, session);
        setIsCheckedIn(true);
      } catch (err) {
        console.log(`[BLOCKCHAIN CHECK] ❌ NO SESSION FOUND`);
        setIsCheckedIn(false);
      }
    } catch (err) {
      console.error('[BLOCKCHAIN CHECK] ERROR:', err);
    }
  };

  // Check blockchain status immediately when component loads or camera changes
  useEffect(() => {
    if (displayId && primaryWallet?.address && connection) {
      console.log(`[CAMERA ID DISPLAY] Component loaded/changed - checking blockchain status`);
      checkBlockchainStatus();
    }
  }, [displayId, primaryWallet?.address, connection]);

  // Add a function to handle status change from modal
  const handleCheckStatusChange = (newStatus: boolean) => {
    console.log("Status change from modal:", newStatus);
    setIsCheckedIn(newStatus);
    // If timelineRef exists, refresh it
    if (timelineRef?.current?.refreshTimeline) {
      timelineRef.current?.refreshTimeline();
    }
  };

  // Add a function to handle modal close with a forced refresh
  const handleModalClose = () => {
    setIsModalOpen(false);
    // Check blockchain status when modal closes
    checkBlockchainStatus();
  };

  // Handle case where ID might not be valid
  const formatId = (id: string | undefined | null) => {
    if (!id) return 'None';
    try {
      return `${id.slice(0, 6)}...${id.slice(-6)}`;
    } catch (_) {
      return id;
    }
  };

  // The default camera PDA for development
  const defaultDevCameraPda = 'EugmfUyT8oZuP9QnCpBicrxjt1RMnavaAQaPW6YecYeA';

  return (
    <div>
      <h2 className="text-xl font-semibold">Camera</h2>
      {!displayId || displayId === 'None' ? (
        <div
          onClick={() => setIsModalOpen(true)}
          className="text-sm text-red-500 font-medium hover:text-red-600 cursor-pointer"
        >
          No camera connected
        </div>
      ) : (
        <div
          onClick={() => setIsModalOpen(true)}
          className="text-sm text-gray-600 hover:text-blue-600 transition-colors cursor-pointer flex items-center"
        >
          <span>id: {formatId(displayId)}</span>
          <span id="check-in-status-icon">
            {isCheckedIn ? (
              <CheckCircle className="w-3.5 h-3.5 ml-1.5 text-green-500" />
            ) : (
              <Link2 className="w-3.5 h-3.5 ml-1.5 text-blue-500" />
            )}
          </span>
        </div>
      )}

      <CameraModal
        isOpen={isModalOpen}
        onClose={handleModalClose}
        onCheckStatusChange={handleCheckStatusChange}
        camera={{
          id: displayId && displayId !== 'None' ? displayId : '',
          owner: selectedCamera?.owner || cameraStatus.owner || '',
          ownerDisplayName: selectedCamera?.metadata?.name || "newProgramCamera",
          model: selectedCamera?.metadata?.model || "pi5",
          isLive: cameraStatus.isLive || false,
          isStreaming: cameraStatus.isStreaming || false,
          status: 'ok',
          activityCounter: selectedCamera?.activityCounter || 226,
          // Add development info when no camera is connected
          showDevInfo: !displayId || displayId === 'None',
          defaultDevCamera: defaultDevCameraPda
        }}
      />
    </div>
  );
};

export function CameraView() {
  const { primaryWallet, user } = useDynamicContext();
  const { cameraId } = useParams<{ cameraId: string }>();
  useEmbeddedWallet();
  const { selectedCamera, setSelectedCamera, fetchCameraById } = useCamera();
  const { program } = useProgram();
  const { connection } = useConnection();
  const timelineRef = useRef<{ 
    addEvent?: (event: Omit<TimelineEvent, 'id'>) => void; 
    refreshTimeline?: () => void;
    refreshEvents?: () => void;
  }>(null);
  const [cameraAccount, setCameraAccount] = useState<string | null>(null);
  const [isLive, setIsLive] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [currentToast, setCurrentToast] = useState<ToastMessage | null>(null);
  const [loading] = useState(false);
  const [, setIsMobileView] = useState(window.innerWidth <= 768);
  const [isCheckedIn, setIsCheckedIn] = useState(false);
  
  // Add state to store video recording transaction signature
  const [recordingTransactionSignature, setRecordingTransactionSignature] = useState<string | null>(null);

  // Add state for gesture monitoring
  const [gestureMonitoring, setGestureMonitoring] = useState(false);
  const gestureCheckIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  // Add state to track gesture controls status changes
  const [gestureControlsEnabled, setGestureControlsEnabled] = useState(false);

  // Helper function to detect if we're using the Jetson camera
  // const isJetsonCamera = (cameraId: string | null): boolean => {
  //   return cameraId === CONFIG.JETSON_CAMERA_PDA;
  // };

  // Update the states for TransactionModal
  const [showTransactionModal, setShowTransactionModal] = useState(false);
  const [transactionData, setTransactionData] = useState<{
    type: 'photo' | 'video' | 'stream' | 'initialize';
    cameraAccount: string;
  } | null>(null);
  
  // Remove local isStreaming state - query hardware instead
  const [hardwareState, setHardwareState] = useState<{
    isStreaming: boolean;
    isRecording: boolean;
    lastUpdated: number;
  }>({
    isStreaming: false,
    isRecording: false,
    lastUpdated: 0
  });
  
  // Polling interval for hardware state
  const hardwareStateInterval = useRef<NodeJS.Timeout>();

  // Add a function to create timeline events with Farcaster profile info
  const addTimelineEvent = (eventType: TimelineEventType, transactionId?: string, mediaUrl?: string) => {
    if (primaryWallet && user) {
      // Create a unique key for this event to prevent duplicates
      const eventKey = `${eventType}_${transactionId}_${Date.now()}`;
      const recentEventsKey = 'recentTimelineEvents';
      
      try {
        // Get recent events from localStorage to check for duplicates
        const recentEventsStr = localStorage.getItem(recentEventsKey) || '[]';
        const recentEvents = JSON.parse(recentEventsStr);
        
        // Check if we've added this event recently (within last 30 seconds)
        const now = Date.now();
        const duplicateEvent = recentEvents.find((event: any) => 
          event.type === eventType && 
          event.transactionId === transactionId &&
          (now - event.timestamp) < 30000 // 30 seconds
        );
        
        if (duplicateEvent) {
          console.log('Skipping duplicate timeline event:', eventType, transactionId);
          return;
        }
        
        // Clean up old events (older than 5 minutes)
        const cleanedEvents = recentEvents.filter((event: any) => 
          (now - event.timestamp) < 300000 // 5 minutes
        );
        
        // Add this event to recent events
        cleanedEvents.push({
          type: eventType,
          transactionId,
          timestamp: now
        });
        
        localStorage.setItem(recentEventsKey, JSON.stringify(cleanedEvents));
      } catch (e) {
        console.warn('Error checking for duplicate events:', e);
      }

      // Get the user's Farcaster profile info if available
      const farcasterCred = user.verifiedCredentials?.find(
        cred => cred.oauthProvider === 'farcaster'
      );

      // Create the timeline event with enriched user info
      const event: Omit<TimelineEvent, 'id'> = {
        type: eventType,
        user: {
          address: primaryWallet.address,
          // Include Farcaster profile info if available
          displayName: farcasterCred?.oauthDisplayName || undefined,
          username: farcasterCred?.oauthUsername || undefined,
          pfpUrl: farcasterCred?.oauthAccountPhotos?.[0] || undefined
        },
        timestamp: Date.now(),
        transactionId,
        mediaUrl,
        cameraId: cameraAccount || undefined
      };

      console.log('Adding timeline event:', {
        type: event.type,
        transactionId: event.transactionId ? `${event.transactionId.slice(0, 8)}...` : 'none',
        mediaUrl: event.mediaUrl ? 'present' : 'none',
        cameraId: event.cameraId,
        userInfo: {
          address: event.user.address,
          displayName: event.user.displayName || '(none)',
          username: event.user.username || '(none)',
          hasPfp: !!event.user.pfpUrl
        }
      });

      // Add the event to the timeline
      if (timelineRef.current?.addEvent) {
        timelineRef.current?.addEvent(event);
      }
    }
  };

  // Load camera from URL params if available - simplify to just set the ID
  useEffect(() => {
    if (!cameraId) return;

    try {
      // Log the camera ID (decoding it first)
      const decodedCameraId = decodeURIComponent(cameraId);
      console.log(`[CameraView] Using camera ID: ${decodedCameraId}`);

      // Store in localStorage for persistence
      localStorage.setItem('directCameraId', decodedCameraId);

      // Set camera account state
      setCameraAccount(decodedCameraId);

      // Attempt to fetch camera data if possible (but don't show errors if it fails)
      if (fetchCameraById) {
        fetchCameraById(decodedCameraId).then(camera => {
          if (camera) {
            console.log(`[CameraView] Camera data loaded:`, camera);
            setSelectedCamera(camera);
          }
        }).catch(err => {
          // Just log the error but don't show to the user
          console.warn(`[CameraView] Non-critical error loading camera data:`, err);
        });
      }
    } catch (error) {
      console.error('[CameraView] Error processing camera ID:', error);
    }
  }, [cameraId, fetchCameraById, setSelectedCamera]);

  // Clear camera account when navigating to the default /app route
  useEffect(() => {
    // If we're on the default /app route (no cameraId in URL), clear the camera account
    if (!cameraId && window.location.pathname === '/app') {
      setCameraAccount(null);
      // Also clear the selected camera in the provider if needed
      if (selectedCamera) {
        setSelectedCamera(null);
      }
    }
  }, [cameraId, setSelectedCamera, selectedCamera]);

  // Also update cameraAccount whenever selectedCamera changes
  useEffect(() => {
    if (selectedCamera) {
      setCameraAccount(selectedCamera.publicKey);
    }
  }, [selectedCamera]);

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      setIsMobileView(window.innerWidth <= 768);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Check if the user is using an embedded wallet
  useEffect(() => {
    if (primaryWallet) {
      const isEmbedded = primaryWallet.connector?.name.toLowerCase() !== 'phantom';
      if (isEmbedded) {
        console.log('Using embedded wallet:', primaryWallet.connector?.name);
      }
    }
  }, [primaryWallet]);

  // Sync UI state with actual camera state when component loads or camera changes
  useEffect(() => {
    if (!cameraAccount) return;

    const syncCameraState = async () => {
      try {
        console.log(`[CameraView] Syncing state for camera: ${cameraAccount}`);
        
        // Check if camera exists in registry
        if (!unifiedCameraService.hasCamera(cameraAccount)) {
          console.log(`[CameraView] Camera not in registry, skipping state sync`);
          return;
        }

        // Get actual camera status
        const statusResponse = await unifiedCameraService.getStatus(cameraAccount);
        if (statusResponse.success && statusResponse.data) {
          console.log(`[CameraView] Camera status:`, statusResponse.data);
          setIsLive(statusResponse.data.isOnline);
          setIsRecording(statusResponse.data.isRecording);
        }

        // Get actual stream status
        const streamResponse = await unifiedCameraService.getStreamInfo(cameraAccount);
        if (streamResponse.success && streamResponse.data) {
          console.log(`[CameraView] Stream status:`, streamResponse.data);
          setIsStreaming(streamResponse.data.isActive);
        }
      } catch (error) {
        console.error(`[CameraView] Error syncing camera state:`, error);
      }
    };

    // Sync state immediately when camera changes
    syncCameraState();
  }, [cameraAccount]);

  // Handle camera update from ActivateCamera component

  const updateToast = (type: 'success' | 'error' | 'info', message: string) => {
    // Skip showing "Failed to fetch" network errors as toasts
    if (type === 'error' && (
      message.includes('Failed to fetch') ||
      message.includes('Network error') ||
      message.includes('Camera error') ||
      message.includes('CORS')
    )) {
      console.warn('Suppressing network error toast:', message);
      return;
    }

    const id = Date.now().toString();
    setCurrentToast({ id, type, message });
  };

  const dismissToast = () => {
    setCurrentToast(null);
  };


  // Helper function to convert action type to event type
  const getEventType = (actionType: string): TimelineEventType => {
    switch (actionType) {
      case 'photo':
        return 'photo_captured';
      case 'video':
        return 'video_recorded';
      case 'stream':
        return isStreaming ? 'stream_ended' : 'stream_started';
      case 'stream_start':
        return 'stream_started';
      case 'stream_stop':
        return 'stream_ended';
      default:
        return 'initialization'; // Changed from 'photo_captured' to avoid fake photo entries
    }
  };

  useEffect(() => {
    const unsubscribe = cameraStatus.subscribe((status) => {
      setIsLive(status.isLive);
      setIsStreaming(status.isStreaming);
    });

    return () => {
      unsubscribe();
    };
  }, []);

  // Force camera status check when we detect Jetson camera
  // useEffect(() => {
  //   if (cameraAccount && isJetsonCamera(cameraAccount)) {
  //     console.log('[CameraView] Detected Jetson camera, forcing status check');
  //     // Force a status check for Jetson camera
  //     cameraStatus.forceCheck();
  //     
  //     // Also test the Jetson camera connection directly
  //     jetsonCameraService.testConnection().then(result => {
  //       console.log('[CameraView] Jetson camera test result:', result);
  //       if (result.success) {
  //         // If the test succeeds, manually set the camera as online
  //         cameraStatus.setOnline(false);
  //       }
  //     }).catch(err => {
  //       console.warn('[CameraView] Jetson camera test failed:', err);
  //     });
  //   }
  // }, [cameraAccount]);

  // Check localStorage for direct camera ID (set by MainContent)
  useEffect(() => {
    const directCameraId = localStorage.getItem('directCameraId');
    const isDefaultRoute = window.location.pathname === '/app' || window.location.pathname === '/app/';

    // Only load the camera ID from localStorage if explicitly on a camera route
    // or if we're already connected (cameraAccount exists)
    if (directCameraId && !cameraAccount && window.location.pathname.includes('/camera/')) {
      console.log(`[CameraView] Found direct camera ID in localStorage: ${directCameraId}`);
      setCameraAccount(directCameraId);
    } else if (isDefaultRoute) {
      // If we're on the default /app route with no specific camera, always ensure storage is cleared
      console.log('[CameraView] On default route - clearing any stored camera ID');
      localStorage.removeItem('directCameraId');

      // Force reset of camera state variables
      if (cameraAccount) setCameraAccount(null);
      if (selectedCamera) setSelectedCamera(null);
    }
  }, [cameraAccount, selectedCamera, setSelectedCamera]);

  // Debug logging for camera route with fix for paths
  useEffect(() => {
    const isDefaultRoute = window.location.pathname === '/app' || window.location.pathname === '/app/';

    console.log('CameraView Debug:', {
      cameraIdParam: cameraId,
      cameraAccount: cameraAccount,
      selectedCamera: selectedCamera?.publicKey || null,
      localStorageCamera: localStorage.getItem('directCameraId'),
      route: window.location.pathname,
      isDefaultRoute
    });

    // Only update localStorage if we have a valid camera ID and are on a camera route
    if (cameraId) {
      localStorage.setItem('directCameraId', cameraId);
    } else if (selectedCamera?.publicKey && window.location.pathname.includes('/camera/')) {
      localStorage.setItem('directCameraId', selectedCamera.publicKey);
    } else if (cameraAccount && window.location.pathname.includes('/camera/')) {
      localStorage.setItem('directCameraId', cameraAccount);
    } else if (isDefaultRoute) {
      // Always clear on the default route
      localStorage.removeItem('directCameraId');

      // Force reset of all camera state if we're still seeing a camera connection
      if (cameraAccount || selectedCamera) {
        console.log('[CameraView] Forcing disconnect on default route');
        setCameraAccount(null);
        if (setSelectedCamera) setSelectedCamera(null);
      }
    }
  }, [cameraId, cameraAccount, selectedCamera, setSelectedCamera]);

  // Debug logging
  useEffect(() => {
    console.log('Debug - selectedCamera:', selectedCamera ? selectedCamera.publicKey : 'null');
    console.log('Debug - cameraAccount state:', cameraAccount);
    console.log('Debug - cameraId param:', cameraId);
  }, [selectedCamera, cameraAccount, cameraId]);

  // Add a special direct loading function that doesn't rely on the useProgram hook
  // to avoid potential circular dependencies or timing issues
  useEffect(() => {
    if (!cameraId) return;

    const loadCameraDirectly = async () => {
      try {
        console.log(`[CameraView] DIRECT LOADING camera for URL param: cameraId="${cameraId}"`);

        // Always decode the cameraId from the URL
        const decodedCameraId = decodeURIComponent(cameraId);
        console.log(`[CameraView] DIRECT LOADING Decoded cameraId="${decodedCameraId}"`);

        // Store in localStorage for persistence
        localStorage.setItem('directCameraId', decodedCameraId);

        // Set the camera account in state
        setCameraAccount(decodedCameraId);

        // Try to load the camera data using the direct method if connection available
        if (connection) {
          console.log(`[CameraView] Loading camera directly using wallet-adapter connection`);
          const camera = await fetchCameraByPublicKey(decodedCameraId, connection);

          if (camera) {
            console.log(`[CameraView] Successfully loaded camera data directly:`, camera);
            setSelectedCamera(camera);
            // No need for connection notification - the UI already shows connection status via the ID and link icon
            return;
          } else {
            console.log(`[CameraView] Failed to load camera data directly`);
          }
        }

        // Fall back to using the camera provider's method if direct loading fails
        if (fetchCameraById) {
          const camera = await fetchCameraById(decodedCameraId);
          if (camera) {
            console.log(`[CameraView] Successfully loaded camera with provider:`, camera);
            setSelectedCamera(camera);
            // Don't show a second success notification to avoid duplicates
          } else {
            console.log(`[CameraView] Camera could not be loaded with provider, showing ID only`);
          }
        }
      } catch (error) {
        console.error('[CameraView] DIRECT LOADING Error:', error);
      }
    };

    loadCameraDirectly();
  }, [cameraId, connection, fetchCameraById, setSelectedCamera]);

  // Button handlers

  // Debug logs for program and connection
  useEffect(() => {
    console.log("Program initialized:", !!program);
    console.log("Program details:", program ? {
      programId: program.programId.toString(),
      provider: !!program.provider,
    } : "No program");

    console.log("Connection initialized:", !!connection);
    console.log("Connection details:", connection ? {
      rpcEndpoint: connection.rpcEndpoint,
    } : "No connection");
  }, [program, connection]);

  // More detailed error logging
  const logDetailedError = (error: any, context: string) => {
    console.error(`Error in ${context}:`, error);
    if (error instanceof Error) {
      console.error(`Name: ${error.name}, Message: ${error.message}`);
      console.error(`Stack: ${error.stack}`);
    } else {
      console.error(`Unknown error type: ${typeof error}`);
    }
  };

  // Update the Simple direct transaction function to return the signature and work with the existing Solana program
  const sendSimpleTransaction = async (actionType: string): Promise<string | undefined> => {
    console.log("DIRECT TRANSACTION FUNCTION - Type:", actionType);
    let retryCount = 0;
    const MAX_RETRIES = 3;
    let currentConnection = connection;

    while (retryCount < MAX_RETRIES) {
      try {
        // Check for wallet, program, and connection
        if (!primaryWallet || !program || !currentConnection) {
          console.error("Missing required components for transaction");
          const missing = [];
          if (!primaryWallet) missing.push("wallet");
          if (!program) missing.push("program");
          if (!currentConnection) missing.push("connection");
          updateToast('error', `Cannot send transaction: missing ${missing.join(", ")}`);
          return undefined;
        }

        // Get camera ID
        const cameraId = cameraAccount || localStorage.getItem('directCameraId');
        if (!cameraId) {
          console.error("No camera ID found");
          updateToast('error', 'No camera ID found');
          return undefined;
        }

        // First, verify user is checked in
        const isCheckedInNow = await checkUserSession();
        if (!isCheckedInNow) {
          updateToast('error', 'You need to check in before performing camera actions');
          return undefined;
        }

        // Create a simplified transaction that will update the session's lastActivity timestamp
        // This is the minimally invasive approach that works with the current program
        try {
          // Get recent blockhash with retry logic
          let blockhash;
          try {
            const { blockhash: newBlockhash } = await currentConnection.getLatestBlockhash('finalized');
            blockhash = newBlockhash;
          } catch (bhError) {
            console.error("Error getting blockhash:", bhError);
            const nextEndpoint = CONFIG.getNextEndpoint();
            console.log(`Switching to RPC endpoint: ${nextEndpoint}`);
            currentConnection = new Connection(nextEndpoint, 'confirmed');
            throw new Error('Failed to get blockhash, retrying with new endpoint');
          }

          // Create a basic transaction that sends a minimal amount of SOL to yourself
          // This serves as a placeholder transaction that will be recorded on-chain
          const userPublicKey = new PublicKey(primaryWallet.address);
          const cameraPublicKey = new PublicKey(cameraId);

          // Find the session PDA
          // eslint-disable-next-line @typescript-eslint/no-unused-vars
          const [_sessionPda] = PublicKey.findProgramAddressSync(
            [
              Buffer.from('session'),
              userPublicKey.toBuffer(),
              cameraPublicKey.toBuffer()
            ],
            CAMERA_ACTIVATION_PROGRAM_ID
          );

          // Create a transaction with a memo instruction that identifies the camera action
          const memoProgram = new PublicKey('MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr');

          // Create the memo data with the action type for on-chain recording
          const memoData = Buffer.from(`camera:${actionType}:${new Date().toISOString()}`);

          // Create the memo instruction
          const memoInstruction = new TransactionInstruction({
            keys: [{ pubkey: userPublicKey, isSigner: true, isWritable: true }],
            programId: memoProgram,
            data: memoData
          });

          // Create the transaction
          const tx = new Transaction();
          tx.add(memoInstruction);

          // Add blockhash and payer
          tx.recentBlockhash = blockhash;
          tx.feePayer = userPublicKey;

          // Sign and send the transaction
          const signer = await (primaryWallet as any).getSigner();
          const signedTx = await signer.signTransaction(tx);
          const signature = await currentConnection.sendRawTransaction(signedTx.serialize());

          // Wait for confirmation
          await currentConnection.confirmTransaction(signature, 'confirmed');

          console.log(`Transaction confirmed: ${signature}`);
          updateToast('success', `${actionType} transaction sent: ${signature.slice(0, 8)}...`);

          // Add the event to the timeline
          addTimelineEvent(getEventType(actionType), signature);

          return signature;
        } catch (error) {
          console.error("Error in transaction:", error);
          logDetailedError(error, "Transaction error");

          // Check if it's a blockhash error
          const errorMessage = error instanceof Error ? error.message : String(error);

          if (errorMessage.includes('Blockhash not found') ||
            errorMessage.includes('block height exceeded') ||
            errorMessage.includes('timeout')) {
            retryCount++;
            if (retryCount < MAX_RETRIES) {
              console.log(`Retrying transaction (attempt ${retryCount + 1}/${MAX_RETRIES})`);
              // Switch to next RPC endpoint
              const nextEndpoint = CONFIG.getNextEndpoint();
              console.log(`Switching to RPC endpoint: ${nextEndpoint}`);
              currentConnection = new Connection(nextEndpoint, 'confirmed');
              continue;
            }
          }

          updateToast('error', `Transaction failed: ${errorMessage}`);
          return undefined;
        }
      } catch (error) {
        console.error("Outer error in transaction:", error);
        logDetailedError(error, "Outer transaction error");

        retryCount++;
        if (retryCount < MAX_RETRIES) {
          console.log(`Retrying entire transaction process (attempt ${retryCount + 1}/${MAX_RETRIES})`);
          continue;
        }

        updateToast('error', `Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
        return undefined;
      }
    }

    updateToast('error', 'Transaction failed after maximum retries');
    return undefined;
  };

  // Expose sendSimpleTransaction to window for Pi5Camera to use
  useEffect(() => {
    (window as any).sendSimpleTransaction = sendSimpleTransaction;
    
    // Cleanup on unmount
    return () => {
      delete (window as any).sendSimpleTransaction;
    };
  }, [sendSimpleTransaction]);

  // Replace the promptForCheckIn function with one that shows the modal
  const promptForCheckIn = (actionType: 'photo' | 'video' | 'stream') => {
    if (!cameraAccount) {
      updateToast('error', 'No camera connected');
      return false;
    }

    // Set transaction data and show modal
    setTransactionData({
      type: actionType,
      cameraAccount: cameraAccount
    });
    setShowTransactionModal(true);

    // Track this as an "attempt" 
    console.log(`User attempted to use camera for ${actionType} without checking in first`);

    return false;
  };

  // Update the checkUserSession function with similar throttling
  const checkUserSession = async (): Promise<boolean> => {
    if (!primaryWallet?.address || !cameraAccount || !program || !connection) {
      console.error("Missing required components for check-in status");
      return false;
    }

    try {
      // Add basic rate limiting - don't check more than once every 3 seconds
      const lastCheckTime = parseInt(localStorage.getItem('lastCheckUserSessionTime') || '0');
      const now = Date.now();
      if (now - lastCheckTime < 3000) {
        // Return the current state without checking
        return isCheckedIn;
      }
      
      localStorage.setItem('lastCheckUserSessionTime', now.toString());

      const userPublicKey = new PublicKey(primaryWallet.address);
      const cameraPublicKey = new PublicKey(cameraAccount);

      // Find the session PDA
      const [sessionPda] = PublicKey.findProgramAddressSync(
        [
          Buffer.from('session'),
          userPublicKey.toBuffer(),
          cameraPublicKey.toBuffer()
        ],
        CAMERA_ACTIVATION_PROGRAM_ID
      );

      try {
        // Try to fetch the session account
        const sessionAccount = await program.account.userSession.fetch(sessionPda);
        if (sessionAccount) {
          if (!isCheckedIn) {
            console.log("[CameraView] Setting checked-in status to TRUE");
            setIsCheckedIn(true);
            // Refresh timeline if status changed
            if (timelineRef.current?.refreshTimeline) {
              timelineRef.current?.refreshTimeline();
            }
          }
          return true;
        }
      } catch (_) {
        console.log("[CameraView] Session account not found, user is not checked in");
        if (isCheckedIn) {
          console.log("[CameraView] Setting checked-in status to FALSE");
          setIsCheckedIn(false);
          // Refresh timeline if status changed
          if (timelineRef.current?.refreshTimeline) {
            timelineRef.current?.refreshTimeline();
          }
        }
        return false;
      }
    } catch (err) {
      console.error("[CameraView] Error checking session status:", err);
      // Don't update state on error to prevent UI flashing
      return isCheckedIn; // Return current state on error
    }

    return isCheckedIn;
  };

  // Update the periodic check-in status check to use a longer interval
  useEffect(() => {
    if (!primaryWallet?.address || !cameraAccount) return;

    console.log("[CameraView] Setting up check-in status monitoring");

    // Check status immediately
    checkUserSession();

    // Set up periodic check every 10 seconds (instead of 3)
    const intervalId = setInterval(() => {
      checkUserSession().then(isCheckedIn => {
        console.log("[CameraView] Periodic check result:", isCheckedIn ? "CHECKED IN" : "NOT CHECKED IN");
      });
    }, 10000);

    // Clean up on unmount
    return () => {
      console.log("[CameraView] Cleaning up check-in status monitor");
      clearInterval(intervalId);
    };
  }, [primaryWallet, cameraAccount, program, connection]);

  // Sync gesture controls state with localStorage
  useEffect(() => {
    const updateGestureControlsState = async () => {
      const enabled = await unifiedCameraService.getGestureControlsStatus();
      console.log("[CameraView] Gesture controls status from localStorage:", enabled);
      setGestureControlsEnabled(enabled);
    };

    // Initial sync
    updateGestureControlsState();

    // Listen for storage changes (when other tabs/components update localStorage)
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'jetson_gesture_controls_enabled') {
        updateGestureControlsState();
      }
    };

    window.addEventListener('storage', handleStorageChange);

    // Also check periodically in case localStorage is updated by the same tab
    const intervalId = setInterval(updateGestureControlsState, 1000);

    return () => {
      window.removeEventListener('storage', handleStorageChange);
      clearInterval(intervalId);
    };
  }, []);

  // Gesture monitoring effect for Jetson cameras
  useEffect(() => {
    // Only monitor gestures if:
    // 1. User is checked in
    // 2. Using Jetson camera
    // 3. Gesture controls are enabled
    const isJetson = cameraAccount && unifiedCameraService.hasCamera(cameraAccount);
    
    console.log("[CameraView] Gesture monitoring conditions:", {
      isCheckedIn,
      cameraAccount,
      isJetson,
      gestureControlsEnabled,
      gestureMonitoring
    });
    
    const shouldMonitorGestures = isCheckedIn && 
                                  cameraAccount && 
                                  isJetson && 
                                  gestureControlsEnabled;

    console.log("[CameraView] Should monitor gestures:", shouldMonitorGestures);

    // Clear any existing interval first
    if (gestureCheckIntervalRef.current) {
      console.log("[CameraView] 🧹 Clearing existing gesture monitoring interval");
      clearInterval(gestureCheckIntervalRef.current);
      gestureCheckIntervalRef.current = null;
    }

    if (shouldMonitorGestures) {
      console.log("[CameraView] ✅ STARTING GESTURE MONITORING - All conditions met!");
      
      // Don't use the gestureMonitoring state to control the interval
      // Just start it directly when conditions are met
      let lastGestureAction: string | null = null;
      let gestureActionCooldown = false;
      
      console.log("[CameraView] 🔄 Setting up gesture check interval...");
      gestureCheckIntervalRef.current = setInterval(async () => {
        try {
          console.log("[CameraView] 👀 Checking for gesture trigger...");
          const gestureCheck = await unifiedCameraService.checkForGestureTrigger(cameraAccount);
          console.log("[CameraView] 📊 Gesture check result:", gestureCheck);
          
          if (gestureCheck.shouldCapture && !gestureActionCooldown) {
            const gesture = gestureCheck.gesture;
            console.log(`[CameraView] 🎯 GESTURE TRIGGER DETECTED: ${gesture}`);
            
            // Prevent the same gesture from triggering multiple times
            if (lastGestureAction !== gesture) {
              lastGestureAction = gesture || null;
              gestureActionCooldown = true;
              
              // Trigger the appropriate action based on gesture
              if (gestureCheck.gestureType === 'photo') {
                updateToast('info', `📸 Gesture detected: ${gesture} - Taking photo...`);
                await handleDirectPhoto();
              } else if (gestureCheck.gestureType === 'video') {
                updateToast('info', `🎥 Gesture detected: ${gesture} - Recording video...`);
                await handleDirectVideo();
              }
              
              // Reset cooldown after 2 seconds
              setTimeout(() => {
                gestureActionCooldown = false;
                lastGestureAction = null;
              }, 2000);
            }
          }
        } catch (error) {
          console.error("[CameraView] Error checking gesture trigger:", error);
        }
      }, 500);
      
      console.log("[CameraView] ✅ Gesture monitoring interval started!");
      
      // Update the state to reflect that monitoring is active
      if (!gestureMonitoring) {
        setGestureMonitoring(true);
      }
      
    } else {
      console.log("[CameraView] ❌ Gesture monitoring conditions not met");
      console.log("[CameraView] Conditions:", {
        isCheckedIn,
        cameraAccount,
        isJetson,
        gestureControlsEnabled
      });
      
      // Update the state to reflect that monitoring is inactive
      if (gestureMonitoring) {
        setGestureMonitoring(false);
      }
    }

    // Cleanup on unmount
    return () => {
      if (gestureCheckIntervalRef.current) {
        console.log("[CameraView] 🧹 Cleaning up gesture monitoring on unmount");
        clearInterval(gestureCheckIntervalRef.current);
        gestureCheckIntervalRef.current = null;
      }
    };
  }, [isCheckedIn, cameraAccount, gestureControlsEnabled]);

  // Update the button handlers to check for check-in status
  const handleDirectPhoto = async () => {
    // Check if there's a camera connected first
    if (!cameraAccount && !selectedCamera) {
      updateToast('error', 'No camera connected. Please connect to a camera first.');
      return;
    }

    const currentCameraId = cameraAccount || selectedCamera?.publicKey;
    if (!currentCameraId) {
      updateToast('error', 'No camera ID available.');
      return;
    }

    // Check if camera exists in registry
    if (!unifiedCameraService.hasCamera(currentCameraId)) {
      updateToast('error', 'Camera not found in registry. Please reconnect.');
      return;
    }

    // Check if user is checked in first
    const isCheckedIn = await checkUserSession();
    if (!isCheckedIn) {
      return promptForCheckIn('photo');
    }

    try {
      updateToast('info', 'Initiating photo capture...');
      
      // First create the blockchain transaction
      const signature = await sendSimpleTransaction('photo');
      if (!signature) {
        updateToast('error', 'Failed to create blockchain transaction');
        return;
      }
      
      updateToast('info', 'Capturing photo...');
      
      try {
        // Connect to camera if not already connected
        const isConnected = await unifiedCameraService.isConnected(currentCameraId);
        if (!isConnected && primaryWallet?.address) {
          await unifiedCameraService.connect(currentCameraId, primaryWallet.address);
        }
        
        const response = await unifiedCameraService.takePhoto(currentCameraId);
        
        console.log('[PHOTO DEBUG] takePhoto response:', response);
        
        if (response.success && response.data?.blob) {
          console.log('[PHOTO DEBUG] Photo capture successful, blob size:', response.data.blob.size);
          updateToast('info', 'Photo captured, uploading to IPFS...');
          
          try {
            // Upload to IPFS
            console.log('[PHOTO DEBUG] Starting IPFS upload...');
            
            const results = await unifiedIpfsService.uploadFile(
              response.data.blob,
              primaryWallet?.address || '',
              'image',
              {
                transactionId: signature,
                cameraId: currentCameraId
              }
            );
            
            console.log('[PHOTO DEBUG] IPFS upload results:', results);
            
            if (results.length > 0) {
              updateToast('success', 'Photo captured and uploaded to IPFS');
              addTimelineEvent('photo_captured', signature);
              
              // Set camera status to online
              cameraStatus.setOnline(false);
            } else {
              console.log('[PHOTO DEBUG] IPFS upload returned empty results');
              updateToast('success', 'Photo captured (upload to IPFS failed)');
              addTimelineEvent('photo_captured', signature);
            }
          } catch (uploadError) {
            console.error('[PHOTO DEBUG] Error uploading to IPFS:', uploadError);
            updateToast('success', 'Photo captured (upload to IPFS failed)');
            addTimelineEvent('photo_captured', signature);
          }
          
          // Refresh the timeline
          if (timelineRef.current?.refreshEvents) {
            timelineRef.current?.refreshEvents();
          }
        } else {
          console.error('[PHOTO DEBUG] Photo capture failed:', response);
          updateToast('error', `Failed to capture photo: ${response.error || 'Unknown error'}`);
          // Still add timeline event for the transaction
          addTimelineEvent('photo_captured', signature);
          if (timelineRef.current?.refreshEvents) {
            timelineRef.current?.refreshEvents();
          }
        }
      } catch (error) {
        console.error('Error capturing photo:', error);
        updateToast('error', `Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
        // Still add timeline event for the transaction
        addTimelineEvent('photo_captured', signature);
        if (timelineRef.current?.refreshEvents) {
          timelineRef.current?.refreshEvents();
        }
      }
    } catch (error) {
      console.error('Error in photo capture process:', error);
      updateToast('error', `Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  const handleDirectVideo = async () => {
    // Check if there's a camera connected first
    if (!cameraAccount && !selectedCamera) {
      updateToast('error', 'No camera connected. Please connect to a camera first.');
      return;
    }

    const currentCameraId = cameraAccount || selectedCamera?.publicKey;
    if (!currentCameraId) {
      updateToast('error', 'No camera ID available.');
      return;
    }

    // Check if user is checked in first
    const isCheckedIn = await checkUserSession();
    if (!isCheckedIn) {
      return promptForCheckIn('video');
    }

    try {
      if (isRecording) {
        // Stop recording (this shouldn't happen with timed recording, but just in case)
        updateToast('info', 'Stopping video recording...');
        setIsRecording(false);
        return;
      }

      // Start timed recording using direct Jetson API
      updateToast('info', 'Initiating video recording...');
      
      // First create the blockchain transaction
      const signature = await sendSimpleTransaction('video');
      if (!signature) {
        updateToast('error', 'Failed to create blockchain transaction');
        return;
      }
      
      setRecordingTransactionSignature(signature);
      updateToast('info', 'Starting video recording (30 seconds) - will auto-stop and upload');
      
      try {
        const jetsonUrl = 'https://jetson.mmoment.xyz';
        const walletAddress = primaryWallet?.address || '9gERcKdpaTNLfFNNYANzs1P73iMHJpVqhvKMKLa6Xvo';
        
        // Step 1: Connect/create session
        const connectResponse = await fetch(`${jetsonUrl}/api/session/connect`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ wallet_address: walletAddress })
        });
        
        if (!connectResponse.ok) {
          throw new Error(`Failed to connect: ${connectResponse.status}`);
        }
        
        const connectData = await connectResponse.json();
        console.log('Session created:', connectData);
        
        // Step 2: Start recording
        const recordResponse = await fetch(`${jetsonUrl}/api/record`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            wallet_address: walletAddress,
            session_id: connectData.session_id
            // Remove duration - let it record indefinitely until we stop it
          })
        });
        
        if (!recordResponse.ok) {
          throw new Error(`Failed to start recording: ${recordResponse.status}`);
        }
        
        const recordData = await recordResponse.json();
        console.log('Recording started:', recordData);
        console.log('⏱️ 30-second timer started, will stop at:', new Date(Date.now() + 30000).toLocaleTimeString());
        
        setIsRecording(true);
        updateToast('success', 'Video recording started (30 seconds) - will auto-stop and upload');
        
        // Add timeline event
        addTimelineEvent('video_recorded', signature);
        if (timelineRef.current?.refreshEvents) {
          timelineRef.current?.refreshEvents();
        }
        
        // Add periodic logging to show recording is active
        const progressInterval = setInterval(() => {
          const elapsed = Math.floor((Date.now() - Date.now()) / 1000);
          console.log(`📹 Recording in progress...`);
        }, 5000);
        
        // Step 3: Stop after exactly 30 seconds
        setTimeout(async () => {
          clearInterval(progressInterval);
          try {
            console.log('⏹️ 30 seconds completed at:', new Date().toLocaleTimeString(), '- stopping recording...');
            updateToast('info', 'Recording completed, processing video...');
            
            const stopResponse = await fetch(`${jetsonUrl}/api/record`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                action: 'stop',
                wallet_address: walletAddress,
                session_id: connectData.session_id
              })
            });
            
            if (stopResponse.ok) {
              const stopData = await stopResponse.json();
              console.log('Recording stopped:', stopData);
              
              if (stopData.filename) {
                // Wait longer for video encoding since it's a 30-second video
                console.log('Waiting for video encoding to complete...');
                updateToast('info', 'Video processing - please wait (this may take 10-15 seconds)...');
                
                // Wait for encoding to complete
                setTimeout(async () => {
                  try {
                    // Download the video file - REQUEST THE .MP4 VERSION!
                    const mp4Filename = stopData.filename.replace('.mov', '.mp4');
                    const videoUrl = `${jetsonUrl}/api/videos/${mp4Filename}`;
                    console.log('Downloading MP4 video from:', videoUrl);
                    const videoResponse = await fetch(videoUrl, {
                      headers: {
                        'Accept': 'video/mp4,video/quicktime,video/*'
                      }
                    });
                    
                    if (videoResponse.ok) {
                      const videoBlob = await videoResponse.blob();
                      console.log('MP4 video downloaded:', videoBlob.size, 'bytes', 'type:', videoBlob.type);
                      
                      // Use the direct Jetson MP4 URL
                      const directVideoUrl = `${jetsonUrl}/api/videos/${mp4Filename}`;
                      console.log('Using direct MP4 video URL:', directVideoUrl);
                      
                      // Upload video to IPFS properly - NO localStorage shortcuts!
                      updateToast('info', 'Video processed successfully, uploading to IPFS...');
                      
                      try {
                        console.log('📤 Uploading video to IPFS:', videoBlob.size, 'bytes');
                        
                        // Upload to IPFS with proper metadata
                        const ipfsResult = await unifiedIpfsService.uploadFile(
                          videoBlob, 
                          primaryWallet?.address || '', 
                          'video',
                          {
                            transactionId: signature,
                            cameraId: currentCameraId
                          }
                        );
                        
                        if (ipfsResult && ipfsResult.length > 0) {
                          console.log('✅ Video uploaded to IPFS successfully:', ipfsResult[0]);
                          updateToast('success', 'Video recorded and uploaded to IPFS successfully!');
                        } else {
                          console.error('❌ IPFS upload failed - no result returned');
                          updateToast('error', 'Video recorded but IPFS upload failed');
                        }
                      } catch (ipfsError) {
                        console.error('❌ IPFS upload error:', ipfsError);
                        updateToast('error', 'Video recorded but IPFS upload failed');
                      }
                    } else {
                      updateToast('error', 'Failed to download MP4 video from camera');
                    }
                  } catch (downloadError) {
                    console.error('Video download error:', downloadError);
                    updateToast('error', 'Failed to process recorded video');
                  } finally {
                    setIsRecording(false);
                    setRecordingTransactionSignature(null);
                    
                    // Refresh timeline
                    if (timelineRef.current?.refreshEvents) {
                      timelineRef.current?.refreshEvents();
                    }
                  }
                }, 10000); // Wait 10s for encoding
                
              } else {
                updateToast('error', 'Recording completed but no video file was created');
                setIsRecording(false);
                setRecordingTransactionSignature(null);
              }
            } else {
              updateToast('error', 'Failed to stop recording');
              setIsRecording(false);
              setRecordingTransactionSignature(null);
            }
          } catch (error) {
            console.error('Recording stop error:', error);
            updateToast('error', 'Error stopping recording');
            setIsRecording(false);
            setRecordingTransactionSignature(null);
          }
        }, 30000); // Record for 30s
        
      } catch (error) {
        console.error('Error in video recording:', error);
        updateToast('error', `Recording failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
        setIsRecording(false);
        setRecordingTransactionSignature(null);
      }
      
    } catch (error) {
      console.error('Error in video recording:', error);
      updateToast('error', `Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  const handleDirectStream = async () => {
    const currentCameraId = cameraAccount || selectedCamera?.publicKey || CONFIG.JETSON_CAMERA_PDA;
    if (!currentCameraId) {
      updateToast('error', 'No camera selected');
      return;
    }

    // Check if user is checked in first
    const isCheckedIn = await checkUserSession();
    if (!isCheckedIn) {
      return promptForCheckIn('stream');
    }

    try {
      console.log(`🔄 [STREAM DEBUG] Current hardware state:`, hardwareState);
      
      if (hardwareState.isStreaming) {
        // Stop streaming
        updateToast('info', 'Stopping stream...');
        console.log(`🛑 [STREAM DEBUG] Attempting to stop stream...`);
        
        // First create the blockchain transaction
        const signature = await sendSimpleTransaction('stream_stop');
        if (!signature) {
          updateToast('error', 'Failed to create blockchain transaction');
          return;
        }
        
        const response = await unifiedCameraService.stopStream(currentCameraId);
        console.log(`🛑 [STREAM DEBUG] Stop stream response:`, response);
        
        if (response.success) {
          // Force immediate hardware state poll
          setTimeout(() => pollHardwareState(), 500);
          updateToast('success', 'Stream stopped');
          // Timeline event already created by sendSimpleTransaction
        } else {
          updateToast('error', `Failed to stop stream: ${response.error || 'Unknown error'}`);
          // Timeline event already created by sendSimpleTransaction
        }
      } else {
        // Start streaming
        updateToast('info', 'Starting stream...');
        console.log(`▶️ [STREAM DEBUG] Attempting to start stream...`);
        
        // First create the blockchain transaction
        const signature = await sendSimpleTransaction('stream_start');
        if (!signature) {
          updateToast('error', 'Failed to create blockchain transaction');
          return;
        }
        
        // Connect to camera if not already connected
        const isConnected = await unifiedCameraService.isConnected(currentCameraId);
        if (!isConnected && primaryWallet?.address) {
          await unifiedCameraService.connect(currentCameraId, primaryWallet.address);
        }
        
        const response = await unifiedCameraService.startStream(currentCameraId);
        console.log(`▶️ [STREAM DEBUG] Start stream response:`, response);
        
        if (response.success) {
          // Force immediate hardware state poll
          setTimeout(() => pollHardwareState(), 500);
          updateToast('success', 'Stream started');
          // Timeline event already created by sendSimpleTransaction
        } else {
          updateToast('error', `Failed to start stream: ${response.error || 'Unknown error'}`);
          // Timeline event already created by sendSimpleTransaction
        }
      }
      
      // Force additional polls at 2s and 5s intervals to catch delayed state changes
      setTimeout(() => pollHardwareState(), 2000);
      setTimeout(() => pollHardwareState(), 5000);
      
      // Refresh the timeline
      if (timelineRef.current?.refreshEvents) {
        timelineRef.current?.refreshEvents();
      }
    } catch (error) {
      console.error('🚨 [STREAM DEBUG] Error handling stream:', error);
      updateToast('error', `Error handling stream: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  // Add a useEffect to handle stream state changes
  useEffect(() => {
    // When streaming starts, clear any lingering "Starting stream..." toast
    if (hardwareState.isStreaming && currentToast?.message?.includes('Starting stream')) {
      dismissToast();
    }
  }, [hardwareState.isStreaming, currentToast]);

  // Helper function to handle stream errors more gracefully

  // Add a simple test function for gesture detection that we can call from console
  (window as any).testGestureAPI = async () => {
    console.log("[GESTURE TEST] Testing gesture API directly...");
    try {
      const currentCameraId = cameraAccount || selectedCamera?.publicKey || CONFIG.JETSON_CAMERA_PDA;
      const result = await unifiedCameraService.getCurrentGesture(currentCameraId);
      console.log("[GESTURE TEST] getCurrentGesture result:", result);
      
      const triggerCheck = await unifiedCameraService.checkForGestureTrigger();
      console.log("[GESTURE TEST] checkForGestureTrigger result:", triggerCheck);
      
      return { getCurrentGesture: result, checkForGestureTrigger: triggerCheck };
    } catch (error) {
      console.error("[GESTURE TEST] Error:", error);
      return { error };
    }
  };

  // Expose debug functions to global scope for testing
  useEffect(() => {
    const currentCameraId = cameraAccount || selectedCamera?.publicKey;
    if (currentCameraId && unifiedCameraService.hasCamera(currentCameraId)) {
      // Get the camera instance from the unified service
      const cameraInstance = (unifiedCameraService as any).cameras?.get(currentCameraId);
      
      if (cameraInstance) {
        (window as any).debugPhotoCapture = () => cameraInstance.debugPhotoCapture?.();
        (window as any).debugVideoRecording = () => cameraInstance.debugVideoRecording?.();
        (window as any).debugStreaming = () => cameraInstance.debugStreaming?.();
        (window as any).debugSession = () => cameraInstance.debugSession?.();
        (window as any).debugStreamDisplay = () => cameraInstance.debugStreamDisplay?.();
        (window as any).debugAllFunctions = () => cameraInstance.debugAllFunctions?.();
        
        // Add debug function to check camera configuration
        (window as any).debugCameraConfig = () => {
          console.log('🔧 === CAMERA CONFIGURATION DEBUG ===');
          console.log('Current Camera ID:', currentCameraId);
          console.log('Camera Instance:', cameraInstance);
          console.log('Camera API URL:', cameraInstance.apiUrl);
          console.log('Camera Type:', cameraInstance.cameraType);
          console.log('CONFIG.JETSON_CAMERA_URL:', CONFIG.JETSON_CAMERA_URL);
          console.log('CONFIG.JETSON_CAMERA_PDA:', CONFIG.JETSON_CAMERA_PDA);
          console.log('All registered cameras:', unifiedCameraService.getAllCameras());
          console.log('=================================');
        };
        
        // Add simple unified service test
        (window as any).testUnifiedService = async () => {
          console.log('🧪 === TESTING UNIFIED SERVICE ===');
          console.log('Current Camera ID:', currentCameraId);
          
          try {
            console.log('1. Testing hasCamera...');
            const hasCamera = unifiedCameraService.hasCamera(currentCameraId);
            console.log('Has camera:', hasCamera);
            
            console.log('2. Testing getStreamInfo...');
            const streamInfo = await unifiedCameraService.getStreamInfo(currentCameraId);
            console.log('Stream info result:', streamInfo);
            
            if (streamInfo.success) {
              console.log('✅ Stream info SUCCESS:', streamInfo.data);
            } else {
              console.log('❌ Stream info FAILED:', streamInfo.error);
            }
          } catch (error) {
            console.error('❌ Test failed:', error);
          }
          console.log('=================================');
        };
        
        console.log('🔧 Debug functions available:');
        console.log('- debugPhotoCapture() - Test photo capture and download');
        console.log('- debugVideoRecording() - Test video recording start/stop');
        console.log('- debugStreaming() - Test streaming start/stop');
        console.log('- debugSession() - Test session connectivity');
        console.log('- debugStreamDisplay() - Test stream info and display');
        console.log('- debugCameraConfig() - Check camera configuration and URLs');
        console.log('- testUnifiedService() - Test unified camera service directly');
        console.log('- debugAllFunctions() - Test all functions comprehensively');
      } else {
        console.log('🔧 Camera instance not found for debug functions');
      }
    }
  }, [cameraAccount, selectedCamera]);

  // Function to poll actual hardware state
  const pollHardwareState = useCallback(async () => {
    const currentCameraId = cameraAccount || selectedCamera?.publicKey || localStorage.getItem('directCameraId');
    if (!currentCameraId) return;

    try {
      const comprehensiveState = await unifiedCameraService.getComprehensiveState(currentCameraId);
      
      if (comprehensiveState.status.success && comprehensiveState.streamInfo.success) {
        const newState = {
          isStreaming: comprehensiveState.streamInfo.data?.isActive || false,
          isRecording: comprehensiveState.status.data?.isRecording || false,
          lastUpdated: Date.now()
        };
        
        // Only update if state actually changed
        setHardwareState(prev => {
          if (prev.isStreaming !== newState.isStreaming || prev.isRecording !== newState.isRecording) {
            console.log('[CameraView] 🔄 Hardware state changed:', {
              wasStreaming: prev.isStreaming,
              nowStreaming: newState.isStreaming,
              wasRecording: prev.isRecording,
              nowRecording: newState.isRecording
            });
            return newState;
          }
          return { ...prev, lastUpdated: newState.lastUpdated };
        });
      }
    } catch (error) {
      console.error('[CameraView] Failed to poll hardware state:', error);
    }
  }, [cameraAccount, selectedCamera?.publicKey]);

  // Set up hardware state polling
  useEffect(() => {
    // Poll immediately
    pollHardwareState();
    
    // Then poll every 3 seconds for responsive UI
    hardwareStateInterval.current = setInterval(pollHardwareState, 3000);
    
    return () => {
      if (hardwareStateInterval.current) {
        clearInterval(hardwareStateInterval.current);
      }
    };
  }, [pollHardwareState]);

  return (
    <>
      <div className="h-full overflow-y-auto pb-40">
        <div className="relative max-w-3xl mx-auto pt-8">
          <ToastContainer message={currentToast} onDismiss={dismissToast} />

          {/* Transaction Modal for embedded wallets */}
          <TransactionModal
            isOpen={showTransactionModal}
            onClose={() => setShowTransactionModal(false)}
            transactionData={transactionData || undefined}
            onSuccess={({ transactionId }) => {
              setShowTransactionModal(false);

              // After successful check-in and action, refresh check-in status
              checkUserSession().then(() => {
                // Create a timeline event with the transaction ID
                if (transactionData) {
                  const eventType = getEventType(transactionData.type);

                  // Use the addTimelineEvent function for consistency
                  addTimelineEvent(eventType, transactionId);

                  // Show success message and toggle stream state if needed
                  updateToast('success', `${transactionData.type.charAt(0).toUpperCase() + transactionData.type.slice(1)} action recorded successfully`);

                  // Hardware state will be updated by polling automatically

                  // Refresh timeline to show latest events
                  if (timelineRef.current?.refreshTimeline) {
                    timelineRef.current?.refreshTimeline();
                  }
                }
              });
            }}
          />

          <div className="bg-white rounded-lg mb-6 px-6">
            <div className="py-4 flex justify-between items-center">
              <CameraIdDisplay
                cameraId={cameraId}
                selectedCamera={selectedCamera}
                cameraAccount={cameraAccount}
                timelineRef={timelineRef}
              />
            </div>
          </div>
          <div className="px-2">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="md:col-span-3 relative">
                <StreamPlayer />

                <div className="hidden sm:flex absolute -right-14 top-0 flex-col h-full z-[45]">
                  {/* Direct buttons for desktop */}
                  <div className="group h-1/2 relative">
                    <button
                      onClick={handleDirectStream}
                      disabled={loading}
                      className="w-16 h-full flex items-center justify-center hover:text-blue-600 text-black transition-colors rounded-xl"
                      aria-label={hardwareState.isStreaming ? "Stop Stream" : "Start Stream"}
                    >
                      {loading ? (
                        <Loader className="w-5 h-5 animate-spin" />
                      ) : isStreaming ? (
                        <StopCircle className="w-5 h-5" />
                      ) : (
                        <Play className="w-5 h-5" />
                      )}
                    </button>
                    <span className="absolute right-full mr-3 top-1/2 -translate-y-1/2 px-3 py-2 bg-black/75 text-white text-sm rounded opacity-0 group-hover:opacity-100 whitespace-nowrap transition-opacity">
                      {loading ? 'Processing...' : hardwareState.isStreaming ? 'Stop Stream' : 'Start Stream'}
                    </span>
                  </div>

                  <div className="group h-1/2 relative">
                    <button
                      onClick={handleDirectPhoto}
                      disabled={loading}
                      className="w-16 h-full flex items-center justify-center hover:text-blue-600 text-gray-800 transition-colors rounded-xl"
                    >
                      {loading ? (
                        <Loader className="w-5 h-5 animate-spin" />
                      ) : (
                        <Camera className="w-5 h-5" />
                      )}
                    </button>
                    <span className="absolute right-full mr-3 top-1/2 -translate-y-1/2 px-3 py-2 bg-black/75 text-white text-sm rounded opacity-0 group-hover:opacity-100 whitespace-nowrap transition-opacity">
                      {loading ? 'Processing...' : 'Take Picture'}
                    </span>
                  </div>

                  <div className="group h-1/2 relative">
                    <button
                      onClick={handleDirectVideo}
                      disabled={loading}
                      className="w-16 h-full flex items-center justify-center hover:text-blue-600 text-gray-800 transition-colors rounded-xl"
                    >
                      {loading ? (
                        <Loader className="w-5 h-5 animate-spin" />
                      ) : (
                        <Video className="w-5 h-5" />
                      )}
                    </button>
                    <span className="absolute right-full mr-3 top-1/2 -translate-y-1/2 px-3 py-2 bg-black/75 text-white text-sm rounded opacity-0 group-hover:opacity-100 whitespace-nowrap transition-opacity">
                      {loading ? 'Processing...' : 'Record Video'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div className="flex-1 mt-2 px-2">
            <CameraControls
              onTakePicture={handleDirectPhoto}
              onRecordVideo={handleDirectVideo}
              onToggleStream={handleDirectStream}
              isLoading={loading}
              isStreaming={isStreaming}
            />
          </div>
          <div className="max-w-3xl mt-6 mx-auto flex flex-col justify-top relative">
            <div className="relative mb-36">
              <div className="flex pl-6 items-center gap-2">
                {!cameraId && !cameraAccount && !selectedCamera ? (
                  <div className="flex items-center gap-2">
                    <span className="relative flex h-3 w-3 -ml-1">
                      {/* Proper prohibited symbol (🚫) */}
                      <span className="relative inline-flex rounded-full h-3 w-3 bg-white border border-gray-400"></span>
                      <span className="absolute inset-0 flex items-center justify-center">
                        <span className="h-[1.5px] w-2 bg-gray-500 rotate-45 absolute"></span>
                        <span className="h-[1.5px] w-2 bg-gray-500 -rotate-45 absolute"></span>
                      </span>
                    </span>
                    <span className="text-gray-500 font-medium">Disconnected</span>
                  </div>
                ) : !isLive ? (
                  <div className="flex items-center gap-2">
                    <span className="relative flex h-3 w-3 -ml-1">
                      <span className="relative inline-flex rounded-full h-3 w-3 bg-gray-400"></span>
                    </span>
                    <span className="text-gray-500 font-medium">Offline</span>
                  </div>
                ) : isStreaming ? (
                  <div className="flex items-center gap-2">
                    <span className="relative flex h-3 w-3 -ml-1">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
                    </span>
                    <span className="text-red-500 font-medium">LIVE</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <span className="relative flex h-3 w-3 -ml-1">
                      <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
                    </span>
                    <span className="text-green-500 font-medium">Online</span>
                  </div>
                )}
              </div>
            </div>

            <div className="absolute mt-12 pb-20 pl-5 left-0 w-full">
              <Timeline ref={timelineRef} variant="camera" cameraId={cameraAccount || undefined} />
              <div
                className="top-0 left-0 right-0 pointer-events-none"
                style={{
                  background: 'linear-gradient(to bottom, rgba(255,255,255,0) 0%, rgba(255,255,255,1) 100%)'
                }}
              />
            </div>

            <div className="relative md:ml-20 ml-16 bg-white">
              <div className="relative pl-4 pr-2 sm:px-4">
                <MediaGallery mode="recent" maxRecentItems={6} cameraId={cameraAccount || undefined} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
