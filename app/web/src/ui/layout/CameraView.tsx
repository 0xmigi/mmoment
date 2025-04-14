import { useDynamicContext, useEmbeddedWallet } from '@dynamic-labs/sdk-react-core';
import { useRef, useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { Timeline } from '../../timeline/Timeline';
import MediaGallery from '../../media/Gallery';
import { CameraControls } from './MobileControls';
import { ToastMessage } from '../../core/types/toast';
import { ToastContainer } from '../feedback/ToastContainer';
import { StreamPlayer } from '../../media/StreamPlayer';
import { useCamera, CameraData, fetchCameraByPublicKey } from '../../camera/CameraProvider';
import { PublicKey, SystemProgram, Connection } from '@solana/web3.js';
import { useProgram } from '../../anchor/setup';
import { cameraActionService } from '../../camera/camera-service';
import { useConnection } from '@solana/wallet-adapter-react';
import { Transaction } from '@solana/web3.js';
import { TransactionModal } from '../../auth/components/TransactionModal';
import { StopCircle, Play, Camera, Video, Loader, Link2 } from 'lucide-react';
import { useCameraStatus } from '../../camera/useCameraStatus';
import { CameraModal } from '../../camera/CameraModal';
import { cameraStatus } from '../../camera/camera-status';
import { CONFIG } from '../../core/config';

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

// Define the result type from wallet transaction
interface TransactionResult {
  signature: string;
}

// Update the CameraIdDisplay component to show "No camera connected" in red and make it clickable
const CameraIdDisplay = ({ cameraId, selectedCamera, cameraAccount }: { 
  cameraId: string | undefined; 
  selectedCamera: CameraData | null;
  cameraAccount: string | null;
}) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const cameraStatus = useCameraStatus(selectedCamera?.publicKey || cameraAccount || cameraId || '');
  
  // Get the direct ID from localStorage if available (most reliable source)
  const directId = localStorage.getItem('directCameraId');
  
  // Determine which ID to display (in order of preference)
  const displayId = selectedCamera?.publicKey || cameraAccount || directId || cameraId;
  
  // Handle case where ID might not be valid
  const formatId = (id: string | undefined | null) => {
    if (!id) return 'None';
    try {
      return `${id.slice(0, 6)}...${id.slice(-6)}`;
    } catch (e) {
      return id;
    }
  };
  
  // The default camera PDA for development
  const defaultDevCameraPda = '5onKAv5c6VdBZ8a7D11XqF79Hdzuv3tnysjv4B2pQWZ2';
  
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
          <Link2 className="w-3.5 h-3.5 ml-1.5 text-green-500" />
        </div>
      )}
      
      <CameraModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
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
  const timelineRef = useRef<any>(null);
  const [cameraAccount, setCameraAccount] = useState<string | null>(null);
  const [isLive, setIsLive] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentToast, setCurrentToast] = useState<ToastMessage | null>(null);
  const [loading, setLoading] = useState(false);
  const [, setIsMobileView] = useState(window.innerWidth <= 768);
  
  // Add state for transaction modal
  const [showTransactionModal, setShowTransactionModal] = useState(false);
  const [transactionData] = useState<{
    type: 'photo' | 'video' | 'stream' | 'initialize';
    cameraAccount: string;
  } | null>(null);
  

  // Add a function to create timeline events with Farcaster profile info
  const addTimelineEvent = (eventType: TimelineEventType, transactionId?: string, mediaUrl?: string) => {
    if (primaryWallet && user) {
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
      timelineRef.current?.addEvent(event);
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
      default:
        return 'photo_captured';
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

  // Update the Simple direct transaction function to return the signature
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
        
        // Create transaction
        const tx = new Transaction();
        
        // Get activity type based on action
        let activityTypeObj;
        switch(actionType) {
          case 'photo': 
            activityTypeObj = { photoCapture: {} };
            break;
          case 'video':
            activityTypeObj = { videoRecord: {} };
            break;
          case 'stream':
            activityTypeObj = { liveStream: {} };
            break;
          default:
            activityTypeObj = { photoCapture: {} };
        }
        
        // Create metadata
        const metadata = JSON.stringify({
          timestamp: new Date().toISOString(),
          action: actionType,
          cameraId
        });
        
        try {
          // Create transaction instruction
          console.log("Creating instruction with activityType:", activityTypeObj);
          const ix = await program.methods
            .recordActivity({
              activityType: activityTypeObj,
              metadata
            })
            .accounts({
              camera: new PublicKey(cameraId),
              owner: new PublicKey(primaryWallet.address),
              systemProgram: SystemProgram.programId,
            })
            .instruction();
          
          // Add instruction to transaction
          tx.add(ix);
          
          // Get recent blockhash with retry logic
          let blockhash;
          try {
            const { blockhash: newBlockhash } = await currentConnection.getLatestBlockhash('finalized');
            blockhash = newBlockhash;
          } catch (bhError) {
            console.error("Error getting blockhash:", bhError);
            // Try the next RPC endpoint
            const nextEndpoint = CONFIG.getNextEndpoint();
            console.log(`Switching to RPC endpoint: ${nextEndpoint}`);
            currentConnection = new Connection(nextEndpoint, 'confirmed');
            throw new Error('Failed to get blockhash, retrying with new endpoint');
          }
          
          tx.recentBlockhash = blockhash;
          tx.feePayer = new PublicKey(primaryWallet.address);
          
          // Sign and send transaction with timeout
          const signer = await (primaryWallet as any).getSigner();
          const signPromise = signer.signAndSendTransaction(tx);
          const timeoutPromise = new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Transaction timeout')), 30000)
          );
          
          const result = await Promise.race([signPromise, timeoutPromise]) as TransactionResult;
          console.log("Transaction sent:", result.signature);
          
          // Confirm transaction
          const confirmation = await currentConnection.confirmTransaction({
            signature: result.signature,
            blockhash,
            lastValidBlockHeight: await currentConnection.getBlockHeight()
          });
          
          if (confirmation.value.err) {
            throw new Error(`Transaction failed: ${confirmation.value.err}`);
          }
          
          // Show success toast
          updateToast('success', `${actionType} transaction sent: ${result.signature.slice(0, 8)}...`);
          
          // Add event to timeline
          addTimelineEvent(getEventType(actionType), result.signature);
          
          return result.signature;
          
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

  // Direct button handlers for testing
  const handleDirectPhoto = () => {
    console.log("DIRECT PHOTO BUTTON PRESSED");
    
    // Check if we have what we need to proceed
    if (!primaryWallet || !cameraAccount) {
      updateToast('error', 'Wallet or camera not connected');
      return;
    }
    
    setLoading(true);
    updateToast('info', 'Processing photo transaction...');
    
    // Add a small delay for visual feedback
    setTimeout(() => {
      sendSimpleTransaction('photo')
        .then(async (signature: string | undefined) => {
          if (signature && primaryWallet?.address) {
            updateToast('info', 'Capturing photo...');
            
            // Set camera status to online immediately for better UX
            cameraStatus.setOnline(false);
            
            const response = await cameraActionService.capturePhoto(signature, primaryWallet.address);
            if (response.success) {
              updateToast('success', 'Photo captured and uploaded to IPFS');
              // Ensure camera shows as online
              cameraStatus.setOnline(false);
            } else {
              updateToast('error', response.error || 'Failed to capture photo');
            }
          }
        })
        .catch((error: unknown) => {
          updateToast('error', `Transaction error: ${error instanceof Error ? error.message : 'Unknown error'}`);
        })
        .finally(() => {
          setLoading(false);
        });
    }, 100);
  };
  
  const handleDirectVideo = () => {
    console.log("DIRECT VIDEO BUTTON PRESSED");
    
    // Check if we have what we need to proceed
    if (!primaryWallet || !cameraAccount) {
      updateToast('error', 'Wallet or camera not connected');
      return;
    }
    
    setLoading(true);
    updateToast('info', 'Recording 30-second video...');
    
    // Add a small delay for visual feedback
    setTimeout(() => {
      sendSimpleTransaction('video')
        .then(async (signature: string | undefined) => {
          if (signature && primaryWallet?.address) {
            updateToast('info', 'Recording 30-second video...');
            
            // Set camera status to online immediately for better UX
            cameraStatus.setOnline(false);
            
            const response = await cameraActionService.recordVideo(signature, primaryWallet.address);
            if (response.success) {
              updateToast('success', '30-second video recorded and uploaded to IPFS');
              // Ensure camera shows as online
              cameraStatus.setOnline(false);
            } else {
              updateToast('error', response.error || 'Failed to record video');
            }
          }
        })
        .catch((error: unknown) => {
          updateToast('error', `Transaction error: ${error instanceof Error ? error.message : 'Unknown error'}`);
        })
        .finally(() => {
          setLoading(false);
        });
    }, 100);
  };
  
  const handleDirectStream = () => {
    console.log("DIRECT STREAM BUTTON PRESSED");
    
    // Check if we have what we need to proceed
    if (!primaryWallet || !cameraAccount) {
      updateToast('error', 'Wallet or camera not connected');
      return;
    }
    
    setLoading(true);
    
    // Clear the current toast before setting a new one
    setCurrentToast(null);
    
    // Use a more descriptive message
    if (isStreaming) {
      updateToast('info', 'Preparing to stop stream...');
    } else {
      updateToast('info', 'Preparing stream...');
    }
    
    if (isStreaming) {
      // For stopping the stream, create a transaction first, THEN call the camera API
      setTimeout(() => {
        sendSimpleTransaction('stream')
          .then(async (signature: string | undefined) => {
            if (signature && primaryWallet?.address) {
              // Update toast with transaction success
              updateToast('info', 'Stopping stream, transaction sent successfully');
              
              try {
                // Now call the API to actually stop the stream
                const response = await cameraActionService.stopStream(primaryWallet.address);
                if (response.success) {
                  setIsStreaming(false);
                  updateToast('success', 'Stream stopped and recorded on-chain');
                  // Update camera status to not streaming
                  cameraStatus.setOnline(false);
                } else {
                  updateToast('error', response.error || 'Failed to stop stream');
                }
              } catch (error) {
                handleStreamError(error);
              }
            }
          })
          .catch((error: unknown) => {
            updateToast('error', `Transaction error: ${error instanceof Error ? error.message : 'Unknown error'}`);
          })
          .finally(() => {
            setLoading(false);
          });
      }, 100);
    } else {
      // For starting the stream, we need a transaction
      setTimeout(() => {
        sendSimpleTransaction('stream')
          .then(async (signature: string | undefined) => {
            if (signature && primaryWallet?.address) {
              // Update toast with more specific info
              updateToast('info', 'Transaction confirmed, starting stream...');
              
              // Set camera status to online and streaming immediately for better UX
              cameraStatus.setOnline(true);
              
              try {
                const response = await cameraActionService.startStream(signature, primaryWallet.address);
                if (response.success) {
                  setIsStreaming(true);
                  
                  // Get stream info if available
                  const streamUrl = response.data?.streamInfo?.streamUrl;
                  const successMessage = streamUrl 
                    ? `Stream started successfully! Stream URL: ${streamUrl.split('/').pop()}`
                    : 'Stream started successfully!';
                  
                  // Replace the toast with success message
                  updateToast('success', successMessage);
                  
                  // Ensure camera shows as online and streaming
                  cameraStatus.setOnline(true);
                } else {
                  // Use the error handler for better messages
                  handleStreamError(response.error || 'Failed to start stream');
                }
              } catch (error) {
                handleStreamError(error);
              }
            }
          })
          .catch((error: unknown) => {
            // Use our error handler for transaction errors too
            handleStreamError(error);
          })
          .finally(() => {
            setLoading(false);
          });
      }, 100);
    }
  };

  // Add a useEffect to handle stream state changes
  useEffect(() => {
    // When streaming starts, clear any lingering "Starting stream..." toast
    if (isStreaming && currentToast?.message?.includes('Starting stream')) {
      dismissToast();
    }
  }, [isStreaming, currentToast]);

  // Helper function to handle stream errors more gracefully
  const handleStreamError = (error: any) => {
    // Extract more helpful information from stream errors
    let errorMessage = 'Unknown streaming error';
    
    if (error instanceof Error) {
      // Common stream errors with better messages
      if (error.message.includes('CORS')) {
        errorMessage = 'Stream blocked by CORS policy. This is a camera configuration issue.';
      } else if (error.message.includes('Failed to fetch')) {
        errorMessage = 'Stream server connection failed. Please check camera hardware.';
      } else if (error.message.includes('timeout')) {
        errorMessage = 'Stream timed out. Camera may be offline or busy.';
      } else {
        errorMessage = error.message;
      }
    } else if (typeof error === 'string') {
      errorMessage = error;
    }
    
    console.error('Stream error:', error);
    updateToast('error', `Stream error: ${errorMessage}`);
    
    // Reset streaming state
    setIsStreaming(false);
    cameraStatus.setOnline(false);
    setLoading(false);
  };

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
              
              // Create a timeline event with the transaction ID
              if (transactionData) {
                const eventType = getEventType(transactionData.type);
                
                // Use the addTimelineEvent function for consistency
                addTimelineEvent(eventType, transactionId);
                
                // Show success message and toggle stream state if needed
                updateToast('success', `${transactionData.type.charAt(0).toUpperCase() + transactionData.type.slice(1)} action recorded successfully`);
                
                // Toggle stream state if it's a stream action
                if (transactionData.type === 'stream') {
                  setIsStreaming(!isStreaming);
                }
              }
            }}
          />
          
          <div className="bg-white rounded-lg mb-6 px-6">
            <CameraIdDisplay 
              cameraId={cameraId}
              selectedCamera={selectedCamera}
              cameraAccount={cameraAccount}
            />
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
                      aria-label={isStreaming ? "Stop Stream" : "Start Stream"}
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
                      {loading ? 'Processing...' : isStreaming ? 'Stop Stream' : 'Start Stream'}
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
